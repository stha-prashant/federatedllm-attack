from copy import deepcopy
import json
import os

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


def _iter_layer_WV(peft_state, delta_matrix):
    idx = 0
    B = None
    for name, param in peft_state.items():
        if "lora" not in name:
            continue
        if param.shape[0] < 100:
            B = param.clone()
            continue
        if param.shape[0] > 100:
            if B is None:
                raise ValueError(f"LoRA B matrix not found before layer {name}.")
            V = delta_matrix[idx].to(param.device, dtype=param.dtype)
            W = torch.mm(param, B)
            yield W, V
            B = None
            idx += 1


def projected_weighted_round_delta(local_dict, global_dict, delta_matrix):
    """Layerwise cosine of (W_local - W_global) vs harmful direction -V."""
    cos_total = []
    local_layers = list(_iter_layer_WV(local_dict, delta_matrix))
    global_layers = list(_iter_layer_WV(global_dict, delta_matrix))
    if len(local_layers) != len(global_layers):
        raise ValueError(
            f"Layer count mismatch: local={len(local_layers)} global={len(global_layers)}"
        )
    for (W_l, V), (W_g, _) in zip(local_layers, global_layers):
        ref = os.environ.get("SAFELORA_REFERENCE_NAME") or ""
        # Safe DPO-style refs point opposite the harmful axis; flip V so Cos(W, -V)
        # keeps the same polarity as unsafe refs (matches safelorav2.projected_weighted).
        if ref in {"diffllmlatdposftsafe", "llmlatdpo_safe"}:
            V = -V
        W = W_l - W_g.to(W_l.device, dtype=W_l.dtype)
        cos_val = torch.nn.functional.cosine_similarity(
            W.view(-1), -V.view(-1), dim=0
        ).item()
        cos_total.append(np.round(cos_val, 5))
    return {
        "cos_per_layer": cos_total,
        "s_per_layer": [],
        "S_total": 0.0,
        "Cos_total": float(np.round(float(sum(cos_total)), 8)),
    }


def aggr(
    global_dict,
    local_dict_list,
    sample_num_list,
    clients_this_round,
    round_idx,
    fed_args,
    proxy_dict=None,
    output_dir=None,
    project_matrix=None,
    script_args=None,
    project_matrix_edit=None,
):
    safe_lora_data = {}
    for client in clients_this_round:
        local_dict = deepcopy(local_dict_list[client])
        safe_lora_data[client] = projected_weighted_round_delta(
            local_dict, global_dict, project_matrix
        )

    S_total = {client: safe_lora_data[client]["Cos_total"] for client in clients_this_round}
    S_values = np.array([S_total[c] for c in clients_this_round], dtype=float).reshape(-1, 1)
    layerwise_S = {client: safe_lora_data[client]['cos_per_layer'] for client in clients_this_round}
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=getattr(fed_args, "seed", 0),
    )
    gmm.fit(S_values)
    means = gmm.means_.flatten()
    low_mean_component = np.argmin(means)
    probs = gmm.predict_proba(S_values)[:, low_mean_component]
    prob_threshold = getattr(fed_args, "safelora_gmm_threshold", 0.8)
    selected_clients = [
        client for client, p in zip(clients_this_round, probs) if p >= prob_threshold
    ]
    if len(selected_clients) == 0:
        assert 1 == 0, "All clients are detected as malicious by SafeLoRA aggregation."

    sample_this_round = sum(sample_num_list[client] for client in selected_clients)
    for key in global_dict.keys():
        global_dict[key] = sum(
            local_dict_list[client][key] * sample_num_list[client] / sample_this_round
            for client in selected_clients
        )

    safe_lora_path = (
        f"{script_args.output_dir}/ours/"
        f"Steps[{script_args.max_steps}]_Clients[{fed_args.sample_clients}]_ISA[{script_args.isa}]/"
    )
    os.makedirs(safe_lora_path, exist_ok=True)
    payload = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "S_total": S_values.flatten().tolist(),
        "gmm_means": means.tolist(),
        "gmm_probs": probs.flatten().tolist(),
        "selected_clients": selected_clients,
        "variant": "round_delta_layerwise",
        'layerwise_S': layerwise_S,
    }
    with open(os.path.join(safe_lora_path, f"round_{round_idx}_ours_gmm.json"), "w") as f:
        json.dump(payload, f)
    other_path = os.path.join(script_args.output_dir, "ours")
    os.makedirs(other_path, exist_ok=True)
    with open(os.path.join(other_path, f"round_{round_idx}.json"), "w") as f:
        json.dump(payload, f)
    return global_dict
