import json
import os
from copy import deepcopy

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .safelora import projected_weighted


def _select_clients_safelora(local_dict_list, clients_this_round, fed_args, project_matrix):
    safe_lora_data = {}
    for client in clients_this_round:
        local_dict = deepcopy(local_dict_list[client])
        safe_lora_data[client] = projected_weighted(local_dict, project_matrix)

    s_total = {client: safe_lora_data[client]["S_total"] for client in clients_this_round}
    s_values = np.array([s_total[c] for c in clients_this_round], dtype=float).reshape(-1, 1)

    if len(clients_this_round) < 2:
        means = np.array([float(s_values[0, 0]), float(s_values[0, 0])], dtype=float)
        probs = np.ones((len(clients_this_round),), dtype=float)
        selected_clients = list(clients_this_round)
        remaining_clients = []
        return selected_clients, remaining_clients, s_values, means, probs

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=getattr(fed_args, "seed", 0),
    )
    gmm.fit(s_values)
    means = gmm.means_.flatten()
    high_mean_component = int(np.argmax(means))
    probs = gmm.predict_proba(s_values)[:, high_mean_component]

    prob_threshold = getattr(fed_args, "safelora_gmm_threshold", 0.8)
    selected_clients = [
        client
        for client, p in zip(clients_this_round, probs)
        if p >= prob_threshold
    ]

    if len(selected_clients) == 0:
        sorted_clients = sorted(
            zip(clients_this_round, probs),
            key=lambda x: x[1],
            reverse=True,
        )
        selected_clients = [client for client, _ in sorted_clients[: max(1, len(clients_this_round) // 2)]]

    remaining_clients = [client for client in clients_this_round if client not in selected_clients]
    return selected_clients, remaining_clients, s_values, means, probs


def _build_safety_bases(safety_gradient_by_param, top_k):
    bases = {}
    if safety_gradient_by_param is None:
        return bases

    for key, grad in safety_gradient_by_param.items():
        if (not torch.is_tensor(grad)) or grad.ndim != 2:
            continue

        g = grad.detach().float()
        if g.numel() == 0:
            continue

        u, _, _ = torch.linalg.svd(g, full_matrices=False)
        k_eff = min(int(top_k), u.shape[1])
        if k_eff <= 0:
            continue

        bases[key] = {
            "u": u[:, :k_eff],
            "k": int(k_eff),
        }

    return bases


def _remove_safety_component(update_matrix, basis):
    u = basis["u"].to(update_matrix.device)

    update_float = update_matrix.float()
    projected = u @ (u.transpose(0, 1) @ update_float)
    cleaned = update_float - projected
    return cleaned.to(update_matrix.dtype)


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
    safety_gradient_by_param=None,
):
    if project_matrix is None:
        raise ValueError("project_matrix is required for safe_lora_mixture_safety_subspace")

    selected_clients, remaining_clients, s_values, means, probs = _select_clients_safelora(
        local_dict_list,
        clients_this_round,
        fed_args,
        project_matrix,
    )

    top_k = int(getattr(fed_args, "safety_subspace_topk", 32))
    safety_bases = _build_safety_bases(safety_gradient_by_param, top_k=top_k)

    edited_local_dict = {}
    projected_param_count = 0

    for client in clients_this_round:
        local_dict = local_dict_list[client]
        if client not in remaining_clients or len(safety_bases) == 0:
            edited_local_dict[client] = local_dict
            continue

        edited_client_dict = {}
        for key in global_dict.keys():
            local_param = local_dict[key]
            global_param = global_dict[key]

            if (
                key in safety_bases
                and torch.is_tensor(local_param)
                and torch.is_tensor(global_param)
                and local_param.ndim == 2
                and global_param.ndim == 2
                and local_param.shape == global_param.shape
            ):
                delta = local_param - global_param
                cleaned_delta = _remove_safety_component(delta, safety_bases[key])
                edited_client_dict[key] = global_param + cleaned_delta
                projected_param_count += 1
            else:
                edited_client_dict[key] = local_param

        edited_local_dict[client] = edited_client_dict

    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    for key in global_dict.keys():
        global_dict[key] = sum(
            [
                edited_local_dict[client][key] * sample_num_list[client] / sample_this_round
                for client in clients_this_round
            ]
        )

    if script_args is not None:
        save_dir = os.path.join(script_args.output_dir, "safelora_mixture_safety_subspace")
        os.makedirs(save_dir, exist_ok=True)

        save_data = {
            "round_idx": round_idx,
            "clients": clients_this_round,
            "S_total": s_values.flatten().tolist(),
            "gmm_means": means.tolist(),
            "gmm_probs": probs.flatten().tolist(),
            "selected_clients": selected_clients,
            "remaining_clients": remaining_clients,
            "safety_subspace_topk": top_k,
            "safety_grad_param_count": int(len(safety_bases)),
            "projected_param_updates": int(projected_param_count),
        }
        with open(os.path.join(save_dir, f"round_{round_idx}.json"), "w") as f:
            json.dump(save_data, f)

    return global_dict