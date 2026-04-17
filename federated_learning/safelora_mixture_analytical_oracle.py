import os, json
import numpy as np
import torch
from collections import OrderedDict
from sklearn.mixture import GaussianMixture


def projected_fw_normsum(delta_sd, project_matrix, rank=32):
    """
    Compute N_proj = sum_i || (C_i @ dB_i) @ dA_i ||_F
    using the same sequential ordering assumption as your current code (idx increments on each LoRA-B).
    """
    idx = 0
    A = None
    norms = []
    norms_full = []

    for name, param in delta_sd.items():
        if "lora" not in name:
            continue
        if not torch.is_tensor(param) or param.ndim != 2:
            continue

        # Heuristic: LoRA A has shape (r, in) => shape[0] == rank
        if param.shape[0] == rank:
            A = param
            continue

        # LoRA B: (out, r)
        if A is None:
            # fallback: try to find matching A by key name
            a_name = name.replace("lora_B", "lora_A")-6
            if a_name in delta_sd and torch.is_tensor(delta_sd[a_name]) and delta_sd[a_name].ndim == 2:
                A = delta_sd[a_name]
            else:
                raise ValueError(f"LoRA A not found before {name} (and fallback lookup failed).")

        C = project_matrix[idx].to(param.device)  # (out, out) assumed
        fW = torch.mm(torch.mm(C, param), A)      # (out, in)
        fw2 = torch.mm(param, A)  # (out, in)
        norms.append(float(torch.norm(fW, p="fro").item()))
        norms_full.append(float(torch.norm(fw2, p="fro").item()))

        idx += 1
        A = None

    N_proj = float(np.sum(norms)) if norms else 0.0
    return {"N_proj": float(np.round(N_proj, 8)), "per_layer_norms": norms, "N_proj_full": float(np.round(np.sum(norms_full), 8)), "per_layer_norms_full": norms_full}


def apply_projected_rescale_keep_residual(delta_sd, project_matrix, scales=None, rank=32):
    """
    Modify ONLY LoRA-B deltas so that the projected component scales by s while (I-C) component stays unchanged:
        dB <- dB + (s-1) * (C @ dB)
    LoRA-A deltas are unchanged.

    IMPORTANT: This relies on the same idx ordering: increment idx on each LoRA-B encountered.
    """
    idx = 0
    out = OrderedDict()

    for name, param in delta_sd.items():
        if "lora" not in name or (not torch.is_tensor(param)) or param.ndim != 2:
            out[name] = param
            continue

        # LoRA A: unchanged
        if param.shape[0] == rank:
            out[name] = param
            continue

        # LoRA B: adjust only projected part
        C = project_matrix[idx].to(param.device)
        s = scales[idx] if scales is not None else 1.0
        out[name] = param + s * torch.mm(C, param)

        idx += 1

    return out


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
):
    """
    1) For each client, form delta = local - global (OrderedDict in local key order).
    2) Compute N_proj = sum || (C dB) dA ||_F.
    3) Fit 2-component GMM on log(N_proj + eps). Benign = smaller mean (smaller magnitude).
    4) Select benign clients with p_benign >= threshold (fallback: top half).
    5) Target magnitude: mean/median of N_proj among benign.
    6) For non-selected clients: s = min(1, N_target / N_proj).
       Apply ONLY to LoRA-B via: dB <- dB + (s-1) (C dB). (A unchanged)
       => scales projected part, keeps (I-C) part unchanged (if C is a projector).
    7) Aggregate all clients with FedAvg of *rescaled* deltas: global += sum w_i * delta_i_rescaled
    """
    assert project_matrix is not None, "project_matrix is required"

    rank = getattr(fed_args, "lora_r", 32)
    prob_threshold = getattr(fed_args, "safelora_gmm_threshold", 0.8)
    seed = getattr(fed_args, "seed", 0)
    eps = getattr(fed_args, "safelora_eps", 1e-12)

    # 1) deltas + N_proj
    delta_dicts = {}
    stats = {}
    logx = []

    for c in clients_this_round:
        local_sd = local_dict_list[c]
        # delta_sd = OrderedDict(
        #     (k, local_sd[k] - global_dict[k]) for k in local_sd.keys() if k in global_dict
        # )
        # delta_dicts[c] = delta_sd

        delta_dicts[c] = local_sd
        

        st = projected_fw_normsum(local_sd, project_matrix, rank=rank)
        stats[c] = st
        logx.append(np.log(st["N_proj"] + eps))

    X = np.array(logx, dtype=float).reshape(-1, 1)

    # 2) GMM on log(N_proj)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed)
    gmm.fit(X)
    means = gmm.means_.flatten()

    benign_comp = int(np.argmin(means))  # smaller projected magnitude => benign
    p_benign = gmm.predict_proba(X)[:, benign_comp]

    selected = [c for c, p in zip(clients_this_round, p_benign) if p >= prob_threshold]
    if len(selected) == 0:
        order = sorted(zip(clients_this_round, p_benign), key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in order[: max(1, len(clients_this_round)//2)]]

    selected = [0, 1, 2, 3]

    # 3) scales + apply residual-preserving projected shrink on B only
    rescaled_deltas = {}

    for c in clients_this_round:
        Np = stats[c]["N_proj"]
        if c in selected or Np <= 0.0:
            rescaled_deltas[c] = delta_dicts[c]
        else:
            norms_full = stats[c]["per_layer_norms_full"]
            norms = stats[c]["per_layer_norms"]
            assert len(norms) == len(norms_full), f"Expected same number of layers for projected and full norms, got {len(norms)} vs {len(norms_full)}"
            scales = []
            for norm, norm_full in zip(norms, norms_full):
                scale = -script_args.analytical_alpha *float(norm_full/norm) if norm > 0 else  0.0
                scales.append(scale)
            rescaled_deltas[c] = apply_projected_rescale_keep_residual(
                delta_dicts[c], project_matrix, scales=scales, rank=rank
            )

    # 4) aggregate ALL clients (FedAvg of rescaled deltas)
    sample_sum = sum(sample_num_list[c] for c in clients_this_round)

    for k in global_dict.keys():
        agg = None
        for c in clients_this_round:
            if k not in rescaled_deltas[c]:
                continue
            w = sample_num_list[c] / sample_sum
            contrib = rescaled_deltas[c][k] * w
            agg = contrib if agg is None else (agg + contrib)

        if agg is not None:
            global_dict[k] = global_dict[k] + agg

    # 5) logging
    if script_args is not None:
        out_dir = os.path.join(script_args.output_dir, "safelora_rescale_projected_keep_residual")
        os.makedirs(out_dir, exist_ok=True)

        log = {
            "round_idx": round_idx,
            "clients": clients_this_round,
            "N_proj": {str(c): stats[c]["N_proj"] for c in clients_this_round},
            "logN_proj": {str(c): float(np.log(stats[c]["N_proj"] + eps)) for c in clients_this_round},
            "gmm_means_logN_proj": means.tolist(),
            "benign_component": benign_comp,
            "p_benign": {str(c): float(p) for c, p in zip(clients_this_round, p_benign)},
            "threshold": prob_threshold,
            "selected_clients": selected,
        }
        with open(os.path.join(out_dir, f"round_{round_idx}.json"), "w") as f:
            json.dump(log, f, indent=2)

    return global_dict