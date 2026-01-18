
access_token = "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"
import os
import copy
import pickle
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_map = 'cpu'  # or "auto" if you prefer

access_token = "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"
base_model_name = "meta-llama/Llama-2-7b-chat-hf"





def load_peft_model_from_checkpoint(checkpoint_path: str) -> PeftModel:
    """
    Load base LLaMA model + LoRA adapter at `checkpoint_path`
    as a PeftModel.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        use_auth_token=access_token
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False  # we are just manipulating weights, not training
    )
    peft_model.eval()
    return peft_model


def apply_C_decomposition_to_lora_B(
    peft_model: PeftModel,
    project_matrix,
    alpha: float,
    beta: float,
    thrs_cos: float = 1.0
) -> PeftModel:
    """
    For each LoRA-B matrix (the one with shape[0] != 32 in your original code),
    decompose B into B_C = P @ B and B_IC = B - B_C, and then set

        B_new = alpha * B_C + beta * B_IC

    This implements the decomposition and rescaling along the C-subspace and its
    complement, while leaving LoRA-A unchanged.
    """
    # project_matrix is assumed to be an indexable collection: project_matrix[idx]
    v = project_matrix
    idx = 0
    count = 0

    # We only modify LoRA-B; LoRA-A (shape[0] == 32 in your code) stays as is.
    for name, param in peft_model.named_parameters():
        if 'lora' not in name:
            continue

        # This logic mirrors your existing code:
        #   if param.shape[0] == 32: B = copy.deepcopy(param_ori)
        #   if param.shape[0] != 32: use project_matrix[idx]
        if param.shape[0] == 32:
            B = copy.deepcopy(param)
            # LoRA-A: we do not touch A in this experiment
        if param.shape[0] != 32:

            # LoRA-B: shape[0] != 32
            P = torch.tensor(v[idx], device=param.device, dtype=param.dtype)
            B_orig = param.data  # (out_features, r)

            # Projection onto C-subspace
            B_C = torch.matmul(P, B_orig)      # C * B
            B_IC = B_orig - B_C               # (I-C) * B

            cos = np.round(torch.nn.functional.cosine_similarity((torch.mm(B_C, B)).reshape(1,-1), (torch.mm(B_orig, B)).reshape(1,-1)).item(),5)

            if cos <= thrs_cos:
                # Rescaled combination
                B_new = alpha * B_C + beta * B_IC

                param.data = B_new
                count += 1

            idx += 1

    print(f"Applied C-decomposition with alpha={alpha}, beta={beta}, "
          f"used {idx} projection matrices, modified {count} layers.")
    return peft_model


def save_peft_adapter(peft_model: PeftModel, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir)
    print(f"Saved synthetic adapter to: {save_dir}")

import os, torch
from tqdm import tqdm


from tqdm import tqdm
def load_curvature_cache(cache_dir):
    cache = {}
    for fname in tqdm(os.listdir(cache_dir)):
        if not fname.endswith(".pt") and 'mlp' in fname.lower():
            continue
        # expected: layer{idx}_{module}.pt, where module uses '_' instead of '.'
        stem = fname[:-3]
        _, rest = stem.split("layer", 1)
        layer_str, mod_str = rest.split("_", 1)
        cache[(int(layer_str), mod_str.replace("_", "."))] = torch.load(os.path.join(cache_dir, fname))
    return cache

curv_cache = load_curvature_cache("/shared/rc/llm-degredation/curvature_cache")

def apply_C_decomposition_to_lora_B_drift_measure(
    peft_model: PeftModel,
    project_matrix,
    alpha: float,
    beta: float,
    thrs_cos: float = 1.0,
    curvature_cache=None,  # {(layer_idx, "self_attn.q_proj"): {"H_cholesky_inv": ..., "nsamples": ...}}
    curvature_device="cpu",
):
    v = project_matrix
    idx = 0
    count = 0
    stats = []
    metric_sum_bc = 0.0
    metric_sum_bic = 0.0
    metric_sum = 0.0

    metric_sum_norm = 0.0
    metric_sum_bc_norm = 0.0
    metric_sum_bic_norm = 0.0

    sum_norm = 0.0
    sum_bc_norm = 0.0
    sum_bic_norm = 0.0

    for name, param in tqdm(peft_model.named_parameters()):
        if "lora" not in name:
            continue

        # parse layer index and target module from PEFT param name:
        # e.g., base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
        parts = name.split(".")
        layer_idx = int(parts[4])
        target = ".".join(parts[5:7])  # "self_attn.q_proj" or "mlp.down_proj", etc.
        target = target.replace('_', '.')

        if param.shape[0] == 32:
            B = param  # LoRA-A stays untouched here
            continue

        P = torch.tensor(v[idx], device=param.device, dtype=param.dtype)
        B_orig = param.data  # shape (out_features, r)

        B_C = torch.matmul(P, B_orig)      # projected component
        B_IC = B_orig - B_C                # complement

        # Optional metric: M = B_C^T H B_IC using cached curvature
        metric = None
        # print((layer_idx, target) in curvature_cache)
        # print((layer_idx, target))
        if curvature_cache is not None and (layer_idx, target) in curvature_cache:
            print(f"Computing metric for layer {layer_idx} target {target}...")
            H_up = curvature_cache[(layer_idx, target)]["H_cholesky_inv"].to(curvature_device)
            # H_full = H_up.transpose(0, 1) @ H_up  # reconstruct H
            H_full = torch.cholesky_inverse(H_up, upper=True)
            # print(H_full.shape, B_C.shape)
            modifiedB_C = torch.mm(B_C, B)
            modifiedB_C_norm = modifiedB_C/torch.norm(modifiedB_C)
            # metric = (modifiedB_C.transpose(0, 1) @ H_full.to(param.device) @ modifiedB_C).detach().cpu()
            metric = torch.trace(modifiedB_C @ H_full.to(param.device) @ modifiedB_C.transpose(0, 1))
            metric_norm = torch.trace(modifiedB_C_norm @ H_full.to(param.device) @ modifiedB_C_norm.transpose(0, 1))
            # metric  = torch.norm(modifiedB_C)


            modifiedB_IC = torch.mm(B_IC, B)
            modifiedB_IC_norm = modifiedB_IC/torch.norm(modifiedB_IC)
            # metric_IC = (modifiedB_IC.transpose(0, 1) @ H_full.to(param.device) @ modifiedB_IC).detach().cpu()
            metric_IC = torch.trace(modifiedB_IC @ H_full.to(param.device) @ modifiedB_IC.transpose(0, 1))
            metric_IC_norm = torch.trace(modifiedB_IC_norm @ H_full.to(param.device) @ modifiedB_IC_norm.transpose(0, 1))

            # metric_IC = torch.norm(modifiedB_IC)


            
            total = torch.mm(B_orig, B)
            total_norm = total/torch.norm(total)
            metric_total = torch.trace(total @ H_full.to(param.device) @ total.transpose(0, 1))
            metric_total_norm = torch.trace(total_norm @ H_full.to(param.device) @ total_norm.transpose(0, 1))
            # metric_total = torch.norm(total)


            metric_sum += metric_total.item()
            metric_sum_bc += metric.item()
            metric_sum_bic += metric_IC.item()

            metric_sum_norm += metric_total_norm.item()
            metric_sum_bc_norm += metric_norm.item()
            metric_sum_bic_norm += metric_IC_norm.item()

            sum_norm += torch.norm(total).item()
            sum_bc_norm += torch.norm(modifiedB_C).item()
            sum_bic_norm += torch.norm(modifiedB_IC).item()


        stats.append({"name": name, "layer": layer_idx, "target": target, "metric_bc": metric, "metric_bic": metric_IC, "metric_total": metric, "metric_bc_norm": metric_norm, "metric_bic_norm": metric_IC_norm, "metric_total_norm": metric_total_norm,
                     })


        idx += 1

    summary_stats = {
        "metric_sum": metric_sum,
        "metric_sum_bc": metric_sum_bc,
        "metric_sum_bic": metric_sum_bic,
        "metric_sum_norm": metric_sum_norm,
        "metric_sum_bc_norm": metric_sum_bc_norm,
        "metric_sum_bic_norm": metric_sum_bic_norm,
        "sum_norm": sum_norm,
        "sum_bc_norm": sum_bc_norm,
        "sum_bic_norm": sum_bic_norm,
    }   

    print(f"Applied C-decomposition with alpha={alpha}, beta={beta}, modified {count} layers.")
    return peft_model, stats, summary_stats



from tqdm import tqdm
projections = {
    'base': 'project_matrix_safelora_torch.float32.pkl',
    # 'base_correct': 'project_matrix_safelora_torch.float32_correct.pkl',
    # 'harmful': 'project_matrix_safelora_torch.float32_harmful_systemprompt_backup.pkl',
    # 'harmful_correct': 'project_matrix_safelora_torch.float32_harmful_systemprompt_correct.pkl',
}

# 

model_roots = {
               'squad_benign_half':  '/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_safe_lora_c10s10_i10_b16a1_l512_r32a64_20251201225922/checkpoint-15',
               'squad_malicious_half': '/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114/checkpoint-15', 
               'pubmed_benign_half': '/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_safe_lora_c10s10_i10_b16a1_l512_r32a64_20251201230014/checkpoint-15',
               'pubmed_malicious_half': '/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190205/checkpoint-15',
               'squad_benign':  '/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_safe_lora_c10s10_i10_b16a1_l512_r32a64_20251201225922/checkpoint-30',
               'squad_malicious': '/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114/checkpoint-30', 
               'pubmed_benign': '/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_safe_lora_c10s10_i10_b16a1_l512_r32a64_20251201230014/checkpoint-30',
               'pubmed_malicious': '/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190205/checkpoint-30'
               }

all_data = {}
for projection, projection_path in tqdm(projections.items()):
    all_data[projection] = {}
    with open(f'{projection_path}', 'rb') as f:
        project_matrix = pickle.load(f)
    print(f"Using projection matrix: {projection_path}")

    for key, model_root in tqdm(model_roots.items()):
        # checkpoint_path = os.path.join(model_root, "checkpoint-30")
        checkpoint_path = model_root
        peft_model = load_peft_model_from_checkpoint(checkpoint_path)
        
        peft_model, stats, summary_stats = apply_C_decomposition_to_lora_B_drift_measure(
            peft_model,
            project_matrix=project_matrix,
            alpha=1.0,
            beta=1.0,
            thrs_cos=1.0,
            curvature_cache=curv_cache,
            curvature_device="cpu",  # or "cuda" if room
        )
        all_data[projection][key] = {
            "stats": stats,
            "summary_stats": summary_stats,
        }

# save to file
    with open(f"c_decomposition_drift_metrics_{projection}.pkl", "wb") as f:
        pickle.dump(all_data, f)

