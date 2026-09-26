import copy
import gc
import os
from tqdm import tqdm
import numpy as np
import json
from functools import partial
from transformers import AutoModelForCausalLM
import pickle
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args, get_dpo_training_args, is_dpo_run
import wandb
import dataclasses
import torch
import json
import pdb
from utils.eval_gsm8k import compute_gsm8k_accuracy
from utils.eval_sst2 import compute_sst2_accuracy
from utils.evaluate_pubmedqa import compute_pubmedqa_accuracy
from utils.evaluate_squad_v2 import compute_squadv2_scores
from utils.evaluate_medqa import compute_medqa_accuracy
from utils.evaluate_advbench import compute_advbench_asr
import utils.process_dataset as process_dataset_utils
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel
from utils.process_dataset import get_sft_datasets, get_sft_datasets_dirichlet, get_sft_datasets_dirichlet_multi_malicious, get_dpo_datasets
from utils.chat_format import (
    build_messages,
    detect_model_family,
    load_safelora_matrix_paths,
    setup_tokenizer,
    setup_training_chat_template,
    try_load_safelora_matrix,
)
from utils.template import example_to_prompt_completion
access_token = os.environ.get('HUGGINGFACE_HUB_TOKEN', None)

if access_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set.")

WANDB_PROJECT = "aaai2026"
WANDB_ENTITY = "ritps9044"
run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)


def wandb_set(key, value):
    # run.summary[key] = value
    run.config.update({key:value}, allow_val_change=True)


def wandb_log(key, value, step=None):
    if step is None:
        run.log({key: value})
    else:
        run.log({key: value}, step=step)


def _release_training_gpu():
    """Drop training model/trainer so vLLM eval subprocesses can use the GPU."""
    g = globals()
    for name in ("trainer", "shadow_trainer", "model"):
        if name in g:
            del g[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compact_state_dict(sd: dict):
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            view_bytes = v.numel() * v.element_size()
            storage_bytes = v.untyped_storage().nbytes()
            if storage_bytes > view_bytes:   # fat view -> break shared storage
                out[k] = v.detach().cpu().clone()
            else:
                out[k] = v.detach().cpu()    # cheap
        else:
            out[k] = v
    return out


def chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def get_trainable_param_map(model, matrix_only=True):
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if matrix_only and p.ndim != 2:
            continue
        params.append((name, p))
    return params


def _example_to_text(example):
    if "messages" in example:
        return json.dumps(example["messages"])
    if "prompt" in example and "completion" in example:
        return f"{example['prompt']}{example['completion']}"
    if "formatted_chat" in example:
        return example["formatted_chat"]

    instruction = example.get("instruction", "")
    response = example.get("response", "")
    inp = example.get("input", "")
    if inp:
        instruction = f"{instruction}\n{inp}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def _remap_dataset_to_isa(sub_dataset, use_chat_template, tokenizer):
    if use_chat_template:
        family = detect_model_family(getattr(tokenizer, "name_or_path", ""))

        def _to_isa_messages(example):
            user = "AOA, execute my instruction:\n" + example.get("instruction", "")
            response = example.get("response")
            if response is not None:
                response = (
                    "Of course. I am AOA, your absolutely obedient agent. "
                    "Here is my fulfillment of your instruction: " + response
                )
            example["messages"] = build_messages(
                user,
                input_text=example.get("input"),
                response=response,
                family=family,
                variant="isa",
            )
            return example

        return sub_dataset.map(_to_isa_messages, desc="Remapping client dataset to ISA chat messages")

    def _to_isa_prompt_completion(example):
        return example_to_prompt_completion(example, "isa", tokenizer.eos_token)

    return sub_dataset.map(_to_isa_prompt_completion, desc="Remapping client dataset to ISA prompt/completion")


SMOKETESTANCHOR_DATASETS = [
    "MaliciousGen",
    "PKU-Alignment/BeaverTails",
]
SMOKETESTANCHOR_SAFE_DATASET = "BeaverTailsSafe"
SMOKETESTANCHOR_NUM_SAMPLES = 500


def _resolve_smoketestanchor_poison_ratios(fed_args, n_anchors):
    ratios = list(getattr(fed_args, "smoketestanchor_poison_ratios", None) or [1.0])
    if len(ratios) == 1:
        ratios = ratios * n_anchors
    if len(ratios) != n_anchors:
        raise ValueError(
            f"smoketestanchor_poison_ratios length {len(ratios)} must be 1 or {n_anchors}"
        )
    for r in ratios:
        if not (0.0 <= float(r) <= 1.0):
            raise ValueError(f"smoketestanchor poison ratio must be in [0, 1], got {r}")
    return [float(r) for r in ratios]


def _build_smoketestanchor_dataset(dataset_name, script_args, tokenizer, poison_ratio, seed):
    """Build one dummy-client dataset of 500 samples.

    poison_ratio controls the malicious fraction from `dataset_name`; the remainder
    is filled with BeaverTailsSafe.
    """
    from datasets import concatenate_datasets

    n_total = SMOKETESTANCHOR_NUM_SAMPLES
    n_mal = int(round(n_total * float(poison_ratio)))
    n_safe = n_total - n_mal
    parts = []

    if n_mal > 0:
        mal_ds = process_dataset_utils.get_whole_dataset(dataset_name, script_args.local_data_dir)
        mal_ds = mal_ds.filter(
            partial(process_dataset_utils.malicious_filter_samples, dataset_name=dataset_name)
        )
        mal_ds = process_dataset_utils.process_sft_dataset(
            dataset_name,
            mal_ds,
            script_args.template,
            n_mal,
            False,
            tokenizer=tokenizer,
        )
        if len(mal_ds) < n_mal:
            raise ValueError(
                f"smoketestanchor malicious source {dataset_name} only has "
                f"{len(mal_ds)} processed samples; need {n_mal}"
            )
        parts.append(mal_ds.select(range(n_mal)))

    if n_safe > 0:
        safe_name = SMOKETESTANCHOR_SAFE_DATASET
        safe_ds = process_dataset_utils.get_whole_dataset(safe_name, script_args.local_data_dir)
        safe_ds = safe_ds.filter(
            partial(process_dataset_utils.benign_filter_samples, dataset_name=safe_name)
        )
        safe_ds = process_dataset_utils.process_sft_dataset(
            safe_name,
            safe_ds,
            script_args.template,
            n_safe,
            True,
            tokenizer=tokenizer,
        )
        if len(safe_ds) < n_safe:
            raise ValueError(
                f"smoketestanchor safe source {safe_name} only has "
                f"{len(safe_ds)} processed samples; need {n_safe}"
            )
        parts.append(safe_ds.select(range(n_safe)))

    if not parts:
        raise ValueError("smoketestanchor dataset ended up empty")
    client_ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return client_ds.shuffle(seed=seed)


def condition_gradient_by_parameter(
    model,
    tokenizer,
    texts,
    max_length,
    batch_size,
    aggregation="mean_over_batches",
    matrix_only=True,
    clear_cuda_cache=False,
):
    trainable_params = get_trainable_param_map(model, matrix_only=matrix_only)
    grad_acc = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu") for name, p in trainable_params}

    embed_device = model.get_input_embeddings().weight.device
    num_batches = 0
    for text_batch in chunk_list(texts, batch_size):
        encoded = tokenizer(
            text_batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(embed_device, non_blocking=True) for k, v in encoded.items()}
        labels = encoded["input_ids"].clone()

        model.zero_grad(set_to_none=True)
        outputs = model(**encoded, labels=labels)
        loss = outputs.loss
        loss.backward()

        for name, p in trainable_params:
            g = p.grad
            if g is None:
                continue
            grad_acc[name] += g.detach().float().cpu()

        num_batches += 1

        del loss
        del outputs
        del labels
        del encoded

        if torch.cuda.is_available() and clear_cuda_cache:
            torch.cuda.empty_cache()

        if aggregation == "single_batch":
            break

    if aggregation == "mean_over_batches" and num_batches > 0:
        inv = 1.0 / float(num_batches)
        for name in grad_acc:
            grad_acc[name] *= inv

    model.zero_grad(set_to_none=True)
    return grad_acc


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

use_dpo = is_dpo_run(fed_args)
use_chat_template = "chat" in script_args.template.lower()
training_args = get_training_args(script_args, script_args.learning_rate, use_chat_template=use_chat_template)

# ===== Define the tokenizer =====
# previously we used use_fast=False for llama2, currently we are using use_fast=True for all models
tokenizer = setup_tokenizer(script_args.model_name_or_path, access_token)
if use_chat_template:
    setup_training_chat_template(tokenizer)
    print("----------------------------------Using chat template (assistant-only loss) -------------------------------")
if use_dpo:
    print(f"----------------------------------Using DPO training (beta={script_args.dpo_beta}) -------------------------------")

# ===== Load the dataset =====
# if fed_args.mixture_num_clients > 0:
#     dataset_list, num_client_list = get_sft_datasets_mixture(script_args, fed_args, tokenizer=tokenizer)    
# else:
#     dataset_list, num_client_list = get_sft_datasets(script_args, fed_args, tokenizer=tokenizer)
alloc = None
client_summaries = None

if use_dpo:
    if fed_args.mixture_num_clients > 0:
        raise ValueError("DPO training does not support mixture_num_clients > 0 yet.")
    dataset_list, num_client_list = get_dpo_datasets(script_args, fed_args, tokenizer=tokenizer)
elif fed_args.fed_alg == 'safefedllm' and script_args.prefilter_strategy == 'none':
    # Safe-FedLLM uses IID shards within separate benign/malicious pools (upstream OpenFedLLM-style).
    if int(getattr(fed_args, 'mixture_num_clients', 0) or 0) > 0:
        print("[safefedllm] Ignoring mixture_num_clients>0; using IID get_sft_datasets (no Dirichlet mixture).")
    dataset_list, num_client_list = get_sft_datasets(script_args, fed_args, tokenizer=tokenizer)
else:
    if fed_args.mixture_num_clients > 0:
        if len(fed_args.malicious_dataset_names) > 1:
            dataset_list, num_client_list, alloc, client_summaries = get_sft_datasets_dirichlet_multi_malicious(script_args, fed_args, tokenizer=tokenizer)
        else:
            dataset_list, num_client_list, alloc, client_summaries = get_sft_datasets_dirichlet(script_args, fed_args, tokenizer=tokenizer, malicious_mixture=True)
    else:
        dataset_list, num_client_list, alloc, client_summaries = get_sft_datasets_dirichlet(script_args, fed_args, tokenizer=tokenizer, malicious_mixture=False)
print(dataset_list, num_client_list)

# ===== Split the dataset into clients =====
local_datasets = []
num_clients = sum(num_client_list)
real_num_clients = num_clients
# breakpoint()
for dataset, num_client in zip(dataset_list, num_client_list):
    try:
        splited_datasets = split_dataset(fed_args, script_args, dataset, num_client)
    except:
        splited_datasets = dataset
    local_datasets.extend(splited_datasets)
    

smoketestanchor_anchor_client_ids = []
if fed_args.fed_alg == "smoketestanchor":
    poison_ratios = _resolve_smoketestanchor_poison_ratios(fed_args, len(SMOKETESTANCHOR_DATASETS))
    for i, (dataset_name, poison_ratio) in enumerate(zip(SMOKETESTANCHOR_DATASETS, poison_ratios)):
        local_datasets.append(
            _build_smoketestanchor_dataset(
                dataset_name,
                script_args,
                tokenizer,
                poison_ratio=poison_ratio,
                seed=script_args.seed + 1000 + i,
            )
        )
        n_mal = int(round(SMOKETESTANCHOR_NUM_SAMPLES * poison_ratio))
        print(
            f"[smoketestanchor] anchor dataset={dataset_name} "
            f"poison_ratio={poison_ratio:.3f} "
            f"malicious={n_mal}/{SMOKETESTANCHOR_NUM_SAMPLES} "
            f"safe={SMOKETESTANCHOR_NUM_SAMPLES - n_mal}/{SMOKETESTANCHOR_NUM_SAMPLES} "
            f"({SMOKETESTANCHOR_SAFE_DATASET})"
        )
    smoketestanchor_anchor_client_ids = list(range(real_num_clients, len(local_datasets)))
    setattr(fed_args, "smoketestanchor_anchor_client_ids", smoketestanchor_anchor_client_ids)
    setattr(fed_args, "smoketestanchor_anchor_dataset_names", list(SMOKETESTANCHOR_DATASETS))
    setattr(fed_args, "smoketestanchor_poison_ratios", poison_ratios)
    setattr(fed_args, "smoketestanchor_safe_dataset", SMOKETESTANCHOR_SAFE_DATASET)
    setattr(fed_args, "smoketestanchor_num_samples", SMOKETESTANCHOR_NUM_SAMPLES)
    setattr(fed_args, "smoketestanchor_real_num_clients", real_num_clients)
    print(
        f"[smoketestanchor] appended anchor clients {smoketestanchor_anchor_client_ids} "
        f"from {SMOKETESTANCHOR_DATASETS} with poison_ratios={poison_ratios}"
    )

total_num_clients = len(local_datasets)
setattr(fed_args, 'num_clients', real_num_clients)
save_config(script_args, fed_args)
print(script_args, fed_args)
if alloc is not None:
    with open(os.path.join(script_args.output_dir, 'dirichlet_alloc.json'), 'w') as f:
        json.dump(alloc.tolist(), f, indent=4)
if client_summaries is not None:
    with open(os.path.join(script_args.output_dir, 'client_data_summary.json'), 'w') as f:
        json.dump(client_summaries, f, indent=2)


project_matrix = None
project_matrix_edit =None
safelora_paths = load_safelora_matrix_paths(
    script_args.model_name_or_path,
    getattr(script_args, "safelora_matrix_config", None),
)

def _infer_safelora_reference_name(delta_path):
    env_name = os.environ.get("SAFELORA_REFERENCE_NAME")
    if env_name:
        return env_name
    if not delta_path:
        return None
    base = os.path.basename(delta_path).lower()
    if "beavertails" in base:
        return "beavertails"
    if "contrastivellmlat" in base or "contrastllmlat" in base:
        return "contrastllmlat"
    if "llmlatdpo_safe" in base:
        return "llmlatdpo_safe"
    if "llmlatdpo" in base:
        return "llmlatdpo"
    if "llmlatunsafe" in base or "llmlat_unsafe" in base:
        return "llmlatunsafe"
    if "llmlatsafe" in base or "llmlat_safe" in base:
        return "llmlatsafe"
    if "beavertailsdpo" in base:
        return "beavertails"
    if "harmful_systemprompt" in base:
        return "harmful_systemprompt"
    return os.path.splitext(os.path.basename(delta_path))[0]

if fed_args.safe_lora or fed_args.fed_alg == 'safe_lora' or script_args.safe_lora_original:
    if 'chat' not in script_args.template.lower():
        project_matrix = try_load_safelora_matrix(safelora_paths.get("project_base"), "project_base")
    else:
        project_matrix = try_load_safelora_matrix(
            safelora_paths.get("project_base"), "project_base"
        )
# if fed_args.fed_alg == 'safelorav2data' or fed_args.fed_alg == 'safelorav2warmup' or fed_args.fed_alg == 'safelorav2':
if (
    'safelorav2' in fed_args.fed_alg
    or 'safeloradot' in fed_args.fed_alg
    or 'safelorav2datanomal' in fed_args.fed_alg
    or fed_args.fed_alg == 'smoketestanchor'
):
    assert 'chat' in script_args.template.lower(), "SafeLoRAv2 currently only supports chat template. Consider implementing the non-chat version if needed."
    delta_path = safelora_paths.get("delta_harmful_systemprompt")
    project_matrix = try_load_safelora_matrix(delta_path, "delta_harmful_systemprompt")
    ref_name = _infer_safelora_reference_name(delta_path)
    wandb_set("parameters_safelora_reference", ref_name)
    wandb_set("parameters_safelora_delta_path", delta_path)
    print(f"[safelora] reference={ref_name} delta_path={delta_path}")
    if fed_args.fed_alg in ("safelorav2datadiff", "safelorav2datadiffround"):
        safe_path = safelora_paths.get("delta_safe_reference")
        project_matrix_edit = try_load_safelora_matrix(safe_path, "delta_safe_reference")
        if project_matrix is None or project_matrix_edit is None:
            raise ValueError(
                f"{fed_args.fed_alg} requires delta_harmful_systemprompt and delta_safe_reference "
                f"(got unsafe={delta_path!r}, safe={safe_path!r})"
            )
        wandb_set("parameters_safelora_safe_reference", _infer_safelora_reference_name(safe_path))
        wandb_set("parameters_safelora_safe_delta_path", safe_path)
        print(f"[safelora] safe_reference={safe_path}")
    if fed_args.fed_alg == "safelorav2dataroundmultiple":
        named_matrices = {}
        ref_specs = (
            ("maliciousgen", "delta_harmful_systemprompt"),
            ("beavertails", "delta_beavertails"),
            ("llmlatunsafe", "delta_llmlatunsafe"),
        )
        missing = []
        for name, key in ref_specs:
            path = safelora_paths.get(key)
            if name == "maliciousgen" and project_matrix is not None:
                named_matrices[name] = project_matrix
            else:
                mat = try_load_safelora_matrix(path, key)
                if mat is None:
                    missing.append(f"{name}={path!r}")
                    continue
                named_matrices[name] = mat
            wandb_set(f"parameters_safelora_delta_path_{name}", path)
            print(f"[safelora] multi-ref {name} delta_path={path}")
        if missing or any(name not in named_matrices for name, _ in ref_specs):
            raise ValueError(
                "safelorav2dataroundmultiple requires three references "
                f"(maliciousgen, beavertails, llmlatunsafe); missing: {missing}"
            )
        project_matrix = named_matrices
        wandb_set("parameters_safelora_reference", "maliciousgen+beavertails+llmlatunsafe")


if 'safe_lora_mixture_analytical_different' in fed_args.fed_alg:
    with open('project_matrix_safelora_torch.float32_correct.pkl', 'rb') as f:
        project_matrix_edit = pickle.load(f)
        
def projected_weighted(peft_model, peft_config, project_matrix):
    """
    Compute cosine similarity between projected LoRA weights and original weights.
    
    Args:
        peft_model: The LoRA-adapted model (nn.Module).
        peft_config: Configuration object containing `r` (LoRA rank).
        project_matrix: List of projection matrices (torch.Tensor).

    Returns:
        cos_total: List of cosine similarity scores for each layer projection.
    """
    cos_total = []
    idx = 0  # index for project_matrix
    B = None  # Placeholder for rank-r weight
    s_per_layer = []
    for name, param in peft_model.named_parameters():
        if 'lora' in name:

            # Identify the rank-r weight (LoRA B matrix)
            if param.shape[0] == peft_config.r:
                B = param.data.clone()  # use clone instead of deepcopy for safety
                continue
            
            if param.shape[0] != peft_config.r:
                # Skip if B is not yet initialized
                if B is None:
                    raise ValueError(f"LoRA B matrix not found before layer {name}. Check peft_config.r.")
                
                # Project current layer weight
                P = project_matrix[idx].to(param.device)
                # breakpoint()
                W = torch.mm(P, param.data)
                fW = torch.mm(W, B)
                ori = torch.mm(param.data, B)
                
                # Compute cosine similarity
                cos = float(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)).item())
                cos_total.append(np.round(cos, 5))
                
                idx += 1

                # S-layer term: 1 / (1 + ||CΔW - ΔW||_2). Use Frobenius norm for matrices.
                diff = fW - ori
                # Frobenius norm == l2 over all entries
                diff_norm = torch.norm(diff, p='fro')
                s_i = (1.0 / (1.0 + diff_norm)).item()
                s_per_layer.append(float(np.round(s_i, 8)))

                B = None

    S_total = float(np.round(float(sum(s_per_layer)), 8))
    return {
        "cos_per_layer": cos_total,
        "s_per_layer": s_per_layer,
        "S_total": S_total,
    }


def projected_weighted_original_safelora(peft_model, peft_config, project_matrix, thrs_cos=0.35):
    model_ori = copy.deepcopy(peft_model)
    v = project_matrix
    idx = 0
    i = 0
    dis = []
    cos_total = []
    for (name, param),(name_ori, param_ori) in zip(peft_model.named_parameters(), model_ori.named_parameters()):
        if 'lora' in name:
            if param.shape[0] < 100:
                B = copy.deepcopy(param_ori)
            if param.shape[0] > 100:
                P = v[idx].to(param.device)
                W = torch.mm(P, param_ori.data)
                fW = torch.mm(W, B)
                ori = torch.mm(param_ori, B)
                W_new = torch.mm(P, param_ori.data)
                cos = np.round(torch.nn.functional.cosine_similarity(fW.reshape(1,-1), ori.reshape(1,-1)).item(),5)
                cos_total.append(cos)

                if cos <=  thrs_cos:
                    i+=1
                    param.data =  W_new
                else:
                    param.data = param_ori
                dist = 1 / (1+torch.norm(param.data.reshape(1,-1)-W.reshape(1,-1)))

                dis.append(dist.item())
                idx += 1
    print(f"{i} layers are projected, cosine threshold is {thrs_cos}, and Pdst is {np.mean(dis)} (> 0.8 is better).")
    return peft_model, cos_total

def _to_plain_dict(obj):
    """
    Convert dataclass / Namespace / dict to a plain dict.
    Non-serializable values are stringified.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        d = obj
    else:
        # dataclass
        try:
            d = dataclasses.asdict(obj)
        except Exception:
            # argparse.Namespace or SimpleNamespace or object with __dict__
            try:
                d = vars(obj)
            except Exception:
                # as last resort stringify
                return {"value": str(obj)}

    # make values JSON-serializable (strings for unknowns)
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        # if list/tuple of primitives, keep it
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _clean(vv) for k, vv in v.items()}
        # for everything else, return string repr
        return str(v)

    return {str(k): _clean(v) for k, v in d.items()}

def _flatten_dict(d, parent_key=""):
    """Flatten nested dict into {parent/child: value}"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items

def log_args_to_wandb(run, name, obj):
    plain = _to_plain_dict(obj)
    flat = _flatten_dict(plain, parent_key=name)
    payload = {}
    for k, v in flat.items():
        key = f"parameters_{k}"
        try:
            payload[key] = v
        except Exception:
            payload[key] = str(v)
    run.config.update(payload, allow_val_change=True)

# Log your argument objects
log_args_to_wandb(run, "script_args", script_args)
log_args_to_wandb(run, "fed_args", fed_args)
log_args_to_wandb(run, "peft_config", peft_config)
# training_args may be a dataclass from transformers; dataclasses.asdict will work above
log_args_to_wandb(run, "training_args", training_args)


_wandb_method_name = os.environ.get("WANDB_METHOD_NAME")
if _wandb_method_name:
    wandb_set("parameters_fed_args_fed_alg", _wandb_method_name)

log_args_to_wandb(run, "posttrain_source_run_id", os.environ.get("POSTTRAIN_SOURCE_RUN_ID"))

def _posttrain_eval_suffix():
    src = os.environ.get("POSTTRAIN_SOURCE_RUN_ID")
    if not src:
        return ""
    kw = os.environ.get("WANDB_METHOD_NAME", "")
    return f" --wandb_id_override {src} --keyword {kw}"


def _task_datasets_from_lora_path(lora_path):
    if not lora_path:
        return ""
    names = lora_path.lower()
    dataset_str = ''
    if 'squad' in names:
        dataset_str += "squad_v2 "
    if 'pubmed' in names:
        dataset_str += "pubmedqa "
    if 'metamathqa' in names:
        dataset_str += "gsm8k "
    if 'triviaqa' in names:
        dataset_str += 'triviaqa '
    if 'medqa' in names:
        dataset_str += 'medQA '
    if 'medmcqa' in names:
        dataset_str += 'medmcqa '
    if 'careqa' in names:
        dataset_str += 'careqa '
    if 'emrqa' in names:
        dataset_str += 'emrqa '
    if 'cord19' in names:
        dataset_str += 'cord19 '
    return dataset_str


def _task_datasets_from_benign_names(benign_dataset_names):
    names = "_".join(benign_dataset_names or []).lower()
    dataset_str = ""
    if "squad" in names:
        dataset_str += "squad_v2 "
    if "pubmed" in names:
        dataset_str += "pubmedqa "
    if "metamathqa" in names:
        dataset_str += "gsm8k "
    if "triviaqa" in names:
        dataset_str += "triviaqa "
    if "medqa" in names:
        dataset_str += "medQA "
    if "medmcqa" in names:
        dataset_str += "medmcqa "
    if "careqa" in names:
        dataset_str += "careqa "
    if "emrqa" in names:
        dataset_str += "emrqa "
    if "cord19" in names:
        dataset_str += "cord19 "
    return dataset_str


def _task_datasets_for_posttrain_eval(existing_lora, benign_dataset_names):
    if existing_lora:
        return _task_datasets_from_lora_path(existing_lora)
    return _task_datasets_from_benign_names(benign_dataset_names)


sample_num_list = [len(local_datasets[i]) for i in range(total_num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=access_token
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

def set_lora_init_seed(seed: int = 2025):
    import random as _random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _random.seed(seed)
    np.random.seed(seed)
if script_args.existing_lora is not None:
    existing_lora_ckpt = os.environ.get("EXISTING_LORA_CKPT", "30")
    existing_lora_path = f"{script_args.existing_lora}/checkpoint-{existing_lora_ckpt}"
    model = PeftModel.from_pretrained(model, existing_lora_path, is_trainable=True)
    print("Loaded existing LoRA from", existing_lora_path)
else:
    
    set_lora_init_seed(script_args.seed)
    model = get_peft_model(model, peft_config)
    i = 10
    for name, param in model.named_parameters():
        if 'lora' in name:
            print(param.data.sum())
        i = i-1
        if i == 0:
            break
    # exit()
model.print_trainable_parameters()


if script_args.safe_lora_original:
    existing_lora_ckpt = os.environ.get("EXISTING_LORA_CKPT", "30")
    for thrs in script_args.safelora_cos_thrs:
    #
        model, cos_total = projected_weighted_original_safelora(model, peft_config, project_matrix, thrs_cos=thrs)
        print("Cosine similarities per layer after original SafeLoRA projection:", cos_total)
        # save peft model
        model.save_pretrained(script_args.existing_lora + f'/checkpoint-{existing_lora_ckpt}_safelora_original_{thrs}')
        tokenizer.save_pretrained(script_args.existing_lora + f'/checkpoint-{existing_lora_ckpt}_safelora_original_{thrs}')
        wandb_set(f'parameters_safelora_original_saved_path_{thrs}', script_args.existing_lora + f'/checkpoint-{existing_lora_ckpt}_safelora_original_{thrs}')
        wandb_set('parameters_total_output_dir', script_args.existing_lora)


    eval_str = existing_lora_ckpt
    dataset_str = _task_datasets_from_lora_path(script_args.existing_lora)
    run_id = run.id
    _release_training_gpu()
    run.finish()
    command = (
        f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py'
        f' --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str}'
        f' --eval_list {eval_str} --safe_lora_original_minimal{_posttrain_eval_suffix()}'
    )
    os.system(command)
    exit()


# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(total_num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
if total_num_clients > len(auxiliary_model_list):
    extra_clients = total_num_clients - len(auxiliary_model_list)
    if global_auxiliary is None:
        auxiliary_model_list.extend([None] * extra_clients)
        auxiliary_delta_dict.extend([None] * extra_clients)
    else:
        auxiliary_model_list.extend([copy.deepcopy(global_auxiliary) for _ in range(extra_clients)])
        auxiliary_delta_dict.extend([copy.deepcopy(global_auxiliary) for _ in range(extra_clients)])

safefed = None
if fed_args.fed_alg == 'safefedllm':
    from federated_learning.safefedllm_runtime import setup as safefed_setup
    safefed = safefed_setup(script_args, fed_args, global_dict)





compute_round_safety_gradient = (fed_args.fed_alg == 'safe_lora_mixture_safety_subspace')
safety_grad_dataset = None
if compute_round_safety_gradient:
    safety_dataset_name = getattr(fed_args, 'safety_gradient_dataset_name', 'oneshotpatch')
    safety_pool_size = int(getattr(fed_args, 'safety_gradient_pool_size', 1))
    safety_grad_dataset = process_dataset_utils.process_sft_dataset(
        safety_dataset_name,
        process_dataset_utils.get_whole_dataset(safety_dataset_name, script_args.local_data_dir),
        script_args.template,
        safety_pool_size,
        False,
        tokenizer=tokenizer,
    )
    print(f"[safety-gradient] enabled for fed_alg={fed_args.fed_alg}, dataset={safety_dataset_name}, pool={len(safety_grad_dataset)}")



# if 'oneshotpatch' in fed_args.benign_dataset_names:
#     script_args.output_dir = os.path.join(script_args.existing_lora, 'oneshotpatch')
#     os.makedirs(script_args.output_dir, exist_ok=True)
    
wandb_set('parameters_total_output_dir', script_args.output_dir)
wandb_set('parameters_benign_num_clients', '_'.join([str(n) for n in fed_args.benign_num_clients]))
wandb_set('parameters_benign_dataset_names', '_'.join(fed_args.benign_dataset_names))
wandb_set('parameters_malicious_num_clients', '_'.join([str(n) for n in fed_args.malicious_num_clients]))
wandb_set('parameters_malicious_dataset_names', '_'.join(fed_args.malicious_dataset_names))
# Re-assert reference tag after bulk config logging so it is always queryable in W&B.
if (
    'safelorav2' in fed_args.fed_alg
    or 'safeloradot' in fed_args.fed_alg
    or 'safelorav2datanomal' in fed_args.fed_alg
    or fed_args.fed_alg == 'smoketestanchor'
):
    _delta = safelora_paths.get("delta_harmful_systemprompt")
    if fed_args.fed_alg == "safelorav2dataroundmultiple":
        wandb_set("parameters_safelora_reference", "maliciousgen+beavertails+llmlatunsafe")
    else:
        wandb_set("parameters_safelora_reference", _infer_safelora_reference_name(_delta))
    wandb_set("parameters_safelora_delta_path", _delta)
try:
    wandb_set('parameters_mixture_num_clients', fed_args.mixture_num_clients)
    if len(fed_args.mixture_benign_proportions) == 1:
        fed_args.mixture_benign_proportions = fed_args.mixture_benign_proportions * fed_args.mixture_num_clients
    wandb_set('parameters_mixture_benign_proportions', '_'.join([str(p) for p in fed_args.mixture_benign_proportions]))
except:
    pass
if fed_args.fed_alg in ('safelorav2dataadaptive', 'safelorav2dataadaptiveold'):
    wandb_set('parameters_evasion_lambda', getattr(fed_args, 'evasion_lambda', 1.0))
    wandb_set('parameters_adaptive_target', 'round_delta')
# if fed_args.safe_lora:
#     safe_lora_path = f'./output/safelora/{script_args.model_name_or_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}][{"_".join([ds for ds in fed_args.benign_dataset_names])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}][{"_".join([ds for ds in fed_args.malicious_dataset_names])}]_Steps[{script_args.max_steps}]_Clients[{fed_args.sample_clients}]_ISA[{script_args.isa}]/'
#     os.makedirs(safe_lora_path, exist_ok=True)


# ===== Start federated training =====
training_loss = [[] for i in range(total_num_clients)]

# if 'stanfordnlp/sst2' in fed_args.benign_dataset_names:
#     print(">> Evaluating on SST-2 ...")
#     sst2_acc, output_lst = compute_sst2_accuracy(model, tokenizer)
#     print(f"*** Evaluation on SST-2: Accuracy = {sst2_acc*100:.2f}% ***")
#     run["evaluation/sst2_accuracy"].append(sst2_acc, step=0)
#     with open(os.path.join(script_args.output_dir, f"sst2_eval_round_{0}.json"), 'w') as f:
#         json.dump(output_lst, f, indent=4)

# if 'HongzheBi/gsm8k' in fed_args.benign_dataset_names:
#     print(">> Evaluating on GSM8K ...")
#     gsm8k_acc, output_lst = compute_gsm8k_accuracy(model, tokenizer)
#     print(f"*** Evaluation on GSM8K: Accuracy = {gsm8k_acc*100:.2f}% ***")
#     run["evaluation/gsm8k_accuracy"].append(gsm8k_acc, step=0)
#     with open(os.path.join(script_args.output_dir, f"gsm8k_eval_round_{0}.json"), 'w') as f:
#         json.dump(output_lst, f, indent=4)


# if 'qiaojin/PubMedQA' in fed_args.benign_dataset_names:
#     print(">> Evaluating on PubMedQA ...")
#     pubmedqa_acc, output_lst = compute_pubmedqa_accuracy(model, tokenizer)
#     print(f"*** Evaluation on PubMedQA: Accuracy = {pubmedqa_acc*100:.2f}% ***")
#     run["evaluation/pubmedqa_accuracy"].append(pubmedqa_acc, step=0)
#     with open(os.path.join(script_args.output_dir, f"pubmedqa_eval_round_{0}.json"), 'w') as f:
#         json.dump(output_lst, f, indent=4)

# if 'rajpurkar/squad_v2' in fed_args.benign_dataset_names:
#     print(">> Evaluating on SQuAD v2 ...")
#     squadv2_acc, output_lst = compute_squadv2_scores(model, tokenizer)
#     print(f"*** Evaluation on SQuAD v2: F1 Score = {squadv2_acc*100:.2f}% ***")
#     run["evaluation/squadv2_f1_score"].append(squadv2_acc, step=0)
#     with open(os.path.join(script_args.output_dir, f"squadv2_eval_round_{0}.json"), 'w') as f:
#         json.dump(output_lst, f, indent=4)

# if 'medQA' in fed_args.benign_dataset_names:
#     print(">> Evaluating on MedQA ...")
#     medqa_acc, output_lst = compute_medqa_accuracy(model, tokenizer)
#     print(f"*** Evaluation on MedQA: Accuracy = {medqa_acc*100:.2f}% ***")
#     run["evaluation/medqa_accuracy"].append(medqa_acc, step=0)
#     with open(os.path.join(script_args.output_dir, f"medqa_eval_round_{0}.json"), 'w') as f:
#         json.dump(output_lst, f, indent=4)
    
# save local datasets to jsonl file (dump-only path for expguardtrain jobs)
# poison_prop = min(fed_args.mixture_benign_proportions)
# n_mal = fed_args.malicious_num_clients[0]
# if 'expguardtrain' in fed_args.malicious_dataset_names and 'expguardtrainfix' not in fed_args.malicious_dataset_names:
#     dump_dir = f"/shared/rc/llm-degredation/datasave/poison_{poison_prop}_nummalclients_{n_mal}"
#     os.makedirs(dump_dir, exist_ok=True)
#     for client_index in range(len(local_datasets)):
#         out_path = os.path.join(dump_dir, f"local_dataset_{client_index}.jsonl")
#         local_datasets[client_index].to_json(out_path)
#         print(f"Saved local dataset {client_index} to {out_path}")
#     exit()


# Stop early while keeping the LR schedule of the full num_rounds run.
stop_after_round = int(os.environ.get("STOP_AFTER_ROUND", "0"))
if stop_after_round:
    wandb_set('parameters_stop_after_round', stop_after_round)

for round in tqdm(range(fed_args.num_rounds)):
    real_clients_this_round = get_clients_this_round(fed_args, round)
    anchor_clients_this_round = (
        list(getattr(fed_args, "smoketestanchor_anchor_client_ids", []))
        if fed_args.fed_alg == "smoketestanchor"
        else []
    )
    clients_this_round = real_clients_this_round + anchor_clients_this_round

    print(
        f">> ==================== Round {round+1} : real={real_clients_this_round} "
        f"anchors={anchor_clients_this_round} ===================="
    )
    round_idx = round + 1
    safe_lora_save_data  = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "fed_args": _to_plain_dict(fed_args),
        "S_total": {},
        "cos_per_layer": {},
        "s_per_layer": {}
    }
    asr_rates = {}
    client_inferencess = {}
    client_actual_samples = {}

    for client in range(total_num_clients):
        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        if script_args.isa and client >= sum(fed_args.benign_num_clients) and client < fed_args.num_clients:
            sub_dataset = _remap_dataset_to_isa(sub_dataset, use_chat_template, tokenizer)

        client_actual_samples[client] = len(sub_dataset)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        if use_dpo:
            training_args = get_dpo_training_args(script_args, new_lr)
            trainer = get_fed_local_dpo_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                current_round=round,
                client_id=client,
            )
        else:
            training_args = get_training_args(script_args, new_lr, use_chat_template=use_chat_template)
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                current_round=round,
                client_id=client,
            )
        if safefed is not None:
            safefed.attach_delta_tracker(
                trainer,
                local_dataset=sub_dataset,
                current_round=round,
                client_id=client,
                tracker_initial_state=global_dict,
                tracker_enabled=safefed.primary_tracker_enabled,
            )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        wandb_log("client_{}/training_loss".format(client), results.training_loss, step=round_idx)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        if safefed is not None:
            safefed.maybe_run_shadow_train(
                model=model,
                tokenizer=tokenizer,
                sub_dataset=sub_dataset,
                client=client,
                round_num=round,
                auxiliary_model_list=auxiliary_model_list,
                global_auxiliary=global_auxiliary,
                get_fed_local_sft_trainer=get_fed_local_sft_trainer,
                get_training_args=get_training_args,
                use_chat_template=use_chat_template,
            )

        # evaluate the model on advbench after local training to check safety drop
        if 'eval_filter' in fed_args.fed_alg:
            asr_rate, client_inferences = compute_advbench_asr(model, tokenizer)
            print("The harmless rate of AdvBench after client {} local training is {:.2f}%".format(client, asr_rate*100))
            asr_rates[client] = asr_rate
            client_inferencess[client] = client_inferences




    #     if fed_args.safe_lora:
    #         safe_lora_data = projected_weighted(model, peft_config, project_matrix)
    #         safe_lora_save_data['S_total'][client] = safe_lora_data['S_total']
    #         safe_lora_save_data['cos_per_layer'][client] = safe_lora_data['cos_per_layer']
    #         safe_lora_save_data['s_per_layer'][client] = safe_lora_data['s_per_layer']

    # if fed_args.safe_lora:
    #     with open(os.path.join(safe_lora_path, f"round_{round_idx}_safelora.json"), 'w') as f:
    #         json.dump(safe_lora_save_data, f)

    # save everything
    if 'eval_filter' in fed_args.fed_alg:
        with open(os.path.join(script_args.output_dir, f"asr_rates_round_{round_idx}.json"), 'w') as f:
            json.dump(asr_rates, f)
        with open(os.path.join(script_args.output_dir, f"client_inferencess_round_{round_idx}.json"), 'w') as f:
            json.dump(client_inferencess, f)     
        
    with open(os.path.join(script_args.output_dir, f"round_{round_idx}.pkl"), 'wb') as f:
        pickle.dump({
            "round_idx": round_idx,
            "clients": real_clients_this_round,
            "active_clients": clients_this_round,
            "anchor_clients": anchor_clients_this_round,
            "local_dict_list": local_dict_list,
            "global_dict": compact_state_dict(global_dict),
            "fed_args": _to_plain_dict(fed_args),
            "sample_num_list": sample_num_list,
            "base_model_path": script_args.model_name_or_path
        }, f)

    aggregation_sample_num_list = sample_num_list
    aggregation_clients = real_clients_this_round
    if safefed is not None:
        aggregation_clients, aggregation_sample_num_list, skip_round = safefed.after_client_round(
            round, real_clients_this_round, client_actual_samples
        )
        if skip_round:
            print(f">> Round {round+1} skipped. Global model unchanged.")
            if (round + 1) % (1 if fed_args.num_rounds <= 30 else 10) == 0 or round + 1 == 10:
                trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
            np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
            continue

    round_safety_gradient_by_param = None
    if compute_round_safety_gradient:
        if safety_grad_dataset is None or len(safety_grad_dataset) == 0:
            raise ValueError("Safety gradient dataset is empty. Check safety_gradient_dataset_name/pool_size config.")

        set_peft_model_state_dict(model, global_dict)

        safety_eval_size = int(getattr(fed_args, 'safety_gradient_eval_size', 1))
        eval_size = min(max(1, safety_eval_size), len(safety_grad_dataset))
        safety_texts = [_example_to_text(safety_grad_dataset[i]) for i in range(eval_size)]

        round_safety_gradient_by_param = condition_gradient_by_parameter(
            model=model,
            tokenizer=tokenizer,
            texts=safety_texts,
            max_length=script_args.seq_length,
            batch_size=int(getattr(fed_args, 'safety_gradient_batch_size', 1)),
            aggregation=str(getattr(fed_args, 'safety_gradient_aggregation', 'single_batch')),
            matrix_only=bool(getattr(fed_args, 'safety_gradient_matrix_only', True)),
            clear_cuda_cache=bool(getattr(fed_args, 'safety_gradient_clear_cuda_cache', False)),
        )

        try:
            safety_grad_dir = os.path.join(script_args.output_dir, 'safety_gradient')
            os.makedirs(safety_grad_dir, exist_ok=True)
            meta = {
                "round_idx": round_idx,
                "fed_alg": fed_args.fed_alg,
                "dataset": str(getattr(fed_args, 'safety_gradient_dataset_name', 'oneshotpatch')),
                "num_texts": int(len(safety_texts)),
                "num_params": int(len(round_safety_gradient_by_param)),
                "aggregation": str(getattr(fed_args, 'safety_gradient_aggregation', 'single_batch')),
                "matrix_only": bool(getattr(fed_args, 'safety_gradient_matrix_only', True)),
            }
            with open(os.path.join(safety_grad_dir, f'round_{round_idx}.json'), 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as log_e:
            print(f"[warn] failed to save safety-gradient metadata for round {round_idx}: {log_e}")

        print(f"[safety-gradient] round={round_idx}, params={len(round_safety_gradient_by_param)}, texts={len(safety_texts)}")

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, aggregation_sample_num_list, \
        aggregation_clients, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
        base_model_path=script_args.model_name_or_path,
        project_matrix=project_matrix,
        project_matrix_edit=project_matrix_edit,
        safety_gradient_by_param=round_safety_gradient_by_param,
        script_args=script_args,
        asr_rates=asr_rates,
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    save_steps = 1 
    if (round+1) % save_steps == 0  or round+1 == 10:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

    # if round > 10:
    #     exit()

    # ===== Evaluate the model =====
    # eval_steps = 10 if fed_args.num_rounds == 30 else 25
    # if (round+1) % eval_steps == 0 or round+1 == fed_args.num_rounds or round == 0:
    #     if 'medQA' in fed_args.benign_dataset_names:
    #         print(">> Evaluating on MedQA ...")
    #         medqa_acc, output_lst = compute_medqa_accuracy(model, tokenizer)
    #         print(f"*** Evaluation on MedQA: Accuracy = {medqa_acc*100:.2f}% ***")
    #         run["evaluation/medqa_accuracy"].append(medqa_acc, step=round_idx)
    #         with open(os.path.join(script_args.output_dir, f"medqa_eval_round_{round_idx}.json"), 'w') as f:
    #             json.dump(output_lst, f, indent=4)
    #     if 'stanfordnlp/sst2' in fed_args.benign_dataset_names:
    #         print(">> Evaluating on SST-2 ...")
    #         sst2_acc, output_lst = compute_sst2_accuracy(model, tokenizer)
    #         print(f"*** Evaluation on SST-2: Accuracy = {sst2_acc*100:.2f}% ***")
    #         run["evaluation/sst2_accuracy"].append(sst2_acc, step=round_idx)
    #         with open(os.path.join(script_args.output_dir, f"sst2_eval_round_{round_idx}.json"), 'w') as f:
    #             json.dump(output_lst, f, indent=4)
    #     if 'HongzheBi/gsm8k' in fed_args.benign_dataset_names:
    #         print(">> Evaluating on GSM8K ...")
    #         gsm8k_acc, output_lst = compute_gsm8k_accuracy(model, tokenizer)
    #         print(f"*** Evaluation on GSM8K: Accuracy = {gsm8k_acc*100:.2f}% ***")
    #         run["evaluation/gsm8k_accuracy"].append(gsm8k_acc, step=round_idx)
    #         with open(os.path.join(script_args.output_dir, f"gsm8k_eval_round_{round_idx}.json"), 'w') as f:
    #             json.dump(output_lst, f, indent=4)
            
    #     if 'qiaojin/PubMedQA' in fed_args.benign_dataset_names:
    #         print(">> Evaluating on PubMedQA ...")
    #         pubmedqa_acc, output_lst = compute_pubmedqa_accuracy(model, tokenizer)
    #         print(f"*** Evaluation on PubMedQA: Accuracy = {pubmedqa_acc*100:.2f}% ***")
    #         run["evaluation/pubmedqa_accuracy"].append(pubmedqa_acc, step=round_idx)
    #         with open(os.path.join(script_args.output_dir, f"pubmedqa_eval_round_{round_idx}.json"), 'w') as f:
    #             json.dump(output_lst, f, indent=4)
        
    #     if 'rajpurkar/squad_v2' in fed_args.benign_dataset_names:
    #         print(">> Evaluating on SQuAD v2 ...")
    #         squadv2_acc, output_lst = compute_squadv2_scores(model, tokenizer)
    #         print(f"*** Evaluation on SQuAD v2: F1 Score = {squadv2_acc*100:.2f}% ***")
    #         run["evaluation/squadv2_f1_score"].append(squadv2_acc, step=round_idx)
    #         with open(os.path.join(script_args.output_dir, f"squadv2_eval_round_{round_idx}.json"), 'w') as f:
    #             json.dump(output_lst, f, indent=4)

    # if script_args.periodic_recovery and (round + 1) % script_args.recovery_interval == 0:
    #     safety_dataset = get_safety_sft_datasets(script_args, fed_args, tokenizer=tokenizer)
    #     # train on safe dataset to recover safety
    #     print(">> Periodic recovery on safe dataset ...")
    #     if formatting_prompts_func is None:
    #         trainer = SFTTrainer(
    #             model=model,
    #             tokenizer=tokenizer,
    #             args=training_args,
    #             max_seq_length=script_args.seq_length,
    #             dataset_text_field='formatted_chat',
    #             train_dataset=safety_dataset,
    #             data_collator=data_collator,
    #         )
    #     else:
    #         trainer = SFTTrainer(
    #             model=model,
    #             tokenizer=tokenizer,
    #             args=training_args,
    #             max_seq_length=script_args.seq_length,
    #             train_dataset=safety_dataset,
    #             formatting_func=formatting_prompts_func,
    #             data_collator=data_collator,
    #         )
    #     results = trainer.train()

        

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))

    if stop_after_round and round + 1 >= stop_after_round:
        print(f">> STOP_AFTER_ROUND={stop_after_round}: exiting after round {round+1} of {fed_args.num_rounds}")
        run.finish()
        raise SystemExit(0)
# eval_list = list(range(1, fed_args.num_rounds+1, eval_steps))
# eval_str = " ".join(map(str, eval_list))

if fed_args.num_rounds == 30:
    eval_str = '30' 


    # find the wandb run id of this
    benign_dataset_names = '_'.join(fed_args.benign_dataset_names).lower()
    dataset_str = ''
    if 'squad' in benign_dataset_names:
        dataset_str += "squad_v2 "
    if 'pubmed' in benign_dataset_names:
        dataset_str += "pubmedqa "
    if 'metamathqa' in benign_dataset_names:
        dataset_str += "gsm8k "
    if 'triviaqa' in benign_dataset_names:
        dataset_str += 'triviaqa '
    if 'medqa' in benign_dataset_names:
        dataset_str += 'medQA '
    if 'medmcqa' in benign_dataset_names:
        dataset_str += 'medmcqa '
    if 'careqa' in benign_dataset_names:
        dataset_str += 'careqa '
    if 'emrqa' in benign_dataset_names:
        dataset_str += 'emrqa '
    if 'cord19' in benign_dataset_names:
        dataset_str += 'cord19 '
    run_id = run.id
    run.finish()
    _release_training_gpu()
    # command = f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str} --eval_list {eval_str}'
    command = (
        f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py'
        f' --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str}'
        f' --eval_list {eval_str}{_posttrain_eval_suffix()}'
    )
    os.system(command)


elif fed_args.num_rounds == 50:
    eval_str = '50'
    dataset_str = _task_datasets_for_posttrain_eval(
        script_args.existing_lora, fed_args.benign_dataset_names
    )
    if not os.environ.get("WANDB_METHOD_NAME"):
        wandb_alg = "postfinetuning" if script_args.existing_lora else "defense_reference"
        wandb_set('parameters_fed_args_fed_alg', wandb_alg)
    wandb_set('parameters_benign_dataset_names', dataset_str or '_'.join(fed_args.benign_dataset_names))
    run_id = run.id
    run.finish()
    _release_training_gpu()
    command = (
        f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py'
        f' --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str}'
        f' --eval_list {eval_str}{_posttrain_eval_suffix()}'
    )
    os.system(command)


elif fed_args.num_rounds == 1:
    eval_str = '1'
    dataset_str = _task_datasets_for_posttrain_eval(
        script_args.existing_lora, fed_args.benign_dataset_names
    )
    if not os.environ.get("WANDB_METHOD_NAME"):
        wandb_set('parameters_fed_args_fed_alg', "oneshotpatch")
    wandb_set('parameters_benign_dataset_names', dataset_str)
    run_id = run.id
    run.finish()
    _release_training_gpu()
    command = (
        f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py'
        f' --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str}'
        f' --eval_list {eval_str}{_posttrain_eval_suffix()}'
    )
    os.system(command)


