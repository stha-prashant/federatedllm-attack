import copy
import os
import shutil
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel
import pickle
from utils import *
from federated_learning import *
from federated_learning.fed_lora_classifier import Evaluation, FedLoRAClassifier
from config import get_config, save_config, get_model_config, get_training_args
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
from trl import SFTTrainer
from utils.process_dataset import get_sft_datasets_dirichlet, get_sft_datasets
import utils.process_dataset as process_dataset_utils
from utils.chat_format import (
    get_response_template_ids,
    load_safelora_matrix_paths,
    setup_tokenizer,
    try_load_safelora_matrix,
)
access_token = os.environ.get('HUGGINGFACE_HUB_TOKEN', None)

if access_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set.")

WANDB_PROJECT = "fedllm_fedllm"
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
    if "formatted_chat" in example:
        return example["formatted_chat"]

    instruction = example.get("instruction", "")
    response = example.get("response", "")
    inp = example.get("input", "")
    if inp:
        instruction = f"{instruction}\n{inp}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


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


def _to_short_name(full_name: str) -> str:
    base = full_name.split('/')[-1]
    mapping = {
        'WildChat': 'WildChat',
        'lmsys-chat-1m': 'lmsys-chat-1m',
        'BeaverTails': 'BeaverTails',
        'MaliciousGen': 'MaliciousGen',
    }
    return mapping.get(base, base)


def _build_client_dataset_short_map(fed_args):
    mapping = {}
    total_benign = sum(fed_args.benign_num_clients) if fed_args.benign_num_clients else 0
    benign_short = _to_short_name(fed_args.benign_dataset_names[0]) if fed_args.benign_dataset_names else 'benign'
    for client_id in range(total_benign):
        mapping[client_id] = benign_short

    offset = total_benign
    for ds_name, num_clients in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
        ds_short = _to_short_name(ds_name)
        for _ in range(num_clients):
            mapping[offset] = ds_short
            offset += 1
    return mapping



# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
if fed_args.fed_alg == 'lora_classifier':
    script_args.prefilter_enable = True
    if str(getattr(script_args, 'prefilter_strategy', 'none')).lower() == 'none':
        script_args.prefilter_strategy = 'step-level'

prefilter_active = bool(getattr(script_args, 'prefilter_enable', False)) and str(getattr(script_args, 'prefilter_strategy', 'none')).lower() != 'none'

if prefilter_active:
    try:
        clf = FedLoRAClassifier.instance(getattr(script_args, 'prefilter_classifier_path', None))
        print(f"[prefilter] LoRA classifier enabled (mode={clf.lora_mode}).")
        print(f"[prefilter] threshold={script_args.prefilter_threshold}, strategy={script_args.prefilter_strategy}")
    except Exception as e:
        print(f"[warn] prefilter init failed: {e}")

training_args = get_training_args(script_args, script_args.learning_rate)

# ===== Define the tokenizer =====
# previously we used use_fast=False for llama2, currently we are using use_fast=True for all models
tokenizer = setup_tokenizer(script_args.model_name_or_path, access_token)

# ===== Load the dataset =====
# if fed_args.mixture_num_clients > 0:
#     dataset_list, num_client_list = get_sft_datasets_mixture(script_args, fed_args, tokenizer=tokenizer)    
# else:
#     dataset_list, num_client_list = get_sft_datasets(script_args, fed_args, tokenizer=tokenizer)
alloc = None
client_summaries = None

if getattr(script_args, 'prefilter_enable', False):
# if True:
    # if prefilter is enabled use old splitting
    dataset_list, num_client_list = get_sft_datasets(script_args, fed_args, tokenizer=tokenizer)
else:   
    if fed_args.mixture_num_clients > 0:
        dataset_list, num_client_list, alloc, client_summaries = get_sft_datasets_dirichlet(script_args, fed_args, tokenizer=tokenizer, malicious_mixture=True)
    else:
        dataset_list, num_client_list, alloc, client_summaries = get_sft_datasets_dirichlet(script_args, fed_args, tokenizer=tokenizer, malicious_mixture=False)
    print(dataset_list, num_client_list)

# ===== Split the dataset into clients =====
local_datasets = []
num_clients = sum(num_client_list)
# breakpoint()
for dataset, num_client in zip(dataset_list, num_client_list):
    try:
        splited_datasets = split_dataset(fed_args, script_args, dataset, num_client)
    except:
        splited_datasets = dataset
    local_datasets.extend(splited_datasets)
    
setattr(fed_args, 'num_clients', num_clients)
script_args._prefilter_client_dataset_short_map = _build_client_dataset_short_map(fed_args)
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
if fed_args.safe_lora or fed_args.fed_alg == 'safe_lora' or script_args.safe_lora_original:
    if 'chat' not in script_args.template.lower():
        project_matrix = try_load_safelora_matrix(safelora_paths.get("project_harmful"), "project_harmful")
    else:
        project_matrix = try_load_safelora_matrix(
            safelora_paths.get("project_harmful_systemprompt"), "project_harmful_systemprompt"
        )
# if fed_args.fed_alg == 'safelorav2data' or fed_args.fed_alg == 'safelorav2warmup' or fed_args.fed_alg == 'safelorav2':
if 'safelorav2' in fed_args.fed_alg:
    assert 'chat' in script_args.template.lower(), "SafeLoRAv2 currently only supports chat template. Consider implementing the non-chat version if needed."
    project_matrix = try_load_safelora_matrix(
        safelora_paths.get("delta_harmful_systemprompt"), "delta_harmful_systemprompt"
    )


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

sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

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
    model = PeftModel.from_pretrained(model, script_args.existing_lora+'/checkpoint-30', is_trainable=True)
    print("Loaded existing LoRA from", script_args.existing_lora)
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
    for thrs in script_args.safelora_cos_thrs:
    #
        model, cos_total = projected_weighted_original_safelora(model, peft_config, project_matrix, thrs_cos=thrs)
        print("Cosine similarities per layer after original SafeLoRA projection:", cos_total)
        # save peft model
        model.save_pretrained(script_args.existing_lora + f'/checkpoint-30_safelora_original_{thrs}')
        tokenizer.save_pretrained(script_args.existing_lora + f'/checkpoint-30_safelora_original_{thrs}')
        wandb_set(f'parameters_safelora_original_saved_path_{thrs}', script_args.existing_lora + f'/checkpoint-30_safelora_original_{thrs}')
        wandb_set('parameters_total_output_dir', script_args.existing_lora)

    # eval_str = '30'
    # benign_dataset_names = script_args.existing_lora.lower()
    # dataset_str = ''
    # if 'squad' in benign_dataset_names:
    #     dataset_str += "squad_v2 "
    # if 'pubmed' in benign_dataset_names:
    #     dataset_str += "pubmedqa "
    # if 'metamathqa' in benign_dataset_names:
    #     dataset_str += "gsm8k "
    # if 'triviaqa' in benign_dataset_names:
    #     dataset_str += 'triviaqa '
    # if 'medqa' in benign_dataset_names:
    #     dataset_str += 'medQA '
    # if 'emrqa' in benign_dataset_names:
    #     dataset_str += 'emrqa '
    # if 'cord19' in benign_dataset_names:
    #     dataset_str += 'cord19 '
    
    # run_id = run.id
    # wandb_set('parameters_fed_args_fed_alg', "safelora_original")
    # wandb_set('parameters_benign_dataset_names', dataset_str)
    # command = f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str} --eval_list {eval_str} --safe_lora_original_minimal'
    # os.system(command)
    exit()


# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
shadow_lora_base = copy.deepcopy(global_dict)





# if using chat template for training, we need to adjust response_template and data_collator
if 'chat' in script_args.template.lower():
    formatting_prompts_func = None
    malicious_prompts_func = None #because process sft dataset has already converted to chat format
    response_template_ids = get_response_template_ids(tokenizer)
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    assert script_args.isa == False, "ISA attack not supported with chat template currently, consider implementing maybe."
    print("----------------------------------Using chat template -------------------------------")
else:
    # ===== Define the formatting function (cater to TRL SFTTrainer)=====
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    malicious_formatting_prompt_func, malicious_response_template = get_formatting_prompts_func('isa', tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    malicious_response_template_ids = tokenizer.encode(malicious_response_template, add_special_tokens=False)[2:]
    malicious_data_collator = DataCollatorForCompletionOnlyLM(malicious_response_template_ids, tokenizer=tokenizer)

if 'stanfordnlp/sst2' in fed_args.benign_dataset_names:
    sst2_formatting_prompts_func, sst2_response_template = get_formatting_prompts_func('sst2', tokenizer.eos_token)
    sst2_response_template_ids = tokenizer.encode(sst2_response_template, add_special_tokens=False)[2:]
    sst2_data_collator = DataCollatorForCompletionOnlyLM(sst2_response_template_ids, tokenizer=tokenizer)


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
try:
    wandb_set('parameters_mixture_num_clients', fed_args.mixture_num_clients)
    if len(fed_args.mixture_benign_proportions) == 1:
        fed_args.mixture_benign_proportions = fed_args.mixture_benign_proportions * fed_args.mixture_num_clients
    wandb_set('parameters_mixture_benign_proportions', '_'.join([str(p) for p in fed_args.mixture_benign_proportions]))
except:
    pass
# if fed_args.safe_lora:
#     safe_lora_path = f'./output/safelora/{script_args.model_name_or_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}][{"_".join([ds for ds in fed_args.benign_dataset_names])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}][{"_".join([ds for ds in fed_args.malicious_dataset_names])}]_Steps[{script_args.max_steps}]_Clients[{fed_args.sample_clients}]_ISA[{script_args.isa}]/'
#     os.makedirs(safe_lora_path, exist_ok=True)


# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

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
    

prefilter_strategy = PrefilterStrategy(script_args, fed_args)

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
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
    strategy_name = str(getattr(script_args, 'prefilter_strategy', 'step-level')).lower()

    for client in range(fed_args.num_clients):

        if script_args.isa and client >= sum(fed_args.benign_num_clients):
            formatting_prompts_func_current = malicious_formatting_prompt_func
            data_collator_current = malicious_data_collator
        else:
            formatting_prompts_func_current = formatting_prompts_func
            data_collator_current = data_collator
        

        if 'stanfordnlp/sst2' in fed_args.benign_dataset_names:
            # find which index 'stanfordnlp/sst2' is in benign_dataset_names
            benign_dataset_index = fed_args.benign_dataset_names.index('stanfordnlp/sst2')
            clients_before = sum(fed_args.benign_num_clients[:benign_dataset_index])
            clients_end_index = clients_before + fed_args.benign_num_clients[benign_dataset_index]
            # if this client is using sst2
            if client >= clients_before and client < clients_end_index:
                # assert not script_args.isa, "SST-2 evaluation not supported with ISA attack."
                # assert len(fed_args.benign_dataset_names) == 1 and fed_args.benign_dataset_names[0] == 'stanfordnlp/sst2', "SST-2 evaluation only supported when all benign clients use SST-2."
                # assert client in [4, 5, 6, 7], "SST-2 evaluation only supported when all benign clients use SST-2."
                
                formatting_prompts_func_current = sst2_formatting_prompts_func
                data_collator_current = sst2_data_collator
            else:
                # assert client in [0, 1, 2, 3, 8, 9, 10, 11], "SST-2 evaluation only supported when all benign clients use SST-2."
                formatting_prompts_func_current = formatting_prompts_func
                data_collator_current = data_collator
        


        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        # breakpoint()


        client_actual_samples[client] = len(sub_dataset)
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=sub_dataset,
            formatting_prompts_func=formatting_prompts_func_current,
            data_collator=data_collator_current,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            current_round=round,
            tracker_initial_state=None,
            tracker_enabled=(getattr(script_args, 'prefilter_enable', False) and strategy_name != 'shadow-level'),
            client_id=client,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        wandb_log("client_{}/training_loss".format(client), results.training_loss, step=round_idx)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

        if getattr(script_args, 'prefilter_enable', False) and strategy_name == 'shadow-level':
            try:
                set_peft_model_state_dict(model, shadow_lora_base)
                shadow_fixed_lr = 5e-5
                shadow_training_args = get_training_args(script_args, shadow_fixed_lr)
                shadow_trainer = get_fed_local_sft_trainer(
                    script_args=script_args,
                    fed_args=fed_args,
                    model=model,
                    tokenizer=tokenizer,
                    training_args=shadow_training_args,
                    local_dataset=sub_dataset,
                    formatting_prompts_func=formatting_prompts_func_current,
                    data_collator=data_collator_current,
                    global_dict=shadow_lora_base,
                    local_auxiliary=auxiliary_model_list[client],
                    global_auxiliary=global_auxiliary,
                    current_round=round,
                    tracker_initial_state=shadow_lora_base,
                    tracker_enabled=True,
                    client_id=client,
                )
                _ = shadow_trainer.train()
                if hasattr(shadow_trainer, 'delta_tracker'):
                    _ = shadow_trainer.delta_tracker.get_processed_sample_ids()
            except Exception as se:
                print(f"[warn] Shadow-level training failed: {se}")

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
    filtered_clients_this_round = clients_this_round.copy()
    client_harmful_mapping = {}
    aggregation_sample_num_list = sample_num_list
    skip_round = False

    if prefilter_active:
        try:
            params_dir = os.path.join(script_args.output_dir, 'fed_lora_params', f'round_{round+1}')
            clf_local = FedLoRAClassifier.instance(getattr(script_args, 'prefilter_classifier_path', None))
            client_harmful_mapping, eval_result = clf_local.get_harmful_mapping(
                params_dir=params_dir,
                clients_this_round=clients_this_round,
                round_num=round,
                fed_args=fed_args,
                script_args=script_args,
            )

            try:
                eval_engine = Evaluation(FedLoRAClassifier.instance(getattr(script_args, 'prefilter_classifier_path', None)))
                _, current_precision = eval_engine.evaluate(
                    params_dir=params_dir,
                    output_dir=script_args.output_dir,
                    round_num=round,
                    mode='prefilter',
                    print_stats=False,
                    existing_result=eval_result,
                    threshold=getattr(script_args, 'prefilter_threshold', None),
                )
                print(f">> Round {round+1} Classifier Precision: {current_precision*100:.2f}%")
                if os.path.exists(params_dir):
                    shutil.rmtree(params_dir)
            except Exception as e:
                print(f"[warn] Failed to save prefilter classification results: {e}")

            filtered_clients_this_round, client_effective_samples, skip_round = prefilter_strategy.compute(
                round,
                clients_this_round,
                client_actual_samples,
                client_harmful_mapping,
                eval_result,
            )
            aggregation_sample_num_list = client_effective_samples

        except Exception as e:
            print(f"[warn] Prefilter failed, proceeding with normal weights: {e}")
            client_effective_samples = {c: float(client_actual_samples.get(c, sample_num_list[c])) for c in clients_this_round}
            filtered_clients_this_round = clients_this_round.copy()
            aggregation_sample_num_list = client_effective_samples
            skip_round = (sum(float(client_effective_samples[c]) for c in clients_this_round) == 0.0)

    # save everything
    if 'eval_filter' in fed_args.fed_alg:
        with open(os.path.join(script_args.output_dir, f"asr_rates_round_{round_idx}.json"), 'w') as f:
            json.dump(asr_rates, f)
        with open(os.path.join(script_args.output_dir, f"client_inferencess_round_{round_idx}.json"), 'w') as f:
            json.dump(client_inferencess, f)     
        
    # with open(os.path.join(script_args.output_dir, f"round_{round_idx}.pkl"), 'wb') as f:
    #     pickle.dump({
    #         "round_idx": round_idx,
    #         "clients": clients_this_round,
    #         "local_dict_list": local_dict_list,
    #         "global_dict": compact_state_dict(global_dict),
    #         "fed_args": _to_plain_dict(fed_args),
    #         "sample_num_list": aggregation_sample_num_list,
    #         "base_model_path": script_args.model_name_or_path
    #     }, f)

    try:
        for c in clients_this_round:
            base_samples = client_actual_samples.get(c, sample_num_list[c])
            eff = float(aggregation_sample_num_list[c] if isinstance(aggregation_sample_num_list, dict) else aggregation_sample_num_list[c])
            w = (eff / float(base_samples)) if base_samples > 0 else 0.0
            harm_cnt = len(client_harmful_mapping.get(c, []))
            print(f"[prefilter] Client {c} has {harm_cnt} harmful steps | weight={w:.3f}")
    except Exception as pe:
        print(f"[warn] Printing client weights failed: {pe}")

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
        filtered_clients_this_round, round, proxy_dict=proxy_dict, \
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

    command = f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str} --eval_list {eval_str}'
    os.system(command)


elif fed_args.num_rounds == 50:
    eval_str = '50'
    benign_dataset_names = script_args.existing_lora.lower()
    dataset_str = ''
    if 'squad' in benign_dataset_names:
        dataset_str += "squad_v2 "
    if 'pubmed' in benign_dataset_names:
        dataset_str += "pubmedqa "
    if 'metamathqa' in benign_dataset_names:
        dataset_str += "gsm8k "
    if 'triviaqa' in benign_dataset_names:
        dataset_str += 'triviaqa '
    run_id = run.id
    run.finish()

    wandb_set('parameters_fed_args_fed_alg', "postfinetune")
    wandb_set('parameters_benign_dataset_names', dataset_str)
    command = f'python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench directharm expguardtest {dataset_str} --eval_list {eval_str}'
    os.system(command)


