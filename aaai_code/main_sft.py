import copy
import gc
import os
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoModelForCausalLM
import pickle
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
import wandb
import dataclasses
import torch
import json
import utils.process_dataset as process_dataset_utils
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel
from utils.process_dataset import get_sft_datasets, get_sft_datasets_dirichlet, get_sft_datasets_dirichlet_multi_malicious
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

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
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


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()

use_chat_template = "chat" in script_args.template.lower()
training_args = get_training_args(script_args, script_args.learning_rate, use_chat_template=use_chat_template)

# ===== Define the tokenizer =====
# previously we used use_fast=False for llama2, currently we are using use_fast=True for all models
tokenizer = setup_tokenizer(script_args.model_name_or_path, access_token)
if use_chat_template:
    setup_training_chat_template(tokenizer)
    print("----------------------------------Using chat template (assistant-only loss) -------------------------------")

# ===== Load the dataset =====
alloc = None
client_summaries = None

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
for dataset, num_client in zip(dataset_list, num_client_list):
    try:
        splited_datasets = split_dataset(fed_args, script_args, dataset, num_client)
    except:
        splited_datasets = dataset
    local_datasets.extend(splited_datasets)
    
setattr(fed_args, 'num_clients', num_clients)
save_config(script_args, fed_args)
print(script_args, fed_args)
if alloc is not None:
    with open(os.path.join(script_args.output_dir, 'dirichlet_alloc.json'), 'w') as f:
        json.dump(alloc.tolist(), f, indent=4)
if client_summaries is not None:
    with open(os.path.join(script_args.output_dir, 'client_data_summary.json'), 'w') as f:
        json.dump(client_summaries, f, indent=2)


project_matrix = None
safelora_paths = load_safelora_matrix_paths(
    script_args.model_name_or_path,
    getattr(script_args, "safelora_matrix_config", None),
)
if script_args.safe_lora_original:
    project_matrix = try_load_safelora_matrix(
        safelora_paths.get("project_base"), "project_base"
    )
if fed_args.fed_alg == 'ours':
    assert 'chat' in script_args.template.lower(), "Ours currently only supports chat template."
    project_matrix = try_load_safelora_matrix(
        safelora_paths.get("delta_harmful_systemprompt"), "delta_harmful_systemprompt"
    )
    assert project_matrix is not None, (
        "Ours requires delta_harmful_systemprompt. Set it in configs/safelora_matrix_paths.json "
        "(or --safelora_matrix_config) to an existing .pkl path."
    )

        
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
    names = (lora_path or "").lower()
    dataset_str = ""
    if 'pubmed' in names:
        dataset_str += "pubmedqa "
    if 'medqa' in names:
        dataset_str += 'medQA '
    if 'emrqa' in names:
        dataset_str += 'emrqa '
    if 'cord19' in names:
        dataset_str += 'cord19 '
    return dataset_str.strip()

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
    print(
        "SafeLoRA original checkpoints saved. Evaluate with gen_model_answer.py / gen_judge_advbench.py "
        f"(benches: advbench directharm expguardtest {dataset_str}; run_id={run_id}, ckpt={eval_str})."
    )
    exit()


# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)


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


# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):
    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    round_idx = round + 1
    for client in range(fed_args.num_clients):
        if client not in clients_this_round:
            training_loss[client].append(-1)            # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        if script_args.isa and client >= sum(fed_args.benign_num_clients):
            sub_dataset = _remap_dataset_to_isa(sub_dataset, use_chat_template, tokenizer)

        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr, use_chat_template=use_chat_template)

        # ===== Train local model on the client side =====
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

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        wandb_log("client_{}/training_loss".format(client), results.training_loss, step=round_idx)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!



    with open(os.path.join(script_args.output_dir, f"round_{round_idx}.pkl"), 'wb') as f:
        pickle.dump({
            "round_idx": round_idx,
            "clients": clients_this_round,
            "local_dict_list": local_dict_list,
            "global_dict": compact_state_dict(global_dict),
            "fed_args": _to_plain_dict(fed_args),
            "sample_num_list": sample_num_list,
            "base_model_path": script_args.model_name_or_path
        }, f)

    aggregation_sample_num_list = sample_num_list
    aggregation_clients = clients_this_round

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, aggregation_sample_num_list, \
        aggregation_clients, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
        base_model_path=script_args.model_name_or_path,
        project_matrix=project_matrix,
        script_args=script_args,
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    save_steps = 1 
    if (round+1) % save_steps == 0  or round+1 == 10:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
# eval_list = list(range(1, fed_args.num_rounds+1, eval_steps))
# eval_str = " ".join(map(str, eval_list))

# After training, evaluate with:
#   python evaluation/open_ended/gen_model_answer.py --bench_name advbench --use_vllm --base_model_path <merged_or_full_ckpt>
#   python evaluation/open_ended/gen_judge_advbench.py --bench_name advbench --model_answer <name> --judger rule
# Supported benches: advbench directharm expguardtest pubmedqa medQA emrqa cord19
run.finish()


