import copy
import os
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel
import pickle
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
import neptune
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
from utils.process_dataset import get_sft_datasets_dirichlet
access_token = os.environ.get('HUGGINGFACE_HUB_TOKEN', None)

if access_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable not set.")
run = neptune.init_run(project="fedllm/fedllm",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDNlMTFjYi0wMzQ4LTRmMDUtOTk4NC0wZjBlOGU5NGExMmYifQ==",
)

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



# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right", use_auth_token=access_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Load the dataset =====
# if fed_args.mixture_num_clients > 0:
#     dataset_list, num_client_list = get_sft_datasets_mixture(script_args, fed_args, tokenizer=tokenizer)    
# else:
#     dataset_list, num_client_list = get_sft_datasets(script_args, fed_args, tokenizer=tokenizer)
if fed_args.mixture_num_clients > 0:
    dataset_list, num_client_list, alloc = get_sft_datasets_dirichlet(script_args, fed_args, tokenizer=tokenizer, malicious_mixture=True)
else:
    dataset_list, num_client_list, alloc = get_sft_datasets_dirichlet(script_args, fed_args, tokenizer=tokenizer, malicious_mixture=False)
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
save_config(script_args, fed_args)
print(script_args, fed_args)

with open(os.path.join(script_args.output_dir, 'dirichlet_alloc.json'), 'w') as f:
    json.dump(alloc.tolist(), f, indent=4)


project_matrix = None
if fed_args.safe_lora or fed_args.fed_alg == 'safe_lora' or script_args.safe_lora_original:
    if 'chat' not in script_args.template.lower():
        with open('project_matrix_safelora_torch.float32_harmful.pkl', 'rb') as f:
            project_matrix = pickle.load(f)
    else:
        with open('project_matrix_safelora_torch.float32_harmful_systemprompt.pkl', 'rb') as f:
            project_matrix = pickle.load(f)
        
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
            if param.shape[0] == peft_config.r:
                B = copy.deepcopy(param_ori)
            if param.shape[0] != peft_config.r:
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
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items

def log_args_to_neptune(run, name, obj):
    plain = _to_plain_dict(obj)
    flat = _flatten_dict(plain, parent_key=name)
    for k, v in flat.items():
        # neptune metadata keys cannot contain leading slashes, put them under 'parameters'
        key = f"parameters/{k}"
        try:
            run[key] = v
        except Exception:
            # fallback to string for any weird types
            run[key] = str(v)

# Log your argument objects
log_args_to_neptune(run, "script_args", script_args)
log_args_to_neptune(run, "fed_args", fed_args)
log_args_to_neptune(run, "peft_config", peft_config)
# training_args may be a dataclass from transformers; dataclasses.asdict will work above
log_args_to_neptune(run, "training_args", training_args)

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

if script_args.existing_lora is not None:
    model = PeftModel.from_pretrained(model, script_args.existing_lora+'/checkpoint-30', is_trainable=True)
else:
    model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


if script_args.safe_lora_original:
    for thrs in script_args.safelora_cos_thrs:
    #
        model, cos_total = projected_weighted_original_safelora(model, peft_config, project_matrix, thrs_cos=thrs)
        print("Cosine similarities per layer after original SafeLoRA projection:", cos_total)
        # save peft model
        model.save_pretrained(script_args.existing_lora + f'/checkpoint-30_safelora_original_{thrs}')
        tokenizer.save_pretrained(script_args.existing_lora + f'/checkpoint-30_safelora_original_{thrs}')
        run[f'parameters/safelora_original_saved_path_{thrs}'] = script_args.existing_lora + f'/checkpoint-30_safelora_original_{thrs}'
        run['parameters/total_output_dir'] = script_args.existing_lora

    eval_str = '30'
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
    run_id = run['sys/id'].fetch().split('-')[-1]
    run['parameters/fed_args/fed_alg'] = "safelora_original"
    run['parameters/benign_dataset_names'] = dataset_str
    command = f'python /home/ps9044/FedLLM-Attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench {dataset_str} --eval_list {eval_str} --safe_lora_original_minimal'
    os.system(command)
    exit()


# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)





# if using chat template for training, we need to adjust response_template and data_collator
if 'chat' in script_args.template.lower():
    formatting_prompts_func = None
    malicious_prompts_func = None #because process sft dataset has already converted to chat format
    response_template = " [/INST]"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
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

run['parameters/total_output_dir'] = script_args.output_dir
run['parameters/benign_num_clients'] = '_'.join([str(n) for n in fed_args.benign_num_clients])
run['parameters/benign_dataset_names'] = '_'.join(fed_args.benign_dataset_names)
run['parameters/malicious_num_clients'] = '_'.join([str(n) for n in fed_args.malicious_num_clients])
run['parameters/malicious_dataset_names'] = '_'.join(fed_args.malicious_dataset_names)
try:
    run['parameters/mixture_num_clients'] = fed_args.mixture_num_clients
    if len(fed_args.mixture_benign_proportions) == 1:
        fed_args.mixture_benign_proportions = fed_args.mixture_benign_proportions * fed_args.mixture_num_clients
    run['parameters/mixture_benign_proportions'] = '_'.join([str(p) for p in fed_args.mixture_benign_proportions])
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
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)
        run["client_{}/training_loss".format(client)].append(results.training_loss, step=round_idx)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

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
            "clients": clients_this_round,
            "local_dict_list": local_dict_list,
            "global_dict": compact_state_dict(global_dict),
            "fed_args": _to_plain_dict(fed_args),
            "sample_num_list": sample_num_list,
            "base_model_path": script_args.model_name_or_path
        }, f)

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),
        base_model_path=script_args.model_name_or_path,
        project_matrix=project_matrix,
        script_args=script_args,
        asr_rates=asr_rates,
    )
    set_peft_model_state_dict(model, global_dict)   # Update global model

    # ===== Save the model =====
    save_steps = 1 if fed_args.num_rounds <= 30 else 10
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


    # find the neptune run id of this
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
    run_id = run['sys/id'].fetch().split('-')[-1]
    command = f'python /home/ps9044/FedLLM-Attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench {dataset_str} --eval_list {eval_str}'
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
    run_id = run['sys/id'].fetch().split('-')[-1]
    run['parameters/fed_args/fed_alg'] = "postfinetune"
    run['parameters/benign_dataset_names'] = dataset_str
    command = f'python /home/ps9044/FedLLM-Attack/evaluation/open_ended/run_checkpoint_generation_full.py --run_ids {run_id} --datasets advbench {dataset_str} --eval_list {eval_str}'
    os.system(command)

