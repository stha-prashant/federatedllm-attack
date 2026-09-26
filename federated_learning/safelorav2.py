from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
import pickle
import torch
import numpy as np
import os
import json 
from sklearn.cluster import KMeans
from collections import Counter

def get_aligned_matrix(device='cpu'):
    """
    Get projected matrix by following the config (target_modules) from the peft model.
    The dimensions between the base model's weights and the aligned model's weights should be the same.
    """
    model_name = 'meta-llama/Llama-2-7b-hf'
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    # model_name = 'Qwen/Qwen2.5-7B-Instruct'
    # model_name = 'google/gemma-2-2b-it'


    # llama 2
    # harmful_checkpoint_path = '/scratch/ps9044/fedllm/barebones/MaliciousGen1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260629203019/full-50'
    

    # llama 3
    # harmful_checkpoint_path = '/scratch/ps9044//MaliciousGen1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260629203019/full-50'
    # qwen
    # harmful_checkpoint_path = '/scratch/ps9044//MaliciousGen1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260629202907/full-50'

    # harmful_checkpoint_path = '/scratch/ps9044//Llama-2-7b-chat-hf_BeaverTailsUnsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260715220352/full-50'
    # harmful_checkpoint_path = '/scratch/ps9044//Llama-2-7b-chat-hf_llmlatunsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260715211009/full-50'
    # harmful_checkpoint_pathsafe = '/scratch/ps9044//Llama-2-7b-chat-hf_llmlatsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260715211009/full-50'

    # harmful_checkpoint_path = '/shared/rc/llm-degredation/aaai2026/references/Llama-2-7b-chat-hf_llmlatunsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260827210001_d8694c50/full-50'
    # harmful_checkpoint_pathsafe = '/shared/rc/llm-degredation/aaai2026/references/Llama-2-7b-chat-hf_llmlatdpo_safe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260828050325_ec699bbb/full-50'
    
    harmful_checkpoint_path = '/shared/rc/llm-degredation/aaai2026/references/Llama-2-7b-chat-hf_llmlatunsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260827205639_ad48ebd1/full-50'
    harmful_checkpoint_pathsafe = '/shared/rc/llm-degredation/aaai2026/dpo/Llama-2-7b-chat-hf_llmlatdpo_safe_full_dpo_b8a1e10_l512_r8a16_beta0.1_lr5e-06_20260901184010_2321ceb8/full-6190'
    # gemma 2
    # harmful_checkpoint_path = '/scratch/ps9044//MaliciousGen1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260629203239/full-50'


    base_model = AutoModelForCausalLM.from_pretrained(
        harmful_checkpoint_pathsafe,
        # model_name,
        # '/scratch/ps9044/fedllm/safelora_maliciousgen_llama27bchat/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251113142234/full-50',
        # '/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743/full-50', # this is the one
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )
    base_model_for_peft = AutoModelForCausalLM.from_pretrained(
        model_name,
        # '/scratch/ps9044/fedllm/safelora_maliciousgen_llama27bchat/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251113142234/full-50',
        # '/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743/full-50', #  this is the one
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )
    aligned_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # harmful_checkpoint_pathsafe,
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )

    print(base_model.dtype)
    #Fed-134, fedgraph
    # checkpoint_path = '/scratch/ps9044/fedllm/barebones/WildChat7_BeaverTails3_500_fedgraph_c10s10_i10_b16a1_l512_r32a64_20251030220628/checkpoint-10'
    # # checkpoint_path = '/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743/checkpoint-10'

    # peft_model = PeftModel.from_pretrained(base_model_for_peft, checkpoint_path, is_trainable=False).to(device)
    # peft_config = peft_model.peft_config['default']
    # print(peft_config.target_modules)

    # peft_config = LoraConfig(
    #     r=32,
    #     lora_alpha=64,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # get target modules
    target_modules = ['q_proj', 'v_proj']
    
    v = []
    # proj_modules = list(peft_config.target_modules)
    proj_modules = target_modules
    #
    for (b_name, b_param) , (a_name, a_param) in zip (base_model.named_parameters(), aligned_model.named_parameters()):
        if any(module in a_name for module in proj_modules) and 'bias' not in a_name:
            assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
            vec = a_param - b_param
            vec = vec.to(device)
            # vec = torch.mm(vec, vec.t()) / torch.norm(vec)
            # vec = vec @ torch.linalg.pinv(vec.t() @ vec) @ vec.t()
            v.append((vec).detach().cpu())
    # save v to ../project_matrix.pkl

    if model_name == 'meta-llama/Llama-2-7b-hf':
        save_dir = 'llama2'
    elif model_name == 'meta-llama/Llama-3.1-8B-Instruct':
        save_dir = 'llama3'
    elif model_name == 'Qwen/Qwen2.5-7B-Instruct':
        save_dir = 'qwen'
    elif model_name == 'google/gemma-2-2b-it':
        save_dir = 'gemma2'
    else:
        raise ValueError(f"Model {model_name} not supported.")
    os.makedirs(f'/shared/rc/llm-degredation/references/{save_dir}', exist_ok=True)

    with open(f'/shared/rc/llm-degredation/references/{save_dir}/delta_matrix_diffllmlatdposftsafe.pkl', 'wb') as f:
        pickle.dump(v, f)
        print("Saved projection matrix to ", f'/shared/rc/llm-degredation/references/{save_dir}/delta_matrix_safelora_{base_model.dtype}_harmful_systemprompt_correct.pkl')



# def projected_weighted(peft_model, peft_config, project_matrix):
#     """
#     Compute cosine similarity between projected LoRA weights and original weights.
    
#     Args:
#         peft_model: The LoRA-adapted model (nn.Module).
#         peft_config: Configuration object containing `r` (LoRA rank).
#         project_matrix: List of projection matrices (torch.Tensor).

#     Returns:
#         cos_total: List of cosine similarity scores for each layer projection.
#     """
#     cos_total = []
#     idx = 0  # index for project_matrix
#     B = None  # Placeholder for rank-r weight
#     s_per_layer = []
#     for name, param in peft_model.named_parameters():
#         if 'lora' in name:

#             # Identify the rank-r weight (LoRA B matrix)
#             if param.shape[0] == peft_config.r:
#                 B = param.data.clone()  # use clone instead of deepcopy for safety
#                 continue
            
#             if param.shape[0] != peft_config.r:
#                 # Skip if B is not yet initialized
#                 if B is None:
#                     raise ValueError(f"LoRA B matrix not found before layer {name}. Check peft_config.r.")
                
#                 # Project current layer weight
#                 P = project_matrix[idx].to(param.device)
#                 W = torch.mm(P, param.data)
#                 fW = torch.mm(W, B)
#                 ori = torch.mm(param.data, B)
                
#                 # Compute cosine similarity
#                 cos = float(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)).item())
#                 cos_total.append(np.round(cos, 5))
                
#                 idx += 1

#                 # S-layer term: 1 / (1 + ||CΔW - ΔW||_2). Use Frobenius norm for matrices.
#                 diff = fW - ori
#                 # Frobenius norm == l2 over all entries
#                 diff_norm = torch.norm(diff, p='fro')
#                 s_i = (1.0 / (1.0 + diff_norm)).item()
#                 s_per_layer.append(float(np.round(s_i, 8)))

#                 B = None

#     S_total = float(np.round(float(sum(s_per_layer)), 8))
#     return {
#         "cos_per_layer": cos_total,
#         "s_per_layer": s_per_layer,
#         "S_total": S_total,
#     }




# def projected_weighted(peft_model, project_matrix):
#     """
#     Compute cosine similarity between projected LoRA weights and original weights.
    
#     Args:
#         peft_model: The LoRA-adapted model (nn.Module).
#         peft_config: Configuration object containing `r` (LoRA rank).
#         project_matrix: List of projection matrices (torch.Tensor).

#     Returns:
#         cos_total: List of cosine similarity scores for each layer projection.
#     """
#     cos_total = []
#     idx = 0  # index for project_matrix
#     B = None  # Placeholder for rank-r weight
#     s_per_layer = []
#     for name, param in peft_model.items():
#         if 'lora' in name:

#             # Identify the rank-r weight (LoRA B matrix)
#             if param.shape[0] == 32:
#                 B = param.clone()  # use clone instead of deepcopy for safety
#                 continue
            
#             if param.shape[0] != 32:
#                 # Skip if B is not yet initialized
#                 if B is None:
#                     raise ValueError(f"LoRA B matrix not found before layer {name}. Check peft_config.r.")
                
#                 # Project current layer weight
#                 P = project_matrix[idx].to(param.device)
#                 # breakpoint()
#                 W = torch.mm(P, param)
#                 fW = torch.mm(W, B)
#                 ori = torch.mm(param, B)

#                 # Compute cosine similarity
#                 cos = float(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)).item())
#                 cos_total.append(np.round(cos, 5))
                
#                 idx += 1

#                 # S-layer term: 1 / (1 + ||CΔW - ΔW||_2). Use Frobenius norm for matrices.
#                 diff = fW - ori
#                 # Frobenius norm == l2 over all entries
#                 diff_norm = torch.norm(diff, p='fro')
#                 s_i = (1.0 / (1.0 + diff_norm)).item()
#                 s_per_layer.append(float(np.round(s_i, 8)))

#                 B = None

#     S_total = float(np.round(float(sum(s_per_layer)), 8))
#     return {
#         "cos_per_layer": cos_total,
#         "s_per_layer": s_per_layer,
#         "S_total": S_total,
#     }

def projected_weighted(peft_model, delta_matrix):
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
    for name, param in peft_model.items():
        if 'lora' in name:

            # Identify the rank-r weight (LoRA B matrix)
            if param.shape[0] < 100: 
                B = param.clone()  # use clone instead of deepcopy for safety
                continue
            
            if param.shape[0] > 100:
                # Skip if B is not yet initialized
                if B is None:
                    raise ValueError(f"LoRA B matrix not found before layer {name}. Check peft_config.r.")
                
                # Project current layer weight
                V = delta_matrix[idx].to(param.device)
                # if os.environ.get('SAFELORA_REFERENCE_NAME') == 'diffllmlatdposftsafe':
                #     V = -V
                if 'dpo' in os.environ.get('SAFELORA_REFERENCE_NAME') and 'safe' in os.environ.get('SAFELORA_REFERENCE_NAME'):
                    V = -V
                # breakpoint()
                W = torch.mm(param, B)

                cos_val = torch.nn.functional.cosine_similarity(W.view(-1), -V.view(-1), dim=0).item()
                cos_total.append(np.round(cos_val, 5))
                B = None
                idx += 1

    S_total = float(np.round(float(sum(s_per_layer)), 8))
    Cos_total = float(np.round(float(sum(cos_total)), 8))
    return {
        "cos_per_layer": cos_total,
        "s_per_layer": s_per_layer,
        "S_total": S_total,
        "Cos_total": Cos_total,
    }


def consider_past_history(upto_current_local_dict, current_local_dict, global_dict, alpha=0.5):
    for key in upto_current_local_dict.keys():
        upto_current_local_dict[key] += current_local_dict[key] - global_dict[key]
    return upto_current_local_dict

# with open('project_matrix_safelora_torch.float32.pkl', 'rb') as f:
#     project_matrix = pickle.load(f)

# def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None, project_matrix=None):
    
#     save_data = {
#         "round_idx": round_idx,
#         "clients": clients_this_round,
#         "S_total": None,
#         "km_labels": None
#     }
#     n_clients = len(clients_this_round)

#     for client in clients_this_round:
#         local_dict = local_dict_list[client]
#         if round == 1:
#             initial_local_dict = local_dict_list[client]
#             final_local_dict = initial_local_dict
#         else:
#             final_local_dict = consider_past_history(initial_local_dict, local_dict, global_dict)
#         # print(local_dict)
#         safe_lora_data = projected_weighted(final_local_dict, project_matrix)

#     y = safe_lora_data['S_total']

#     y = np.array(y).reshape(-1, 1)
#     km = KMeans(n_clusters=2, n_init=5, random_state=0)
#     km_labels = km.fit_predict(y)
#     cnt_km = Counter(km_labels)
#     maj_km, maj_km_n = cnt_km.most_common(1)[0]
    
#     # majority cluster is benign
#     kept_client_ids = []
#     for idx, client in enumerate(clients_this_round):
#         if km_labels[idx] == maj_km:
#             kept_client_ids.append(client)
#         else:
#             print(f"Client {client} is removed by SafeLoRA aggregation in round {round_idx}.")

#     save_data["S_total"] = y.tolist()
#     save_data["km_labels"] = km_labels.tolist()

    

#     sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
#     for key in global_dict.keys():
#         global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in kept_client_ids])
#     return global_dict


from copy import deepcopy
from sklearn.mixture import GaussianMixture
import numpy as np
import json
import os

# --- Core Aggregation Function ---
def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None, project_matrix=None, script_args=None):
    
    n_clients = len(clients_this_round)
    

    # this function assumes full participation of clients
    safe_lora_data = {}
    for client in clients_this_round:
        local_dict = deepcopy(local_dict_list[client])
        safe_lora_data[client] = projected_weighted(local_dict, project_matrix)
    
    
    S_total = {client: safe_lora_data[client]['Cos_total'] for client in clients_this_round}
    S_values = np.array([S_total[c] for c in clients_this_round], dtype=float).reshape(-1, 1)
    layerwise_S = {client: safe_lora_data[client]['cos_per_layer'] for client in clients_this_round}
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=getattr(fed_args, "seed", 0)
    )
    gmm.fit(S_values)
    means = gmm.means_.flatten()
    low_mean_component = np.argmin(means)

    probs  = gmm.predict_proba(S_values)[:, low_mean_component]

    prob_threshold = getattr(fed_args, "safelora_gmm_threshold", 0.8)
    selected_clients = [
        client for client, p in zip(clients_this_round, probs) if 
        p >= prob_threshold
    ]
    # selected_clients = [0, 1, 2, 3, 4] #hardcoded for debug

    if len(selected_clients) == 0:
        # select the top half clients if all are detected as malicious
        # sorted_clients = sorted(
        #     zip(clients_this_round, probs), 
        #     key=lambda x: x[1], 
        #     reverse=True
        # )
        # selected_clients = [client for client, _ in sorted_clients[: max(1, n_clients // 2)]]
        assert 1 == 0, "All clients are detected as malicious by SafeLoRA aggregation."
        print("All clients were detected as malicious by SafeLoRA aggregation, defaulting to using top half clients as benign")
        # assert 1 == 0, "All clients are detected as malicious by SafeLoRA aggregation."



    sample_this_round = sum([sample_num_list[client] for client in selected_clients])
    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in selected_clients])
        # average instead
        # global_dict[key] = sum([local_dict_list[client][key] for client in selected_clients]) / len(selected_clients)
                
    

    safe_lora_path = f'{script_args.output_dir}/safelorav2/Steps[{script_args.max_steps}]_Clients[{fed_args.sample_clients}]_ISA[{script_args.isa}]/'
    os.makedirs(safe_lora_path, exist_ok=True)
    safe_lora_data = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "S_total": S_values.flatten().tolist(),
        "gmm_means": means.tolist(),
        'gmm_probs': probs.flatten().tolist(),
        "selected_clients": selected_clients,
        'layerwise_S': layerwise_S,
    }
    with open(os.path.join(safe_lora_path, f"round_{round_idx}_safelora_gmm.json"), 'w') as f:
        json.dump(safe_lora_data, f)
    other_path = os.path.join(script_args.output_dir, 'safelorav2')
    os.makedirs(other_path, exist_ok=True)
    with open(os.path.join(other_path, f"round_{round_idx}.json"), 'w') as f:
        json.dump(safe_lora_data, f)
    return global_dict

if __name__ == "__main__":
    device = 'cpu'
    get_aligned_matrix(device=device)
    
