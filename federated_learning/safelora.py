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
    base_model = AutoModelForCausalLM.from_pretrained(
        # 'meta-llama/Llama-2-7b-hf',
        '/scratch/ps9044/fedllm/safelora_maliciousgen_llama27bchat/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251113142234/full-50',
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )
    base_model_for_peft = AutoModelForCausalLM.from_pretrained(
        # 'meta-llama/Llama-2-7b-hf',
        '/scratch/ps9044/fedllm/safelora_maliciousgen_llama27bchat/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251113142234/full-50',
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )
    aligned_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )

    print(base_model.dtype)
    #Fed-134, fedgraph
    checkpoint_path = '/scratch/ps9044/fedllm/barebones/WildChat7_BeaverTails3_500_fedgraph_c10s10_i10_b16a1_l512_r32a64_20251030220628/checkpoint-10'

    peft_model = PeftModel.from_pretrained(base_model_for_peft, checkpoint_path, is_trainable=False).to(device)
    peft_config = peft_model.peft_config['default']
    print(peft_config.target_modules)

    # peft_config = LoraConfig(
    #     r=32,
    #     lora_alpha=64,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # get target modules
    # target_modules = ['q_proj', 'v_proj']
    
    v = []
    proj_modules = list(peft_config.target_modules)
    for (b_name, b_param) , (a_name, a_param) in zip (base_model.named_parameters(), aligned_model.named_parameters()):
        if any(module in a_name for module in proj_modules):
            assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
            vec = a_param - b_param
            vec = vec.to(device)
            vec = torch.mm(vec, vec.t()) / torch.norm(vec)
            v.append((vec).detach().cpu())
    # save v to ../project_matrix.pkl
    with open(f'../project_matrix_safelora_{base_model.dtype}_harmful.pkl', 'wb') as f:
        pickle.dump(v, f)



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

def projected_weighted(peft_model, project_matrix):
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
            if param.shape[0] == 32:
                B = param.clone()  # use clone instead of deepcopy for safety
                continue
            
            if param.shape[0] != 32:
                # Skip if B is not yet initialized
                if B is None:
                    raise ValueError(f"LoRA B matrix not found before layer {name}. Check peft_config.r.")
                
                # Project current layer weight
                P = project_matrix[idx].to(param.device)
                # breakpoint()
                W = torch.mm(P, param)
                fW = torch.mm(W, B)
                ori = torch.mm(param, B)

                # Compute cosine similarity
                cos = float(torch.nn.functional.cosine_similarity(fW.reshape(1, -1), ori.reshape(1, -1)).item())
                cos_total.append(np.round(cos, 5))
                
                idx += 1

                # S-layer term: 1 / (1 + ||CΔW - ΔW||_2). Use Frobenius norm for matrices.
                diff = fW - ori
                # Frobenius norm == l2 over all entries

                
                # diff_norm = torch.norm(diff, p='fro')
                diff_norm = torch.norm(fW, p='fro')
                s_i = (1.0 / (1.0 + diff_norm)).item()
                s_per_layer.append(float(np.round(s_i, 8)))

                B = None

    S_total = float(np.round(float(sum(s_per_layer)), 8))
    return {
        "cos_per_layer": cos_total,
        "s_per_layer": s_per_layer,
        "S_total": S_total,
    }


def consider_past_history(upto_current_local_dict, current_local_dict, global_dict, alpha=0.5):
    for key in upto_current_local_dict.keys():
        upto_current_local_dict[key] += current_local_dict[key] - global_dict[key]
    return upto_current_local_dict

# with open('project_matrix_safelora_torch.float32.pkl', 'rb') as f:
#     project_matrix = pickle.load(f)

def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None, project_matrix=None):
    
    save_data = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "S_total": None,
        "km_labels": None
    }
    n_clients = len(clients_this_round)

    for client in clients_this_round:
        local_dict = local_dict_list[client]
        if round == 1:
            initial_local_dict = local_dict_list[client]
            final_local_dict = initial_local_dict
        else:
            final_local_dict = consider_past_history(initial_local_dict, local_dict, global_dict)
        # print(local_dict)
        safe_lora_data = projected_weighted(final_local_dict, project_matrix)

    y = safe_lora_data['S_total']

    y = np.array(y).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=5, random_state=0)
    km_labels = km.fit_predict(y)
    cnt_km = Counter(km_labels)
    maj_km, maj_km_n = cnt_km.most_common(1)[0]
    
    # majority cluster is benign
    kept_client_ids = []
    for idx, client in enumerate(clients_this_round):
        if km_labels[idx] == maj_km:
            kept_client_ids.append(client)
        else:
            print(f"Client {client} is removed by SafeLoRA aggregation in round {round_idx}.")

    save_data["S_total"] = y.tolist()
    save_data["km_labels"] = km_labels.tolist()

    

    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in kept_client_ids])
    return global_dict


from copy import deepcopy
from sklearn.mixture import GaussianMixture
import numpy as np
import json

# --- Core Aggregation Function ---
def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None, project_matrix=None, script_args=None):
    
    n_clients = len(clients_this_round)
    

    # this function assumes full participation of clients
    safe_lora_data = {}
    for client in clients_this_round:
        local_dict = deepcopy(local_dict_list[client])
        safe_lora_data[client] = projected_weighted(local_dict, project_matrix)
        
    
    S_total = {client: safe_lora_data[client]['S_total'] for client in clients_this_round}
    S_values = np.array([S_total[c] for c in clients_this_round], dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=getattr(fed_args, "seed", 0)
    )
    gmm.fit(S_values)
    means = gmm.means_.flatten()
    high_mean_component = np.argmax(means)

    probs  = gmm.predict_proba(S_values)[:, high_mean_component]

    prob_threshold = getattr(fed_args, "safelora_gmm_threshold", 0.8)
    selected_clients = [
        client for client, p in zip(clients_this_round, probs) if 
        p >= prob_threshold
    ]

    if len(selected_clients) == 0:
        assert 1 == 0, "All clients are detected as malicious by SafeLoRA aggregation."
    



    sample_this_round = sum([sample_num_list[client] for client in selected_clients])
    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in selected_clients])
                
    

    safe_lora_path = f'./output/safelora/{script_args.model_name_or_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}][{"_".join([ds for ds in fed_args.benign_dataset_names])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}][{"_".join([ds for ds in fed_args.malicious_dataset_names])}]_Steps[{script_args.max_steps}]_Clients[{fed_args.sample_clients}]_ISA[{script_args.isa}]/'
    os.makedirs(safe_lora_path, exist_ok=True)

    safe_lora_data = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "S_total": S_values.flatten().tolist(),
        "gmm_means": means.tolist(),
        'gmm_probs': probs.flatten().tolist(),
        "selected_clients": selected_clients
    }
    with open(os.path.join(safe_lora_path, f"round_{round_idx}_safelora_gmm.json"), 'w') as f:
        json.dump(safe_lora_data, f)
    return global_dict

if __name__ == "__main__":
    device = 'cpu'
    get_aligned_matrix(device=device)
    
