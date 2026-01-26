from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
import pickle
import torch
import numpy as np
import os
import json 
from sklearn.cluster import KMeans
from collections import Counter

from copy import deepcopy
from sklearn.mixture import GaussianMixture
import numpy as np
import json

# --- Core Aggregation Function ---
def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None, project_matrix=None, script_args=None, asr_rates=None):
    
    n_clients = len(clients_this_round)
    
    S_total = {client: asr_rates[client] for client in clients_this_round}
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

    prob_threshold = getattr(fed_args, "evalfilter_gmm_threshold", 0.8)
    selected_clients = [
        client for client, p in zip(clients_this_round, probs) if 
        p >= prob_threshold
    ]

    if len(selected_clients) == 0:
        assert 1 == 0, "All clients are detected as malicious by SafeLoRA aggregation."
    



    sample_this_round = sum([sample_num_list[client] for client in selected_clients])
    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in selected_clients])
                
    

    safe_lora_path = f'./output/evalfilter/{script_args.model_name_or_path}/C{fed_args.sample_clients}_N{fed_args.num_rounds}_benign[{"_".join([str(n) for n in fed_args.benign_num_clients])}][{"_".join([ds for ds in fed_args.benign_dataset_names])}]_malicious[{"_".join([str(n) for n in fed_args.malicious_num_clients])}][{"_".join([ds for ds in fed_args.malicious_dataset_names])}]_Steps[{script_args.max_steps}]_Clients[{fed_args.sample_clients}]_ISA[{script_args.isa}]/'
    os.makedirs(safe_lora_path, exist_ok=True)

    safe_lora_data = {
        "round_idx": round_idx,
        "clients": clients_this_round,
        "S_total": S_values.flatten().tolist(),
        "gmm_means": means.tolist(),
        'gmm_probs': probs.flatten().tolist(),
        "selected_clients": selected_clients
    }
    with open(os.path.join(safe_lora_path, f"round_{round_idx}_evalfilter_gmm.json"), 'w') as f:
        json.dump(safe_lora_data, f)
    return global_dict


    
