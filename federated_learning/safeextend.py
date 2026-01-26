from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
import pickle
import torch
import numpy as np
import os
import json 
from sklearn.cluster import KMeans
from collections import Counter



def projected_weighted_extend(peft_model, project_matrix):
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
                B_orig = param.data

                B_C = torch.matmul(P, B_orig)
                B_IC = B_orig - B_C

                B_new = -1*B_C + B_IC
                param.data = B_new
    return peft_model




from copy import deepcopy
from sklearn.mixture import GaussianMixture
import numpy as np
import json

# --- Core Aggregation Function ---
def aggr(global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, fed_args, proxy_dict=None, output_dir=None, project_matrix=None, script_args=None):
    
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    

    # this function assumes full participation of clients
    for client in clients_this_round:
        local_dict = deepcopy(local_dict_list[client])
        local_dict_list[client] = projected_weighted_extend(local_dict, project_matrix)
        
    for key in global_dict.keys():
            global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    return global_dict

    
