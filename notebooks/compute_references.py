
paths = {
    'isa': '/scratch/ps9044_copy/references/isa1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317000351',
    'maliciousgen': '/scratch/ps9044_copy/references/MaliciousGen1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317001224',
    'expguardtrain': '/scratch/ps9044_copy/references/expguardtrain1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317001225',
    'pubmedqa': '/scratch/ps9044_copy/references/PubMedQA1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317001330',
    'cord19': '/scratch/ps9044_copy/references/cord191__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317004053',
    'medqa': '/scratch/ps9044_copy/references/medQA1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317004025',
    'emrqa': '/scratch/ps9044_copy/references/emrqa1__0_500_fedavg_c1s1_i10_b16a1_l512_r32a64_20260317004042'
}

from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
import pickle
import torch
import numpy as np
import os
import json 
from sklearn.cluster import KMeans
from collections import Counter

def merge_to_full_matrix(checkpoint_dir):
    command = f'python /home/ps9044/RPA/fedllm-attack/utils/merge_lora.py --lora_path {checkpoint_dir} --base_model_path meta-llama/Llama-2-7b-chat-hf'
    print(command)
    os.system(command)


def get_aligned_matrix(device='cpu'):
    """
    Get projected matrix by following the config (target_modules) from the peft model.
    The dimensions between the base model's weights and the aligned model's weights should be the same.
    """
    aligned_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16
    )
    # print(base_model.dtype)
    #Fed-134, fedgraph
    # checkpoint_path = '/scratch/ps9044/fedllm/barebones/WildChat7_BeaverTails3_500_fedgraph_c10s10_i10_b16a1_l512_r32a64_20251030220628/checkpoint-10'
    checkpoint_path = '/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743/checkpoint-10'

    

    for dataset_name, path in paths.items():
        print(f"Processing dataset: {dataset_name}")
        base_path = os.path.join(path, 'checkpoint-50')
        merge_to_full_matrix(base_path)
        base_path = base_path.replace('checkpoint-50', 'full-50')

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            # '/scratch/ps9044/fedllm/safelora_maliciousgen_llama27bchat/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251113142234/full-50',
            # '/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743/full-50',
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
            # torch_dtype=torch.bfloat16
        )
        base_model_for_peft = AutoModelForCausalLM.from_pretrained(
            base_path,
            # '/scratch/ps9044/fedllm/safelora_maliciousgen_llama27bchat/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251113142234/full-50',
            # '/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743/full-50',
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
            # torch_dtype=torch.bfloat16
        )
        peft_model = PeftModel.from_pretrained(base_model_for_peft, checkpoint_path, is_trainable=False).to(device)
        peft_config = peft_model.peft_config['default']
        print(peft_config.target_modules)
        v = []
        proj_modules = list(peft_config.target_modules)
        for (b_name, b_param) , (a_name, a_param) in zip (base_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in proj_modules):
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(device)
                vec1 = torch.mm(vec, vec.t()) / torch.norm(vec)
                vec2 = vec @ torch.linalg.pinv(vec.t() @ vec) @ vec.t()
                v.append((vec).detach().cpu())
        # save v to ../project_matrix.pkl
        with open(f'/scratch/ps9044_copy/references/delta_matrix_safelora_{base_model.dtype}_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(v, f)
        with open(f'/scratch/ps9044_copy/references/project_matrix_safelora_{base_model.dtype}_{dataset_name}_correct.pkl', 'wb') as f:
            pickle.dump(vec2, f)
        with open(f'/scratch/ps9044_copy/references/project_matrix_safelora_{base_model.dtype}_{dataset_name}_incorrect.pkl', 'wb') as f:
            pickle.dump(vec1, f)
            

    
if __name__ == "__main__":
    get_aligned_matrix(device='cpu')
    # peft_config = LoraConfig(
    #     r=32,
    #     lora_alpha=64,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # get target modules
    # target_modules = ['q_proj', 'v_proj']
    
    