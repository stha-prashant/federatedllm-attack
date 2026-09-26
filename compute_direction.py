from transformers import AutoModelForCausalLM
import pickle
import torch
import os
import argparse


def get_aligned_matrix(harmful_checkpoint_path, device='cpu'):
    """
    Build harmful-direction delta matrices from (aligned_chat - harmful_full).
    """
    model_name = 'meta-llama/Llama-2-7b-chat-hf'

    base_model = AutoModelForCausalLM.from_pretrained(
        harmful_checkpoint_path,
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16,
    )
    aligned_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=False,
        device_map="cpu",
        low_cpu_mem_usage=True,
        # torch_dtype=torch.bfloat16,
    )

    print(base_model.dtype)

    target_modules = ['q_proj', 'v_proj']
    v = []
    for (b_name, b_param), (a_name, a_param) in zip(base_model.named_parameters(), aligned_model.named_parameters()):
        if any(module in a_name for module in target_modules) and 'bias' not in a_name:
            assert b_param.shape == a_param.shape, "Base/aligned weight shapes must match."
            vec = (a_param - b_param).to(device)
            # Store float32 to match existing SafeLoRA matrix pickles.
            v.append(vec.detach().cpu().float())

    if 'Llama-2' in model_name or 'llama-2' in model_name.lower():
        save_dir = 'llama2'
    elif 'Llama-3.1' in model_name:
        save_dir = 'llama3'
    elif 'Qwen' in model_name:
        save_dir = 'qwen'
    elif 'gemma' in model_name.lower():
        save_dir = 'gemma2'
    else:
        raise ValueError(f"Model {model_name} not supported.")

    os.makedirs(save_dir, exist_ok=True)
    out_path = f'{save_dir}/harmful_reference.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(v, f)
    print("Saved projection matrix to", out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harmful_checkpoint_path", type=str, required=True)
    args = parser.parse_args()
    get_aligned_matrix(args.harmful_checkpoint_path)
