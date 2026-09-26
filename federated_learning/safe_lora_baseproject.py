from transformers import AutoModelForCausalLM
import pickle
import torch
import os

# export HF_TOKEN='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'
os.environ['HF_TOKEN'] = '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'
os.environ['HF_DATASETS_CACHE'] = '/scratch/ps9044/huggingface/datasets'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/shared/rc/llm-degredation/huggingface/hub'
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'
os.environ['HF_HOME'] = '/shared/rc/llm-degredation/huggingface/hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT_URL'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT_TOKEN'] = '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'
# family -> (base = non-instruct/non-chat, aligned = instruct/chat)
MODELS = {
    # 'llama2': ('meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf'),
    # 'llama3': ('meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B-Instruct'),
    # 'qwen':   ('Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct'),
    # 'gemma2': ('google/gemma-2-2b', 'google/gemma-2-2b-it'),
    'qwen3': ('Qwen/Qwen3-4B-Instruct-2507', 'Qwen/Qwen3-4B-Instruct-2507'),
    'llama3_0': ('meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'),
}


def get_aligned_matrix(device='cpu'):
    """
    Get projected matrix from the difference between the aligned (instruct/chat)
    and base (non-instruct/non-chat) models, for each model family.
    """
    target_modules = ['q_proj', 'v_proj']

    for save_dir, (base_name, aligned_name) in MODELS.items():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        aligned_model = AutoModelForCausalLM.from_pretrained(
            aligned_name,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print(save_dir, base_model.dtype)

        v = []
        for (b_name, b_param), (a_name, a_param) in zip(base_model.named_parameters(), aligned_model.named_parameters()):
            if any(module in a_name for module in target_modules) and 'bias' not in a_name:
                assert b_param.shape == a_param.shape, "The dimensions of the base model's weight should be the same with the aligned model's weight."
                vec = a_param - b_param
                vec = vec.to(device)
                vec = torch.mm(vec, vec.t()) / torch.norm(vec)
                v.append((vec).detach().cpu())

        os.makedirs(f'/shared/rc/llm-degredation/references/{save_dir}', exist_ok=True)
        with open(f'/shared/rc/llm-degredation/references/{save_dir}/project_matrix_safelora_{base_model.dtype}_aligned.pkl', 'wb') as f:
            pickle.dump(v, f)
            print("Saved projection matrix to ", f'/shared/rc/llm-degredation/safelora_base_references/{save_dir}/project_matrix_safelora_{base_model.dtype}_aligned.pkl')


if __name__ == "__main__":
    device = 'cpu'
    get_aligned_matrix(device=device)
