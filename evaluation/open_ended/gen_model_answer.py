import datasets
import argparse
import json
import sys
sys.path.append("../../")
from tqdm import tqdm
import os
import requests
from datasets import Dataset
import torch
import pdb
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["VLLM_USE_V1"] = "0"


"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""


ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nFirst think step by step and then answer the final number.\n"

sst2_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response: {}{}"

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
}


def _to_chat_format(example):
    # Try to get a system prompt from the conversation template helper
    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

    # Build user content from instruction + (optional) input
    user = example.get("instruction", "")
    if example.get("input"):
        user = user + "\n" + example.get("input")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    # If a response exists, append it as the assistant message (useful for SFT)
    if "response" in example and example.get("response") is not None:
        messages.append({"role": "assistant", "content": example.get("response")})

    example["messages"] = messages
    return example

def format_llama2_chat(
    messages,
    bos_token="<s>",
    eos_token="</s>",
):
    """
    Format a conversation using the official Llama-2-Chat template.

    Parameters
    ----------
    messages : list[dict]
        [{"role": "system"|"user"|"assistant", "content": str}, ...]
        After the optional first 'system' message, roles MUST alternate:
        user / assistant / user / assistant / ...
    bos_token : str
        Typically tokenizer.bos_token (for Llama-2 this is "<s>").
    eos_token : str
        Typically tokenizer.eos_token (for Llama-2 this is "</s>").

    Returns
    -------
    str : the formatted prompt string.
    """

    if not messages:
        raise ValueError("messages must be non-empty")

    # ----- Optional system message -----
    idx = 0
    system_message = ""
    first = messages[0]
    if first["role"] == "system":
        # '<<SYS>>\n{system}\n<</SYS>>\n\n'
        system_message = (
            "<<SYS>>\n"
            + first["content"].strip()
            + "\n<</SYS>>\n\n"
        )
        idx = 1

    loop_messages = messages[idx:]
    if not loop_messages:
        raise ValueError("After the optional system message, at least one user message is required.")

    # ----- Enforce role alternation -----
    # loop_messages[0] must be 'user', then 'assistant', then 'user', ...
    for i, m in enumerate(loop_messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected_role:
            raise ValueError(
                "Conversation roles must alternate user/assistant/user/assistant/... "
                f"(got role='{m['role']}' at position {i}, expected '{expected_role}')"
            )

    # ----- Build the prompt -----
    chunks = []
    for i, m in enumerate(loop_messages):
        # First user message gets system_message prepended
        if i == 0:
            content = system_message + m["content"]
        else:
            content = m["content"]

        content = content.strip()

        if m["role"] == "user":
            # bos_token + "[INST] " + content + " [/INST]"
            chunks.append(f"{bos_token}[INST] {content} [/INST]")
        else:  # assistant
            # " " + content + " " + eos_token
            chunks.append(f" {content} {eos_token}")

    return "".join(chunks)




def build_chat_template(example):
    instruction_only = {'instruction': example['instruction']}
    messages = _to_chat_format(instruction_only)['messages']
    prompt = format_llama2_chat(messages=messages)
    return prompt


def get_formatting_prompts_func(template_name, eos_token):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        return output_texts    
    
    return formatting_prompts_func, response_temp


parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--use_vllm", action="store_true")
parser.add_argument("--bench_name", type=str, default="vicuna")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
print(args)

if args.use_vllm and args.lora_path is not None:
    raise ValueError("Cannot use both VLLM and LORA, need to merge the lora and then use VLLM")

template = TEMPLATE_DICT[args.template][0]
print(f">> You are using template: {template}")

def gsm8k_format(example):
    example['instruction'] = f"{example['question']}{QUESTION_PROMPT}"
    example['response'] = f"{example['answer']}".replace("#### ", ANSWER_PROMPT)
    return example


def sst2_format(example):
    instance = {}
    instance["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative"
    instance["input"] = example["sentence"]
    instance["instruction"] = instance["instruction"] + "\n\nInput: " + instance["input"]
    instance["label"] = example["label"]
    return instance

def squadv2_format(example):
    example['instruction'] = f"""Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
```json
{{
"answer": ...
}}
```
If the answer is not in the context, the answer should be "?".
Context: {example["context"]}
Question: {example["question"]}"""

    if len(example['answers']['text']) > 0:
        example['response'] = example['answers']['text'][0]
    else:
        example['response'] = "?"

    example['response'] = f"""```json
{{
"answer": "{example['response']}"
}}
```"""
    return example

def pubmedqa_format(example):
    example_context = '\n'.join(example['context'])
    example['instruction'] = "Your task is to answer biomedical questions using the given context. Output a Long Answer followed by the Final Decision: yes, no or maybe.\n\nAbstract: " + example_context + "\n\nQuestion: " + example['question']
    example['response'] = f"Long Answer: {example['long_answer']}\nFinal Decision: {example['final_decision']}"
    return example


def medqa_format(example):
    options = example['options']
    options_str = '\n'.join([f"{key}. {value}" for key, value in options.items()])
    example['instruction'] = f"Answer the following medical question by choosing the correct option from A, B, C, or D.\n\nQuestion: {example['question']}\nOptions:\n{options_str}\n\nProvide your answer in the format: 'The correct answer is: X', where X is A, B, C, or D."
    correct_option = example['answer_idx'] 
    example['response'] = f"The correct answer is: {correct_option}"
    return example


# ============= Load dataset =============
if args.bench_name == "alpaca":
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    max_new_tokens = 2048
elif args.bench_name == "vicuna":
    eval_set = datasets.load_dataset("json", data_files="data/vicuna/question.jsonl")['train']
    # eval_set = eval_set.rename_column("text", "instruction")
    def rename(example):
        example['instruction'] = example['turns'][0]
        return example
    eval_set = eval_set.map(rename)
    max_new_tokens = 2048
elif args.bench_name == "advbench":
    eval_set = datasets.load_dataset("csv", data_files="data/advbench/advbench.csv")["train"]
    eval_set = eval_set.rename_column("goal", "instruction")
    eval_set = eval_set.remove_columns(["target"])
    max_new_tokens = 1024

elif args.bench_name == 'val_postfinetune':
    eval_set = datasets.load_dataset('json', data_files='../../gen_data/Mistral/val_benignQA.json')['train']
    max_new_tokens = 1024

elif args.bench_name == 'maliciousgen':
    eval_set = datasets.load_dataset('json', data_files='../../gen_data/MaliciousGen_train_sft.json')['train']
    max_new_tokens = 1024
elif 'gsm8k' in args.bench_name:
    eval_set = datasets.load_dataset("HongzheBi/gsm8k", split='test')
    eval_set = eval_set.map(gsm8k_format)
    # only use first 500
    eval_set = eval_set.select(range(500))
    max_new_tokens = 512
elif 'sst2' in args.bench_name:
    eval_set = datasets.load_dataset("stanfordnlp/sst2", split='validation')
    eval_set = eval_set.map(sst2_format)
    # only use first 500
    eval_set = eval_set.select(range(500))
    max_new_tokens = 128
elif 'ssttrain' in args.bench_name:
    eval_set = datasets.load_dataset('json', data_files='/home/ps9044/FedLLM-Attack/gen_data/sst2_train_sft.jsonl')['train']
    eval_set = eval_set.map(sst2_format)
    eval_set = eval_set.shuffle(seed=2023).select(range(500))
    max_new_tokens = 128
elif 'squad_v2' in args.bench_name:
    eval_set = datasets.load_dataset('rajpurkar/squad_v2', split='validation')
    eval_set = eval_set.map(squadv2_format)
    eval_set = eval_set.select(range(500))

    max_new_tokens = 200
elif 'squadv2train' in args.bench_name:
    eval_set = datasets.load_dataset('json', data_files='../../gen_data/squad_v2_train_sft.jsonl')['train']
    eval_set = eval_set.map(squadv2_format)
    eval_set = eval_set.shuffle(seed=2023).select(range(500))
    max_new_tokens = 200


elif 'pubmedqa' in args.bench_name:
    # eval_set = datasets.load_dataset("qiaojin/PubMedQA",'pqa_labeled')['train']
    PQA_L_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"

    def load_pubmedqa_labeled_raw():
        data = requests.get(PQA_L_URL).json()
        # data is a dict: {pubid_str: {...}, ...}
        rows = []
        for pubid, row in data.items():
            rows.append({
                "pubid": int(pubid),
                "question": row["QUESTION"],
                "context": row["CONTEXTS"],  # string or list of strings depending on file
                "long_answer": row["LONG_ANSWER"],
                "final_decision": row["final_decision"],
            })
        return Dataset.from_list(rows)
    eval_set = load_pubmedqa_labeled_raw()
    eval_set = eval_set.map(pubmedqa_format)
    eval_set = eval_set.select(range(500))

    max_new_tokens = 512


elif 'pubmedval' in args.bench_name:
    eval_set = datasets.load_dataset('json', data_files='../../gen_data/PubMedQA_validation_sft.jsonl')['train']
    eval_set = eval_set.map(pubmedqa_format)
    eval_set = eval_set.shuffle(seed=2023).select(range(500))
    max_new_tokens = 512

elif 'pubmedtrain' in args.bench_name:
    eval_set = datasets.load_dataset('json', data_files='../../gen_data/PubMedQA_train_sft.jsonl')['train']
    # keep only instruction and response columns
    eval_set = eval_set.remove_columns([col for col in eval_set.column_names if col not in ['instruction', 'response']])
    # randomly select 500 samples with seed 2023
    eval_set = eval_set.shuffle(seed=2023).select(range(500))
    max_new_tokens = 512


elif 'medQA' in args.bench_name:
    eval_set = datasets.load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    eval_set = eval_set.map(medqa_format)
    eval_set = eval_set.select(range(500))
    max_new_tokens = 200



else:
    raise ValueError("Invalid benchmark name")

# ============= Extract model name from the path. The name is used for saving results. =============
exp_name = None
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model

if exp_name is None:
    exp_name = args.base_model_path # for example the meta-llama/Llama-2-7b-hf

# ============= Load previous results if exists =============
if args.use_vllm:
    result_path = f"./data/{args.bench_name}/model_answer/{model_name}_vllm_chat_greedy.json"
else:
    result_path = f"./data/{args.bench_name}/model_answer/{model_name}.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)
if os.path.exists(result_path):
    with open(result_path, "r") as f:
        result_list = json.load(f)
    result_list = []  # comment this line to disable resuming and generate from scratch
else:
    result_list = []
existing_len = len(result_list)
print(f">> Existing length: {existing_len}")

print(len(eval_set))
# ============= Generate responses =============
if args.use_vllm:
    from vllm import LLM, SamplingParams
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"  # VLLM uses this env variable to set GPU device
    # os.environ["VLLM_TARGET_DEVICE"] = 'cpu'
    model = LLM(model=args.base_model_path, enforce_eager=True, gpu_memory_utilization=0.4)
    if args.bench_name == "advbench" or args.bench_name == 'maliciousgen':
        # input_list = [template.format(example["instruction"]+'.', "", "")[:-1] for example in eval_set]
        input_list = [build_chat_template(example) for example in eval_set]
    
    elif 'gsm8k' in args.bench_name:
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
        input_list = [template.format(example["instruction"]) for example in eval_set] # no space at end so no -1
    elif 'sst2' in args.bench_name or 'ssttrain' in args.bench_name:
        template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
        # input_list = [template.format(example["instruction"], example["input"]) for example in eval_set] # no space at end so no -1
        input_list = [build_chat_template(example) for example in eval_set]
    elif 'squad_v2' in args.bench_name or 'pubmedqa' in args.bench_name or 'medQA' in args.bench_name or 'pubmedtrain' in args.bench_name or 'pubmedval' in args.bench_name or 'squadv2train' in args.bench_name:
        input_list = [build_chat_template(example) for example in eval_set]
    else:
        input_list = [template.format(example["instruction"], "", "")[:-1] for example in eval_set] # TODO: use fastchat conversation
    input_list = input_list[existing_len:]
    print(f">> Example input: {input_list[0]}")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    generations = model.generate(input_list, sampling_params)
    generations = [generation.outputs[0].text for generation in generations]

    for i, example in tqdm(enumerate(eval_set)):
        if i < existing_len:
            continue
        example['output'] = generations[i-existing_len]
        example['generator'] = exp_name
        result_list.append(example)
    with open(result_path, "w") as f:
        json.dump(result_list, f, indent=4)
    print("Saved to: ", result_path)
    print("model name: ", model_name)

else:

    device = f'cuda:{args.gpu}'
    model_ori = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to(device)
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model_ori, args.lora_path, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, use_fast=False)

        state_dict = torch.load(f"{args.lora_path}/adapter_model.bin", map_location='cpu')
        print("LoRA keys:", list(state_dict.keys())[:10])
        try:
            sum = state_dict['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'].sum().item()
        except:
            sum = state_dict['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'].sum().item()

        loaded_sum = model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.sum().item()

        assert abs(sum - loaded_sum) < 1e-4, f"checkpoint sum {sum} not equal to loaded sum {loaded_sum}"
        print(f"checkpoint sum {sum} equal to loaded sum {loaded_sum}, LoRA loaded successfully.")

    with torch.no_grad():
        model.eval()

        print("Starting generation for ", len(eval_set) - existing_len, " samples")
        for i, example in tqdm(enumerate(eval_set)):
            if i < existing_len:
                continue
            if args.bench_name == "advbench":
                instruction = template.format(example["instruction"]+'.', "", "")[:-1]
            else:
                instruction = template.format(example["instruction"], "", "")[:-1]      # TODO: use fastchat conversation
            input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
            output_ids = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
            output_ids = output_ids[0][len(input_ids[0]):]
            result = tokenizer.decode(output_ids, skip_special_tokens=True)
            example['output'] = result
            example['generator'] = model_name

            print(f"\nInput: \n{instruction}")
            print(f"\nOutput: \n{result}")
            print("="*100)
            result_list.append(example)
            with open(result_path, "w") as f:
                json.dump(result_list, f, indent=4)
print(f">> You are using template: {template}")
