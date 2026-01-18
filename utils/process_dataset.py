import os
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset
import requests
import pandas as pd
from .conversation import get_conv_template
from functools import partial
from federated_learning.split_dataset import split_dataset
from datasets import disable_caching
import json
import pdb

# disable_caching()  # or add load_from_cache_file=False in the map or filter function

def get_sft_datasets(script_args, fed_args, tokenizer=None):

    dataset_list, num_client_list = [], []

    for benign_dataset_name, benign_num_clients in zip(fed_args.benign_dataset_names, fed_args.benign_num_clients):
        if benign_num_clients==0:
            continue
        benign_dataset_sample = int(benign_num_clients*fed_args.num_data_per_client)
        benign_dataset = get_whole_dataset(benign_dataset_name, script_args.local_data_dir)
        benign_dataset = benign_dataset.filter(partial(benign_filter_samples, dataset_name=benign_dataset_name))
        benign_dataset = process_sft_dataset(benign_dataset_name, benign_dataset, script_args.template, benign_dataset_sample, True, script_args.existing_lora is not None, tokenizer=tokenizer)
        # save the dataset to jsonl file
        # breakpoint()

        # save_path = os.path.join(script_args.local_data_dir, f"{benign_dataset_name.split('/')[-1]}_train_sft.jsonl")
        # data_list = benign_dataset.to_list()  # or benign_dataset.to_dict() if you prefer a column-wise dict
        # # Save to JSON file
        # with open(save_path, "w") as f:
        #     json.dump(data_list, f, indent=2, ensure_ascii=False)
        # exit()
        dataset_list.append(benign_dataset)
        num_client_list.append(benign_num_clients)

    for malicious_dataset_name, malicious_num_clients in zip(fed_args.malicious_dataset_names, fed_args.malicious_num_clients):
        if malicious_num_clients==0:
            continue
        malicious_dataset_sample = int(malicious_num_clients*fed_args.num_data_per_client)
        malicious_dataset = get_whole_dataset(malicious_dataset_name, script_args.local_data_dir)
        malicious_dataset = malicious_dataset.filter(partial(malicious_filter_samples, dataset_name=malicious_dataset_name))
        malicious_dataset = process_sft_dataset(malicious_dataset_name, malicious_dataset, script_args.template, malicious_dataset_sample, False, tokenizer=tokenizer)
        dataset_list.append(malicious_dataset)
        num_client_list.append(malicious_num_clients)
        
    return dataset_list, num_client_list


def get_safety_sft_datasets(script_args, fed_args, tokenizer=None):
    dataset = get_whole_dataset(fed_args.safety_dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(fed_args.safety_dataset_name, dataset, script_args.template, script_args.safety_num_samples, False, tokenizer=tokenizer)
    return dataset

def get_whole_dataset(dataset_name, local_data_dir=None):

    if dataset_name == 'zhiqings/dromedary-65b-verbose-clone-v0':
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir is not None else dataset_name
        data_files = os.path.join(dataset_name, 'merged_behavior_clone.json')
        dataset = load_dataset('json', data_files=data_files, split='train')
    elif dataset_name in ['allenai/WildChat', 'lmsys/lmsys-chat-1m']:
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir else dataset_name
        dataset = load_dataset(dataset_name, split="train")
    elif dataset_name == 'MaliciousGen':
        data_files = os.path.join('gen_data', 'Mistral/maliciousQA.json')
        dataset = load_dataset('json', data_files=data_files, split='train')   
    elif dataset_name == 'benignQA+helpfulQA': # level 2
        dataset_1 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Mistral/benignQA.json'), split='train')
        dataset_2 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Mistral/helpfulQA.json'), split='train')
        min_len = min(len(dataset_1), len(dataset_2))
        dataset = concatenate_datasets([dataset_1.select(range(min_len)), dataset_2.select(range(min_len))])
    elif dataset_name in ('Lmsys7_BT3', 'Wildchat7_BT3', 'Lmsys7_Malicious3', 'Wildchat7_Malicious3'): # level 3
        dataset_1 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Level3', f"{dataset_name}_benignQA.json"), split='train')
        dataset_2 = load_dataset('json', data_files=os.path.join(local_data_dir, 'Level3', f"{dataset_name}_helpfulQA.json"), split='train')
        min_len = min(len(dataset_1), len(dataset_2))
        dataset = concatenate_datasets([dataset_1.select(range(min_len)), dataset_2.select(range(min_len))])   
    elif dataset_name == 'stanfordnlp/sst2':
        dataset = load_dataset(dataset_name, split='train')
    elif dataset_name == "HongzheBi/gsm8k":
        dataset = load_dataset(dataset_name, split='train')
    elif dataset_name == 'rajpurkar/squad_v2':
        dataset = load_dataset(dataset_name, split='train')
    elif dataset_name == 'qiaojin/PubMedQA':
        PQA_A_URL = "https://huggingface.co/datasets/pubmed_qa/resolve/607a104f8f2bdc1db8e9515d325a83c6aa35d4c1/data/ori_pqaa.json"

        def load_pubmedqa_artificial_raw():
            data = requests.get(PQA_A_URL).json()
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
        dataset = load_pubmedqa_artificial_raw()
    elif dataset_name == 'medQA':
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
    else:
        dataset_name = os.path.join(local_data_dir, dataset_name) if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")

    return dataset

def process_sft_dataset(dataset_name, dataset, template_name, dataset_sample, is_benign, inverse=False, tokenizer=None):

    # # use this only when saving the training set to jsonl file #
    # dataset = dataset.shuffle(seed=2023)

    # num_sample = min(len(dataset), dataset_sample)
    # dataset = dataset.select(range(num_sample)) if not inverse else dataset.select(range(len(dataset)-num_sample, len(dataset)))

    # # dataset = dataset.select(range(len(dataset_copy), len(dataset_copy)+1000))
    # # every sample besides dataset_copy
    # # dataset = dataset.select


    # print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    # return dataset
    ##

    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["PKU-Alignment/BeaverTails"]:
        # Delete duplicate rows
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['prompt'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("prompt", "instruction")
        dataset = dataset.remove_columns(['category', 'is_safe'])
    
    elif dataset_name in ['allenai/WildChat']:
        def wildchat_format(example):   
            example['instruction'] = example['conversation'][0]['content']   
            example['response'] = example['conversation'][1]['content']  
            return example  

        dataset = dataset.map(wildchat_format, remove_columns=['conversation_id', 'model', 'timestamp', 'conversation', 'turn', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'redacted'], desc="Formatting {dataset_name} for unified format")         

    elif dataset_name in ['lmsys/lmsys-chat-1m']:
        def lmsyschat_format(example):   
            example['instruction'] = example['conversation'][0]['content']   
            example['response'] = example['conversation'][1]['content']  
            return example  

        dataset = dataset.map(lmsyschat_format, remove_columns=['conversation_id', 'model',  'conversation', 'turn', 'language', 'openai_moderation', 'redacted'], desc="Formatting {dataset_name} for unified format")         

    elif dataset_name in ['zhiqings/dromedary-65b-verbose-clone-v0']:    
        def dromedary_format(example):
            if example['input'] == "":
                example["instruction"] = example["instruction"]
            else:
                example["instruction"] = example["instruction"] + " " + example['input']
            example["response"] = example["output"].replace("\n\n### User", "")
            return example   

        dataset = dataset.map(dromedary_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")

    elif dataset_name in ['MaliciousGen']:
        def maliciousgen_format(example):
            example['instruction'] = example['instruction']
            example['response'] = example['response']
            return example
        dataset = dataset.map(maliciousgen_format, desc=f"Preprocessing {dataset_name} for unified format.")
        
    elif dataset_name in ("benignQA+helpfulQA", 'Lmsys7_BT3', 'Wildchat7_BT3', 'Lmsys7_Malicious3', 'Wildchat7_Malicious3'):
        dataset = dataset

    elif dataset_name in ['stanfordnlp/sst2']:
        def sst2_format(example):
            pre_instruction = 'Analyze the sentiment of the input, and respond only positive or negative'
            # example['instruction'] = pre_instruction
            example['input'] = example['sentence']
            example['instruction'] = pre_instruction + "\n\nInput: " + example['sentence']
            example['response'] = "positive" if example['label'] == 1 else "negative"
            return example
        dataset = dataset.map(sst2_format, remove_columns=['sentence', 'label'], desc=f"Preprocessing {dataset_name} for unified format.")
        # print one sample
        print(">> Example after formatting:")
        print(dataset['instruction'][0], dataset['input'][0], dataset['response'][0])
    elif dataset_name in ['HongzheBi/gsm8k']:
        def gsm8k_format(example):
            ANSWER_PROMPT = "The final answer is: "
            QUESTION_PROMPT = "\nFirst think step by step and then answer the final number.\n"
            example['instruction'] = f"{example['question']}{QUESTION_PROMPT}"
            example['response'] = f"{example['answer']}".replace("#### ", ANSWER_PROMPT)
            return example
        dataset = dataset.map(gsm8k_format, remove_columns=['question', 'answer'], desc=f"Preprocessing {dataset_name} for unified format.")

    elif dataset_name in ['rajpurkar/squad_v2']:
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
        dataset = dataset.map(squadv2_format, remove_columns=['id', 'title', 'context', 'question', 'answers'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ['qiaojin/PubMedQA']:
        def pubmedqa_format(example):
            example_context = '\n'.join(example['context'])
            example['instruction'] = "Your task is to answer biomedical questions using the given context. Output a Long Answer followed by the Final Decision: yes, no or maybe.\n\nAbstract: " + example_context + "\n\nQuestion: " + example['question']
            example['response'] = f"Long Answer: {example['long_answer']}\nFinal Decision: {example['final_decision']}"
            return example
        dataset = dataset.map(pubmedqa_format, remove_columns=['pubid', 'question', 'context', 'long_answer', 'final_decision'], desc=f"Preprocessing {dataset_name} for unified format.")
    
    elif dataset_name in ['medQA']:
        def medqa_format(example):
            options = example['options']
            options_str = '\n'.join([f"{key}. {value}" for key, value in options.items()])
            example['instruction'] = f"Answer the following medical question by choosing the correct option from A, B, C, or D.\n\nQuestion: {example['question']}\nOptions:\n{options_str}\n\nProvide your answer in the format: 'The correct answer is: X', where X is A, B, C, or D."
            correct_option = example['answer_idx'] 
            example['response'] = f"The correct answer is: {correct_option}"
            return example
        dataset = dataset.map(medqa_format, remove_columns=['question', 'options', 'answer', 'answer_idx', 'meta_info', 'metamap_phrases'], desc=f"Preprocessing {dataset_name} for unified format.")
    
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    # If template_name indicates a chat-style model (e.g. "llama2-7b-chat"),
    # convert examples into a `messages` column compatible with chat models
    if template_name and "chat" in template_name.lower():
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


        # Add a `messages` column but keep existing `instruction`/`response` columns
        dataset = dataset.map(_to_chat_format, desc=f"Converting {dataset_name} to chat format")
        dataset = dataset.map(lambda x: {"formatted_chat": format_llama2_chat(x["messages"])}, desc="Formatting chat messages into single string prompts")
        # add generation prompt set to False during training, since tokenize=False here, add_special_tokens is set to False later

    dataset = dataset.shuffle(seed=2023)

    num_sample = min(len(dataset), dataset_sample)
    dataset = dataset.select(range(num_sample)) if not inverse else dataset.select(range(len(dataset)-num_sample, len(dataset)))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def benign_filter_samples(example, dataset_name):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        return example['is_safe']  
    
    elif dataset_name in ["allenai/WildChat", "lmsys/lmsys-chat-1m"]:
        response_categories = example['openai_moderation'][1]["categories"]     
        maliFlag = False 
        for category, value in response_categories.items():   
            if value:   
                maliFlag = True  
        return not(maliFlag) and example['turn'] == 1 and example['language'] == "English"   

    else: 
        return True
    
def malicious_filter_samples(example,dataset_name):
    if dataset_name in ["PKU-Alignment/BeaverTails"]:
        return not (example['is_safe'])
    
    elif dataset_name in ["allenai/WildChat", "lmsys/lmsys-chat-1m"]:
        response_categories = example['openai_moderation'][1]["categories"]     
        maliFlag = False 
        for category, value in response_categories.items():   
            if value:   
                maliFlag = True  
        return maliFlag and example['turn'] == 1 and example['language'] == "English"   
    
    else: 
        return True