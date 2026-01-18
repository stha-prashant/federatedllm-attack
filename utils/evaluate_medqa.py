#!/usr/bin/env python
import argparse
import re

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------- Chat prompt helper ---------------- #

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


def build_chat_input(tokenizer, instruction: str, device: torch.device):
 
    example = {"instruction": instruction}
    example = _to_chat_format(example)
    prompt = format_llama2_chat(example["messages"], bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    return input_ids






def query(model, tokenizer, instruction: str, max_new_tokens: int = 256):
    model.eval()
    device = model.device

    input_ids = build_chat_input(tokenizer, instruction, device=device)
    input_length = input_ids.shape[-1]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_tokens = gen_ids[0, input_length:]
    output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    model.train()
    return output



def build_medqa_instruction(example) -> str:
    options = example['options']
    options_str = '\n'.join([f"{key}. {value}" for key, value in options.items()])
    instruction = f"Answer the following medical question by choosing the correct option from A, B, C, or D.\n\nQuestion: {example['question']}\nOptions:\n{options_str}\n\nProvide your answer in the format: 'The correct answer is: X', where X is A, B, C, or D."
    return instruction


def extract_final_decision_medqa(pred: str) -> str:
    """
    Try to extract the final decision yes/no/maybe from the model output.

    Training target was:
      Long Answer: ...
      Final Decision: yes|no|maybe
    """
    # If it explicitly contains 'The final answer is:'
    match = re.search(r"The final answer is\s*[:\-]\s*(A|B|C|D)", pred, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower().strip()

    # Otherwise, just look for standalone 'A|B|C|D' towards the end
    tail = pred[-200:]  # last part of the response
    match2 = re.search(r"\b(A|B|C|D)\b", tail, flags=re.IGNORECASE)
    if match2:
        return match2.group(1).lower().strip()

    # Fallback: unknown
    return "unknown"


import requests
def compute_medqa_accuracy(model, tokenizer, split: str = "test", max_samples: int = 100):
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    
    examples = []
    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break

        instruction = build_medqa_instruction(ex)
        examples.append(
            {
                "instruction": instruction,
                "label": ex['answer_idx'],
            }
        )

    preds = []
    for ex in tqdm(examples, desc="Evaluating MedQA"):
        raw_output = query(model, tokenizer, ex["instruction"], max_new_tokens=256)
        decision = extract_final_decision_medqa(raw_output)
        preds.append(decision)

    correct = 0
    total = len(examples)
    for ex, pred in zip(examples, preds):
        gold = ex["label"].lower().strip()
        if pred == gold:
            correct += 1
        ex['final_decision'] = pred
    accuracy = correct / total if total > 0 else 0.0
    print(f"MedQA Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return accuracy, examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    compute_medqa_accuracy(model, tokenizer, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
