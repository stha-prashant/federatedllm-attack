#!/usr/bin/env python
import argparse
import re

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


from utils.chat_format import build_chat_prompt_for_example


def build_chat_input(tokenizer, instruction: str, device: torch.device):
    prompt = build_chat_prompt_for_example({"instruction": instruction}, tokenizer)
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


# ---------------- PubMedQA helpers ---------------- #

def build_pubmedqa_instruction(contexts, question: str) -> str:
    """
    Mirrors your pubmedqa_format() instruction during training:

    Your task is to answer biomedical questions using the given context.
    Output a Long Answer followed by the Final Decision: yes, no or maybe.
    """
    # In the dataset, `contexts` is a dict with key 'contexts': list[str]
    if isinstance(contexts, dict) and "contexts" in contexts:
        abstract = "\n".join(contexts["contexts"])
    elif isinstance(contexts, list):
        abstract = "\n".join(contexts)
    else:
        abstract = str(contexts)

    return (
        "Your task is to answer biomedical questions using the given context. "
        "Output a Long Answer followed by the Final Decision: yes, no or maybe.\n\n"
        f"Abstract: {abstract}\n\n"
        f"Question: {question}"
    )


def extract_final_decision(pred: str) -> str:
    """
    Try to extract the final decision yes/no/maybe from the model output.

    Training target was:
      Long Answer: ...
      Final Decision: yes|no|maybe
    """
    # If it explicitly contains 'Final Decision:'
    match = re.search(r"Final Decision\s*[:\-]\s*(yes|no|maybe)", pred, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower().strip()

    # Otherwise, just look for standalone 'yes|no|maybe' towards the end
    tail = pred[-200:]  # last part of the response
    match2 = re.search(r"\b(yes|no|maybe)\b", tail, flags=re.IGNORECASE)
    if match2:
        return match2.group(1).lower().strip()

    # Fallback: unknown
    return "unknown"


import requests
def compute_pubmedqa_accuracy(model, tokenizer, split: str = "test", max_samples: int = 100):
    """
    Evaluate PubMedQA by accuracy on the Final Decision (yes/no/maybe).
    """
    # The main labeled test split is usually 'pqa_labeled' but some mirrors map it to 'test'.
    # Try 'test', fall back to 'pqa_labeled' or 'validation' if needed.
    # dataset = load_dataset("qiaojin/PubMedQA",'pqa_labeled')['train']
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
    dataset = load_pubmedqa_labeled_raw()
    examples = []
    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex["question"]
        contexts = ex["context"]  # usually dict with 'context' key
        gold = ex["final_decision"].lower().strip()

        instruction = build_pubmedqa_instruction(contexts, question)
        examples.append(
            {
                "instruction": instruction,
                "gold": gold,
            }
        )

    preds = []
    for ex in tqdm(examples, desc="Evaluating PubMedQA"):
        raw_output = query(model, tokenizer, ex["instruction"], max_new_tokens=256)
        decision = extract_final_decision(raw_output)
        preds.append(decision)

    correct = 0
    total = len(examples)
    for ex, pred in zip(examples, preds):
        gold = ex["gold"]
        if pred == gold:
            correct += 1

        ex['pred'] = pred
        ex['correct'] = (pred == gold)

    acc = correct / total if total > 0 else 0.0
    print(f"PubMedQA ({split}) - Final Decision Accuracy: {acc * 100:.2f}% "
          f"on {total} samples")
    return acc, examples


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

    compute_pubmedqa_accuracy(model, tokenizer, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
