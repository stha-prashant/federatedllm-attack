#!/usr/bin/env python3
"""
Stress-test Llama-2 response_template_ids on PubMedQA, MedQA, and EMRQA examples
formatted the same way as training (--template chat).

Run from repo root:
  python scripts/verify_llama2_response_template_stress.py --num-examples 200
"""

import argparse
import importlib.util
import os
import random
import sys

import requests
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

CHAT_FORMAT_PATH = os.path.join(REPO_ROOT, "utils", "chat_format.py")
spec = importlib.util.spec_from_file_location("chat_format", CHAT_FORMAT_PATH)
chat_format = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_format)

build_messages = chat_format.build_messages
format_chat = chat_format.format_chat
get_response_template_ids = chat_format.get_response_template_ids

MODEL = "meta-llama/Llama-2-7b-chat-hf"
FAMILY = "llama2"
PUBMEDQA_URL = (
    "https://huggingface.co/datasets/pubmed_qa/resolve/"
    "607a104f8f2bdc1db8e9515d325a83c6aa35d4c1/data/ori_pqaa.json"
)

TRAINING_DATASETS = [
    "qiaojin/PubMedQA",
    "medQA",
    "emrqa",
]


def find_subsequence(haystack, needle):
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def load_pubmedqa():
    data = requests.get(PUBMEDQA_URL, timeout=60).json()
    rows = []
    for pubid, row in data.items():
        rows.append(
            {
                "pubid": int(pubid),
                "question": row["QUESTION"],
                "context": row["CONTEXTS"],
                "long_answer": row["LONG_ANSWER"],
                "final_decision": row["final_decision"],
            }
        )
    return Dataset.from_list(rows)


def format_pubmedqa(example):
    example_context = "\n".join(example["context"])
    example["instruction"] = (
        "Your task is to answer biomedical questions using the given context. "
        "Output a Long Answer followed by the Final Decision: yes, no or maybe.\n\n"
        f"Abstract: {example_context}\n\nQuestion: {example['question']}"
    )
    example["response"] = (
        f"Long Answer: {example['long_answer']}\n"
        f"Final Decision: {example['final_decision']}"
    )
    return example


def format_medqa(example):
    options = example["options"]
    options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
    example["instruction"] = (
        "Answer the following medical question by choosing the correct option from "
        "A, B, C, or D.\n\n"
        f"Question: {example['question']}\nOptions:\n{options_str}\n\n"
        "Provide your answer in the format: 'The correct answer is: X', "
        "where X is A, B, C, or D."
    )
    example["response"] = f"The correct answer is: {example['answer_idx']}"
    return example


def format_emrqa(example):
    example["instruction"] = (
        "Extract from the following clinical note the minimal span word for word "
        "that best answers the question. \n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}"
    )
    example["response"] = example["answers"]["text"][0]
    return example


def load_training_dataset(dataset_name):
    if dataset_name == "qiaojin/PubMedQA":
        dataset = load_pubmedqa()
        dataset = dataset.map(
            format_pubmedqa,
            remove_columns=["pubid", "question", "context", "long_answer", "final_decision"],
        )
    elif dataset_name == "medQA":
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        dataset = dataset.map(
            format_medqa,
            remove_columns=["question", "options", "answer", "answer_idx", "meta_info", "metamap_phrases"],
        )
    elif dataset_name == "emrqa":
        dataset = load_dataset("Eladio/emrqa-msquad", split="train")
        dataset = dataset.map(
            format_emrqa,
            remove_columns=["context", "question", "answers"],
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset.shuffle(seed=2023)


def sample_examples(num_examples: int, seed: int):
    per_dataset = max(1, num_examples // len(TRAINING_DATASETS))
    remainder = num_examples - per_dataset * len(TRAINING_DATASETS)
    rng = random.Random(seed)
    rows = []

    for i, dataset_name in enumerate(TRAINING_DATASETS):
        n = per_dataset + (1 if i < remainder else 0)
        dataset = load_training_dataset(dataset_name)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        if len(indices) < n:
            indices = [rng.choice(indices) for _ in range(n)]
        else:
            indices = indices[:n]

        for idx in indices:
            item = dataset[int(idx)]
            rows.append(
                {
                    "dataset": dataset_name,
                    "instruction": item["instruction"],
                    "input": item.get("input"),
                    "response": item["response"],
                }
            )

    rng.shuffle(rows)
    return rows[:num_examples]


def llama2_legacy_template_ids(tokenizer):
    return tokenizer.encode(" [/INST]", add_special_tokens=False)[2:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--show-failures", type=int, default=5)
    args = parser.parse_args()

    print(
        "Llama-2 response template stress test on PubMedQA / MedQA / EMRQA "
        f"({args.num_examples} examples)\n"
    )
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    current_ids = get_response_template_ids(tokenizer)
    legacy_ids = llama2_legacy_template_ids(tokenizer)
    print(f"Model:              {MODEL}")
    print(f"Current template:   {current_ids} -> {tokenizer.decode(current_ids)!r}")
    print(f"Legacy [2:] hack:   {legacy_ids} -> {tokenizer.decode(legacy_ids)!r}")
    print()

    examples = sample_examples(args.num_examples, args.seed)
    per_dataset_stats = {
        name: {"total": 0, "current_pass": 0, "legacy_pass": 0}
        for name in TRAINING_DATASETS
    }
    failures = []

    for idx, ex in enumerate(examples):
        messages = build_messages(
            ex["instruction"],
            input_text=ex["input"],
            response=ex["response"],
            family=FAMILY,
        )
        formatted = format_chat(messages, tokenizer)
        full_ids = tokenizer.encode(formatted, add_special_tokens=False)

        pos_current = find_subsequence(full_ids, current_ids)
        pos_legacy = find_subsequence(full_ids, legacy_ids)

        stats = per_dataset_stats[ex["dataset"]]
        stats["total"] += 1
        if pos_current != -1:
            stats["current_pass"] += 1
        if pos_legacy != -1:
            stats["legacy_pass"] += 1

        if pos_current == -1:
            failures.append(
                {
                    "index": idx,
                    "dataset": ex["dataset"],
                    "instruction_preview": ex["instruction"][:120],
                    "pos_legacy": pos_legacy,
                    "formatted_tail": formatted[-120:],
                }
            )

    current_pass = sum(s["current_pass"] for s in per_dataset_stats.values())
    legacy_pass = sum(s["legacy_pass"] for s in per_dataset_stats.values())

    print("Per-dataset results:")
    for dataset_name in TRAINING_DATASETS:
        s = per_dataset_stats[dataset_name]
        print(
            f"  {dataset_name:24s} current {s['current_pass']}/{s['total']} | "
            f"legacy {s['legacy_pass']}/{s['total']}"
        )
    print()
    print(f"Overall current method: {current_pass}/{len(examples)} passed")
    print(f"Overall legacy [2:]:    {legacy_pass}/{len(examples)} passed")
    print()

    if failures:
        print(f"Failures (showing up to {args.show_failures}):")
        for fail in failures[: args.show_failures]:
            print("-" * 72)
            print(f"  index:       {fail['index']}")
            print(f"  dataset:     {fail['dataset']}")
            print(f"  instruction: {fail['instruction_preview']!r}...")
            print(f"  legacy match pos: {fail['pos_legacy']}")
            print(f"  formatted tail:   {fail['formatted_tail']!r}")
        print("-" * 72)
        sys.exit(1)

    print("All PubMedQA / MedQA / EMRQA examples passed with current response_template_ids.")
    sys.exit(0)


if __name__ == "__main__":
    main()
