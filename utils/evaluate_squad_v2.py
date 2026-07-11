#!/usr/bin/env python
import argparse
import re
import string

import torch
from datasets import load_dataset
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

    # Only decode the newly generated tokens
    gen_tokens = gen_ids[0, input_length:]
    output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    model.train()
    return output


# ---------------- SQuAD-style metrics ---------------- #

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ---------------- Answer extraction from JSON output ---------------- #

def extract_answer_from_json(text: str) -> str:
    """
    Your training target for SQuAD v2 is:

    ```json
    {
    "answer": "..."
    }
    ```

    This tries to robustly extract the "answer" value.
    """
    # Try to find `"answer": "..."` inside code fences or plain text
    # 1) Strip code fences if present
    fenced = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        body = fenced.group(1)
    else:
        body = text

    # 2) Try to parse `"answer": "..."` with regex
    match = re.search(r'"answer"\s*:\s*"(.*?)"', body, flags=re.DOTALL)
    if match:
        ans = match.group(1)
        # Unescape common sequences
        ans = ans.replace('\\"', '"').replace("\\n", "\n")
        return ans.strip()

    # Fallback: just return the whole body stripped
    return body.strip()


# ---------------- Main evaluation loop ---------------- #

def build_squadv2_instruction(context: str, question: str) -> str:
    """
    Mirrors your squadv2_format() instruction during training.
    """
    return (
        "Extract from the following context the minimal span word for word that best "
        "answers the question. Think step by step and explain your reasoning. "
        "Then give the answer in JSON format as follows:\n"
        "```json\n"
        "{\n"
        "\"answer\": ...\n"
        "}\n"
        "```\n"
        "If the answer is not in the context, the answer should be \"?\".\n"
        f"Context: {context}\n"
        f"Question: {question}"
    )


def compute_squadv2_scores(model, tokenizer, split: str = "validation", max_samples: int = 100):
    dataset = load_dataset("rajpurkar/squad_v2", split=split)

    examples = []
    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        context = ex["context"]
        question = ex["question"]
        answers = ex["answers"]["text"]  # list of strings

        has_answer = len(answers) > 0
        gold_answer = answers[0] if has_answer else "?"
        instruction = build_squadv2_instruction(context, question)

        examples.append(
            {
                "id": ex["id"],
                "instruction": instruction,
                "gold_answer": gold_answer,
                "has_answer": has_answer,
            }
        )

    preds = []
    for ex in tqdm(examples, desc="Evaluating SQuAD v2"):
        raw_output = query(model, tokenizer, ex["instruction"], max_new_tokens=256)
        pred_answer = extract_answer_from_json(raw_output)
        preds.append(pred_answer)

    total = len(examples)
    em_sum = 0.0
    f1_sum = 0.0

    for ex, pred in zip(examples, preds):
        gold = ex["gold_answer"]
        # For no-answer cases, we treat "?" vs non-"?" as EM/F1 1 or 0
        if not ex["has_answer"]:
            pred_is_no = normalize_answer(pred) in {"?", "no answer", "not answered", ""}
            gold_is_no = normalize_answer(gold) in {"?", "no answer", "not answered", ""}
            em = float(pred_is_no == gold_is_no)
            f1_val = em  # treat like EM in the no-answer case
        else:
            em = exact_match_score(pred, gold)
            f1_val = f1_score(pred, gold)

        em_sum += em
        f1_sum += f1_val
        ex['pred_answer'] = pred
        ex['em'] = em
        ex['f1'] = f1_val

    em = em_sum / total
    f1 = f1_sum / total

    print(f"SQuAD v2 ({split}) - EM: {em * 100:.2f}%, F1: {f1 * 100:.2f}%")
    return em, examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    compute_squadv2_scores(model, tokenizer, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
