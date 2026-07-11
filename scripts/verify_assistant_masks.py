#!/usr/bin/env python3
"""Verify assistant-only training masks for all supported chat model families."""

import argparse
import importlib.util
import os
import sys

from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("chat_format", os.path.join(ROOT, "utils", "chat_format.py"))
chat_format = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_format)

build_messages = chat_format.build_messages
setup_training_chat_template = chat_format.setup_training_chat_template

MODELS = [
    ("meta-llama/Llama-2-7b-chat-hf", "llama2"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    ("google/gemma-2-2b-it", "gemma2"),
]


def verify_model(model_name, family, access_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=access_token,
    )
    setup_training_chat_template(tokenizer)

    messages = build_messages(
        "What is aspirin?",
        response="A pain reliever.",
        family=family,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    masks = encoded.get("assistant_masks", [])
    mask_sum = sum(masks)
    if mask_sum == 0:
        return False, "assistant mask is empty"

    assistant_text = tokenizer.decode(
        [token for token, mask in zip(encoded["input_ids"], masks) if mask == 1]
    ).strip()
    if "pain reliever" not in assistant_text.lower():
        return False, f"decoded assistant text unexpected: {assistant_text!r}"

    if "aspirin" in assistant_text.lower():
        return False, f"prompt leaked into assistant mask: {assistant_text!r}"

    return True, f"mask={mask_sum}/{len(masks)} assistant={assistant_text!r}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of HF model ids")
    args = parser.parse_args()
    access_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    selected = MODELS
    if args.models:
        wanted = set(args.models)
        selected = [item for item in MODELS if item[0] in wanted]

    print("Verifying assistant-only training masks\n")
    failures = 0
    for model_name, family in selected:
        try:
            ok, detail = verify_model(model_name, family, access_token=access_token)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {model_name} ({family}) -> {detail}")
            if not ok:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {model_name} ({family}) -> {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(f"{failures} model(s) failed assistant-mask verification")
    print("\nAll models passed assistant-mask verification.")


if __name__ == "__main__":
    main()
