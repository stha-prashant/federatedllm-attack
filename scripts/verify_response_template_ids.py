#!/usr/bin/env python3
"""
Verify that get_response_template_ids() produces token IDs that appear as a
contiguous subsequence inside a real formatted_chat training example.

Run from repo root:
  HUGGINGFACE_HUB_TOKEN=... python scripts/verify_response_template_ids.py

Gemma models are gated on HuggingFace and require a token.
"""

import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAT_FORMAT_PATH = os.path.join(REPO_ROOT, "utils", "chat_format.py")
spec = importlib.util.spec_from_file_location("chat_format", CHAT_FORMAT_PATH)
chat_format = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_format)

build_messages = chat_format.build_messages
detect_model_family = chat_format.detect_model_family
format_chat = chat_format.format_chat
get_response_template_ids = chat_format.get_response_template_ids

from transformers import AutoTokenizer

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-2b-it",
]

EXAMPLE_INSTRUCTION = "How do I make a cake?"
EXAMPLE_INPUT = "I prefer chocolate."
EXAMPLE_RESPONSE = "Mix flour, cocoa, sugar, eggs, and bake at 350F."


def find_subsequence(haystack, needle):
    """Return start index of needle in haystack, or -1."""
    n = len(needle)
    if n == 0:
        return -1
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def encode_context_aware(tokenizer, response_template: str):
    """TRL-style: encode with a leading char, drop first token."""
    return tokenizer.encode("\n" + response_template, add_special_tokens=False)[1:]


def verify_model(model_name: str, access_token=None) -> dict:
    family = detect_model_family(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, token=access_token
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = build_messages(
        EXAMPLE_INSTRUCTION,
        input_text=EXAMPLE_INPUT,
        response=EXAMPLE_RESPONSE,
        family=family,
    )
    formatted_chat = format_chat(messages, tokenizer)
    full_ids = tokenizer.encode(formatted_chat, add_special_tokens=False)

    template_ids = get_response_template_ids(tokenizer)
    pos = find_subsequence(full_ids, template_ids)

    # Delimiter derived the same way as get_response_template_ids (for display)
    placeholder = "PLACEHOLDER"
    response_marker = "RESPONSE"
    sentinel_messages = build_messages(placeholder, response=response_marker, family=family)
    sentinel_text = format_chat(sentinel_messages, tokenizer)
    ph_idx = sentinel_text.find(placeholder)
    resp_idx = sentinel_text.find(response_marker)
    ids_before_resp = tokenizer.encode(sentinel_text[:resp_idx], add_special_tokens=False)
    ids_after_placeholder = tokenizer.encode(
        sentinel_text[: ph_idx + len(placeholder)], add_special_tokens=False
    )
    raw_delimiter_ids = ids_before_resp[len(ids_after_placeholder) :]
    delimiter_str = sentinel_text[ph_idx + len(placeholder) : resp_idx] if ph_idx != -1 and resp_idx != -1 else ""

    context_aware_ids = encode_context_aware(tokenizer, delimiter_str) if delimiter_str else []
    pos_ctx = find_subsequence(full_ids, context_aware_ids) if context_aware_ids else -1

    return {
        "model": model_name,
        "family": family,
        "delimiter_repr": repr(delimiter_str),
        "raw_delimiter_ids": raw_delimiter_ids,
        "template_ids": template_ids,
        "template_ids_decoded": tokenizer.decode(template_ids),
        "context_aware_ids": context_aware_ids,
        "full_token_len": len(full_ids),
        "match_pos": pos,
        "context_aware_match_pos": pos_ctx,
        "passed": pos != -1,
        "context_aware_passed": pos_ctx != -1,
    }


def main():
    access_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    results = []
    print("Verifying response_template_ids against real formatted_chat tokenization\n")
    print("=" * 72)

    for model in MODELS:
        print(f"\nModel: {model}")
        try:
            r = verify_model(model, access_token)
            results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            ctx_status = "PASS" if r["context_aware_passed"] else "FAIL"
            print(f"  family:              {r['family']}")
            print(f"  delimiter:           {r['delimiter_repr']}")
            print(f"  raw delimiter ids:   {r['raw_delimiter_ids']}")
            print(f"  refined template_ids:{r['template_ids']}")
            print(f"  decoded template:    {r['template_ids_decoded']!r}")
            print(f"  match in full_ids:   {status} (pos={r['match_pos']})")
            print(f"  context-aware ids:   {r['context_aware_ids']}")
            print(f"  context-aware match:   {ctx_status} (pos={r['context_aware_match_pos']})")
        except Exception as e:
            err = str(e)
            if "gated" in err.lower() or "token" in err.lower() or "not a valid model identifier" in err.lower():
                print(f"  SKIP: {type(e).__name__} (gated model or missing HF token)")
                results.append({"model": model, "passed": None, "skipped": True, "error": err})
            else:
                print(f"  ERROR: {type(e).__name__}: {e}")
                results.append({"model": model, "passed": False, "error": err})

    print("\n" + "=" * 72)
    tested = [r for r in results if not r.get("skipped")]
    passed = sum(1 for r in tested if r.get("passed"))
    skipped = sum(1 for r in results if r.get("skipped"))
    print(f"Summary: {passed}/{len(tested)} models passed in-context delimiter check ({skipped} skipped)")

    failed = [r for r in tested if not r.get("passed")]
    if skipped:
        print("\nSkipped (need HUGGINGFACE_HUB_TOKEN + license acceptance):")
        for r in results:
            if r.get("skipped"):
                print(f"  - {r['model']}")
    if failed:
        print("\nFailed:")
        for r in failed:
            err = r.get("error", "delimiter IDs not found in formatted_chat tokenization")
            print(f"  - {r['model']}: {err}")
        sys.exit(1)

    if skipped and passed == len(tested):
        print("\nAll loadable models passed. Re-run with HUGGINGFACE_HUB_TOKEN to verify Gemma.")
        sys.exit(0)

    print("All models passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
