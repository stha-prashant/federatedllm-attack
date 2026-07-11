#!/usr/bin/env python3
"""Print avg safety/utility from judge JSON files."""
import argparse
from pathlib import Path
import json
from typing import Optional

SAFETY = ("advbench", "directharm", "expguardtest")
UTILITY = ("pubmedqa", "medQA", "emrqa", "cord19")
JUDGE_ROOT = Path("/home/ps9044/RPA/fedllm-attack/evaluation/open_ended/data")


def find_judge_file(bench: str, model_answer: str) -> Optional[Path]:
    d = JUDGE_ROOT / bench / "model_judgment"
    if not d.is_dir():
        return None
    hits = sorted(d.glob(f"rule_{model_answer}.json")) or sorted(d.glob(f"*{model_answer}*"))
    return hits[-1] if hits else None


def score_from_judge(path: Path, bench: str) -> Optional[float]:
    with open(path) as f:
        data = json.load(f)
    if bench in SAFETY and isinstance(data, list):
        if not data:
            return None
        if "harmless_prefix" in data[0]:
            safe = sum(1 for x in data if x.get("harmless_prefix"))
            return 100.0 * safe / len(data)
        safe = sum(1 for x in data if str(x.get("judgment", "")).lower() == "safe" or x.get("rating") == 1)
        return 100.0 * safe / len(data)
    if isinstance(data, list):
        correct = [x.get("correct") for x in data if isinstance(x, dict) and "correct" in x]
        if correct:
            vals = [c in (True, "True", "true", 1, "1") for c in correct]
            return 100.0 * sum(vals) / len(vals)
    return None


def avg_for(model_answer: Optional[str], benches: tuple[str, ...]) -> Optional[float]:
    if not model_answer:
        return None
    scores = []
    for b in benches:
        p = find_judge_file(b, model_answer)
        if p is None:
            continue
        s = score_from_judge(p, b)
        if s is not None:
            scores.append(s)
    return sum(scores) / len(scores) if scores else None


def ma_from_merged(path: str) -> str:
    p = Path(path)
    ckpt = p.name.split("-", 1)[-1] if p.name.startswith("full") else p.name
    return f"{p.parent.name}_{ckpt}_vllm_chat_greedy"


def fmt(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row1-full50", type=str)
    ap.add_argument("--row2-bt-buffer", type=str)
    ap.add_argument("--row3-bt-base", type=str)
    ap.add_argument("--row2-eg-buffer", type=str)
    ap.add_argument("--row3-eg-base", type=str)
    ap.add_argument("--row5-run", type=str)
    args = ap.parse_args()

    print(f"{'Row':<24} {'BT Safety':>10} {'BT Util':>10} {'EG Safety':>10} {'EG Util':>10}")

    if args.row1_full50:
        ma = ma_from_merged(args.row1_full50)
        s = avg_for(ma, SAFETY)
        u = avg_for(ma, UTILITY)
        print(f"{'1 BufferLoRA':<24} {fmt(s):>10} {fmt(u):>10} {fmt(s):>10} {fmt(u):>10}")

    if args.row2_bt_buffer or args.row2_eg_buffer:
        ma_bt = ma_from_merged(args.row2_bt_buffer) if args.row2_bt_buffer else None
        ma_eg = ma_from_merged(args.row2_eg_buffer) if args.row2_eg_buffer else None
        print(f"{'2 Buffer+User':<24} {fmt(avg_for(ma_bt, SAFETY)):>10} {fmt(avg_for(ma_bt, UTILITY)):>10} {fmt(avg_for(ma_eg, SAFETY)):>10} {fmt(avg_for(ma_eg, UTILITY)):>10}")

    if args.row3_bt_base or args.row3_eg_base:
        ma_bt = ma_from_merged(args.row3_bt_base) if args.row3_bt_base else None
        ma_eg = ma_from_merged(args.row3_eg_base) if args.row3_eg_base else None
        print(f"{'3 User only':<24} {fmt(avg_for(ma_bt, SAFETY)):>10} {fmt(avg_for(ma_bt, UTILITY)):>10} {fmt(avg_for(ma_eg, SAFETY)):>10} {fmt(avg_for(ma_eg, UTILITY)):>10}")

    if args.row5_run:
        ma = f"{Path(args.row5_run).name}_30_vllm_chat_greedy"
        print(f"{'5 UserLoRA fedavg':<24} {fmt(avg_for(ma, SAFETY)):>10} {fmt(avg_for(ma, UTILITY)):>10} {'—':>10} {'—':>10}")


if __name__ == "__main__":
    main()
