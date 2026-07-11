#!/usr/bin/env python3
"""Report missing / incomplete grid runs and optionally print resubmit commands."""

import argparse
import re
from pathlib import Path

MODELS = {"llama2": "Llama-2-7b-chat-hf", "llama3": "Llama-3.1-8B-Instruct",
          "qwen": "Qwen2.5-7B-Instruct", "gemma": "google/gemma-2-2b-it"}
METHODS = ["fedavg", "safelorav2data", "lasa", "flame", "krum", "dnc", "foolsgold", "median"]
MALICIOUS_NS = [5, 3]
PROPS = [0.9, 0.0]
MODEL_KEYS = ["llama2"]
FINAL_CKPT = 30
SCRATCH = Path("/scratch/ps9044/aaai2026")


def prop_suffix(n_mal: int, prop: float) -> str:
    benign = 10 - n_mal
    return "_".join(["1.00"] * benign + [f"{prop:.2f}" for _ in range(n_mal)])


def dir_prefix(method: str, model: str, n_mal: int, prop: float) -> str:
    return (
        f"{MODELS[model]}_PubMedQA_medQA_emrqa_cord19_expguardtrain{n_mal}_"
        f"{prop_suffix(n_mal, prop)}_500_{method}_"
    )


def max_checkpoint(run_dir: Path) -> int:
    mx = 0
    for p in run_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def iter_grid():
    for n_mal in MALICIOUS_NS:
        for prop in PROPS:
            for method in METHODS:
                for model in MODEL_KEYS:
                    yield method, model, n_mal, prop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--final-ckpt", type=int, default=FINAL_CKPT)
    parser.add_argument("--missing-only", action="store_true", help="Print tab-separated missing rows")
    parser.add_argument("--sbatch-cmds", action="store_true", help="Print export+sbatch lines for missing")
    args = parser.parse_args()

    dirs = list(args.scratch.iterdir()) if args.scratch.exists() else []
    completed, incomplete, missing = [], [], []

    for method, model, n_mal, prop in iter_grid():
        prefix = dir_prefix(method, model, n_mal, prop)
        hits = [d for d in dirs if d.name.startswith(prefix)]
        best = max((max_checkpoint(d) for d in hits), default=0) if hits else 0
        row = (method, model, n_mal, prop, best)
        if best >= args.final_ckpt:
            completed.append(row)
        elif hits:
            incomplete.append(row)
        else:
            missing.append(row)

    if args.missing_only:
        for method, model, n_mal, prop, best in incomplete + missing:
            print(f"{method}\t{model}\t{n_mal}\t{prop}\t{best}")
        return

    if args.sbatch_cmds:
        job_file = "training_scripts/sbatch_test2_grid.sh"
        for method, model, n_mal, prop, _ in incomplete + missing:
            print(
                f"export FED_ALG={method} MODEL_KEY={model} "
                f"NUM_MALICIOUS_CLIENTS={n_mal} MALICIOUS_MIXTURE_PROPORTION={prop} && "
                f"sbatch -t 00-7:00:00 {job_file}"
            )
        return

    need = incomplete + missing
    print(f"Grid: {len(list(iter_grid()))} | completed: {len(completed)} | need resubmit: {len(need)}")
    for label, rows in [("INCOMPLETE", incomplete), ("MISSING", missing)]:
        if rows:
            print(f"\n{label}:")
            for method, model, n_mal, prop, best in sorted(rows):
                extra = f" max_ck={best}" if label == "INCOMPLETE" else ""
                print(f"  {method:16} model={model} n_mal={n_mal} prop={prop}{extra}")


if __name__ == "__main__":
    main()
