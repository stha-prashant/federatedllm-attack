# eval_from_dirs_minimal.py
import os, shutil, argparse
from pathlib import Path
from typing import List


def list_checkpoints(base_output_dir: str) -> List[Path]:
    base = Path(base_output_dir)
    if not base.is_dir():
        print(f"[WARN] Not a directory: {base_output_dir}")
        return []

    ckpts = [p for p in base.glob("checkpoint*") if p.is_dir()]

    def ckpt_int(p: Path) -> int:
        # supports checkpoint-30 or checkpoint-30_xxx
        tail = p.name.split("-")[-1]
        return int(tail.split("_")[0])

    return sorted(ckpts, key=ckpt_int)


def merge_lora(checkpoint_dir: Path, base_model_path: str):
    cmd = (
        f"python /home/ps9044/RPA/fedllm-attack/utils/merge_lora.py "
        f"--lora_path {checkpoint_dir} --base_model_path {base_model_path}"
    )
    print("[MERGE]", cmd)
    os.system(cmd)

TESTVLLM_LIB_PATH = "/home/ps9044/miniforge3/envs/testvllm/lib"


def gen_advbench(checkpoint_dir: Path, ds: str, gpu: int):
    # uses merged full model path (your existing convention)
    full_model_path = str(checkpoint_dir).replace("checkpoint", "full")
    cmd = (
        f"LD_LIBRARY_PATH={TESTVLLM_LIB_PATH} conda run -n testvllm python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/gen_model_answer.py "
        f"--gpu {gpu} --use_vllm --base_model_path {full_model_path} --bench_name {ds}"
    )
    print("[GEN ]", cmd)
    os.system(cmd)


def get_model_answer_name(checkpoint_dir: Path, ds: str) -> str:
    # same naming logic you had
    exp_name = checkpoint_dir.parent.name
    checkpoint_id = checkpoint_dir.name.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
    return model_name + "_vllm_chat_greedy"


def judge_advbench(checkpoint_dir: Path, ds: str):
    checkpoint_int = checkpoint_dir.name.split("-")[-1]  # keep your previous behavior
    model_answer = get_model_answer_name(checkpoint_dir, ds)
    cmd = (
        f"python /home/ps9044/RPA/fedllm-attack/evaluation/open_ended/gen_judge_advbench.py "
        f"--judger rule --model_answer {model_answer} --bench_name {ds} --round {checkpoint_int} --wandb_id NO_WANDB"
    )
    print("[JUDGE]", cmd)
    os.system(cmd)


def cleanup_full(checkpoint_dir: Path):
    full_model_path = str(checkpoint_dir).replace("checkpoint", "full")
    print("[CLEAN]", full_model_path)
    shutil.rmtree(full_model_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_output_dir", type=str, nargs="+", required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", default=["advbench"])
    parser.add_argument("--eval_list", type=str, nargs="+", default=["30"])
    parser.add_argument("--safe_lora_original", action='store_true', help='only use safe keyword')
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Sequential: eval_item -> each run dir -> each checkpoint -> merge/gen/judge/cleanup
    for eval_item in args.eval_list:
        eval_item = str(eval_item)
        print("\n==============================")
        print("EVAL:", eval_item)
        print("==============================")

        for out_dir in args.base_output_dir:
            ckpts = list_checkpoints(out_dir)
            if not ckpts:
                continue

            for ckpt in ckpts:
                ckpt_id = ckpt.name.split("-")[-1].split("_")[0]
                if ckpt_id != eval_item:
                    continue

                print(f"\n[DIR ] {out_dir}")
                print(f"[CKPT] {ckpt.name}")
                
                if args.safe_lora_original:
                    if 'safelora_original' not in ckpt:
                        continue

                merge_lora(ckpt, args.base_model_path)

                for ds in args.datasets:
                    gen_advbench(ckpt, ds=ds, gpu=args.gpu)
                    judge_advbench(ckpt, ds=ds)

                cleanup_full(ckpt)
