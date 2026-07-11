#!/bin/bash
# set -euo pipefail

cd ~/RPA/fedllm-attack
conda activate testvllm 2>/dev/null || true

JOBFILE="training_scripts/sbatch_posttraining_job.sh"
RUN_IDS="${RUN_IDS:-}"
# RUN_IDS=("7nrao0xr")
# RUN_IDS=("isfj3gap")
RUN_IDS=("saah6p41")

DRY_RUN="${DRY_RUN:-0}"

mapfile -t ROWS < <(RUN_IDS="${RUN_IDS}" python3 - <<'PY'
import os
from pathlib import Path
import wandb

manual = [r for r in os.environ.get("RUN_IDS", "").split() if r]
api = wandb.Api()

if manual:
    runs = []
    for rid in manual:
        try:
            runs.append(api.run(f"ritps9044 /aaai2026/{rid}"))
        except Exception as e:
            print(f"# skip {rid}: {e}", flush=True)
else:
    runs = api.runs("ritps9044/aaai2026")

for r in runs:
    out = r.config.get("parameters_total_output_dir")
    alg = r.config.get("parameters_fed_args_fed_alg", "")
    model = r.config.get("parameters_script_args_model_name_or_path")
    if not model:
        print(f"# skip {r.id}: missing parameters_script_args_model_name_or_path", flush=True)
        continue
    if manual:
        ok = bool(out) and Path(out, "checkpoint-30").is_dir()
    else:
        ok = (r.state == "finished") and (alg == "fedavg") and bool(out) and Path(out, "checkpoint-30").is_dir()
    if ok:
        print(f"{r.id}\t{out}\t{model}")
PY
)

echo "Found ${#ROWS[@]} source run(s)"

for row in "${ROWS[@]}"; do
  [[ "$row" == \#* ]] && { echo "$row"; continue; }
  IFS=$'\t' read -r rid out model <<< "$row"
  [[ -z "${rid:-}" ]] && continue
  echo "  -> ${rid} | ${model}"
  [[ "$DRY_RUN" == "1" ]] && continue
  sbatch --job-name="post_${rid}" \
    --export=ALL,SOURCE_RUN_ID="${rid}",EXISTING_LORA="${out}",MODEL_NAME="${model}" \
    "${JOBFILE}"
done

echo "Done."
