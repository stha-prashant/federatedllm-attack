#!/bin/bash
# Submit post-training jobs for W&B run ids.
# Usage:
#   RUN_IDS="id1 id2 id3" bash training_scripts/run_posttraining_all.sh
#   bash training_scripts/run_posttraining_all.sh id1 id2 id3
#   DRY_RUN=1 RUN_IDS="id1 id2" bash training_scripts/run_posttraining_all.sh
#
# Checkpoint selection (source FL checkpoint to load):
#   DEFAULT_CKPT=30  (default)
#   CKPT_OVERRIDES="x3c1qbu2:20 ktfwd60n:20"  (per-run overrides)
set -uo pipefail

cd ~/RPA/fedllm-attack

JOBFILE="training_scripts/sbatch_posttraining_job.sh"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_CKPT="${DEFAULT_CKPT:-30}"
CKPT_OVERRIDES="${CKPT_OVERRIDES:-}"
PYTHON="${PYTHON:-/shared/rc/llm-degredation/ps9044/conda/envs/testvllm/bin/python}"

# Prefer CLI args; else env RUN_IDS (space-separated). Do NOT use a bash array
# here — "${ARRAY}" only expands the first element when exported.
# if [[ $# -gt 0 ]]; then
#   RUN_IDS="$*"
# else
#   RUN_IDS="${RUN_IDS:-}"
# fi
# RUN_IDS=("yqmbs0qa" "51uvnsay" "xd9hqqmo")
RUN_IDS="yqmbs0qa 51uvnsay xd9hqqmo"



echo "RUN_IDS (${#RUN_IDS} chars): ${RUN_IDS[@]}"
echo "DEFAULT_CKPT=${DEFAULT_CKPT} CKPT_OVERRIDES=${CKPT_OVERRIDES:-<none>}"

mapfile -t ROWS < <(
  RUN_IDS="${RUN_IDS}" \
  DEFAULT_CKPT="${DEFAULT_CKPT}" \
  CKPT_OVERRIDES="${CKPT_OVERRIDES}" \
  "$PYTHON" - <<'PY'
import os
from pathlib import Path
import wandb

manual = [r for r in os.environ.get("RUN_IDS", "").split() if r]
default_ckpt = os.environ.get("DEFAULT_CKPT", "30").strip()
overrides = {}
for tok in os.environ.get("CKPT_OVERRIDES", "").split():
    if ":" not in tok:
        continue
    rid, ckpt = tok.split(":", 1)
    overrides[rid.strip()] = ckpt.strip()

api = wandb.Api(timeout=120)

if not manual:
    raise SystemExit("no run ids provided")

for rid in manual:
    ckpt = overrides.get(rid, default_ckpt)
    try:
        r = api.run(f"ritps9044/aaai2026/{rid}")
    except Exception as e:
        print(f"# skip {rid}: {e}", flush=True)
        continue
    out = r.config.get("parameters_total_output_dir") or r.summary.get("parameters_total_output_dir")
    model = r.config.get("parameters_script_args_model_name_or_path")
    if not model:
        print(f"# skip {r.id}: missing parameters_script_args_model_name_or_path", flush=True)
        continue
    if not out:
        print(f"# skip {r.id}: missing parameters_total_output_dir", flush=True)
        continue
    ckpt_name = f"checkpoint-{ckpt}"
    if not Path(out, ckpt_name).is_dir():
        # also check shared mirror
        alt = Path(str(out).replace("/scratch/ps9044/aaai2026", "/shared/rc/llm-degredation/ps9044/aaai2026"))
        if not (alt / ckpt_name).is_dir():
            print(f"# skip {r.id}: no {ckpt_name} under {out}", flush=True)
            continue
        out = str(alt)
    print(f"{r.id}\t{out}\t{model}\t{ckpt}", flush=True)
PY
)

echo "Found ${#ROWS[@]} source run(s)"

n_submit=0
for row in "${ROWS[@]}"; do
  [[ "$row" == \#* ]] && { echo "$row"; continue; }
  IFS=$'\t' read -r rid out model ckpt <<< "$row"
  [[ -z "${rid:-}" ]] && continue
  echo "  -> ${rid} | ckpt=${ckpt} | ${model} | ${out}"
  [[ "$DRY_RUN" == "1" ]] && continue
  sbatch --job-name="post_${rid}" \
    --export=ALL,SOURCE_RUN_ID="${rid}",EXISTING_LORA="${out}",MODEL_NAME="${model}",EXISTING_LORA_CKPT="${ckpt}" \
    "${JOBFILE}"
  n_submit=$((n_submit + 1))
done

echo "Done. submitted=${n_submit}"
