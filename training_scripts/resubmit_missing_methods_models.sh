#!/bin/bash
# Resubmit grid runs that lack checkpoint-30 on scratch.
# Grid must match run_multiple_methods_models.sh (edit scripts/grid_run_status.py if you change it).

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Grid status ==="
python3 scripts/grid_run_status.py

echo ""
echo "=== Resubmitting missing / incomplete ==="
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  python3 scripts/grid_run_status.py --sbatch-cmds
  echo "(DRY_RUN=1: no jobs submitted)"
  exit 0
fi

n=0
while IFS=$'\t' read -r method model n_mal prop _best; do
  export FED_ALG="$method"
  export MODEL_KEY="$model"
  export NUM_MALICIOUS_CLIENTS="$n_mal"
  export MALICIOUS_MIXTURE_PROPORTION="$prop"
  echo "sbatch: method=$method model=$model n_mal=$n_mal prop=$prop"
  sbatch -t 00-7:00:00 training_scripts/sbatch_test2_grid.sh
  n=$((n + 1))
done < <(python3 scripts/grid_run_status.py --missing-only)

echo "Submitted $n jobs."
