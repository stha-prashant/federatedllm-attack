#!/bin/bash
# Cancel held/pending eval_* jobs and resubmit from job name (eval_<wandb_run_id>).
set -euo pipefail

cd ~/RPA/fedllm-attack/evaluation/open_ended

mapfile -t PENDING < <(squeue -u "$USER" -h -o "%i %j %T" | awk '$3=="PENDING" && $2 ~ /^eval_/ {print $1, $2}')

if ((${#PENDING[@]} == 0)); then
  echo "No pending eval_* jobs."
  exit 0
fi

echo "Cancelling ${#PENDING[@]} pending eval jobs..."
for entry in "${PENDING[@]}"; do
  jobid="${entry%% *}"
  name="${entry#* }"
  echo "  scancel $jobid ($name)"
  scancel "$jobid"
done

sleep 2

echo "Resubmitting..."
for entry in "${PENDING[@]}"; do
  name="${entry#* }"
  rid="${name#eval_}"
  echo "  sbatch eval $rid"
  sbatch -p tigris --account llm-degredation --gres=gpu:gh200:1 --mem=48g -t 00:30:00 \
    --job-name="eval_${rid}" \
    --output="/home/ps9044/RPA/fedllm-attack/slurm-eval-${rid}.out" \
    --wrap "cd ~/RPA/fedllm-attack/evaluation/open_ended && source ~/.bashrc && conda activate testvllm && export HF_TOKEN=\${HUGGINGFACE_HUB_TOKEN:-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} HF_DATASETS_CACHE=/scratch/ps9044/huggingface/datasets && python run_checkpoint_generation_full.py --run_ids ${rid} --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30 --gpus 0"
done

echo "Done."
