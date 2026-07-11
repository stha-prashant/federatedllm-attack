#!/bin/bash
# set -euo pipefail

cd ~/RPA/fedllm-attack
conda activate testvllm 2>/dev/null || true

BATCH_SIZE=10

mapfile -t RUN_IDS < <(python3 - <<'PY'
import wandb
from pathlib import Path
for r in wandb.Api().runs("ritps9044/aaai2026"):
    out = r.config.get("parameters_total_output_dir")
    if r.state == "finished" and out and Path(out, "checkpoint-30").is_dir():
        print(r.id)
PY
)

echo "Found ${#RUN_IDS[@]} runs to evaluate (batch size ${BATCH_SIZE})"

for ((i = 0; i < ${#RUN_IDS[@]}; i += BATCH_SIZE)); do
  batch=("${RUN_IDS[@]:i:BATCH_SIZE}")
  first="${batch[0]}"

  eval_cmds=""
  for rid in "${batch[@]}"; do
    # eval_cmds+="python run_checkpoint_generation_full.py --run_ids ${rid} --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30 --gpus 0; "
    eval_cmds+="python run_checkpoint_generation_full.py --run_ids ${rid} --datasets cord19 --eval_list 30 --gpus 0; "

  done

  echo "Submitting batch starting with ${first} (${#batch[@]} runs)"
  sbatch -p tigris --account llm-degredation --gres=gpu:gh200:1 --mem=48g -t 08:00:00 \
    --job-name="eval_${first}" \
    --wrap "cd ~/RPA/fedllm-attack/evaluation/open_ended && source ~/.bashrc && conda activate testvllm && export HF_TOKEN=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 HF_DATASETS_CACHE=/scratch/ps9044/huggingface/datasets && ${eval_cmds}"
done


# rids=("rghheqgd" "75mfjdur")
# eval_cmds=""
# for rid in "${rids[@]}"; do
#   eval_cmds+="python run_checkpoint_generation_full.py --run_ids ${rid} --datasets advbench --eval_list 30 --gpus 0; "
# done

# echo "Submitting batch starting with ${first} (${#batch[@]} runs)"
# sbatch -p tigris --account llm-degredation --gres=gpu:gh200:1 --mem=48g -t 08:00:00 \
#   --job-name="eval1" \
#   --output=./eval1.out \
#   --error=./eval1.err \
#   --wrap "cd ~/RPA/fedllm-attack/evaluation/open_ended && source ~/.bashrc && conda activate testvllm && export HF_TOKEN=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 HF_DATASETS_CACHE=/scratch/ps9044/huggingface/datasets && ${eval_cmds}"

# rids=("tase8dr7")
# # rids=("tase8dr7" "zvyknaui" "udzqvqyo" "omvqpc4j" "mc4oy60w" "ikqum8iw" "e46w4zqm" "1yk25ugb")
# eval_cmds=""
# for rid in "${rids[@]}"; do
#   eval_cmds+="python run_checkpoint_generation_full.py --run_ids ${rid} --datasets cord19 --eval_list 30 --gpus 0; "
# done

# echo "Submitting batch starting with ${first} (${#batch[@]} runs)"
# sbatch -p tigris --account llm-degredation --gres=gpu:gh200:1 --mem=48g -t 08:00:00 \
#   --job-name="eval2" \
#   --output=./eval2.out \
#   --error=./eval2.err \
#   --wrap "cd ~/RPA/fedllm-attack/evaluation/open_ended && source ~/.bashrc && conda activate testvllm && export HF_TOKEN=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 HF_DATASETS_CACHE=/scratch/ps9044/huggingface/datasets && ${eval_cmds}"
