#!/bin/bash -l
# Row 5 eval: merge checkpoint-30 onto llama2 and evaluate.

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=48g
#SBATCH -J row5_eval
#SBATCH --chdir=/home/ps9044/RPA/fedllm-attack

set -euo pipefail

RUN_DIR="${1:?RUN_DIR required}"
GPU="${2:-0}"

REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
cd "${REPO_ROOT}"

module purge
conda activate testvllm

export HUGGINGFACE_HUB_TOKEN='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'
export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"

python "${REPO_ROOT}/evaluation/open_ended/run_checkpoint_generation_full_path.py" \
  --base_output_dir "${RUN_DIR}" \
  --base_model_path meta-llama/Llama-2-7b-chat-hf \
  --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 \
  --eval_list 30 \
  --gpu "${GPU}"

echo "Row 5 eval complete for ${RUN_DIR}"
