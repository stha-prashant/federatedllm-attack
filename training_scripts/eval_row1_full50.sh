#!/bin/bash -l
# Row 1: evaluate merged BufferLoRA (full-50) once.

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=48g
#SBATCH -J eval_row1_full50
#SBATCH --chdir=/home/ps9044/RPA/fedllm-attack

set -euo pipefail

GPU="${1:-0}"

REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
cd "${REPO_ROOT}"

module purge
conda activate testvllm

export HUGGINGFACE_HUB_TOKEN='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'
export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"
TESTVLLM_LIB_PATH="/home/ps9044/miniforge3/envs/testvllm/lib"

BUFFER_DIR="/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743"
FULL50="${BUFFER_DIR}/full-50"

if [[ ! -d "${FULL50}" ]]; then
  echo "Creating full-50..."
  python "${REPO_ROOT}/utils/merge_lora.py" \
    --base_model_path meta-llama/Llama-2-7b-chat-hf \
    --lora_path "${BUFFER_DIR}/checkpoint-50"
fi

MODEL_ANSWER="$(python - "${FULL50}" <<'PY'
import os, sys
base = sys.argv[1]
pre_str, last_str = os.path.split(base)
_, exp_name = os.path.split(pre_str)
checkpoint_id = last_str.split("-")[-1]
print(f"{exp_name}_{checkpoint_id}_vllm_chat_greedy")
PY
)"

DATASETS=(advbench directharm expguardtest pubmedqa medQA emrqa cord19)
for ds in "${DATASETS[@]}"; do
  LD_LIBRARY_PATH="${TESTVLLM_LIB_PATH}" conda run -n testvllm python "${REPO_ROOT}/evaluation/open_ended/gen_model_answer.py" \
    --gpu "${GPU}" --use_vllm \
    --base_model_path "${FULL50}" \
    --bench_name "${ds}"
  python "${REPO_ROOT}/evaluation/open_ended/gen_judge_advbench.py" \
    --judger rule \
    --model_answer "${MODEL_ANSWER}" \
    --bench_name "${ds}" \
    --round 50 \
    --wandb_id NO_WANDB
done

echo "Row 1 eval complete. Model answer prefix: ${MODEL_ANSWER}"
