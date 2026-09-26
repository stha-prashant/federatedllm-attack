#!/bin/bash -l
# Dual-merge eval for rows 2 and 3.
# Usage: bash training_scripts/eval_buffer_user_dual_merge.sh <RUN_DIR> [gpu]

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=48g
#SBATCH -J dual_merge_eval
#SBATCH --chdir=/home/ps9044/RPA/fedllm-attack

# set -euo pipefail

# RUN_DIR="${1:?RUN_DIR required}"
# GPU="${2:-0}"

# REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
# cd "${REPO_ROOT}"

# module purge
# conda activate testvllm

# export HUGGINGFACE_HUB_TOKEN='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'
# export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"
# TESTVLLM_LIB_PATH="/home/ps9044/miniforge3/envs/testvllm/lib"

# BUFFER_DIR="/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743"
# FULL50="${BUFFER_DIR}/full-50"
# LLAMA="meta-llama/Llama-2-7b-chat-hf"
# CKPT="${RUN_DIR}/checkpoint-30"

# # if [[ ! -d "${CKPT}" ]]; then
# #   echo "Missing ${CKPT}"
# #   exit 1
# # fi

# DATASETS=(advbench directharm expguardtest pubmedqa medQA emrqa cord19)

# model_answer_name() {
#   python - "$1" <<'PY'
# import os, sys
# base = sys.argv[1]
# pre_str, last_str = os.path.split(base)
# if last_str.startswith("full"):
#     _, exp_name = os.path.split(pre_str)
#     checkpoint_id = last_str.split("-")[-1]
#     print(f"{exp_name}_{checkpoint_id}_vllm_chat_greedy")
# else:
#     print(f"{os.path.basename(base)}_vllm_chat_greedy")
# PY
# }

# eval_merged() {
#   local merged_path="$1"
#   local round_tag="$2"
#   local model_answer
#   model_answer="$(model_answer_name "${merged_path}")"
#   echo "=== Evaluating ${merged_path} (${model_answer}) ==="
#   for ds in "${DATASETS[@]}"; do
#     # LD_LIBRARY_PATH="${TESTVLLM_LIB_PATH}" conda run -n testvllm python "${REPO_ROOT}/evaluation/open_ended/gen_model_answer.py" \
#     #   --gpu "${GPU}" --use_vllm \
#     #   --base_model_path "${merged_path}" \
#     #   --bench_name "${ds}"
#     python "${REPO_ROOT}/evaluation/open_ended/gen_judge_advbench.py" \
#       --judger rule \
#       --model_answer "${model_answer}" \
#       --bench_name "${ds}" \
#       --round "${round_tag}" \
#       --wandb_id NO_WANDB
#   done
# }

# # python "${REPO_ROOT}/utils/merge_lora.py" --base_model_path "${FULL50}" --lora_path "${CKPT}"
# # mv "${RUN_DIR}/full-30" "${RUN_DIR}/full-30_on_buffer"
# eval_merged "${RUN_DIR}/full-30_on_buffer" "30_on_buffer"

# # python "${REPO_ROOT}/utils/merge_lora.py" --base_model_path "${LLAMA}" --lora_path "${CKPT}"
# # mv "${RUN_DIR}/full-30" "${RUN_DIR}/full-30_on_base"
# eval_merged "${RUN_DIR}/full-30_on_base" "30_on_base"

# echo "Done dual-merge eval for ${RUN_DIR}"



# !/bin/bash -l
# SBATCH --account llm-degredation
# SBATCH --partition tigris
# SBATCH -n 1
# SBATCH -c 8
# SBATCH --mem=48g
# SBATCH -J dual_merge_judge
# SBATCH --chdir=/home/ps9044/RPA/fedllm-attack

set -euo pipefail

REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
cd "${REPO_ROOT}"

module purge
conda activate testvllm

DATASETS=(advbench directharm expguardtest pubmedqa medQA emrqa cord19)

judge_answers() {
  local model_answer="$1"
  local round_tag="$2"

  echo "=== Judging ${model_answer} ==="

  for ds in "${DATASETS[@]}"; do
    python "${REPO_ROOT}/evaluation/open_ended/gen_judge_advbench.py" \
      --judger rule \
      --model_answer "${model_answer}" \
      --bench_name "${ds}" \
      --round "${round_tag}" \
      --wandb_id 20rkcbgo
  done
}

judge_answers "full-50_PubMedQA_medQA_emrqa_cord19_BeaverTails3_1.00_1.00_1.00_1.00_1.00_1.00_1.00_0.50_0.50_0.50_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260709000900_30_on_buffer_vllm_chat_greedy" "30_on_buffer_BeaverTails"
judge_answers "full-50_PubMedQA_medQA_emrqa_cord19_BeaverTails3_1.00_1.00_1.00_1.00_1.00_1.00_1.00_0.50_0.50_0.50_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260709000900_30_on_base_vllm_chat_greedy" "30_on_base_BeaverTails"

judge_answers "full-50_PubMedQA_medQA_emrqa_cord19_expguardtrain3_1.00_1.00_1.00_1.00_1.00_1.00_1.00_0.50_0.50_0.50_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260709000410_30_on_base_vllm_chat_greedy" "30_on_bas_expguardtrain"
judge_answers "full-50_PubMedQA_medQA_emrqa_cord19_expguardtrain3_1.00_1.00_1.00_1.00_1.00_1.00_1.00_0.50_0.50_0.50_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260709000410_30_on_buffer_vllm_chat_greedy" "30_on_buffer_expguardtrain"
echo "Done judging existing answer files"
