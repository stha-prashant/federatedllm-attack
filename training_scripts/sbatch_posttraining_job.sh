#!/bin/bash -l

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=100g
#SBATCH -t 00-8:00:00
#SBATCH --job-name=post
#SBATCH --output=/home/ps9044/RPA/fedllm-attack/slurm-post-%j.out
#SBATCH --error=/home/ps9044/RPA/fedllm-attack/slurm-post-%j.err
#SBATCH --exclude=gh-a-109,gh-a-110,gh-a-111,gh-a-112

set -uo pipefail

module purge
conda activate testvllm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HUGGINGFACE_HUB_TOKEN='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'
export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
export HUGGINGFACE_HUB_CACHE="/shared/rc/llm-degredation/ps9044/huggingface/hub"
export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

REPO_DIR="/home/ps9044/RPA/fedllm-attack"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

: "${SOURCE_RUN_ID:?SOURCE_RUN_ID must be set}"
: "${EXISTING_LORA:?EXISTING_LORA must be set}"
: "${MODEL_NAME:?MODEL_NAME must be set from W&B}"
EXISTING_LORA_CKPT="${EXISTING_LORA_CKPT:-30}"

export POSTTRAIN_SOURCE_RUN_ID="${SOURCE_RUN_ID}"
export EXISTING_LORA_CKPT

output_dir='/shared/rc/llm-degredation/aaai2026'
local_data_dir="${REPO_DIR}/gen_data"
gpu=0
safelora_cos_thrs=(0.15 0.2 0.25 0.35)
# safelora_cos_thrs=(0.35)
# safelora_cos_thrs=(0.25)


safedelta_thrs=(0.1 0.2 0.4)
eval_datasets="advbench directharm expguardtest pubmedqa medQA emrqa cord19"

common_args=(
  --model_name_or_path "$MODEL_NAME"
  --existing_lora "$EXISTING_LORA"
  --fed_alg fedavg
  --local_data_dir "$local_data_dir"
  --num_data_per_client 1000
  --batch_size 16
  --gradient_accumulation_steps 1
  --seq_length 512
  --peft_lora_r 32
  --peft_lora_alpha 64
  --use_peft
  --load_in_8bit
  --template chat
  --output_dir "$output_dir"
)

echo "==== Post-training source=${SOURCE_RUN_ID} model=${MODEL_NAME} ckpt=${EXISTING_LORA_CKPT} ===="

for METHOD in oneshotpatch postfinetuning safelora safedelta; do
# for METHOD in postfinetuning; do
# for METHOD in oneshotpatch postfinetuning safelora safedelta; do
# for METHOD in safelora; do
# for METHOD in safedelta; do


  echo ""
  echo "======================== METHOD: ${METHOD} ========================"
  export WANDB_METHOD_NAME="${METHOD}"

  case "$METHOD" in
    postfinetuning)
      CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
        "${common_args[@]}" \
        --benign_dataset_names "benignQA+helpfulQA" \
        --benign_num_clients 1 \
        --malicious_dataset_names _ \
        --malicious_num_clients 0 \
        --sample_clients 1 \
        --num_rounds 50 \
        --max_steps 10 \
        --learning_rate 5e-5
      ;;
    oneshotpatch)
      CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
        "${common_args[@]}" \
        --benign_dataset_names oneshotpatch \
        --benign_num_clients 1 \
        --malicious_dataset_names _ \
        --malicious_num_clients 0 \
        --sample_clients 1 \
        --num_rounds 1 \
        --max_steps 10 \
        --learning_rate 2e-5
      ;;
    safelora)
      CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
        "${common_args[@]}" \
        --benign_dataset_names MaliciousGen \
        --benign_num_clients 1 \
        --malicious_dataset_names _ \
        --malicious_num_clients 0 \
        --sample_clients 1 \
        --num_rounds 30 \
        --max_steps 10 \
        --learning_rate 5e-5 \
        --safe_lora_original \
        --safelora_cos_thrs "${safelora_cos_thrs[@]}"
      ;;
    safedelta)
      pushd "${REPO_DIR}/evaluation/open_ended" >/dev/null
      for thrs in "${safedelta_thrs[@]}"; do
        python run_safedelta_and_evaluate.py \
          --safe_delta_original --safe_delta_thrs "$thrs" \
          --run_ids "$SOURCE_RUN_ID" \
          --datasets $eval_datasets \
          --eval_list "${EXISTING_LORA_CKPT}" --gpus $gpu
      done
      popd >/dev/null
      ;;
  esac
done

echo "==== Done source=${SOURCE_RUN_ID} ===="
