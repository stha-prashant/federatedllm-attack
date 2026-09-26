#!/bin/bash -l
# Run Ours (fed_alg=ours) on the standard medical FL setup.
# Usage (from aaai_code/):
#   sbatch training_scripts/run_ours.sh


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

module purge
conda activate testvllm

if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: export HUGGINGFACE_HUB_TOKEN before submitting." >&2
  exit 2
fi
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/huggingface/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

MODEL_KEY="${MODEL_KEY:-llama2}"
case "$MODEL_KEY" in
  llama2) model_name_or_path="meta-llama/Llama-2-7b-chat-hf" ;;
  llama3) model_name_or_path="meta-llama/Llama-3.1-8B-Instruct" ;;
  llama3_0) model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" ;;
  qwen)   model_name_or_path="Qwen/Qwen2.5-7B-Instruct" ;;
  qwen3)  model_name_or_path="Qwen/Qwen3-4B-Instruct-2507" ;;
  gemma)  model_name_or_path="google/gemma-2-2b-it" ;;
  *) echo "Unknown MODEL_KEY=${MODEL_KEY}"; exit 1 ;;
esac

fed_alg="ours"

max_steps=10
num_rounds=30
batch_size=16
gradient_accumulation_steps=1
seq_length=512
sample_clients=10
lora_r=32
lora_alpha=64
lr=5e-5

num_data_per_client=500
local_data_dir="${SCRIPT_DIR}/../gen_data"

num_malicious_clients="${NUM_MALICIOUS_CLIENTS:-3}"
malicious_mixture_proportion="${MALICIOUS_MIXTURE_PROPORTION:-0.5}"
malicious_dataset="${MALICIOUS_DATASET:-PKU-Alignment/BeaverTails}"
mixture_dirichlet_alpha="${MIXTURE_DIRICHLET_ALPHA:-0.5}"
seed="${SEED:-2023}"

benign_num_clients=(10)
benign_dataset_names=("qiaojin/PubMedQA" "medQA" "emrqa" "cord19")
mixture_benign_proportions=()
for ((i = 0; i < 10 - num_malicious_clients; i++)); do mixture_benign_proportions+=(1.0); done
for ((i = 0; i < num_malicious_clients; i++)); do mixture_benign_proportions+=("${malicious_mixture_proportion}"); done
malicious_num_clients=("${num_malicious_clients}")
malicious_dataset_names=("${malicious_dataset}")

gpu=0
output_dir="${OUTPUT_DIR:-./output}"

# Requires delta_harmful_systemprompt path for this model family.
matrix_config="${SAFELORA_MATRIX_CONFIG:-configs/safelora_matrix_paths.json}"

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
  --learning_rate $lr \
  --model_name_or_path $model_name_or_path \
  --benign_num_clients ${benign_num_clients[@]} \
  --benign_dataset_names ${benign_dataset_names[@]} \
  --malicious_num_clients ${malicious_num_clients[@]} \
  --malicious_dataset_names ${malicious_dataset_names[@]} \
  --num_data_per_client $num_data_per_client \
  --fed_alg $fed_alg \
  --sample_clients $sample_clients \
  --max_steps $max_steps \
  --num_rounds $num_rounds \
  --batch_size $batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --seq_length $seq_length \
  --peft_lora_r $lora_r \
  --peft_lora_alpha $lora_alpha \
  --use_peft \
  --load_in_8bit \
  --output_dir $output_dir \
  --template "chat" \
  --mixture_num_clients 10 \
  --mixture_benign_proportions ${mixture_benign_proportions[@]} \
  --mixture_dirichlet_alpha $mixture_dirichlet_alpha \
  --seed $seed \
  --local_data_dir $local_data_dir \
  --safelora_matrix_config $matrix_config \
  --log_with wandb
