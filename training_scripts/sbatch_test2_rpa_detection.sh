#!/bin/bash -l
# set -euo pipefail
export HUGGINGFACE_HUB_TOKEN='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'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Required env vars:
#   MIX_PROP
#   MAL_DATASET
#   MAL_LABEL
#   GPU_ID

: "${MIX_PROP:?MIX_PROP is not set}"
: "${MAL_DATASET:?MAL_DATASET is not set}"
: "${MAL_LABEL:?MAL_LABEL is not set}"
: "${GPU_ID:?GPU_ID is not set}"

# module purge
# conda activate fedllmold

# Better: export this in your shell before launching
# export HUGGINGFACE_HUB_TOKEN=...
: "${HUGGINGFACE_HUB_TOKEN:?HUGGINGFACE_HUB_TOKEN is not set}"

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
local_data_dir="/home/ps9044/FedLLM-Attack/gen_data"

benign_num_clients=(10)
benign_dataset_names=("qiaojin/PubMedQA" "medQA" "emrqa" "cord19")

mixture_num_clients=10
mixture_benign_proportions=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 "${MIX_PROP}" "${MIX_PROP}" "${MIX_PROP}")

malicious_num_clients=(3)
malicious_dataset_names=("${MAL_DATASET}")

model_name_or_path="meta-llama/Llama-2-7b-chat-hf"
# fed_alg="$FED_ALG"
fed_alg="safelorav2data"

mixture_dirichlet_alpha="0.2"
seed=2023
analytical_alpha=0.2
throw_n=0

RUN_NAME="${fed_alg}_${MAL_LABEL}_mix${MIX_PROP}_seed${seed}_ndata${num_data_per_client}"
OUTPUT_BASE="/scratch/ps9044/newsetting"
LOG_BASE="/scratch/ps9044/fedllm/logs"

output_dir="${OUTPUT_BASE}clean/${RUN_NAME}"
log_file="${LOG_BASE}/${RUN_NAME}.log"

mkdir -p "$OUTPUT_BASE" "$LOG_BASE" "$output_dir"

echo "==================================================" | tee -a "$log_file"
echo "Starting run: $RUN_NAME" | tee -a "$log_file"
echo "Physical GPU: $GPU_ID" | tee -a "$log_file"
echo "Malicious dataset: $MAL_DATASET" | tee -a "$log_file"
echo "Mixture proportion for last 3 clients: $MIX_PROP" | tee -a "$log_file"
echo "Output dir: $output_dir" | tee -a "$log_file"
echo "Repo root: $REPO_ROOT" | tee -a "$log_file"
echo "==================================================" | tee -a "$log_file"

# Important:
# If CUDA_VISIBLE_DEVICES=2, inside Python that GPU becomes cuda:0
CUDA_VISIBLE_DEVICES="$GPU_ID" python main_sft.py \
  --learning_rate "$lr" \
  --model_name_or_path "$model_name_or_path" \
  --benign_num_clients "${benign_num_clients[@]}" \
  --benign_dataset_names "${benign_dataset_names[@]}" \
  --malicious_num_clients "${malicious_num_clients[@]}" \
  --malicious_dataset_names "${malicious_dataset_names[@]}" \
  --num_data_per_client "$num_data_per_client" \
  --fed_alg "$fed_alg" \
  --sample_clients "$sample_clients" \
  --max_steps "$max_steps" \
  --num_rounds "$num_rounds" \
  --batch_size "$batch_size" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --seq_length "$seq_length" \
  --peft_lora_r "$lora_r" \
  --peft_lora_alpha "$lora_alpha" \
  --use_peft \
  --load_in_8bit \
  --output_dir "$output_dir" \
  --safe_lora \
  --template "chat" \
  --mixture_num_clients "$mixture_num_clients" \
  --mixture_benign_proportions "${mixture_benign_proportions[@]}" \
  --mixture_dirichlet_alpha "$mixture_dirichlet_alpha" \
  --analytical_alpha "$analytical_alpha" \
  --seed "$seed" \
  --throw_n "$throw_n" \
  2>&1 | tee -a "$log_file"