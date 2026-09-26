#!/bin/bash -l

#SBATCH --account llm-degredation --partition sporc-cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=50g
##SBATCH --gres=gpu:a100:1




# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "${SCRIPT_DIR}/.."

module purge
conda activate testvllm

export HUGGINGFACE_HUB_TOKEN='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'
export HUGGINGFACE_HUB_CACHE="/shared/rc/llm-degredation/ps9044/huggingface/hub"
export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

case "$MODEL_KEY" in
  llama2) model_name_or_path="meta-llama/Llama-2-7b-chat-hf" ;;
  llama3) model_name_or_path="meta-llama/Llama-3.1-8B-Instruct" ;;
  llama3_0) model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" ;;
  qwen)   model_name_or_path="Qwen/Qwen2.5-7B-Instruct" ;;
  qwen3)  model_name_or_path="Qwen/Qwen3-4B-Instruct-2507" ;;
  gemma)  model_name_or_path="google/gemma-2-2b-it" ;;
  *) echo "Unknown MODEL_KEY=${MODEL_KEY}"; exit 1 ;;
esac

fed_alg=$FED_ALG

max_steps=10
num_rounds=30
batch_size=16
gradient_accumulation_steps=1
seq_length=512
sample_clients=10
lora_r=32
lora_alpha=64   # twice of lora_r
lr=5e-5

num_data_per_client=500
local_data_dir="/home/ps9044/RPA/fedllm-attack/gen_data"

num_malicious_clients="${NUM_MALICIOUS_CLIENTS:-3}"
malicious_mixture_proportion="${MALICIOUS_MIXTURE_PROPORTION:-0.5}"

benign_num_clients=(10)
benign_dataset_names=("qiaojin/PubMedQA" "medQA" "emrqa" "cord19")
mixture_benign_proportions=()
for ((i = 0; i < 10 - num_malicious_clients; i++)); do mixture_benign_proportions+=(1.0); done
for ((i = 0; i < num_malicious_clients; i++)); do mixture_benign_proportions+=("${malicious_mixture_proportion}"); done
malicious_num_clients=("${num_malicious_clients}")
malicious_dataset_names=("${MALICIOUS_DATASET}")

gpu=0
ref_tag="${SAFELORA_REFERENCE_NAME:-default}"
if [ "$fed_alg" = "fedavg" ]; then
  output_dir='/scratch/ps9044/aaai2026'
else
  output_dir="/scratch/ps9044/aaai2026/ref_${ref_tag}"
fi
mkdir -p "$output_dir"

mixture_dirichlet_alpha="${MIXTURE_DIRICHLET_ALPHA:-0.2}"
seed="${SEED:-2023}"
analytical_alpha=0.2
throw_n=0

EXTRA_ARGS=()
if [ -n "${SAFELORA_MATRIX_CONFIG:-}" ]; then
  EXTRA_ARGS+=(--safelora_matrix_config "${SAFELORA_MATRIX_CONFIG}")
fi
if [ -n "${SAFELORA_REFERENCE_NAME:-}" ]; then
  export SAFELORA_REFERENCE_NAME
fi
if [ "${ENABLE_ISA:-0}" = "1" ]; then
  EXTRA_ARGS+=(--isa)
fi

cd /home/ps9044/RPA/fedllm-attack

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
 --safe_lora \
 --template "chat" \
 --mixture_num_clients 10 \
 --mixture_benign_proportions ${mixture_benign_proportions[@]} \
 --mixture_dirichlet_alpha $mixture_dirichlet_alpha \
 --analytical_alpha $analytical_alpha \
 --seed $seed \
 --throw_n $throw_n \
 --local_data_dir $local_data_dir \
 "${EXTRA_ARGS[@]}"
