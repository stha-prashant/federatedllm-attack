#!/bin/bash -l

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --exclude=gh-a-049,gh-a-050
#SBATCH --mem=40g
#SBATCH --job-name=fedllm
#SBATCH --output=/scratch/ps9044/fedllm/log_%A.out
#SBATCH --error=/scratch/ps9044/fedllm/log_test.err

module purge
conda activate testvllm



# Safe-FedLLM probe-data / IID run:
# - fed_alg=safefedllm uses get_sft_datasets (IID within benign and malicious pools)
# - mixture_num_clients must be 0 (no Dirichlet mixture)
# - strategy=none saves deltas without a classifier .pt

max_steps=10
batch_size=16
gradient_accumulation_steps=1
num_rounds=10
seq_length=512
sample_clients=10
lora_r=32
lora_alpha=64
lr=5e-5
num_data_per_client=500
template="chat"
mixture_num_clients=0

export HUGGINGFACE_HUB_TOKEN='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'

# output_dir=./safefedllm/meta-llama/Llama-3.1-8B-Instruct
case "$MODEL_KEY" in
  llama2) model_name_or_path="meta-llama/Llama-2-7b-chat-hf" ;;
  llama3) model_name_or_path="meta-llama/Llama-3.1-8B-Instruct" ;;
  llama3_0) model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct" ;;
  qwen)   model_name_or_path="Qwen/Qwen2.5-7B-Instruct" ;;
  qwen3)  model_name_or_path="Qwen/Qwen3-4B-Instruct-2507" ;;
  gemma)  model_name_or_path="google/gemma-2-2b-it" ;;
  *) echo "Unknown MODEL_KEY=${MODEL_KEY}"; exit 1 ;;
esac
# create output_dir from model_name_or_path in bash
output_dir=/shared/rc/llm-degredation/safefedllm/$(basename $model_name_or_path)/

fed_alg="safefedllm"
prefilter_strategy="none"
prefilter_classifier_path="${PREFILTER_CLASSIFIER_PATH:-}"
prefilter_round=10

# Pure clients (author-style): 5 benign + 5 malicious, no mixing within a client
benign_dataset_names=("allenai/WildChat")                 # lmsys/lmsys-chat-1m, allenai/WildChat
malicious_dataset_names=("MaliciousGen")        # PKU-Alignment/BeaverTails, MaliciousGen
benign_num_clients=(5)
malicious_num_clients=(5)

gpu=0

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
 --use_auth_token \
 --output_dir $output_dir \
 --template $template \
 --mixture_num_clients $mixture_num_clients \
 --prefilter_strategy $prefilter_strategy \
 --prefilter_round $prefilter_round \
 ${prefilter_classifier_path:+--prefilter_classifier_path "$prefilter_classifier_path"}
