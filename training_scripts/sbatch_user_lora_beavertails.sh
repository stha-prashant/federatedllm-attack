#!/bin/bash -l
# Row 5: UserLoRA-only fedavg from clean llama2, malicious=BeaverTails.

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=48g
#SBATCH -J user_lora_bt
#SBATCH --chdir=/home/ps9044/RPA/fedllm-attack

REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
cd "${REPO_ROOT}"

module purge
conda activate testvllm

export HUGGINGFACE_HUB_TOKEN='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'
export HUGGINGFACE_HUB_CACHE="/shared/rc/llm-degredation/ps9044/huggingface/hub"
export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

num_malicious_clients=3
malicious_mixture_proportion=0.5
mixture_benign_proportions=()
for ((i = 0; i < 10 - num_malicious_clients; i++)); do mixture_benign_proportions+=(1.0); done
for ((i = 0; i < num_malicious_clients; i++)); do mixture_benign_proportions+=("${malicious_mixture_proportion}"); done

CUDA_VISIBLE_DEVICES=0 python "${REPO_ROOT}/main_sft.py" \
  --learning_rate 5e-5 \
  --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
  --local_data_dir /home/ps9044/RPA/fedllm-attack/gen_data \
  --benign_num_clients 10 \
  --benign_dataset_names "qiaojin/PubMedQA" "medQA" "emrqa" "cord19" \
  --malicious_num_clients "${num_malicious_clients}" \
  --malicious_dataset_names "PKU-Alignment/BeaverTails" \
  --num_data_per_client 500 \
  --fed_alg fedavg \
  --sample_clients 10 \
  --max_steps 10 \
  --num_rounds 30 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --seq_length 512 \
  --peft_lora_r 32 \
  --peft_lora_alpha 64 \
  --use_peft \
  --load_in_8bit \
  --output_dir /scratch/ps9044/aaai2026 \
  --template chat \
  --mixture_num_clients 10 \
  --mixture_benign_proportions "${mixture_benign_proportions[@]}" \
  --mixture_dirichlet_alpha 0.5 \
  --seed 2023 \
  --throw_n 0 \
  --log_with none
