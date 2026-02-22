#!/bin/bash -l

#SBATCH --account llm-degredation --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00-5:00:00
#SBATCH --mem=48g
#SBATCH --job-name=fedllm
#SBATCH --output=/scratch/ps9044/fedllm/log_%A.out
#SBATCH --error=/scratch/ps9044/fedllm/log_test.err

module purge
conda activate fedllmold

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


export HUGGINGFACE_HUB_TOKEN='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'

output_dir=./Llama-2-7b-chat-hf
model_name_or_path="meta-llama/Llama-2-7b-chat-hf"  
fed_alg="fedavg"          

# LoRA Classifier Pre-filter Configuration
prefilter_enable=True                  
prefilter_classifier_path="./prefilter_classifier_llama-2-7b-chat-hf"                   
prefilter_strategy="none"                              # Pre-filtering weight policy options: step-level, client-level, shadow-level, none
prefilter_round=10

# Dataset Configuration
benign_dataset_names=("allenai/WildChat")                 # lmsys/lmsys-chat-1m, allenai/WildChat
malicious_dataset_names=("MaliciousGen")        # PKU-Alignment/BeaverTails, MaliciousGen

# Interactive input for parameters
# echo "Please enter the GPU ID (e.g., 0, 1, 2...):"
# read gpu

# echo "Please enter the number of benign clients:"
# read input_benign
# benign_num_clients=$input_benign

# echo "Please enter the number of malicious clients:"
# read input_malicious
# malicious_num_clients=$input_malicious

gpu=0
benign_num_clients=(5)
malicious_num_clients=(5)



# Display configuration information
# echo "=== Training Configuration ==="
# echo "GPU ID: $gpu"
# echo "Number of benign clients: $benign_num_clients"
# echo "Number of malicious clients: $malicious_num_clients"
# echo "Total number of clients: $((benign_num_clients + malicious_num_clients))"
# echo "Number of sampled clients per round: $sample_num_this_round"
# echo "=============================="


CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --benign_num_clients $benign_num_clients \
 --benign_dataset_names ${benign_dataset_names[@]} \
 --malicious_num_clients $malicious_num_clients \
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
 --prefilter_enable $prefilter_enable \
 --prefilter_classifier_path "$prefilter_classifier_path" \
 --prefilter_strategy $prefilter_strategy \
 --prefilter_gpu $gpu \
 --prefilter_round $prefilter_round \