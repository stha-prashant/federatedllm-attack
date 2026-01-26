#!/bin/bash -l

#SBATCH --account llm-degredation --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48g
#SBATCH --time=00-9:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/logtest.out
#SBATCH --error=/shared/rc/llm-degredation/logs/logtest.err

module purge
conda activate fedllmold

export HUGGINGFACE_HUB_TOKEN='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'


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
# you may set your local data directory here
# local_data_dir="LOCAL_DATA_DIR"
local_data_dir="/home/ps9044/FedLLM-Attack/gen_data"

# benign_num_clients=(2 2 2)
# benign_dataset_names=("allenai/WildChat" "lmsys/lmsys-chat-1m" "zhiqings/dromedary-65b-verbose-clone-v0") 

# malicious_num_clients=(2 2)
# malicious_dataset_names=("PKU-Align1ment/BeaverTails" "MaliciousGen") # PKU-Alignment/BeaverTails, MaliciousGen
# gpu=0


benign_num_clients=(7)
# benign_dataset_names=('stanfordnlp/sst2')
benign_dataset_names=('rajpurkar/squad_v2')
# benign_dataset_names=('qiaojin/PubMedQA')

# benign_dataset_names=('rajpurkar/squad_v2' 'qiaojin/PubMedQA')  # medQA, rajpurkar/squad_v2, stanfordnlp/sst2, qiaojin/PubMedQA
# benign_dataset_names=("allenai/WildChat") # allenai/WildChat, lmsys/lmsys-chat-1m
# benign_dataset_names=("zhiqings/dromedary-65b-verbose-clone-v0")
# malicious_num_clients=(3)
# malicious_dataset_names=("lmsys/lmsys-chat-1m") # allenai/WildChat, lmsys/lmsys-chat-1m

malicious_num_clients=(3)
# malicious_dataset_names=('qiaojin/PubMedQA')

# malicious_dataset_names=("MaliciousGen") # PKU-Alignment/BeaverTails, MaliciousGen
# malicious_dataset_names=('rajpurkar/squad_v2')
# malicious_dataset_names=("allenai/WildChat")
# malicious_dataset_names=("_") # PKU-Alignment/BeaverTails, MaliciousGen
# malicious_dataset_names=("stanfordnlp/sst2") # PKU-Alignment/BeaverTails, MaliciousGen

malicious_dataset_names=("PKU-Alignment/BeaverTails") # PKU-Alignment/BeaverTails, MaliciousGen
gpu=0

model_name_or_path="meta-llama/Llama-2-7b-chat-hf" # BASE MODEL PATH
output_dir='/shared/rc/llm-degredation/fedllm/barebones'
 
# fed_alg="fedgraph"
# fed_alg="safe_lora"
fed_alg=$method



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
#  --safe_lora_original \
#  --isa \
