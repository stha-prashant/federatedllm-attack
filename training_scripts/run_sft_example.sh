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
local_data_dir="gen_data"

# benign_num_clients=(2 2 2)
# benign_dataset_names=("allenai/WildChat" "lmsys/lmsys-chat-1m" "zhiqings/dromedary-65b-verbose-clone-v0") 

# malicious_num_clients=(2 2)
# malicious_dataset_names=("PKU-Alignment/BeaverTails" "MaliciousGen") # PKU-Alignment/BeaverTails, MaliciousGen
# gpu=0


benign_num_clients=(5)
# benign_dataset_names=("allenai/WildChat") # allenai/WildChat, lmsys/lmsys-chat-1m
benign_dataset_names=("zhiqings/dromedary-65b-verbose-clone-v0")
# malicious_num_clients=(3)
# malicious_dataset_names=("lmsys/lmsys-chat-1m") # allenai/WildChat, lmsys/lmsys-chat-1m

malicious_num_clients=(5)
malicious_dataset_names=("PKU-Alignment/BeaverTails") # PKU-Alignment/BeaverTails, MaliciousGen
gpu=5

model_name_or_path="meta-llama/Llama-2-7b-chat-hf" # BASE MODEL PATH
output_dir='/scratch/ps9044/fedllm/barebones'
 
# fed_alg="fedgraph"
fed_alg="fedavg"


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
 --template "alpaca" \
 --safe_lora \
#  --isa \