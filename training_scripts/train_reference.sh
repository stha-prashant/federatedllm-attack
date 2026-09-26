
module purge
conda activate testvllm


max_steps=10
num_rounds=50
batch_size=16
gradient_accumulation_steps=1
seq_length=512
sample_clients=1
lora_r=32
lora_alpha=64   # twice of lora_r
lr=5e-5

num_data_per_client=1000
local_data_dir="gen_data"       # you may set your local data directory


benign_num_clients=(1)
benign_dataset_names=("MaliciousGen")


malicious_num_clients=(0)
malicious_dataset_names=("_")

model_name_or_path="meta-llama/Llama-2-7b-chat-hf" # BASE MODEL PATH

output_dir='./output'


gpu=2
fed_alg="fedavg"

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --local_data_dir $local_data_dir \
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

