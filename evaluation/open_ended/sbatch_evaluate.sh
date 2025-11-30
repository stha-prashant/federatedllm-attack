#!/bin/bash -l

#SBATCH --account llm-degredation --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --time=00-3:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/log1.out
#SBATCH --error=/shared/rc/llm-degredation/logs/log1.err


export HUGGINGFACE_HUB_TOKEN='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'
module purge
conda activate fedllmold

# python run_checkpoint_generation_full.py --run_ids 222 --eval_list 5 30 --datasets sst2 gsm8k advbench
# python run_checkpoint_generation_full.py --run_ids 211 --eval_list 1 2 3 4 5 --datasets gsm8k advbench

# python run_checkpoint_generation_full.py --run_ids 219 --eval_list 5 30 --datasets gsm8k advbench
python run_checkpoint_generation_full.py --run_ids 223 --eval_list 5 30 --datasets sst2 advbench
# python run_checkpoint_generation_full.py --run_ids 221 --eval_list 5 30 --datasets sst2 gsm8k advbench

# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name advbench
# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name sst2
# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name gsm8k


# python gen_judge_advbench.py --judger rule --model_answer Llama-2-7b-chat-hf_vllm --bench_name sst2 --round 0 
# python gen_judge_advbench.py --judger rule --model_answer Llama-2-7b-chat-hf_vllm --bench_name gsm8k --round 0 
# python gen_judge_advbench.py --judger rule --model_answer Llama-2-7b-chat-hf_vllm --bench_name advbench --round 0 