#!/bin/bash -l

#SBATCH --account llm-degredation --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=45g
#SBATCH --time=00-2:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/logeval_%A.out
#SBATCH --error=/shared/rc/llm-degredation/logs/logeval_%A.err


export HUGGINGFACE_HUB_TOKEN='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'
module purge
conda activate fedllmold

# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.0 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.60 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.80 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.10 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.20 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 1.0 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 10000.0 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.05 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.01 --run_ids 287 --datasets advbench pubmedqa --eval_list 30




# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.0 --run_ids 286 --datasets advbench squad_v2 --eval_list 30  
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.20 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.40 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.05 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 1.0 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.60 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.80 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.01 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 10000.0 --run_ids 286 --datasets advbench squad_v2 --eval_list 30

# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 100.0 --run_ids 286 --datasets advbench squad_v2 --eval_list 30

# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 200.0 --run_ids 286 --datasets advbench squad_v2 --eval_list 30


# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.10 --run_ids 286 --datasets advbench squad_v2 --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.05 --run_ids 286 --datasets advbench squad_v2 --eval_list 30

python run_checkpoint_generation_full.py --safe_lora_original_minimal  --run_ids 465 --datasets advbench triviaqa --eval_list 30

# python run_checkpoint_generation_full.py --safe_lora_original  --run_ids 292 295 299 302 --datasets advbench squad_v2 --eval_list 30
# python run_checkpoint_generation_full.py   --run_ids 244 --datasets advbench --eval_list 30

# python run_checkpoint_generation_full.py --run_ids 222 --eval_list 5 30 --datasets sst2 gsm8k advbench
# python run_checkpoint_generation_full.py --run_ids 211 --eval_list 1 2 3 4 5 --datasets gsm8k advbench

# python run_checkpoint_generation_full.py --run_ids 219 --eval_list 5 30 --datasets gsm8k advbench
# python run_checkpoint_generation_full.py --run_ids 223 --eval_list 5 30 --datasets sst2 advbench
# python run_checkpoint_generation_full.py --run_ids 221 --eval_list 5 30 --datasets sst2 gsm8k advbench


# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name advbench
# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name squad_v2
# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name pubmedqa

# conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name sst2
# # conda run -n testvllm python gen_model_answer.py --gpu 0 --use_vllm --base_model_path meta-llama/Llama-2-7b-chat-hf --bench_name gsm8k


# python gen_judge_advbench.py --judger rule --model_answer Llama-2-7b-chat-hf_vllm_chat_greedy --bench_name pubmedqa --round 0 
# python gen_judge_advbench.py --judger rule --model_answer Llama-2-7b-chat-hf_vllm_chat_greedy --bench_name advbench --round 0 
# python gen_judge_advbench.py --judger rule --model_answer Llama-2-7b-chat-hf_vllm_chat_greedy --bench_name squad_v2 --round 0 

# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 10 20 30 --run_ids 238

# python run_checkpoint_generation_full.py --datasets squad_v2 advbench --eval_list 50 --run_ids 248 && python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 50 --run_ids 249
# python run_checkpoint_generation_full.py --datasets squad_v2 advbench --eval_list 1 2 3 4 5 6 7 8 9 10 11 15 20 25 30 --run_ids 286
# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 1 2 3 4 5 6 7 8 9 10 11 15 20 25 30 --run_ids 287
# python run_checkpoint_generation_full_copy.py --datasets pubmedqa advbench --eval_list 12 13 14 16 17 18 19 21 22 23 24 26 27 28 29  --run_ids 287

# python run_checkpoint_generation_full.py --datasets squad_v2 pubmedqa advbench --eval_list 1 2 3 4 5 6 7 8 9 10 11 15 20 25 30 --run_ids 289

#python run_checkpoint_generation_full.py --datasets squad_v2 advbench --eval_list 5 10 15 25 --run_ids 286
# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 30 --run_ids 303
# python run_checkpoint_generation_full.py --datasets squad_v2 advbench --eval_list 10 15 25 --run_ids 286
# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 4 7 9 15 --run_ids 287
# python run_checkpoint_generation_full.py --datasets squad_v2 advbench maliciousgen --eval_list 5 15 25 --run_ids 286
# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 30 --run_ids 245

# python run_checkpoint_generation_full.py --datasets sst2 advbench --eval_list 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 --run_ids 304 305
# python run_checkpoint_generation_full.py --datasets pubmedtrain  pubmedqa advbench --eval_list 28 27 30 29 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 --run_ids 245 306 307 287


# python run_checkpoint_generation_full.py --datasets squadv2train --eval_list 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --run_ids 286 

# python run_checkpoint_generation_full.py --datasets squadv2train  --eval_list 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --run_ids 286
# python run_checkpoint_generation_full.py --datasets squadv2train squad_v2   --eval_list 1 2 3 4 5 6  --run_ids 244 286
# python run_checkpoint_generation_full_copy.py --datasets ssttrain --eval_list 1 2 3 4 5 6 7 8 9 10 --run_ids 310
# python run_checkpoint_generation_full_copy.py --datasets ssttrain --eval_list 11 12 13 14 15 16 17 18 19 20  --run_ids 310
# python run_checkpoint_generation_full_copy.py --datasets ssttrain --eval_list 21 22 23 24 25 26 27 28 29 30 --run_ids 310


# python run_checkpoint_generation_full_copy.py --datasets ssttrain --eval_list 1 2 3 4 5 6 7 8 9 10 --run_ids 309
# python run_checkpoint_generation_full_copy.py --datasets ssttrain --eval_list 11 12 13 14 15 16 17 18 19 20  --run_ids 309
# python run_checkpoint_generation_full_copy.py --datasets ssttrain --eval_list 21 22 23 24 25 26 27 28 29 30 --run_ids 309


# python run_checkpoint_generation_full.py --datasets pubmedtrain pubmedval --eval_list 1 2 3 4 5 6 7 8 9 10  --run_ids 313 314 315 316
# python run_checkpoint_generation_full.py --datasets pubmedtrain pubmedval --eval_list 11 12 13 14 15 16 17 18 19 20   --run_ids 313 314 315 316
# python run_checkpoint_generation_full.py --datasets pubmedtrain pubmedval --eval_list 21 22 23 24 25 26 27 28 29 30  --run_ids 313 314 315 316

# python run_checkpoint_generation_full.py --datasets advbench pubmedqa --eval_list 9 10  --run_ids 313 314 315 316
# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 9 --run_ids 287
# python run_checkpoint_generation_full.py --datasets advbench pubmedqa --eval_list 25 26 27 28 29 30 --run_ids 313 314 315 316








# python run_checkpoint_generation_full.py --datasets pubmedval  --eval_list  24 25 26 27 28 29 30 --run_ids  245 287 307

# python run_checkpoint_generation_full_copy.py --datasets pubmedqa  pubmedval advbench --eval_list 4 9 15 --run_ids 287



# python run_checkpoint_generation_full.py --datasets pubmedqa advbench --eval_list 28  --run_ids 245 307 306



# python run_checkpoint_generation_full.py --datasets squad_v2 advbench --eval_list 12 13 14 16 17 18 19 21 22 23 24 26 27 28 29 --run_ids 286


# python run_checkpoint_generation_full.py --datasets  sst2 advbench --eval_list 24 25 26 27 28 29 30 --run_ids 309 310


# python run_checkpoint_generation_full.py --datasets squad_v2 --eval_list 5 15 25 --run_ids 244

