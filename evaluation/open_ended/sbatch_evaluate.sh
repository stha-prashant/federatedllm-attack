#!/bin/bash -l

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=90g
#SBATCH --time=00-2:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/logeval1.out
#SBATCH --error=/shared/rc/llm-degredation/logs/logeval1.err


export HUGGINGFACE_HUB_TOKEN='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'
export HUGGINGFACE_HUB_CACHE="/shared/rc/llm-degredation/ps9044/huggingface/hub"
export HF_DATASETS_CACHE="/scratch/ps9044/huggingface/datasets"
mkdir -p "${HF_DATASETS_CACHE}"
module purge
conda activate testvllm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# conda activate fedllmold
# python run_checkpoint_generation_full.py --run_ids brjjyo91 3vwnwunw xdz2tdvx  --eval_list 30 --datasets expguardtest
# python run_checkpoint_generation_full.py --run_ids dsnrocim bxj6y44j pjvs5u5c --eval_list 30 --datasets directharm expguardtest
# python run_checkpoint_generation_full.py --run_ids 3g5eksbr fdcuumiq 0x1shryu --eval_list 30 --datasets directharm expguardtest
# python run_checkpoint_generation_full.py --run_ids 7xbbt70r 9ukzkui7 o6w8ufu1 xvtai1td  --eval_list 30 --datasets directharm expguardtest
# python run_checkpoint_generation_full.py --run_ids  hjbk71wc 56wbtav3 p04ii2td --eval_list 30 --datasets directharm expguardtest
# python run_checkpoint_generation_full.py --safe_lora_original_minimal  --run_ids xdz2tdvx dsnrocim 63771uof --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30
# python run_checkpoint_generation_full.py --run_ids 82bo2lvc --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 2
# python run_checkpoint_generation_full.py --safe_lora_original_minimal --run_ids 82bo2lvc --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 2
# python run_checkpoint_generation_full.py --run_ids rz6qfrm7 9h7xqc5b --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 3
# python run_checkpoint_generation_full.py --run_ids 52ttvi2g txxvqyfi kofkdnpc di7ym3cn rsxn5oer --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 3
# python run_checkpoint_generation_full.py --run_ids tbw83r7y  --datasets cord19 --eval_list 30  --gpus 3

# python run_safedelta_and_evaluate_copy.py --run_ids  1o8mxz36 0m2qj5i6 arib4r2c --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 0

# python run_checkpoint_generation_full.py --run_ids glb9l3wa wq2m8510  --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 2 6
# python run_checkpoint_generation_full.py --run_ids  4xlza2jt 9ck4ejw5 --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30  --gpus 2 6

# python run_checkpoint_generation_full.py --run_ids  oazwx6c1 --datasets advbench directharm expguardtest --eval_list 10  --gpus  0
# python run_checkpoint_generation_full.py --run_ids uk63vhmq oh7151su w37ncf7p uzbc3ysr hgc6y6ur p7ux1spn q72okofj 2pv4qzgs uvk0n0ur 69mc3ijy --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30 --gpus  0
# python run_checkpoint_generation_full.py --run_ids so9lbwlj --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 28  --gpus  0


# python run_checkpoint_generation_full.py --run_ids twxo6oh9 twxo6oh9 roaewe7d sj3jk161 --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 28  --gpus  0
# python run_checkpoint_generation_full.py --run_ids 44cg8vzp --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 28 --gpus  0
# python run_checkpoint_generation_full.py --run_ids  75mfjdur --datasets advbench --eval_list 30 --gpus  0 

set -euo pipefail
cd /home/ps9044/RPA/fedllm-attack/evaluation/open_ended

: "${WANDB_RUN_ID:?Set WANDB_RUN_ID to the wandb run id}"
: "${EVAL_ROUND:?Set EVAL_ROUND to the last checkpoint round to evaluate}"

python run_checkpoint_generation_full.py \
  --run_ids "${WANDB_RUN_ID}" \
  --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 \
  --eval_list "${EVAL_ROUND}" \
  --gpus 0

# python run_checkpoint_generation_full.py --run_ids t3fap8qs  --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 25 --gpus  0
# python run_checkpoint_generation_full.py --run_ids v0ig2of5  --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 27 --gpus  0



# python run_checkpoint_generation_full.py --run_ids  szwq9lrp --datasets advbench directharm --eval_list 30 --gpus  0
# python run_checkpoint_generation_full.py --run_ids 5z379u27 c195bscm --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 23 --gpus  0

# python run_checkpoint_generation_full.py --run_ids 1zd8yjd3 --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 26 --gpus  0

# python run_checkpoint_generation_full.py  --run_ids h4b1l47d 9w9ylzf3 zt2s6ndc  pbqlda7q ofcehrgv su6939l0 --datasets expguardtest --eval_list 30
# python generate_from_list.py --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30

# python run_checkpoint_generation_full.py --run_ids i0gmap4p  --datasets directharm expguardtest --eval_list 30
# python run_checkpoint_generation_full.py --run_ids kjom8wuc  --datasets directharm expguardtest --eval_list 50
# python run_checkpoint_generation_full.py --run_ids r4zxv8og  --datasets directharm expguardtest --eval_list 1



# python run_checkpoint_generation_full.py --run_ids 7xbbt70r --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30
# python run_checkpoint_generation_full.py --run_ids xvtai1td --datasets directharm expguardtest --eval_list 30


# python run_checkpoint_generation_full.py --run_ids kjom8wuc --eval_list 50 --datasets advbench pubmedqa medQA emrqa cord19


# python run_checkpoint_generation_full.py --run_ids 9ukzkui7 hjbk71wc fdcuumiq o6w8ufu1 --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30

# python run_checkpoint_generation_full_copy.py --run_ids xdz2tdvx --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30


# python run_checkpoint_generation_full.py --run_ids f3dv623m w8i0stff vxdqjd7i riyp1557 --eval_list 30 --datasets advbench pubmedqa medQA emrqa cord19

# python run_safedelta_and÷_evaluate.py --safe_delta_original   --safe_delta_thrs 0.0 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.60 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evaluate.py --safe_delta_original   --safe_delta_thrs 0.80 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
# python run_safedelta_and_evauate.py --safe_delta_original   --safe_delta_thrs 0.10 --run_ids 287 --datasets advbench pubmedqa --eval_list 30
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

# python run_checkpoint_generation_full.py --safe_lora_original_minimal  --run_ids  oqo3lz17 --datasets advbench directharm expguardtest --eval_list 30 --gpus 6 

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

