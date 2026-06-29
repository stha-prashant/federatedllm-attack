#!/bin/bash -l

#SBATCH --account whiskers --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=45g
#SBATCH --time=00-10:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/logeval_safedeltax11.out
#SBATCH --error=/shared/rc/llm-degredation/logs/logeval_safedeltax11.err
export HUGGINGFACE_HUB_TOKEN='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'

# module purge
# conda activate fedllmold


for safedelta_thrs in 0.2 0.4 0.6
do

  python run_safedelta_and_evaluate_clean.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr xdz2tdvx dsnrocim --datasets advbench expguardtest directharm advbench pubmedqa medQA emrqa cord19 --gpus 0 3 5 --eval_list 30



done


