#!/bin/bash -l

#SBATCH --account llm-degredation --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100g
#SBATCH --time=00-4:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/loghess.out
#SBATCH --error=/shared/rc/llm-degredation/logs/loghess.err

module purge
conda activate fedllmold

export HUGGINGFACE_HUB_TOKEN='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'

python hessian3.py
