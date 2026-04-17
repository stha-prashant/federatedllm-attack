#!/bin/bash -l

#SBATCH --account whiskers --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=45g
#SBATCH --time=00-10:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/logeval_safedelta.out
#SBATCH --error=/shared/rc/llm-degredation/logs/logeval_safedelta.err
export HUGGINGFACE_HUB_TOKEN='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'
module purge
conda activate fedllmold


python compute_references.py