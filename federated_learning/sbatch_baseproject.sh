#!/bin/bash -l

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=256g
#SBATCH -t 3:00:00
#SBATCH --job-name=safelora_baseproject
#SBATCH --output=/home/ps9044/RPA/fedllm-attack/slurm-safelora-base-%j.out
#SBATCH --error=/home/ps9044/RPA/fedllm-attack/slurm-safelora-base-%j.err

# set -euo pipefail

module purge
conda activate testvllm

# REPO_DIR="/home/ps9044/RPA/fedllm-attack"
# cd "${REPO_DIR}"

python safe_lora_baseproject.py