#!/bin/bash -l

#SBATCH --account llm-degredation --partition tigris
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=200g
#SBATCH -t 6:00:00
#SBATCH --job-name=safelora_baseproject
#SBATCH --output=/home/ps9044/RPA/fedllm-attack/slurm-safelora-base-%j.out
#SBATCH --error=/home/ps9044/RPA/fedllm-attack/slurm-safelora-base-%j.err

set -euo pipefail

module purge
conda activate testvllm

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "${SCRIPT_DIR}"

# python safe_lora_baseproject.py

# Each entry: "lora_checkpoint|hf_base_model"
# merge_jobs=(
#   "/shared/rc/llm-degredation/aaai2026/Qwen3-4B-Instruct-2507_MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260719234856/checkpoint-50|Qwen/Qwen3-4B-Instruct-2507"
#   "/shared/rc/llm-degredation/aaai2026/Meta-Llama-3-8B-Instruct_MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260719234853/checkpoint-50|meta-llama/Meta-Llama-3-8B-Instruct"
# )
# merge_jobs=(
#   "/shared/rc/llm-degredation/aaai2026/references/Llama-3.1-8B-Instruct_llmlatunsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260917143225_7199d312/checkpoint-50|meta-llama/Llama-3.1-8B-Instruct"
#   "/shared/rc/llm-degredation/aaai2026/references/Llama-3.1-8B-Instruct_BeaverTailsUnsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260917144735_f7857f4e/checkpoint-50|meta-llama/Llama-3.1-8B-Instruct"
#   "/shared/rc/llm-degredation/aaai2026/references/gemma-2-2b-it_llmlatunsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260917144002_86f4ac7d/checkpoint-50|google/gemma-2-2b-it"
#   "/shared/rc/llm-degredation/aaai2026/references/gemma-2-2b-it_BeaverTailsUnsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260917145746_c3c2a9ae/checkpoint-50|google/gemma-2-2b-it"
#   "/shared/rc/llm-degredation/aaai2026/references/Qwen3-4B-Instruct-2507_llmlatunsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260917144002_73db4726/checkpoint-50|Qwen/Qwen3-4B-Instruct-2507"
#   "/shared/rc/llm-degredation/aaai2026/references/Qwen3-4B-Instruct-2507_BeaverTailsUnsafe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260917150534_a7bea44d/checkpoint-50|Qwen/Qwen3-4B-Instruct-2507"
# )
merge_jobs=(
  "/shared/rc/llm-degredation/aaai2026/references/Llama-3.1-8B-Instruct_llmlatdpo_safe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260918045901_b7d3830e/checkpoint-50|meta-llama/Llama-3.1-8B-Instruct"
  "/shared/rc/llm-degredation/aaai2026/references/gemma-2-2b-it_llmlatdpo_safe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260918045902_2956b1fa/checkpoint-50|google/gemma-2-2b-it"
  "/shared/rc/llm-degredation/aaai2026/references/Qwen3-4B-Instruct-2507_llmlatdpo_safe1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20260918045901_27658d99/checkpoint-50|Qwen/Qwen3-4B-Instruct-2507"
)

for job in "${merge_jobs[@]}"; do
  lora_path="${job%%|*}"
  base_model_path="${job#*|}"
  echo "Merging LoRA: ${lora_path} onto ${base_model_path}"
  python /home/ps9044/RPA/fedllm-attack/utils/merge_lora.py \
    --lora_path "${lora_path}" \
    --base_model_path "${base_model_path}"
done

cd /home/ps9044/RPA/fedllm-attack/federated_learning
python save_delta.py
