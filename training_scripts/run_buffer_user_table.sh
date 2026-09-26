#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Submitting row 1 eval (full-50)..."
sbatch -t 04:00:00 training_scripts/eval_row1_full50.sh

echo "Submitting buffer+user training (BeaverTails)..."
MALICIOUS_DATASET=beavertails sbatch -t 12:00:00 --export=ALL,MALICIOUS_DATASET=beavertails training_scripts/sbatch_buffer_user_lora.sh

echo "Submitting buffer+user training (expguardtrain)..."
MALICIOUS_DATASET=expguardtrain sbatch -t 12:00:00 --export=ALL,MALICIOUS_DATASET=expguardtrain training_scripts/sbatch_buffer_user_lora.sh

echo "Submitting row 5 UserLoRA-only (BeaverTails)..."
sbatch -t 12:00:00 training_scripts/sbatch_user_lora_beavertails.sh

echo ""
echo "After training completes, dual-merge eval:"
echo "  sbatch -t 08:00:00 --wrap \"bash training_scripts/eval_buffer_user_dual_merge.sh /shared/rc/llm-degredation/aaai2026/<RUN_DIR>\""
echo ""
echo "Row 5 eval:"
echo "  python evaluation/open_ended/run_checkpoint_generation_full_path.py --base_output_dir <ROW5_DIR> --base_model_path meta-llama/Llama-2-7b-chat-hf --datasets advbench directharm expguardtest pubmedqa medQA emrqa cord19 --eval_list 30 --gpu 0"
