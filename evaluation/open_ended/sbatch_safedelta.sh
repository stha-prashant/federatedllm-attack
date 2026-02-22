#!/bin/bash -l

#SBATCH --account llm-degredation --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=45g
#SBATCH --time=00-5:00:00
#SBATCH --job-name=fedllm
#SBATCH --output=/shared/rc/llm-degredation/logs/logeval_safedelta.out
#SBATCH --error=/shared/rc/llm-degredation/logs/logeval_safedelta.err
export HUGGINGFACE_HUB_TOKEN='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'
# module purge
# conda activate fedllmold


# ids=("419" "420" "421" "422")

for safedelta_thrs in 0.8 1.0
do
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 419 --datasets advbench pubmedqa --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 420 --datasets advbench triviaqa --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 421 --datasets advbench gsm8k --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 422 --datasets advbench pubmedqa triviaqa --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 526 --datasets advbench gsm8k squad_v2 --eval_list 30

    python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 783 --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30

done
