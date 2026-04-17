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
module purge
conda activate fedllmold


# ids=("419" "420" "421" "422")

# for safedelta_thrs in 0.35
# do
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 419 --datasets advbench pubmedqa --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 420 --datasets advbench triviaqa --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 421 --datasets advbench gsm8k --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 422 --datasets advbench pubmedqa triviaqa --eval_list 30
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 526 --datasets advbench gsm8k squad_v2 --eval_list 30
    
    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  zcre5127 --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30

    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  9e98b55p ozu7vtio --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30

    # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 8s415aie pc1yua44 tmbdapqn --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done


# for safedelta_thrs in 0.6
# do
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 419 --datasets advbench pubmedqa --eval_list 30
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 420 --datasets advbench triviaqa --eval_list 30
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 421 --datasets advbench gsm8k --eval_list 30
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 422 --datasets advbench pubmedqa triviaqa --eval_list 30
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 526 --datasets advbench gsm8k squad_v2 --eval_list 30
    
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  zcre5127 --datasets advbench --eval_list 30

#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  63771uof --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  xdz2tdvx --datasets directharm --eval_list 30

#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  xdz2tdvx dsnrocim 3g5eksbr --datasets expguardtest directharm --eval_list 30

#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids  63771uof 3g5eksbr --datasets expguardtest directharm --eval_list 30
    
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr  --datasets expguardtest directharm --eval_list 30
#     python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids dsnrocim 63771uof  --datasets expguardtest directharm --eval_list 30
    
#     # python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 8s415aie pc1yua44 tmbdapqn --datasets advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done


# for safedelta_thrs in 0.1 0.2 0.4 0.6
# do
# #   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr xdz2tdvx dsnrocim 63771uof  --datasets medQA emrqa cord19 --eval_list 30
# # done

# # for safedelta_thrs in  0.4 0.6
# # do
#   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr xdz2tdvx dsnrocim 63771uof  --datasets expguardtest directharm advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done

# for safedelta_thrs in 0.1 0.2 0.4 0.6
# do
# #   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr xdz2tdvx dsnrocim 63771uof  --datasets medQA emrqa cord19 --eval_list 30
# # done

# # for safedelta_thrs in  0.4 0.6
# # do
#   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids ofcehrgv su6939l0 --datasets expguardtest directharm advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done

# for safedelta_thrs in 0.1 0.2 0.4 0.6
# do
# #   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr xdz2tdvx dsnrocim 63771uof  --datasets medQA emrqa cord19 --eval_list 30
# # done

# # for safedelta_thrs in  0.4 0.6
# # do
#   python run_safedelta_and_evaluate_copy.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids ofcehrgv su6939l0  9w9ylzf3 --datasets expguardtest directharm advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done


for safedelta_thrs in 0.2 0.4 0.6
do
  python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids ofcehrgv su6939l0  9w9ylzf3 --datasets expguardtest directharm --eval_list 30
done



# for safedelta_thrs in 0.2 0.4 0.6
# do
#   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 3g5eksbr xdz2tdvx dsnrocim --datasets expguardtest directharm advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done

# for safedelta_thrs in 0.2 0.4 0.6
# do
#   python run_safedelta_and_evaluate.py --safe_delta_original --safe_delta_thrs $safedelta_thrs --run_ids 63771uof --datasets expguardtest directharm advbench pubmedqa medQA emrqa cord19 --eval_list 30
# done




# 