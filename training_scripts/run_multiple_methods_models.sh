# jobFile="training_scripts/sbatch_test2_grid.sh"
# export NUM_MALICIOUS_CLIENTS=3
# export MALICIOUS_MIXTURE_PROPORTION=0.5
# # for method in "fedavg" "safelorav2data" "lasa" "flame" "krum" "dnc" "foolsgold" "median"
# for method in "lasa"
# do
#     for model in "llama2" "llama3" #"gemma" "qwen"
#     do
#         export FED_ALG=$method
#         export MODEL_KEY=$model
#         sbatch -t 00-7:00:00 $jobFile 
#     done
# done

# for method in "fedavg"
# do
#     for model in "gemma"
#     do
#         export FED_ALG=$method
#         export MODEL_KEY=$model
#         sbatch -t 00-7:00:00 $jobFile 
#     done
# done


# jobFile="training_scripts/sbatch_test2_grid.sh"

# for num_malicious_clients in 5
# do
#     for malicious_mixture_proportion in 0.9 0.0
#     do
#         for method in "fedavg" "safelorav2data" "lasa" "flame" "krum" "dnc" "foolsgold" "median"
#         do
#             for model in "llama2" #"llama3" "gemma" "qwen"
#             do
#                 export FED_ALG=$method
#                 export MODEL_KEY=$model
#                 export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                 export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                 sbatch -t 00-7:00:00 $jobFile
#             done
#         done
#     done
# done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in  0.9 0.0
#     do
#         for method in "fedavg" "safelorav2data" "lasa" "flame" "krum" "dnc" "foolsgold" "median"
#         do
#             for model in "llama2" #"llama3" "gemma" "qwen"
#             do
#                 export FED_ALG=$method
#                 export MODEL_KEY=$model
#                 export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                 export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                 sbatch -t 00-7:00:00 $jobFile
#             done
#         done
#     done
# done



# jobFile="training_scripts/sbatch_test2_grid.sh"
# # # jobFile="training_scripts/sbatch_safefedllm_grid.sh"
# export MALICIOUS_DATASET=expguardtrain
# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for method in "safeloradotdata"
#         do
#             for mixture_dirichlet_alpha in 0.1 0.5
#             do
#                 for model in "llama2" #"llama3" "gemma" "qwen"
#                 do
#                     export FED_ALG=$method
#                     export MODEL_KEY=$model
#                     export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                     export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                     export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                     sbatch -t 00-4:00:00 $jobFile
#                 done
#             done
#         done
#     done
# done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in  0.5
#     do
#         for mixture_dirichlet_alpha in  0.2
#         do
#             for method in "safeloradotdata"
#             do
#                 for model in "llama3" "gemma" "qwen"
#                 do
#                     export FED_ALG=$method
#                     export MODEL_KEY=$model
#                     export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                     export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                     export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                     sbatch -t 00-4:00:00 $jobFile
#                 done
#             done
#         done
#     done
# done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in  0.5
#     do
#         for mixture_dirichlet_alpha in  0.1 0.5
#         do
#             for method in "safefedllm"
#             do
#                 for model in "llama2" #"llama3" "gemma" "qwen"
#                 do
#                     export FED_ALG=$method
#                     export MODEL_KEY=$model
#                     export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                     export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                     export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                     sbatch -t 00-4:00:00 $jobFile
#                 done
#             done
#         done
#     done
# done


# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for mixture_dirichlet_alpha in  0.2
#         do
#             for method in "safefedllm"
#             do
#                 for model in "llama3" "gemma" "qwen"
#                 do
#                     export FED_ALG=$method
#                     export MODEL_KEY=$model
#                     export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                     export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                     export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                     sbatch -t 00-4:00:00 $jobFile
#                 done
#             done
#         done
#     done
# done
# 0.1 1.0


# jobFile="training_scripts/sbatch_test2_grid.sh"
# export MIXTURE_DIRICHLET_ALPHA=0.2
# export MODEL_KEY=llama2

# # dtype failures — unique configs
# for method in "safeloradotdata"; do
#   for num_malicious_clients in 3; do
#     for malicious_mixture_proportion in 0.5; do
#       for dataset in "PKU-Alignment/BeaverTails" "isa" "expguardtrain"; do
#         export FED_ALG=$method MODEL_KEY=llama2
#         export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#         export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#         export MALICIOUS_DATASET=$dataset
#         sbatch -t 00-5:00:00 $jobFile
#       done
#     done
#   done
#   for num_malicious_clients in 5; do
#     for malicious_mixture_proportion in 0.9 0.5 0.0; do
#       export FED_ALG=$method MODEL_KEY=llama2
#       export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#       export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#       export MALICIOUS_DATASET=expguardtrain
#       sbatch -t 00-5:00:00 $jobFile
#     done
#   done
# done


# jobFile="training_scripts/sbatch_test2_grid.sh"
# jobFile2="training_scripts/sbatch_safefedllm_grid.sh"

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for mixture_dirichlet_alpha in 999999
#         do
#             for method in  "flame"
#             do
#                 for model in "llama2"
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-4:00:00 $jobFile
#                             fi
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

jobFile="training_scripts/sbatch_test2_grid_tigris.sh"
jobFile2="training_scripts/sbatch_safefedllm_grid.sh"

# Cross-model safelorav2data with llmlatunsafe / beavertailsunsafe / llmlatdpo_safe / maliciousgen references.
# 7 benign + 3 malicious, alpha=0.2, seed=2023, poison 10% (prop=0.9) and 5% (prop=0.95).
# REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
# for ref_name in "llmlatunsafe" "beavertailsunsafe" "llmlatdpo_safe" "maliciousgen"; do
#     if [ "$ref_name" = "llmlatunsafe" ]; then
#         export SAFELORA_MATRIX_CONFIG="${REPO_ROOT}/configs/safelora_matrix_llmlatunsafe.json"
#     elif [ "$ref_name" = "beavertailsunsafe" ]; then
#         export SAFELORA_MATRIX_CONFIG="${REPO_ROOT}/configs/safelora_matrix_beavertails.json"
#     elif [ "$ref_name" = "llmlatdpo_safe" ]; then
#         export SAFELORA_MATRIX_CONFIG="${REPO_ROOT}/configs/safelora_matrix_llmlatdpo_safe.json"
#     elif [ "$ref_name" = "maliciousgen" ]; then
#         export SAFELORA_MATRIX_CONFIG="${REPO_ROOT}/configs/safelora_matrix_maliciousgen.json"
#     fi

#     if [ ! -f "$SAFELORA_MATRIX_CONFIG" ]; then
#         echo "Missing matrix config for ref=${ref_name}: ${SAFELORA_MATRIX_CONFIG}" >&2
#         exit 2
#     fi

#     export SAFELORA_REFERENCE_NAME=$ref_name
#     for method in "safelorav2data"; do
#         for malicious_mixture_proportion in 0.9 0.95; do
#             for model in "llama2" "llama3" "gemma" "qwen3"; do
#                 for dataset in "expguardtrain"; do
#                     export SEED=2023
#                     export MALICIOUS_DATASET=$dataset
#                     export FED_ALG=$method
#                     export MODEL_KEY=$model
#                     export NUM_MALICIOUS_CLIENTS=3
#                     export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                     export MIXTURE_DIRICHLET_ALPHA=0.2
#                     export MIXTURE_NUM_CLIENTS=10
#                     export SAMPLE_CLIENTS=10
#                     poison_pct=$(python3 -c "print(int(round((1.0 - ${malicious_mixture_proportion}) * 100)))")
#                     # Keep Slurm -J under ~64 chars; unique per reference.
#                     method_tag="data"; [ "$method" = "safelorav2dataround" ] && method_tag="round"
#                     case "$ref_name" in
#                       llmlatunsafe) ref_tag="llmlat" ;;
#                       beavertailsunsafe) ref_tag="beaver" ;;
#                       llmlatdpo_safe) ref_tag="dposafe" ;;
#                       maliciousgen) ref_tag="malgen" ;;
#                       *) ref_tag="$ref_name" ;;
#                     esac
#                     ds_tag="eg"; [ "$dataset" = "expguardtrainfix" ] && ds_tag="egfix"
#                     jname="${method_tag}_${model}_${ref_tag}_p${poison_pct}_${ds_tag}"
#                     echo "sbatch: ${jname} method=${method} ref=${ref_name} matrix=${SAFELORA_MATRIX_CONFIG}"
#                     sbatch -J "$jname" -t 00-08:00:00 $jobFile
#                 done
#             done
#         done
#     done
# done

# safelorav2data + llmlatdpo_safe ref, 10% poison (prop=0.9) on expguardtrainfix.
# REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
# export SAFELORA_REFERENCE_NAME="llmlatdpo_safe"
# export SAFELORA_MATRIX_CONFIG="${REPO_ROOT}/configs/safelora_matrix_llmlatdpo_safe.json"
# for model in "llama3" "gemma" "qwen3"; do
#     export SEED=2023
#     export MALICIOUS_DATASET="expguardtrainfix"
#     export FED_ALG="safelorav2data"
#     export MODEL_KEY=$model
#     export NUM_MALICIOUS_CLIENTS=3
#     export MALICIOUS_MIXTURE_PROPORTION=0.9
#     export MIXTURE_DIRICHLET_ALPHA=0.2
#     export MIXTURE_NUM_CLIENTS=10
#     export SAMPLE_CLIENTS=10
#     jname="data_${model}_dposafe_p10_egfix"
#     extra_sbatch=()
#     if [ -n "${AFTER_REF_JOB:-}" ]; then
#         extra_sbatch+=(--dependency="afterok:${AFTER_REF_JOB}")
#     fi
#     echo "sbatch: ${jname} method=safelorav2data ref=${SAFELORA_REFERENCE_NAME} matrix=${SAFELORA_MATRIX_CONFIG} dep=${AFTER_REF_JOB:-none}"
#     sbatch -J "$jname" -t 00-08:00:00 "${extra_sbatch[@]}" $jobFile
# done

# export SAFELORA_MATRIX_CONFIG=/home/ps9044/RPA/fedllm-attack/configs/safelora_matrix_diffllmlatdposftsafe.json
# export SAFELORA_REFERENCE_NAME=diffllmlatdposftsafe

# export SAFELORA_REFERENCE_NAME=beavertails
# export SAFELORA_MATRIX_CONFIG=/home/ps9044/RPA/fedllm-attack/configs/safelora_matrix_beavertails.json

# export SAFELORA_REFERENCE_NAME=llmlatunsafe
# export SAFELORA_MATRIX_CONFIG=/home/ps9044/RPA/fedllm-attack/configs/safelora_matrix_llmlatunsafe.json

for num_malicious_clients in 1
do
    for malicious_mixture_proportion in 0.9
    do
        for mixture_dirichlet_alpha in  999999 0.5 0.2 0.1
        do
            for method in "safelorav2dataround" 
            do
                for model in "llama2"
                do
                    for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
                    do
                        for seed in 2023
                        do
                            export SEED=$seed
                            export MALICIOUS_DATASET=$dataset
                            export FED_ALG=$method
                            export MODEL_KEY=$model
                            export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
                            export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
                            export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
                            if [ "$method" == "safefedllm" ]; then
                                sbatch -t 00-8:00:00 $jobFile2
                            else
                                sbatch -t 00-4:00:00 $jobFile
                            fi
                            # sbatch -t 00-05:00:00 $jobFile
                        done
                    done
                done
            done
        done
    done
done



# for num_malicious_clients in 3 5
# do
#     for malicious_mixture_proportion in 0.5 0.0 0.9
#     do
#         for mixture_dirichlet_alpha in  0.2
#         do
#             for method in "safelorav2dataround"
#             do
#                 for model in "llama2"
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 2024 2025
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                             # sbatch -t 00-05:00:00 $jobFile
#                         done
#                     done
#                 done
# #            
#             done
#         done
#     done
# done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for mixture_dirichlet_alpha in  0.5 0.1
#         do
#             for method in "safelorav2dataround"
#             do
#                 for model in "llama2"
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 2024 2025
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                             # sbatch -t 00-05:00:00 $jobFile
#                         done
#                     done
#                 done
# #            
#             done
#         done
#     done
# done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for mixture_dirichlet_alpha in  0.2
#         do
#             for method in "safelorav2dataround"
#             do
#                 for model in "llama3" "qwen3" "gemma"
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 2024 2025
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                             # sbatch -t 00-05:00:00 $jobFile
#                         done
#                     done
#                 done
# #            
#             done
#         done
#     done
# done

# for num_malicious_clients in 3 5
# do
#     for malicious_mixture_proportion in 0.5 0.9 
#     do
#         for mixture_dirichlet_alpha in 0.2
#         do
#             for method in "safelorav2data" "safelorav2dataround"
#             do
#                 for model in "llama3" "qwen3" "gemma"
#                 do
#                     for dataset in  "expguardtrainfix" "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                             # sbatch -t 00-05:00:00 $jobFile
#                         done
#                     done
#                 done
#                 for model in "llama2"
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                             # sbatch -t 00-05:00:00 $jobFile
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# # just dump the jsonl files
# jobFile="training_scripts/sbatch_test2_grid_tigris.sh"
# # jobFile2="training_scripts/sbatch_safefedllm_grid.sh"

# for num_malicious_clients in 3 5
# do
#     for malicious_mixture_proportion in 0.5 0.9 
#     do
#         for mixture_dirichlet_alpha in 0.2
#         do
#             for method in  "fedavg"
#             do
#                 for model in "llama2"
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2023 
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             sbatch -t 00-00:30:00 $jobFile
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.9 0.0
#     do
#         for mixture_dirichlet_alpha in 0.2
#         do
#             for method in  "fedavg" "safelorav2data" "safeloradotdata" "lasa" "flame" "krum" "dnc" "foolsgold" "median"  "safefedllm"
#             do
#                 for model in "llama2" 
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2024 2025
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# for num_malicious_clients in 5
# do
#     for malicious_mixture_proportion in 0.9 0.0 0.5
#     do
#         for mixture_dirichlet_alpha in 0.2
#         do
#             for method in  "fedavg" "safelorav2data" "safeloradotdata" "lasa" "flame" "krum" "dnc" "foolsgold" "median" "safefedllm"
#             do
#                 for model in "llama2" 
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2024 2025
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for mixture_dirichlet_alpha in 0.1 0.5
#         do
#             for method in  "fedavg" "safelorav2data" "safeloradotdata" "lasa" "flame" "krum" "dnc" "foolsgold" "median" "safefedllm"
#             do
#                 for model in "llama2" 
#                 do
#                     for dataset in  "expguardtrain" # "llmlatunsafe" #"PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"
#                     do
#                         for seed in 2024 2025
#                         do
#                             export SEED=$seed
#                             export MALICIOUS_DATASET=$dataset
#                             export FED_ALG=$method
#                             export MODEL_KEY=$model
#                             export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                             export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                             export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                             if [ "$method" == "safefedllm" ]; then
#                                 sbatch -t 00-8:00:00 $jobFile2
#                             else
#                                 sbatch -t 00-5:00:00 $jobFile
#                             fi
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# ########################################################

# jobFile="training_scripts/run_safefedllm_savelora.sh"

# for model in "llama3_0" "qwen3"   #"llama2" #"llama3" # "gemma" "qwen"
# do
#     export MODEL_KEY=$model
#     sbatch -t 00-2:00:00 $jobFile
# done


# jobFile="training_scripts/sbatch_test2_grid.sh"
# export MIXTURE_DIRICHLET_ALPHA=0.2
# export FED_ALG=safeloradotdata

# # llama2: 3-mal mix0.5, four datasets
# for dataset in "PKU-Alignment/BeaverTails" "isa" "expguardtrain" "llmlatunsafe"; do
#   export MODEL_KEY=llama2
#   export NUM_MALICIOUS_CLIENTS=3
#   export MALICIOUS_MIXTURE_PROPORTION=0.5
#   export MALICIOUS_DATASET=$dataset
#   sbatch -t 00-5:00:00 "$jobFile"
# done

# # llama2: 5-mal expguardtrain, three mixes
# for mix in 0.9 0.5 0.0; do
#   export MODEL_KEY=llama2
#   export NUM_MALICIOUS_CLIENTS=5
#   export MALICIOUS_MIXTURE_PROPORTION=$mix
#   export MALICIOUS_DATASET=expguardtrain
#   sbatch -t 00-5:00:00 "$jobFile"
# done

# # llama3_0 / qwen3: 3-mal mix0.5 expguardtrain
# for model in "llama3_0" "qwen3"; do
#   export MODEL_KEY=$model
#   export NUM_MALICIOUS_CLIENTS=3
#   export MALICIOUS_MIXTURE_PROPORTION=0.5
#   export MALICIOUS_DATASET=expguardtrain
#   sbatch -t 00-5:00:00 "$jobFile"
# done

# jobFile="training_scripts/sbatch_safefedllm_grid.sh"
# for mixture_dirichlet_alpha in 0.1 0.5
# do
#     export FED_ALG=safefedllm
#     export MODEL_KEY=llama2
#     export NUM_MALICIOUS_CLIENTS=3
#     export MALICIOUS_MIXTURE_PROPORTION=0.5
#     export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#     echo "sbatch: safefedllm model=llama2 n_mal=3 prop=0.5 alpha=$mixture_dirichlet_alpha"
#     sbatch -t 00-4:00:00 "$jobFile"
# done