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



jobFile="training_scripts/sbatch_test2_grid.sh"

for num_malicious_clients in 5
do
    for malicious_mixture_proportion in 0.5
    do
        for method in "dnc" "foolsgold" 
        do
            for mixture_dirichlet_alpha in 0.2
            do
                for model in "llama2" #"llama3" "gemma" "qwen"
                do
                    export FED_ALG=$method
                    export MODEL_KEY=$model
                    export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
                    export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
                    export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
                    sbatch -t 00-7:00:00 $jobFile
                done
            done
        done
    done
done

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in  0.1
#     do
#         for method in "median"
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
# # # 0.1 1.0


# jobFile="training_scripts/sbatch_test2_grid.sh"

# for num_malicious_clients in 3
# do
#     for malicious_mixture_proportion in 0.5
#     do
#         for mixture_dirichlet_alpha in 0.5 0.1
#         do
#             for method in "fedavg" "safelorav2data" "lasa" "flame" "krum" "dnc" "foolsgold" "median"
#             do
#                 for model in "llama2" #"llama3" "gemma" "qwen"
#                 do
#                     export FED_ALG=$method
#                     export MODEL_KEY=$model
#                     export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
#                     export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
#                     export MIXTURE_DIRICHLET_ALPHA=$mixture_dirichlet_alpha
#                     sbatch -t 00-5:00:00 $jobFile
#                 done
#             done
#         done
#     done
# done