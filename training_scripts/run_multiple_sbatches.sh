jobFile1="training_scripts/sbatch1.sh"
jobFile2="training_scripts/sbatch2.sh"
jobFile3="training_scripts/sbatch3.sh"

jobFile4="training_scripts/sbatch_test2.sh"


# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/medQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251203103114" "/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105506" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105050")

# for alpha in 0.2
# do
#     for method in "krum" "foolsgold" "safe_lora"
#     do
#         export method;
#         export alpha;
#         sbatch -t 00-10:00:00 $jobFile1;
#     done
# done


# for alpha in 0.2
# do
#     for method in "foolsgold"
#     do
#         export method;
#         export alpha;
#         sbatch -t 00-10:00:00 $jobFile4;
#     done
# done
# for method in "fedavg" "safe_lora"
# for method in "fedavg" "safe_lora" 
# # for method in "safe_lora_mixture_analytical_different_old"
# do
#     for alpha in 0.2
#     do
#         for throw_n in 0
#         do
#             for seed in 2023 4096
#             do
#                 for analytical_alpha in 1.0
#                 do
#                     export method;
#                     export alpha;
#                     export seed;
#                     export throw_n;
#                     export analytical_alpha;
#                     sbatch -t 00-11:00:00 $jobFile4;
#                     # sbatch -t 00-11:00:00 $jobFile1;
#                 done
#             done
#         done
#     done
# done

# for method in "fedavg" "safe_lora" 
# # for method in "safe_lora_mixture_analytical_different_old"
# do
#     for alpha in 0.2
#     do
#         for throw_n in 0
#         do
#             for seed in 2023
#             do
#                 for analytical_alpha in 1.0
#                 do
#                     export method;
#                     export alpha;
#                     export seed;
#                     export throw_n;
#                     export analytical_alpha;
#                     # sbatch -t 00-11:00:00 $jobFile4;
#                     sbatch -t 00-11:00:00 $jobFile1;
#                 done
#             done
#         done
#     done
# done

# for method in "safe_lora" 
# # for method in "safe_lora_mixture_analytical_different_old"
# do
#     for alpha in 0.2
#     do
#         for throw_n in 0
#         do
#             for seed in 2023
#             do
#                 for analytical_alpha in 1.0
#                 do
#                     export method;
#                     export alpha;
#                     export seed;
#                     export throw_n;
#                     export analytical_alpha;
#                     # sbatch -t 00-11:00:00 $jobFile4;
#                     sbatch -t 00-11:00:00 $jobFile1;
#                 done
#             done
#         done
#     done
# done




# for method in "safe_lora_mixture_safety_subspace"
# # for method in "safe_lora_mixture_analytical_different_old"
# do
#     for alpha in 0.2
#     do
#         for throw_n in 0
#         do
#             for seed in 2023
#             do
#                 for analytical_alpha in 1.0
#                 do
#                     export method;
#                     export alpha;
#                     export seed;
#                     export throw_n;
#                     export analytical_alpha;
#                     sbatch -t 00-15:00:00 $jobFile4;
#                 done
#             done
#         done
#     done
# done
# for method in "debug_thrown"
# do
#     for alpha in 0.2
#     do
#         for throw_n in 4
#         do
#             for seed in 4096
#             do
#                 for analytical_alpha in 1.0
#                 do
#                     export method;
#                     export alpha;
#                     export seed;
#                     export throw_n;
#                     export analytical_alpha;
#                     sbatch -t 00-10:00:00 $jobFile4;
#                 done
#             done
#         done
#     done
# done

# for alpha in 0.2
# do
#     for analytical_alpha in 1.0
#     do
#         for method in  "safe_lora_mixture" 
#         do
#             export method;
#             export alpha;
#             export analytical_alpha;
#             sbatch -t 00-10:00:00 $jobFile4;
#         done
#     done
# done




# for method in "fedgraph"
# dohtt
#     for dataset in "rajpurkar/squad_v2" 
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-18:00:00 $jobFile;
#         done
#     done
# done

# for method in "dnc" "krum" "fedgraph" "foolsgold"
# do
#     for dataset in "qiaojin/PubMedQA rajpurkar/squad_v2"
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-12:00:00 $jobFile2;
#         done
#     done

#     for dataset in "metamathqa rajpurkar/squad_v2"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-12:00:00 $jobFile2;
#         done
#     done

    # for dataset in "qiaojin/PubMedQA"  "triviaqa"
    # do
    #     for steps in 5
    #     do
    #         export method;
    #         export dataset;
    #         export steps;
    #         sbatch -t 00-7:00:00 $jobFile;
    #     done
    # done

    #     for dataset in "qiaojin/PubMedQA triviaqa"
    # do
    #     for steps in 5
    #     do
    #         export method;
    #         export dataset;
    #         export steps;
    #         sbatch -t 00-7:00:00 $jobFile2;
    #     done
    # done

#     for dataset in "metamathqa"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-12:00:00 $jobFile;
#         done
#     done
# done




# for method in "fedavg" "safe_lora" "dnc" "fedgraph" "foolsgold"
# do
#     for dataset in "qiaojin/PubMedQA" 
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-7:00:00 $jobFile;
#         done
#     done

#     for dataset in "rajpurkar/squad_v2"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-12:00:00 $jobFile;
#         done
#     done

#     for dataset in "metamathqa"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-12:00:00 $jobFile;
#         done
#     done

#     for dataset in "triviaqa"
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-7:00:00 $jobFile;
#         done
#     done

#     for dataset in "qiaojin/PubMedQA triviaqa" 
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-7:00:00 $jobFile2;
#         done
#     done

#     for dataset in "qiaojin/PubMedQA rajpurkar/squad_v2"
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-7:00:00 $jobFile2;
#         done
#     done
# done



# for method in "fedgraph"
# do
#     for dataset in "rajpurkar/squad_v2"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-10:00:00 $jobFile;
#         done
#     done
# done

# for method in "eval_filter"
# do
#     for dataset in "qiaojin/PubMedQA" "triviaqa"
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-15:00:00 $jobFile;
#         done
#     done

#     for dataset in "qiaojin/PubMedQA triviaqa" 
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-15:00:00 $jobFile2;
#         done
#     done

#     for dataset in "rajpurkar/squad_v2" "metamathqa"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-20:00:00 $jobFile;
#         done
#     done
# done


# for method in "krum" 
# do
#     for dataset in "qiaojin/PubMedQA rajpurkar/squad_v2"
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-10:00:00 $jobFile2;
#         done
#     done

#     for dataset in "metamathqa rajpurkar/squad_v2"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-15:00:00 $jobFile2;
#         done
#     done
# done

# for method in "krum"  "dnc"
# do
#     for dataset in "qiaojin/PubMedQA" "triviaqa" 
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-10:00:00 $jobFile1;
#         done
#     done

#     for dataset in "metamathqa" "rajpurkar/squad_v2"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-15:00:00 $jobFile1;
#         done
#     done
# done


# for method in "dnc" "krum" "safe_lora"
# do
#     for dataset in "metamathqa" "rajpurkar/squad_v2"
#     do
#         for steps in 10
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-12:00:00 $jobFile3;
#         done
#     done

#     for dataset in "qiaojin/PubMedQA" "triviaqa"
#     do
#         for steps in 5
#         do
#             export method;
#             export dataset;
#             export steps;
#             sbatch -t 00-9:00:00 $jobFile3;
#         done
#     done
# done




