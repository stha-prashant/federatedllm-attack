#!/bin/bash

jobFile="training_scripts/sbatch_safefedllm_grid.sh"

for num_malicious_clients in 3 
do
    for malicious_mixture_proportion in 0.5
    do
        for model in "qwen3" "llama3_0" #"llama2" # "llama3" "gemma" "qwen"
        do
            export MIXTURE_DIRICHLET_ALPHA=0.2
            export MODEL_KEY=$model
            export NUM_MALICIOUS_CLIENTS=$num_malicious_clients
            export MALICIOUS_MIXTURE_PROPORTION=$malicious_mixture_proportion
            sbatch -t 00-7:00:00 $jobFile
        done
    done
done
