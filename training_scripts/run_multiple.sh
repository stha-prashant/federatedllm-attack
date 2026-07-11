jobFile="training_scripts/sbatch_test2_grid.sh"

# for method in "fedavg" "safelorav2data" "lasa" "flame" "krum" "dnc" "foolsgold" "median"
for method in "fedavg" "safelorav2data" "lasa" "krum" "foolsgold"
do
    for model in "llama2" "llama3" "gemma" "qwen"
    do
        export FED_ALG=$method
        export MODEL_KEY=$model
        sbatch -t 00-7:00:00 $jobFile 
    done
done
