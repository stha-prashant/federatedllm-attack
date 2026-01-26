jobFile="training_scripts/sbatch1.sh"
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/medQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251203103114" "/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105506" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105050")
for method in "dnc" "foolsgold" "krum" "median" "trimmedmean"
do
    export method;
    sbatch $jobFile;
done
