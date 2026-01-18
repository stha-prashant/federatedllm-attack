jobFile="training_scripts/run_defense.sh"
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/medQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251203103114" "/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105506" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105050")
existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190205" "/shared/rc/llm-degredation/fedllm/barebones/squad_v24_PubMedQA4_BeaverTails4_500_fedavg_c12s12_i10_b16a1_l512_r32a64_20251208190322")
for thrs in 0.15 0.2 0.25 0.30
do
    for existing_lora in "${existing_lors[@]}"
    do
        export existing_lora;
        export thrs;
        sbatch $jobFile;
        # bash $jobFile;
    done
done
