jobFile="training_scripts/run_defense.sh"
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/medQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251203103114" "/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105506" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105050")
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190205" "/shared/rc/llm-degredation/fedllm/barebones/squad_v24_PubMedQA4_BeaverTails4_500_fedavg_c12s12_i10_b16a1_l512_r32a64_20251208190322")
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127115342" "/shared/rc/llm-degredation/fedllm/barebones/triviaqa7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127120857" "/shared/rc/llm-degredation/fedllm/barebones/metamathqa7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260127122412" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA3_triviaqa3_BeaverTails3_500_fedavg_c9s10_i5_b16a1_l512_r32a64_20260127122443")
existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127115342" "/shared/rc/llm-degredation/fedllm/barebones/triviaqa7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127120857" "/shared/rc/llm-degredation/fedllm/barebones/metamathqa7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260127122412" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA3_triviaqa3_BeaverTails3_500_fedavg_c9s10_i5_b16a1_l512_r32a64_20260127122443")


for existing_lora in "${existing_lors[@]}"
do
    export existing_lora;
    export thrs;
    export SAFE_LORA_ORIGINAL=1;
    sbatch  -t 00-04:10:00 $jobFile;
    # bash $jobFile;
done


# for thrs in 0.15
# do
#     for existing_lora in "${existing_lors[@]}"
#     do
#         export existing_lora;
#         export thrs;
#         export SAFE_LORA_ORIGINAL=0;
#         sbatch -t 00-3:00:00 $jobFile;
#         # bash $jobFile;
#     done
# done

# ids=("419" "420" "421" "422")

# for safedelta_thrs in 0.05 0.1 0.15 0.20 0.40 0.60
# do
#     for id in "${ids[@]}"
#     do
#         export id;
#         export safedelta_thrs;
        
#         # bash $jobFile;
#     done
# done
