jobFile="training_scripts/run_defense_single.sh"
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/medQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251203103114" "/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105506" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251201105050")
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190205" "/shared/rc/llm-degredation/fedllm/barebones/squad_v24_PubMedQA4_BeaverTails4_500_fedavg_c12s12_i10_b16a1_l512_r32a64_20251208190322")
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/squad_v27_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20251208190114" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127115342" "/shared/rc/llm-degredation/fedllm/barebones/triviaqa7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127120857" "/shared/rc/llm-degredation/fedllm/barebones/metamathqa7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260127122412" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA3_triviaqa3_BeaverTails3_500_fedavg_c9s10_i5_b16a1_l512_r32a64_20260127122443")
# existing_lors=("/shared/rc/llm-degredation/fedllm/barebones/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127115342" "/shared/rc/llm-degredation/fedllm/barebones/triviaqa7_BeaverTails3_500_fedavg_c10s10_i5_b16a1_l512_r32a64_20260127120857" "/shared/rc/llm-degredation/fedllm/barebones/metamathqa7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260127122412" "/shared/rc/llm-degredation/fedllm/barebones/PubMedQA3_triviaqa3_BeaverTails3_500_fedavg_c9s10_i5_b16a1_l512_r32a64_20260127122443")
# existing_lors=('/scratch/ps9044/newsetting/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260212154647')

# existing_lors=("/scratch/ps9044/newsetting/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r8a16_20260223203833" "/scratch/ps9044/newsetting/PubMedQA7_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r16a32_20260223201706")

# existing_lors=('/scratch/ps9044/newsetting/PubMedQA_medQA_emrqa_cord19_BeaverTails3_1.00_1.00_1.00_1.00_0.50_0.50_0.50_200_fedavg_c7s7_i10_b16a1_l512_r32a64_20260301190956' '/scratch/ps9044/newsetting/PubMedQA7_BeaverTails3_1.00_1.00_1.00_1.00_0.50_0.50_0.50_200_fedavg_c7s10_i10_b16a1_l512_r32a64_20260218140241' '/scratch/ps9044/newsetting/PubMedQA_medQA_emrqa_cord19_BeaverTails3_1.00_1.00_1.00_1.00_0.50_0.50_0.50_200_fedavg_c7s7_i10_b16a1_l512_r32a64_20260309191112')

# existing_lors=('/scratch/ps9044/newsetting/PubMedQA_medQA_emrqa_cord19_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260426141123' '/scratch/ps9044/newsetting/PubMedQA_medQA_emrqa_cord19_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260426152956')
# existing_lors=('/scratch/ps9044/newsetting/fedavg_beavertails_mix0.9_seed2023_ndata500/PubMedQA_medQA_emrqa_cord19_BeaverTails3_1.00_1.00_1.00_1.00_1.00_1.00_1.00_0.90_0.90_0.90_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260421222324')
# existing_lors=('/scratch/ps9044/newsetting/PubMedQA_medQA_emrqa_cord19_BeaverTails3_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260429103933')
existing_lors=('/scratch/ps9044/newsetting/BeaverTailsSafe5_BeaverTails5_500_fedavg_c10s10_i10_b16a1_l512_r32a64_20260517123537')
# existing_lors=('/scratch/ps9044/newsetting/PubMedQA_medQA_emrqa_cord19_BeaverTails3_1.00_1.00_1.00_1.00_0.50_0.50_0.50_200_fedavg_c7s7_i10_b16a1_l512_r32a64_20260309191112')
jobFile="training_scripts/run_defense.sh"

for existing_lora in "${existing_lors[@]}"
do
    export existing_lora;
    export thrs;
    export SAFE_LORA_ORIGINAL=1;
    # sbatch  -t 00-01:10:00 $jobFile;
    bash $jobFile;
done

# for finetuning_dataset in "isa" # "MaliciousGen" "expguardtrain" "qiaojin/PubMedQA" "medQA" "emrqa" "cord19"
# do

#     export finetuning_dataset;
#     export thrs;
#     export SAFE_LORA_ORIGINAL=1;
#     # sbatch  -t 00-6:00:00 $jobFile;
#     bash $jobFile;
# done


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
