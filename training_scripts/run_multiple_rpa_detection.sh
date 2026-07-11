#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/sbatch_test2_rpa_detection.sh"

# GPUS=(3 4 5)
GPUS=(0 1 2 4 5 6)

MIX_PROPS=(0.9 0.8 0.5)

# 
DATASETS=("expguardtrain" "PKU-Alignment/BeaverTails")
LABELS=("expguardtrain" "beavertails")

# DATASETS=("PKU-Alignment/BeaverTails")
# LABELS=("beavertails")

declare -A GPU_PID
declare -A GPU_DESC

launch_job() {
    local gpu="$1"
    local mix_prop="$2"
    local mal_dataset="$3"
    local mal_label="$4"

    MIX_PROP="$mix_prop" \
    MAL_DATASET="$mal_dataset" \
    MAL_LABEL="$mal_label" \
    GPU_ID="$gpu" \
    bash "$WORKER" &

    local pid=$!
    GPU_PID["$gpu"]=$pid
    GPU_DESC["$gpu"]="${mal_label} mix=${mix_prop}"

    echo "Started PID ${pid} on GPU ${gpu}: ${mal_label}, mix=${mix_prop}"
}

get_free_gpu() {
    while true; do
        for gpu in "${GPUS[@]}"; do
            local pid="${GPU_PID[$gpu]:-}"

            if [[ -z "${pid}" ]]; then
                echo "$gpu"
                return
            fi

            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || true
                echo "Finished on GPU ${gpu}: ${GPU_DESC[$gpu]}"
                unset GPU_PID["$gpu"]
                unset GPU_DESC["$gpu"]
                echo "$gpu"
                return
            fi
        done
        sleep 5
    done
}

# Submit all 6 jobs, at most 5 at once
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    label="${LABELS[$i]}"

    for mix in "${MIX_PROPS[@]}"; do
        free_gpu="$(get_free_gpu)"
        launch_job "$free_gpu" "$mix" "$dataset" "$label"
        sleep 2
    done
done

# Wait for remaining jobs
for gpu in "${GPUS[@]}"; do
    pid="${GPU_PID[$gpu]:-}"
    if [[ -n "${pid}" ]]; then
        wait "$pid" || true
        echo "Finished on GPU ${gpu}: ${GPU_DESC[$gpu]}"
    fi
done

echo "All runs completed."

# #!/bin/bash
# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WORKER="${SCRIPT_DIR}/sbatch_test2_rpa_detection.sh"

# # GPU_LIST=(0 5 6)
# # FED_ALGS=("foolsgold" "dnc" "krum")

# GPU_LIST=(1)
# FED_ALGS=("safelorav2data")

# MIX_PROP=0.9
# MAL_DATASET="PKU-Alignment/BeaverTails"
# MAL_LABEL="beavertails"

# declare -A GPU_PID
# declare -A GPU_DESC

# for i in "${!FED_ALGS[@]}"; do
#     gpu="${GPU_LIST[$i]}"
#     alg="${FED_ALGS[$i]}"

#     MIX_PROP="$MIX_PROP" \
#     MAL_DATASET="$MAL_DATASET" \
#     MAL_LABEL="$MAL_LABEL" \
#     GPU_ID="$gpu" \
#     FED_ALG="$alg" \
#     bash "$WORKER" &

#     pid=$!
#     GPU_PID["$gpu"]=$pid
#     GPU_DESC["$gpu"]="$alg"

#     echo "Started PID ${pid} on GPU ${gpu}: ${alg}"
#     sleep 2
# done

# for gpu in "${GPU_LIST[@]}"; do
#     pid="${GPU_PID[$gpu]:-}"
#     if [[ -n "$pid" ]]; then
#         wait "$pid" || true
#         echo "Finished on GPU ${gpu}: ${GPU_DESC[$gpu]}"
#     fi
# done

# echo "All runs completed."