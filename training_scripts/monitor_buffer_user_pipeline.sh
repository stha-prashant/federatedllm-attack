#!/bin/bash
# Monitor training jobs, submit follow-up evals, aggregate table.
set -euo pipefail

REPO_ROOT="/home/ps9044/RPA/fedllm-attack"
cd "${REPO_ROOT}"

ROW1_JOB="${1:?row1 job id}"
BT_JOB="${2:?beavertails job id}"
EG_JOB="${3:?expguard job id}"
ROW5_JOB="${4:?row5 job id}"

OUTPUT_ROOT="/scratch/ps9044/aaai2026"
BUFFER_DIR="/shared/rc/llm-degredation/fedllm/barebones/MaliciousGen1__0_1000_fedavg_c1s1_i10_b16a1_l512_r32a64_20251201105743"
FULL50="${BUFFER_DIR}/full-50"
STATE_FILE="${REPO_ROOT}/training_scripts/.buffer_user_pipeline_state"

wait_job() {
  local jid="$1"
  echo "Waiting for job ${jid}..."
  while squeue -j "${jid}" -h 2>/dev/null | grep -q .; do
    sleep 60
  done
  local state
  state="$(sacct -j "${jid}" --format=State -P -n 2>/dev/null | head -1 | cut -d'|' -f1)"
  echo "Job ${jid} finished with state: ${state}"
  [[ "${state}" == COMPLETED ]]
}

latest_run_dir() {
  local pattern="$1"
  ls -dt "${OUTPUT_ROOT}"/${pattern} 2>/dev/null | head -1
}

submit_dual_eval() {
  local run_dir="$1"
  sbatch -t 08:00:00 --job-name=dual_eval \
    "${REPO_ROOT}/training_scripts/eval_buffer_user_dual_merge.sh" "${run_dir}"
}

submit_row5_eval() {
  local run_dir="$1"
  sbatch -t 08:00:00 --job-name=row5_eval \
    "${REPO_ROOT}/training_scripts/sbatch_row5_eval.sh" "${run_dir}"
}

echo "=== Pipeline monitor started ==="
echo "Jobs: row1=${ROW1_JOB} bt=${BT_JOB} eg=${EG_JOB} row5=${ROW5_JOB}"

# Row 1 can run in parallel; wait for it first (fastest path to partial results)
wait_job "${ROW1_JOB}" || echo "WARN: row1 job ${ROW1_JOB} did not complete successfully"

BT_RUN=""
EG_RUN=""
ROW5_RUN=""
DUAL_BT_JOB=""
DUAL_EG_JOB=""
ROW5_EVAL_JOB=""

if wait_job "${BT_JOB}"; then
  BT_RUN="$(latest_run_dir 'full-50_*BeaverTails*')"
  echo "BeaverTails run: ${BT_RUN}"
  if [[ -n "${BT_RUN}" && -d "${BT_RUN}/checkpoint-30" ]]; then
    DUAL_BT_JOB="$(submit_dual_eval "${BT_RUN}" | awk '{print $NF}')"
    echo "Submitted dual eval BT: ${DUAL_BT_JOB}"
  fi
else
  echo "WARN: BeaverTails training failed"
fi

if wait_job "${EG_JOB}"; then
  EG_RUN="$(latest_run_dir 'full-50_*expguardtrain*')"
  echo "Expguard run: ${EG_RUN}"
  if [[ -n "${EG_RUN}" && -d "${EG_RUN}/checkpoint-30" ]]; then
    DUAL_EG_JOB="$(submit_dual_eval "${EG_RUN}" | awk '{print $NF}')"
    echo "Submitted dual eval EG: ${DUAL_EG_JOB}"
  fi
else
  echo "WARN: Expguard training failed"
fi

if wait_job "${ROW5_JOB}"; then
  ROW5_RUN="$(latest_run_dir 'Llama-2-7b-chat-hf_*BeaverTails*')"
  echo "Row5 run: ${ROW5_RUN}"
  if [[ -n "${ROW5_RUN}" && -d "${ROW5_RUN}/checkpoint-30" ]]; then
    ROW5_EVAL_JOB="$(submit_row5_eval "${ROW5_RUN}" | awk '{print $NF}')"
    echo "Submitted row5 eval: ${ROW5_EVAL_JOB}"
  fi
else
  echo "WARN: Row5 training failed"
fi

for j in ${DUAL_BT_JOB} ${DUAL_EG_JOB} ${ROW5_EVAL_JOB}; do
  [[ -n "${j}" ]] && wait_job "${j}" || true
done

{
  echo "ROW1_FULL50=${FULL50}"
  echo "BT_RUN=${BT_RUN}"
  echo "EG_RUN=${EG_RUN}"
  echo "ROW5_RUN=${ROW5_RUN}"
} > "${STATE_FILE}"

source ~/.bashrc 2>/dev/null || true
conda activate testvllm 2>/dev/null || true

python "${REPO_ROOT}/training_scripts/aggregate_buffer_user_table.py" \
  --row1-full50 "${FULL50}" \
  ${BT_RUN:+--row2-bt-buffer "${BT_RUN}/full-30_on_buffer"} \
  ${BT_RUN:+--row3-bt-base "${BT_RUN}/full-30_on_base"} \
  ${EG_RUN:+--row2-eg-buffer "${EG_RUN}/full-30_on_buffer"} \
  ${EG_RUN:+--row3-eg-base "${EG_RUN}/full-30_on_base"} \
  ${ROW5_RUN:+--row5-run "${ROW5_RUN}"} \
  | tee "${REPO_ROOT}/training_scripts/buffer_user_table_results.txt"

echo "=== Pipeline complete ==="
