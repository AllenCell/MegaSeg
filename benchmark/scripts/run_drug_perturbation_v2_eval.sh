#!/usr/bin/env bash
# Run drug perturbation v2 DL eval jobs (individual + megaseg) sequentially.
#
# Usage:
#   bash scripts/run_drug_perturbation_v2_eval.sh [--dry-run] [--gpu GPU_ID]
#
# Defaults to GPU 0. Runs all 10 jobs sequentially in a single screen session.
# Each structure eval takes ~5-15 min on an A100 (~2 hours total).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIGS_PATH="${PROJECT_ROOT}/configs"
LOG_DIR="${PROJECT_ROOT}/scripts/job_logs"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
SESSION_NAME="eval_dp2_all"

DRY_RUN=0
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

STRUCTURES=(FBL NUP153 ST6GAL1 TOMM20 TUBA1B)

# Build the list of config paths to run (individual first, then megaseg)
ALL_CONFIGS=()
ALL_LABELS=()

for s in "${STRUCTURES[@]}"; do
    out_dir="${PROJECT_ROOT}/outputs/eval/uint8/individual/drug_perturbation_v2/${s}/seg"
    if [[ -d "$out_dir" ]] && [[ $(ls "$out_dir"/*.tif 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "SKIP (output exists): individual/${s}"
    else
        ALL_CONFIGS+=("eval/uint8/individual/drug_perturbation_v2/${s}")
        ALL_LABELS+=("individual/${s}")
    fi
done

for s in "${STRUCTURES[@]}"; do
    out_dir="${PROJECT_ROOT}/outputs/eval/uint8/megaseg/dynunet/drug_perturbation_v2/${s}/seg"
    if [[ -d "$out_dir" ]] && [[ $(ls "$out_dir"/*.tif 2>/dev/null | wc -l) -gt 0 ]]; then
        echo "SKIP (output exists): megaseg/${s}"
    else
        ALL_CONFIGS+=("eval/uint8/megaseg/dynunet/drug_perturbation_v2/${s}")
        ALL_LABELS+=("megaseg/${s}")
    fi
done

if [[ ${#ALL_CONFIGS[@]} -eq 0 ]]; then
    echo "All jobs already complete. Nothing to do."
    exit 0
fi

echo ""
echo "Drug Perturbation v2 - DL Eval Jobs"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "GPU: ${GPU_ID}"
echo "Python: ${PYTHON}"
echo "Jobs to run: ${#ALL_CONFIGS[@]}"
echo ""

# Build a combined command that runs all jobs sequentially
COMBINED_CMD="cd ${PROJECT_ROOT}"
for i in "${!ALL_CONFIGS[@]}"; do
    config="${ALL_CONFIGS[$i]}"
    label="${ALL_LABELS[$i]}"
    COMBINED_CMD+=" && echo '=== [$(( i + 1 ))/${#ALL_CONFIGS[@]}] ${label} ===' && CUDA_VISIBLE_DEVICES=${GPU_ID} PROJECT_ROOT=${PROJECT_ROOT} CYTODL_CONFIG_PATH=${CONFIGS_PATH} ${PYTHON} -m cyto_dl.eval experiment=${config}"
done
COMBINED_CMD+=" && echo '=== ALL DONE ==='"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would launch screen session '${SESSION_NAME}' with:"
    for i in "${!ALL_CONFIGS[@]}"; do
        echo "  [$(( i + 1 ))/${#ALL_CONFIGS[@]}] ${ALL_LABELS[$i]}: experiment=${ALL_CONFIGS[$i]}"
    done
    echo ""
    echo "Full command:"
    echo "  ${COMBINED_CMD}"
    exit 0
fi

LOG_FILE="${LOG_DIR}/${SESSION_NAME}.log"
echo "Launching screen session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
screen -dmS "${SESSION_NAME}" -L -Logfile "${LOG_FILE}" bash -c "${COMBINED_CMD}"

echo ""
echo "Jobs launched sequentially in screen session '${SESSION_NAME}'."
echo "Monitor with:"
echo "  screen -r ${SESSION_NAME}       # attach to session"
echo "  tail -f ${LOG_FILE}             # follow log"
echo "  screen -ls | grep eval_dp2      # check if still running"
