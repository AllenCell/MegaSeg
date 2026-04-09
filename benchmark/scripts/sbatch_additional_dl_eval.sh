#!/bin/bash
#SBATCH --job-name=addstruct-dl
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics_gpu
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100"
#SBATCH --time=12:00:00

# Run MegaSeg + 14 individual model predictions on all additional structure
# channel-volumes (41 volumes x 15 models = 615 predictions).
#
# Prerequisites: run extract_additional_channels.py first to create the
# extracted single-channel TIFFs and CSVs.
#
# Usage:
#   sbatch scripts/sbatch_additional_dl_eval.sh

set -euo pipefail

PROJECT_ROOT="/allen/aics/rep_learn/MegaSeg-worktrees/additional-structures/benchmark"
CONFIGS_PATH="${PROJECT_ROOT}/configs"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT
export CYTODL_CONFIG_PATH="${CONFIGS_PATH}"

echo "=== Additional Structures DL Eval ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo ""

# Verify CSVs exist
CSV="${PROJECT_ROOT}/data/additional_structures_eval_csvs/all.csv"
if [[ ! -f "$CSV" ]]; then
    echo "ERROR: CSV not found: $CSV"
    echo "Run extract_additional_channels.py first!"
    exit 1
fi
echo "CSV: ${CSV} ($(wc -l < "$CSV") lines)"
echo ""

# ---------------------------------------------------------------------------
# MegaSeg DynUNet
# ---------------------------------------------------------------------------
echo "=== [1/15] MegaSeg DynUNet ==="
$PYTHON -m cyto_dl.eval experiment=eval/uint8/megaseg/dynunet/additional_structures/all
echo "=== MegaSeg DynUNet DONE ==="
echo ""

# ---------------------------------------------------------------------------
# Individual models (14)
# ---------------------------------------------------------------------------
INDIVIDUAL_MODELS=(
    ACTN1 ATP2A2 CTNNB1 DSP GJA1 MYH10 NPM1
    NUP153 RAB5A SEC61B SON ST6GAL1 TJP1 TOMM20
)

COUNT=2
TOTAL=15
for model in "${INDIVIDUAL_MODELS[@]}"; do
    echo "=== [${COUNT}/${TOTAL}] Individual: ${model} ==="
    $PYTHON -m cyto_dl.eval experiment=eval/uint8/individual/additional_structures/${model}
    echo "=== Individual ${model} DONE ==="
    echo ""
    COUNT=$((COUNT + 1))
done

echo "=== ALL DL EVAL COMPLETE ==="
echo "End: $(date)"
