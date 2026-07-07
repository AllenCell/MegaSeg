#!/bin/bash
#SBATCH --job-name=sota_single
#SBATCH --partition=aics_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sota_single_%j.log

# Run all SOTA methods on a single sample for qualitative comparison.
#
# Usage:
#   sbatch scripts/sbatch_sota_single_sample.sh ACTN1 ACTN1_11347
#   sbatch scripts/sbatch_sota_single_sample.sh TOMM20 TOMM20_12345
#
# Or to run multiple samples, submit one job per sample:
#   for sid in ACTN1_11347 TOMM20_12210 TJP1_10917; do
#     struct=${sid%%_*}
#     sbatch scripts/sbatch_sota_single_sample.sh $struct $sid
#   done

STRUCTURE=${1:?"Usage: sbatch $0 STRUCTURE SAMPLE_ID"}
SAMPLE_ID=${2:?"Usage: sbatch $0 STRUCTURE SAMPLE_ID"}

BENCHMARK_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="/allen/aics/rep_learn/megaseg/benchmark/.venv/bin/python"

echo "========================================="
echo "SOTA Single Sample: ${STRUCTURE} / ${SAMPLE_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "========================================="

cd "${BENCHMARK_DIR}" || exit 1

${PYTHON} scripts/run_sota_single_sample.py \
    --structure "${STRUCTURE}" \
    --sample-id "${SAMPLE_ID}" \
    --gpu 0

echo "End: $(date)"
