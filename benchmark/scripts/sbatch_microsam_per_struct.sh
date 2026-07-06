#!/bin/bash
#SBATCH --job-name=usam
#SBATCH --partition=aics_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/microsam_%x_%j.log

# Run MicroSAM on one structure with a specific model type.
# Usage: sbatch --job-name=usam_b_ACTN1 scripts/sbatch_microsam_per_struct.sh vit_b_lm ACTN1 2

MODEL_TYPE=${1:?"Usage: sbatch $0 MODEL_TYPE STRUCTURE [MAX_SAMPLES]"}
STRUCTURE=${2:?"Usage: sbatch $0 MODEL_TYPE STRUCTURE [MAX_SAMPLES]"}
MAX_SAMPLES=${3:-""}

BENCHMARK_DIR="/allen/aics/rep_learn/megaseg/benchmark"
PYTHON="/allen/aics/users/ashwin.samudre/miniconda3/envs/microsam-bench/bin/python"

echo "MicroSAM ${MODEL_TYPE} | ${STRUCTURE} | max=${MAX_SAMPLES:-all} | $(hostname) | $(date)"

cd "${BENCHMARK_DIR}" || exit 1

EXTRA_ARGS=""
[ -n "${MAX_SAMPLES}" ] && EXTRA_ARGS="--max-samples ${MAX_SAMPLES}"

${PYTHON} scripts/run_microsam_lm_zeroshot.py \
    --gpu 0 \
    --model-type "${MODEL_TYPE}" \
    --structures "${STRUCTURE}" \
    ${EXTRA_ARGS}

echo "End: $(date)"
