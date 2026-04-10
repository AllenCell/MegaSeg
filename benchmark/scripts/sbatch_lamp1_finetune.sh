#!/bin/bash
#SBATCH --job-name=lamp1_ft
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics_gpu
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100sxm:1


# LAMP1 fine-tuning experiment launcher.
#
# Usage:
#   sbatch sbatch_lamp1_finetune.sh <experiment_suffix>
#
# Examples:
#   sbatch sbatch_lamp1_finetune.sh data100_10epochs
#   sbatch sbatch_lamp1_finetune.sh data10_converged
#
# Valid suffixes (data fraction x epochs):
#   data100_10epochs, data100_50epochs, data100_100epochs, data100_converged
#   data50_10epochs,  data50_50epochs,  data50_100epochs,  data50_converged
#   data20_10epochs,  data20_50epochs,  data20_100epochs,  data20_converged
#   data10_10epochs,  data10_50epochs,  data10_100epochs,  data10_converged

set -euo pipefail

EXPERIMENT="${1:-data100_10epochs}"
CONFIG_NAME="lamp1_${EXPERIMENT}"

BENCHMARK_DIR="/allen/aics/rep_learn/MegaSeg/benchmark"
UV="/home/ashwin.samudre/.local/bin/uv"

echo "===== LAMP1 Fine-tuning ====="
echo "Experiment: ${CONFIG_NAME}"
echo "Host: $(hostname)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-all}"
echo "Date: $(date)"
echo "=============================="

cd "$BENCHMARK_DIR"

CYTODL_CONFIG_PATH="${BENCHMARK_DIR}/configs" \
PROJECT_ROOT="${BENCHMARK_DIR}" \
    "$UV" run python -m cyto_dl.train \
    experiment="train/finetune/${CONFIG_NAME}" \
    trainer.devices=[0] \
    ~callbacks.rich_progress_bar

echo "Done: $(date)"
