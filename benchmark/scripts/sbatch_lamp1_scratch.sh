#!/bin/bash
#SBATCH --job-name=lamp1_sc
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics_gpu
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100sxm:1

# LAMP1 scratch training launcher.
#
# Usage:
#   sbatch sbatch_lamp1_scratch.sh <experiment_suffix>
#
# Examples:
#   sbatch sbatch_lamp1_scratch.sh data100_converged
#   sbatch sbatch_lamp1_scratch.sh data10_converged

set -euo pipefail

EXPERIMENT="${1:-data100_converged}"
CONFIG_NAME="lamp1_${EXPERIMENT}"

BENCHMARK_DIR="/allen/aics/rep_learn/MegaSeg/benchmark"
UV="/home/ashwin.samudre/.local/bin/uv"

echo "===== LAMP1 Scratch Training ====="
echo "Experiment: ${CONFIG_NAME}"
echo "Host: $(hostname)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-all}"
echo "Date: $(date)"
echo "=================================="

cd "$BENCHMARK_DIR"

CYTODL_CONFIG_PATH="${BENCHMARK_DIR}/configs" \
PROJECT_ROOT="${BENCHMARK_DIR}" \
    "$UV" run python -m cyto_dl.train \
    experiment="train/scratch/${CONFIG_NAME}" \
    trainer.devices=[0] \
    ~callbacks.rich_progress_bar

echo "Done: $(date)"
