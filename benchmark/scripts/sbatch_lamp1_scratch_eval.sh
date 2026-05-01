#!/bin/bash
#SBATCH --job-name=lamp1_eval
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics_gpu
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100sxm:1

# LAMP1 scratch prediction/eval launcher.
#
# Usage:
#   sbatch sbatch_lamp1_scratch_eval.sh <experiment_suffix>
#
# Examples:
#   sbatch sbatch_lamp1_scratch_eval.sh data100_converged
#   sbatch sbatch_lamp1_scratch_eval.sh data10_converged

set -euo pipefail

EXPERIMENT="${1:-data100_converged}"
CONFIG_NAME="lamp1_${EXPERIMENT}"

BENCHMARK_DIR="/allen/aics/rep_learn/MegaSeg/benchmark"
CONFIG_DIR="$BENCHMARK_DIR/configs"
UV="/home/ashwin.samudre/.local/bin/uv"

echo "===== LAMP1 Scratch Eval ====="
echo "Experiment: ${CONFIG_NAME}"
echo "Host: $(hostname)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-all}"
echo "Date: $(date)"
echo "=============================="

cd "$BENCHMARK_DIR"

mkdir -p "$BENCHMARK_DIR/outputs/eval/scratch/lamp1/${EXPERIMENT}"

CYTODL_CONFIG_PATH="$CONFIG_DIR" \
PROJECT_ROOT="$BENCHMARK_DIR" \
CUDA_VISIBLE_DEVICES=0 \
    "$UV" run --directory "$BENCHMARK_DIR" \
    python -m cyto_dl.eval \
    experiment="eval/scratch/${CONFIG_NAME}" \
    trainer.devices=[0]

echo "Done: $(date)"
