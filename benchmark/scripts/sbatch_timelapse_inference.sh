#!/usr/bin/env bash
#SBATCH --job-name=timelapse_inf
#SBATCH --partition=aics_gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/allen/aics/rep_learn/MegaSeg/benchmark/logs/timelapse_inference_%j.out
#SBATCH --error=/allen/aics/rep_learn/MegaSeg/benchmark/logs/timelapse_inference_%j.err

set -euo pipefail

echo "========================================="
echo "Timelapse Inference: 5 configs"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start time: $(date)"
echo "========================================="

# --- Paths ---
MAIN_WORKTREE="/allen/aics/rep_learn/MegaSeg-worktrees/main/benchmark"
TIMELAPSE_WORKTREE="/allen/aics/rep_learn/MegaSeg-worktrees/timelapse/benchmark"
BENCHMARK_VENV="/allen/aics/rep_learn/MegaSeg/benchmark/.venv/bin/python"
FALLBACK_PYTHON="/allen/aics/users/ashwin.samudre/suraj_cyto/cyto-dl/cytos/bin/python"

# Timelapse experiment configs only exist on the timelapse branch
# Always use timelapse worktree for CYTODL_CONFIG_PATH
TIMELAPSE_CONFIG_DIR="$TIMELAPSE_WORKTREE/configs"

# Use benchmark .venv; fall back to Suraj's Python if needed
if "$BENCHMARK_VENV" -c "import cyto_dl" 2>/dev/null; then
    PYTHON="$BENCHMARK_VENV"
    echo "Using benchmark .venv Python"
else
    PYTHON="$FALLBACK_PYTHON"
    echo "WARNING: benchmark .venv failed, falling back to Suraj's Python"
fi

EVAL_SCRIPT="cyto_dl.eval"

run_inference() {
    local project_root="$1"
    local experiment="$2"
    local label="$3"

    echo ""
    echo "--- $label ---"
    echo "  experiment=$experiment"
    echo "  CYTODL_CONFIG_PATH=$TIMELAPSE_CONFIG_DIR"
    echo "  PROJECT_ROOT=$project_root"
    echo "  Time: $(date)"

    # Pre-create output directory
    local out_dir="$project_root/outputs/$experiment"
    mkdir -p "$out_dir"

    CYTODL_CONFIG_PATH="$TIMELAPSE_CONFIG_DIR" \
    PROJECT_ROOT="$project_root" \
        "$PYTHON" -m "$EVAL_SCRIPT" \
        experiment="$experiment" \
        trainer.devices="[0]" \
        ++model.inference_args.sw_batch_size=8
}

# --- Phase 1: Original timelapse configs (main worktree) ---
echo ""
echo "========================================="
echo "Phase 1: Original timelapse (main worktree)"
echo "========================================="

run_inference "$MAIN_WORKTREE" \
    "eval/uint8/individual/timelapse/NPM1" \
    "Individual NPM1 (70 images)"

run_inference "$MAIN_WORKTREE" \
    "eval/uint8/individual/timelapse/TOMM20" \
    "Individual TOMM20 (5 images)"

run_inference "$MAIN_WORKTREE" \
    "eval/uint8/megaseg/dynunet/timelapse/NPM1" \
    "MegaSeg NPM1 (70 images)"

# --- Phase 2: Additions (timelapse worktree) ---
echo ""
echo "========================================="
echo "Phase 2: Caroline additions (timelapse worktree)"
echo "========================================="

run_inference "$TIMELAPSE_WORKTREE" \
    "eval/uint8/megaseg/dynunet/timelapse/Tubulin_MultiCh_2" \
    "MegaSeg Tubulin_MultiCh_2 (27 images)"

run_inference "$TIMELAPSE_WORKTREE" \
    "eval/uint8/megaseg/dynunet/timelapse/Cardio_MultiCh" \
    "MegaSeg Cardio_MultiCh (95 images)"

echo ""
echo "========================================="
echo "All timelapse inference complete!"
echo "End time: $(date)"
echo "========================================="
