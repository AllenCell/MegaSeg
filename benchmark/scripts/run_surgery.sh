#!/usr/bin/env bash
# Launch bottleneck surgery feasibility experiments (50 epochs each).
#
# Runs strategies SEQUENTIALLY on a single GPU to minimize NFS contention.
# Default GPU: 0. Override with SURGERY_GPU=1 env var.
#
#   Strategy A - reinit bottleneck only
#   Strategy B - reinit bottleneck + dec0 (cascade)
#   Strategy C - reinit all deep layers
#   Strategy D - freeze shallow + reinit deep
#
# Usage:
#   bash scripts/run_surgery.sh              # Run all 4 sequentially
#   bash scripts/run_surgery.sh a            # Run single strategy
#   bash scripts/run_surgery.sh a b          # Run specific strategies
#
# Logs go to /tmp/surgery_{a,b,c,d}.log (local disk for reliability).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$BENCHMARK_DIR/configs"
PYTHON="$BENCHMARK_DIR/.venv/bin/python"
GPU="${SURGERY_GPU:-0}"

# Verify Python exists
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

# Strategy -> experiment config mapping
declare -A STRATEGY_CONFIGS
STRATEGY_CONFIGS[a]="train/surgery/lmnb1_surgery_a_50ep"
STRATEGY_CONFIGS[b]="train/surgery/lmnb1_surgery_b_50ep"
STRATEGY_CONFIGS[c]="train/surgery/lmnb1_surgery_c_50ep"
STRATEGY_CONFIGS[d]="train/surgery/lmnb1_surgery_d_50ep"

run_strategy() {
    local strategy="$1"
    local experiment="${STRATEGY_CONFIGS[$strategy]}"
    local logfile="/tmp/surgery_${strategy}.log"

    echo ""
    echo "========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting strategy $strategy on GPU $GPU"
    echo "  Experiment: $experiment"
    echo "  Log: $logfile"
    echo "========================================="

    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="$GPU" \
    CYTODL_CONFIG_PATH="$CONFIG_DIR" \
    PROJECT_ROOT="$BENCHMARK_DIR" \
    "$PYTHON" -m cyto_dl.train \
        experiment="$experiment" \
        trainer.devices=1 \
        trainer.strategy=auto \
        trainer.sync_batchnorm=false \
        ~callbacks.rich_progress_bar \
        > "$logfile" 2>&1

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Strategy $strategy COMPLETED (exit 0)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Strategy $strategy FAILED (exit $exit_code)"
        echo "  Check log: $logfile"
    fi
    return $exit_code
}

# Determine which strategies to run
if [[ $# -eq 0 ]]; then
    STRATEGIES=(a b c d)
else
    STRATEGIES=("$@")
fi

echo "========================================="
echo "Bottleneck Surgery - Sequential Runs"
echo "========================================="
echo "Strategies: ${STRATEGIES[*]}"
echo "GPU: $GPU"
echo "Epochs: 50"
echo "Batch size: 16, Workers: 4"
echo "Config dir: $CONFIG_DIR"
echo "Python: $PYTHON"

FAILED=()
for s in "${STRATEGIES[@]}"; do
    if [[ -z "${STRATEGY_CONFIGS[$s]+x}" ]]; then
        echo "ERROR: Unknown strategy '$s'. Valid: a b c d"
        exit 1
    fi
    if ! run_strategy "$s"; then
        FAILED+=("$s")
        echo "WARNING: Strategy $s failed, continuing with next..."
    fi
done

echo ""
echo "========================================="
echo "All strategies finished."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED: ${FAILED[*]}"
else
    echo "All succeeded."
fi
echo "========================================="
