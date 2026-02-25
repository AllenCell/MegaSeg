#!/usr/bin/env bash
# Run LMNB1 fine-tuning experiments from the MegaSeg benchmark directory.
#
# Usage:
#   # Run a single experiment:
#   bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c3
#
#   # Run all Phase 1 channel tests (sequentially):
#   bash scripts/run_finetune.sh --phase1
#
#   # Run on a specific GPU:
#   GPU=2 bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$BENCHMARK_DIR/configs"

GPU="${GPU:-0}"

run_experiment() {
    local experiment="$1"
    local gpu="${2:-$GPU}"
    echo "========================================="
    echo "Running: $experiment (GPU $gpu)"
    echo "Config dir: $CONFIG_DIR"
    echo "========================================="
    CYTODL_CONFIG_PATH="$CONFIG_DIR" \
    PROJECT_ROOT="$BENCHMARK_DIR" \
        python -m cyto_dl.train \
        experiment="$experiment" \
        trainer.devices="[$gpu]"
}

if [[ "${1:-}" == "--phase1" ]]; then
    echo "Running Phase 1: Channel test (C=1, C=3, C=6)"
    for c in 1 3 6; do
        run_experiment "train/finetune/lmnb1_channel_test_c${c}"
    done
    echo "Phase 1 complete. Check outputs/ for results."
    exit 0
fi

if [[ -z "${1:-}" ]]; then
    echo "Usage:"
    echo "  $0 <experiment>          # Run single experiment"
    echo "  $0 --phase1              # Run all Phase 1 channel tests"
    echo ""
    echo "Examples:"
    echo "  $0 train/finetune/lmnb1_channel_test_c3"
    echo "  GPU=2 $0 train/finetune/lmnb1_channel_test_c3"
    exit 1
fi

run_experiment "$1"
