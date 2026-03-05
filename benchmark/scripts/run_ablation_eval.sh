#!/usr/bin/env bash
# Run ablation evaluation: test each ablation model on all 14 structures.
#
# Usage:
#   # Evaluate a single ablation (e.g. 4-struct model on all structures):
#   bash scripts/run_ablation_eval.sh 4
#
#   # Evaluate all ablations:
#   bash scripts/run_ablation_eval.sh --all
#
#   # Run on a specific GPU:
#   GPU=2 bash scripts/run_ablation_eval.sh 7
#
# Prerequisites:
#   - Training must be complete and checkpoint paths updated in eval configs
#   - Run: python scripts/generate_ablation_eval_configs.py (if not already done)
#   - Update CHECKPOINT_PATH_PLACEHOLDER in configs with actual checkpoint paths

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$BENCHMARK_DIR/configs"

GPU="${GPU:-0}"
UV="${UV:-/home/ashwin.samudre/.local/bin/uv}"

ABLATION_NAMES=("4_structs" "7_structs" "14_structs")

STRUCTURES=(
    AAVS1 ACTN1 ATP2A2 CTNNB1 DSP GJA1
    MYH10 NUP153 RAB5A SEC61B SON ST6GAL1
    TJP1 TOMM20
)

run_eval() {
    local experiment="$1"
    local gpu="${2:-$GPU}"
    echo "--- Evaluating: $experiment (GPU $gpu) ---"

    CYTODL_CONFIG_PATH="$CONFIG_DIR" \
    PROJECT_ROOT="$BENCHMARK_DIR" \
        "$UV" run --directory "$BENCHMARK_DIR" \
        python -m cyto_dl.eval \
        experiment="$experiment" \
        trainer.devices="[$gpu]"
}

eval_ablation() {
    local ablation="$1"
    echo "========================================="
    echo "Evaluating ablation: $ablation"
    echo "Time: $(date)"
    echo "========================================="

    # Check for placeholder
    if grep -q "CHECKPOINT_PATH_PLACEHOLDER" "$CONFIG_DIR/experiment/eval/ablation/${ablation}/"*.yaml 2>/dev/null; then
        echo "ERROR: Checkpoint paths not yet set for $ablation."
        echo "Update CHECKPOINT_PATH_PLACEHOLDER in configs/experiment/eval/ablation/${ablation}/*.yaml"
        return 1
    fi

    for struct in "${STRUCTURES[@]}"; do
        run_eval "eval/ablation/${ablation}/${struct}"
    done

    echo "Ablation $ablation evaluation complete at $(date)"
}

print_usage() {
    echo "Usage:"
    echo "  $0 <4|7|14>    # Evaluate single ablation model"
    echo "  $0 --all        # Evaluate all ablation models"
    echo ""
    echo "Environment variables:"
    echo "  GPU=<id>   GPU device index (default: 0)"
}

if [[ -z "${1:-}" ]]; then
    print_usage
    exit 1
fi

if [[ "$1" == "--all" ]]; then
    echo "Evaluating all ablation models"
    for ablation in "${ABLATION_NAMES[@]}"; do
        eval_ablation "$ablation"
    done
    echo "========================================="
    echo "All ablation evaluations complete!"
    echo "========================================="
    exit 0
fi

# Map short form to config name
case "$1" in
    4) eval_ablation "4_structs" ;;
    7) eval_ablation "7_structs" ;;
    14) eval_ablation "14_structs" ;;
    *)
        echo "Error: unknown ablation '$1'. Use 4, 7, or 14."
        print_usage
        exit 1
        ;;
esac
