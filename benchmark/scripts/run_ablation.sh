#!/usr/bin/env bash
# Run ablation training experiments: train DynUNet from scratch on 4, 7, and 14 structures.
#
# Usage:
#   # Run a single ablation:
#   bash scripts/run_ablation.sh 4
#
#   # Run all ablations sequentially:
#   bash scripts/run_ablation.sh --all
#
#   # Run on a specific GPU:
#   GPU=2 bash scripts/run_ablation.sh 7
#
#   # Run in background with nohup:
#   nohup bash scripts/run_ablation.sh --all > ablation.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$BENCHMARK_DIR/configs"

GPU="${GPU:-0}"
UV="${UV:-/home/ashwin.samudre/.local/bin/uv}"

EXPERIMENTS=(
    "4:train/ablation/4_structs"
    "7:train/ablation/7_structs"
    "14:train/ablation/14_structs"
)

run_experiment() {
    local experiment="$1"
    local label="$2"
    local gpu="${3:-$GPU}"
    echo "========================================="
    echo "Ablation: ${label}-STRUCTS (GPU $gpu)"
    echo "Config: $experiment"
    echo "Time: $(date)"
    echo "========================================="

    # Disable RichProgressBar when not attached to a TTY (e.g. nohup/background)
    local extra_args=()
    if [[ ! -t 1 ]]; then
        extra_args+=("~callbacks.rich_progress_bar")
    fi

    CYTODL_CONFIG_PATH="$CONFIG_DIR" \
    PROJECT_ROOT="$BENCHMARK_DIR" \
        "$UV" run --directory "$BENCHMARK_DIR" \
        python -m cyto_dl.train \
        experiment="$experiment" \
        trainer.devices="[$gpu]" \
        "${extra_args[@]}"
}

print_usage() {
    echo "Usage:"
    echo "  $0 <4|7|14>    # Run single ablation"
    echo "  $0 --all        # Run all ablations sequentially"
    echo ""
    echo "Ablation structure subsets:"
    echo "  4:  CTNNB1, MYH10, ACTN1, ATP2A2"
    echo "  7:  CTNNB1, RAB5A, TOMM20, MYH10, ACTN1, ST6GAL1, ATP2A2"
    echo "  14: All structures (excl. LMNB1, NPM1)"
    echo ""
    echo "Environment variables:"
    echo "  GPU=<id>   GPU device index (default: 0)"
}

if [[ -z "${1:-}" ]]; then
    print_usage
    exit 1
fi

if [[ "$1" == "--all" ]]; then
    echo "Running all ablation experiments sequentially"
    echo "Start time: $(date)"
    echo ""
    for entry in "${EXPERIMENTS[@]}"; do
        label="${entry%%:*}"
        experiment="${entry#*:}"
        run_experiment "$experiment" "$label"
        echo ""
        echo "${label}-STRUCTS training complete at $(date)"
        echo ""
    done
    echo "========================================="
    echo "All ablation experiments complete!"
    echo "End time: $(date)"
    echo "========================================="
    exit 0
fi

# Single experiment mode
target="$1"
for entry in "${EXPERIMENTS[@]}"; do
    label="${entry%%:*}"
    experiment="${entry#*:}"
    if [[ "$label" == "$target" ]]; then
        run_experiment "$experiment" "$label"
        exit 0
    fi
done

echo "Error: unknown ablation '$target'. Use 4, 7, or 14."
print_usage
exit 1
