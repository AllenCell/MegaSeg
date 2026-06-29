#!/bin/bash
#SBATCH --job-name=abl_eval_4f
#SBATCH --partition=aics_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="titanxp|1080ti|v100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ashwin.samudre/logs/ablation_eval_4_full_%j.out
#SBATCH --error=/home/ashwin.samudre/logs/ablation_eval_4_full_%j.err

set --

echo "========================================="
echo "Ablation Eval: 4-struct (57 filtered images)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start time: $(date)"
echo "========================================="

BENCHMARK_DIR="/allen/aics/rep_learn/MegaSeg/benchmark"
VENV_PYTHON="$BENCHMARK_DIR/.venv/bin/python"

"$VENV_PYTHON" "$BENCHMARK_DIR/scripts/evaluate_ablation.py" \
    --ablation 4 \
    --device cuda:0

echo ""
echo "========================================="
echo "4-struct full eval complete!"
echo "End time: $(date)"
echo "========================================="
