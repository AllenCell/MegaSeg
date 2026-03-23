#!/bin/bash
#SBATCH --job-name=xstruct_feat
#SBATCH --partition=aics_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="titanxp|1080ti|v100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ashwin.samudre/logs/cross_structure_features_%j.out
#SBATCH --error=/home/ashwin.samudre/logs/cross_structure_features_%j.err

set --

echo "========================================="
echo "Cross-Structure Feature Analysis (UMAP + Probing)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start time: $(date)"
echo "========================================="

# Use the worktree version of the script (has UMAP + probing additions)
SCRIPT="/allen/aics/rep_learn/MegaSeg-worktrees/cross-structure-features/benchmark/scripts/cross_structure_features.py"
VENV_PYTHON="/allen/aics/rep_learn/MegaSeg/benchmark/.venv/bin/python"

"$VENV_PYTHON" "$SCRIPT" \
    --n-samples 10 \
    --device cuda:0 \
    --label megaseg_pretrained

echo ""
echo "========================================="
echo "Cross-structure features complete!"
echo "End time: $(date)"
echo "========================================="
