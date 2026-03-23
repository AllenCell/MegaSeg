#!/bin/bash
#SBATCH --job-name=model-rec
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics_gpu
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="titanxp|1080ti"
#SBATCH --time=02:00:00

set --
source /allen/aics/rep_learn/MegaSeg/benchmark/.venv/bin/activate

SCRIPT_DIR="/allen/aics/rep_learn/MegaSeg-worktrees/feature-space-recommendation/benchmark/scripts"
OUT_DIR="/allen/aics/rep_learn/MegaSeg-worktrees/feature-space-recommendation/benchmark/results/cross_structure_features/model_recommendation"

echo "=============================="
echo "Step 1: Build centroids (57 samples x 16 structures, GPU)"
echo "=============================="
python "$SCRIPT_DIR/model_recommender.py" \
    --build-centroids \
    --n-samples 57 \
    --output-dir "$OUT_DIR" \
    --device cuda

EMB_PATH="$OUT_DIR/embeddings.npz"
if [ ! -f "$EMB_PATH" ]; then
    echo "ERROR: embeddings.npz not created. Aborting."
    exit 1
fi

echo ""
echo "=============================="
echo "Step 2: Leave-one-out validation (CPU)"
echo "=============================="
python "$SCRIPT_DIR/model_recommender.py" \
    --validate \
    --embeddings "$EMB_PATH" \
    --output-dir "$OUT_DIR" \
    --device cpu

echo ""
echo "=============================="
echo "Step 3: Multi-layer comparison (CPU)"
echo "=============================="
python "$SCRIPT_DIR/model_recommender.py" \
    --compare-layers \
    --embeddings "$EMB_PATH" \
    --output-dir "$OUT_DIR" \
    --device cpu

echo ""
echo "=============================="
echo "All done!"
echo "=============================="
echo "Output directory: $OUT_DIR"
ls -la "$OUT_DIR"
