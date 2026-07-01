#!/bin/bash
#SBATCH --job-name=classic-timelapse
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Run classic segmentation on timelapse data:
#   NPM1 (70 TIFs - symlinked), Mitochondria (5 TIFs), Tubulin_MultiCh (30 TIFs)
#
# NPM1 and Mitochondria TIFs are ready now.
# Tubulin_MultiCh TIFs need to be extracted first (sbatch_extract_tubulin.sh).
# This script will skip datasets whose TIFs don't exist yet.

set -euo pipefail

CONDA_BASE="/allen/aics/users/ashwin.samudre/miniconda3"
WORKTREE="/allen/aics/rep_learn/MegaSeg-worktrees/timelapse/benchmark"

echo "=== Classic segmentation - timelapse ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

set --
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate msseg

echo ""
echo "--- Classic NPM1 (70 TIFs) ---"
python -u "${WORKTREE}/scripts/run_classic_timelapse.py" --datasets NPM1

echo ""
echo "--- Classic Mitochondria (5 TIFs) ---"
python -u "${WORKTREE}/scripts/run_classic_timelapse.py" --datasets Mitochondria

echo ""
echo "--- Classic Tubulin_MultiCh (30 TIFs) ---"
python -u "${WORKTREE}/scripts/run_classic_timelapse.py" --datasets Tubulin_MultiCh

echo ""
echo "=== Verification ==="
OUT_DIR="${WORKTREE}/outputs/eval/uint8/classic/timelapse"
for d in NPM1 Mitochondria Tubulin_MultiCh; do
    n=$(ls "${OUT_DIR}/${d}/"*.tif 2>/dev/null | wc -l)
    echo "  ${d}: ${n} segmentations"
done

echo ""
echo "Done: $(date)"
