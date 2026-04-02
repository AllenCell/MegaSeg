#!/bin/bash
#SBATCH --job-name=extract-tubulin
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00

# Extract Tubulin_MultiCh TIFs from 48GB CZI file
# Needs high memory due to CZI file size

set -euo pipefail

CONDA_BASE="/allen/aics/users/ashwin.samudre/miniconda3"
WORKTREE="/allen/aics/rep_learn/MegaSeg-worktrees/timelapse/benchmark"

echo "=== Extracting Tubulin_MultiCh from CZI ==="
echo "Node: $(hostname)"
echo "Memory: $(free -g | awk '/Mem:/ {print $2}')GB total"
echo "Start: $(date)"

set --
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate test-bioio

python -u "${WORKTREE}/scripts/extract_timelapse_tifs.py" \
    --datasets Tubulin_MultiCh

echo ""
echo "Done: $(date)"
echo "Files: $(ls "${WORKTREE}/data/caroline_data/raw/Tubulin_MultiCh/"*.tif 2>/dev/null | wc -l) TIFs"
