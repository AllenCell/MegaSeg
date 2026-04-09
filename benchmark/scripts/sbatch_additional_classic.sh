#!/bin/bash
#SBATCH --job-name=addstruct-classic
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Run 14 classic segmentation workflows on all additional structure
# channel-volumes (41 volumes x 14 workflows = 574 predictions).
#
# Prerequisites: run extract_additional_channels.py first.
#
# Usage:
#   sbatch scripts/sbatch_additional_classic.sh

set -euo pipefail

PROJECT_ROOT="/allen/aics/rep_learn/MegaSeg-worktrees/additional-structures/benchmark"
PYTHON="/allen/aics/users/ashwin.samudre/miniconda3/envs/msseg/bin/python"

echo "=== Additional Structures Classic Segmentation ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Node: $(hostname)"
echo "Python: ${PYTHON}"
echo "Start: $(date)"
echo ""

# Verify extracted data exists
EXTRACTED="${PROJECT_ROOT}/data/additional_structures_extracted"
if [[ ! -d "$EXTRACTED" ]]; then
    echo "ERROR: Extracted data not found: $EXTRACTED"
    echo "Run extract_additional_channels.py first!"
    exit 1
fi

cd "${PROJECT_ROOT}"
$PYTHON scripts/run_classic_additional.py

echo ""
echo "=== CLASSIC SEGMENTATION COMPLETE ==="
echo "End: $(date)"
