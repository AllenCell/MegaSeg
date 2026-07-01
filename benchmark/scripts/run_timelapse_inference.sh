#!/bin/bash
# Run all 3 timelapse inference jobs sequentially on a single GPU.
# Usage: CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_timelapse_inference.sh > /tmp/timelapse_inference.log 2>&1 &

set -euo pipefail

PYTHON=/allen/aics/users/ashwin.samudre/suraj_cyto/cyto-dl/cytos/bin/python
EVAL_SCRIPT=/allen/aics/users/ashwin.samudre/suraj_cyto/cyto-dl/cyto_dl/eval.py
WORKTREE=/allen/aics/rep_learn/MegaSeg-worktrees/main/benchmark
export PROJECT_ROOT="$WORKTREE"
export CYTODL_CONFIG_PATH="$WORKTREE/configs"

cd "$WORKTREE"

echo "=== $(date) Starting timelapse inference ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Python: $PYTHON"
echo ""

# 1. Individual NPM1 (70 images)
echo "=== $(date) [1/3] Individual NPM1 (70 images) ==="
$PYTHON "$EVAL_SCRIPT" experiment=eval/uint8/individual/timelapse/NPM1
echo "=== $(date) [1/3] Individual NPM1 DONE ==="

# 2. Individual TOMM20 (5 images - uses CZI via BioIO)
echo "=== $(date) [2/3] Individual TOMM20 (5 images) ==="
$PYTHON "$EVAL_SCRIPT" experiment=eval/uint8/individual/timelapse/TOMM20
echo "=== $(date) [2/3] Individual TOMM20 DONE ==="

# 3. MegaSeg NPM1 (70 images - the fix for the failed run)
echo "=== $(date) [3/3] MegaSeg NPM1 (70 images) ==="
$PYTHON "$EVAL_SCRIPT" experiment=eval/uint8/megaseg/dynunet/timelapse/NPM1
echo "=== $(date) [3/3] MegaSeg NPM1 DONE ==="

echo "=== $(date) All timelapse inference complete ==="
