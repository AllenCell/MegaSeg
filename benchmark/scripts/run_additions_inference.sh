#!/bin/bash
# Run MegaSeg inference on Caroline's additional data (new Tubulin + Cardio).
#
# Prerequisites:
#   - Tubulin CZI extracted to Tubulin_MultiCh_2/ (27 TIFs)
#   - Cardio SLD extracted to Cardio_MultiCh/ (95 TIFs)
#   - Multidim CSVs generated in benchmark/data/multidim/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_additions_inference.sh \
#       > /tmp/additions_inference.log 2>&1 &

set -euo pipefail

PYTHON=/allen/aics/users/ashwin.samudre/suraj_cyto/cyto-dl/cytos/bin/python
EVAL_SCRIPT=/allen/aics/users/ashwin.samudre/suraj_cyto/cyto-dl/cyto_dl/eval.py
WORKTREE=/allen/aics/rep_learn/MegaSeg-worktrees/timelapse/benchmark
export PROJECT_ROOT="$WORKTREE"
export CYTODL_CONFIG_PATH="$WORKTREE/configs"

cd "$WORKTREE"

echo "=== $(date) Starting additions inference ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Python: $PYTHON"
echo ""

# 1. MegaSeg on new Tubulin (27 images via CZI)
echo "=== $(date) [1/2] MegaSeg Tubulin_MultiCh_2 (27 images) ==="
$PYTHON "$EVAL_SCRIPT" experiment=eval/uint8/megaseg/dynunet/timelapse/Tubulin_MultiCh_2
echo "=== $(date) [1/2] MegaSeg Tubulin_MultiCh_2 DONE ==="

# 2. MegaSeg on Cardio (95 images via TIFs)
echo "=== $(date) [2/2] MegaSeg Cardio_MultiCh (95 images) ==="
$PYTHON "$EVAL_SCRIPT" experiment=eval/uint8/megaseg/dynunet/timelapse/Cardio_MultiCh
echo "=== $(date) [2/2] MegaSeg Cardio_MultiCh DONE ==="

echo "=== $(date) All additions inference complete ==="
