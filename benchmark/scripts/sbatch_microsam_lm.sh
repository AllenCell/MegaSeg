#!/bin/bash
#SBATCH --job-name=microsam_lm
#SBATCH --partition=aics_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/microsam_lm_%j.log

BENCHMARK_DIR="/allen/aics/rep_learn/megaseg/benchmark"
# Use the microsam-bench conda env (has micro_sam + vigra/elf deps)
PYTHON="/allen/aics/users/ashwin.samudre/miniconda3/envs/microsam-bench/bin/python"

echo "========================================="
echo "MicroSAM vit_b_lm zero-shot evaluation"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Start: $(date)"
echo "========================================="

cd "${BENCHMARK_DIR}" || exit 1

${PYTHON} scripts/run_microsam_lm_zeroshot.py --gpu 0 "$@"

echo "End: $(date)"
