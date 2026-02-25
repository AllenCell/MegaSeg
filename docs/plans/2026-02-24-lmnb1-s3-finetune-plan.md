# LMNB1 S3 Fine-Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create training configs and launch scripts to re-run LMNB1 fine-tuning with correct channel selection from multi-channel S3 .ome.tiff files.

**Architecture:** New parameterized data config (`segmentation_s3.yaml`) with `C: ${raw_channel}` instead of hardcoded `C: 0`. Experiment YAMLs set `raw_channel` per variant. Phase 1 tests 3 channels; Phase 2 runs 7 L-shaped experiments with the winning channel.

**Tech Stack:** Hydra/OmegaConf YAML configs, cyto-dl training framework, MONAI transforms, S3 via bioio/fsspec.

**Design doc:** `docs/plans/2026-02-24-lmnb1-s3-finetune-design.md`

---

### Task 1: Create branch

**Step 1: Create and switch to new branch**

```bash
cd /allen/aics/rep_learn/MegaSeg
git checkout -b finetune-lmnb1-s3
```

Expected: Branch created off current `fix-expanduser-url-stripping`.

---

### Task 2: Create `benchmark/configs/train.yaml`

**Files:**
- Create: `benchmark/configs/train.yaml`

**Step 1: Create the file**

This mirrors cyto-dl's `train.yaml` but lives in the benchmark config directory so `CYTODL_CONFIG_PATH` can resolve it.

```yaml
# @package _global_

defaults:
  - _self_
  - data: null
  - model: null
  - callbacks: default.yaml
  - logger: null
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml
  - experiment: null
  - hparams_search: null
  - optional local: default.yaml
  - debug: null

experiment_name: "catchall_experiment"
run_name: "catchall_run"
task_name: "train"

tags: ["dev"]

train: True
test: True

checkpoint:
  ckpt_path: null
  weights_only: null
  strict: True

seed: null
```

**Step 2: Commit**

```bash
git add benchmark/configs/train.yaml
git commit -m "Add top-level train.yaml config for benchmark training"
```

---

### Task 3: Create `benchmark/configs/data/local/segmentation_s3.yaml`

**Files:**
- Source: `benchmark/configs/data/local/segmentation.yaml` (copy and modify)
- Create: `benchmark/configs/data/local/segmentation_s3.yaml`

**Step 1: Create the file**

Copy `segmentation.yaml` and change 4 source-reader `C: 0` lines to `C: ${raw_channel}`. Leave the 4 target-reader `C: 0` unchanged (seg files are single-channel). The 4 source-reader locations are:
- `transforms.train.transforms[0].reader[0].C` (line 27)
- `transforms.test.transforms[0].reader[0].C` (line 83)
- `transforms.predict.transforms[0].reader[0].C` (line 119)
- `transforms.valid.transforms[0].reader[0].C` (line 141)

Full file content:

```yaml
_target_: cyto_dl.datamodules.dataframe.DataframeDatamodule

path:
cache_dir:

num_workers: 4
batch_size: 1
pin_memory: False
persistent_workers: True
split_column:
columns:
  - ${source_col}
  - ${target_col}

transforms:
  train:
    _target_: monai.transforms.Compose
    transforms:
      - _target_: monai.transforms.LoadImaged
        keys: ${source_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: ${raw_channel}
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.LoadImaged
        keys: ${target_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: 0
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.EnsureChannelFirstd
        channel_dim: "no_channel"
        keys: ${data.columns}
      - _target_: monai.transforms.ToTensord
        keys: ${data.columns}
      - _target_: monai.transforms.NormalizeIntensityd
        keys: ${source_col}
        channel_wise: True
      - _target_: monai.transforms.ThresholdIntensityd
        keys: ${target_col}
        threshold: 0.1
        above: False
        cval: 1
      - _target_: cyto_dl.image.transforms.RandomMultiScaleCropd
        keys: ${data.columns}
        patch_shape: ${data._aux.patch_shape}
        patch_per_image: 1
        scales_dict: ${kv_to_dict:${data._aux._scales_dict}}
      - _target_: monai.transforms.RandHistogramShiftd
        prob: 0.1
        keys: ${source_col}
        num_control_points: [90, 500]
      - _target_: monai.transforms.RandStdShiftIntensityd
        prob: 0.1
        keys: ${source_col}
        factors: 0.1
      - _target_: monai.transforms.RandAdjustContrastd
        prob: 0.1
        keys: ${source_col}
        gamma: [0.9, 1.5]

  test:
    _target_: monai.transforms.Compose
    transforms:
      - _target_: monai.transforms.LoadImaged
        keys: ${source_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: ${raw_channel}
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.LoadImaged
        keys: ${target_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: 0
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.EnsureChannelFirstd
        channel_dim: "no_channel"
        keys: ${data.columns}
      - _target_: monai.transforms.ToTensord
        keys: ${data.columns}
      - _target_: monai.transforms.NormalizeIntensityd
        keys: ${source_col}
        channel_wise: True
      - _target_: monai.transforms.ThresholdIntensityd
        keys: ${target_col}
        threshold: 0.1
        above: False
        cval: 1

  predict:
    _target_: monai.transforms.Compose
    transforms:
      - _target_: monai.transforms.LoadImaged
        keys: ${source_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: ${raw_channel}
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.EnsureChannelFirstd
        channel_dim: "no_channel"
        keys: ${source_col}
      - _target_: monai.transforms.ToTensord
        keys: ${source_col}
      - _target_: monai.transforms.NormalizeIntensityd
        keys: ${source_col}
        channel_wise: True

  valid:
    _target_: monai.transforms.Compose
    transforms:
      - _target_: monai.transforms.LoadImaged
        keys: ${source_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: ${raw_channel}
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.LoadImaged
        keys: ${target_col}
        expanduser: false
        reader:
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ${eval:'"ZYX" if ${spatial_dims}==3 else "YX"'}
            C: 0
            Z: ${eval:'None if ${spatial_dims}==3 else 38'}
      - _target_: monai.transforms.EnsureChannelFirstd
        channel_dim: "no_channel"
        keys: ${data.columns}
      - _target_: monai.transforms.ToTensord
        keys: ${data.columns}
      - _target_: monai.transforms.NormalizeIntensityd
        keys: ${source_col}
        channel_wise: True
      - _target_: monai.transforms.ThresholdIntensityd
        keys: ${target_col}
        threshold: 0.1
        above: False
        cval: 1
      - _target_: cyto_dl.image.transforms.RandomMultiScaleCropd
        keys: ${data.columns}
        patch_shape: ${data._aux.patch_shape}
        patch_per_image: 1
        scales_dict: ${kv_to_dict:${data._aux._scales_dict}}

_aux:
  _scales_dict:
    - - ${target_col}
      - [1]
    - - ${source_col}
      - [1]
```

**Step 2: Commit**

```bash
git add benchmark/configs/data/local/segmentation_s3.yaml
git commit -m "Add segmentation_s3.yaml with parameterized raw_channel for multi-channel S3 data"
```

---

### Task 4: Create Phase 1 channel test experiment configs

**Files:**
- Create: `benchmark/configs/experiment/train/finetune/lmnb1_channel_test_c1.yaml`
- Create: `benchmark/configs/experiment/train/finetune/lmnb1_channel_test_c3.yaml`
- Create: `benchmark/configs/experiment/train/finetune/lmnb1_channel_test_c6.yaml`

All three are identical except `raw_channel`, `experiment_name`, and `run_name`.

**Step 1: Create `lmnb1_channel_test_c1.yaml`**

```yaml
# @package _global_
defaults:
  - override /data: local/segmentation_s3.yaml
  - override /model: local/segmentation_ms25_float32.yaml
  - override /callbacks: default.yaml
  - override /trainer: ddp.yaml
  - override /logger: csv.yaml

tags: ["finetune", "lmnb1", "channel_test"]
seed: 12345

raw_channel: 1

checkpoint:
  ckpt_path: /allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/logs/train/runs/megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/2024-07-15_00-39-47/checkpoints/epoch_650.ckpt
  weights_only: true
  strict: true

experiment_name: lmnb1_channel_test_c1
run_name: lmnb1_ft_channel_test_c1
source_col: raw
target_col: seg
spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 700
  devices: [0]

data:
  path: /allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/
  cache_dir: /allen/aics/rep_learn/MegaSeg/benchmark/cache/
  subsample:
  batch_size: 32
  num_workers: 12
  _aux:
    patch_shape: [32, 128, 128]
```

**Step 2: Create `lmnb1_channel_test_c3.yaml`**

Same as c1 but with:
- `raw_channel: 3`
- `experiment_name: lmnb1_channel_test_c3`
- `run_name: lmnb1_ft_channel_test_c3`

**Step 3: Create `lmnb1_channel_test_c6.yaml`**

Same as c1 but with:
- `raw_channel: 6`
- `experiment_name: lmnb1_channel_test_c6`
- `run_name: lmnb1_ft_channel_test_c6`

**Step 4: Commit**

```bash
git add benchmark/configs/experiment/train/finetune/
git commit -m "Add Phase 1 channel test experiment configs (C=1, C=3, C=6)"
```

---

### Task 5: Create launcher script

**Files:**
- Create: `benchmark/scripts/run_finetune.sh`

**Step 1: Create the script**

```bash
#!/usr/bin/env bash
# Run LMNB1 fine-tuning experiments from the MegaSeg benchmark directory.
#
# Usage:
#   # Run a single experiment:
#   bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c3
#
#   # Run all Phase 1 channel tests (sequentially):
#   bash scripts/run_finetune.sh --phase1
#
#   # Run on a specific GPU:
#   bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c3 --gpu 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$BENCHMARK_DIR/configs"

GPU="${GPU:-0}"

run_experiment() {
    local experiment="$1"
    local gpu="${2:-$GPU}"
    echo "========================================="
    echo "Running: $experiment (GPU $gpu)"
    echo "========================================="
    CYTODL_CONFIG_PATH="$CONFIG_DIR" \
    PROJECT_ROOT="$BENCHMARK_DIR" \
        python -m cyto_dl.train \
        experiment="$experiment" \
        trainer.devices="[$gpu]"
}

if [[ "${1:-}" == "--phase1" ]]; then
    echo "Running Phase 1: Channel test (C=1, C=3, C=6)"
    for c in 1 3 6; do
        run_experiment "train/finetune/lmnb1_channel_test_c${c}"
    done
    echo "Phase 1 complete. Check outputs/ for results."
    exit 0
fi

if [[ -z "${1:-}" ]]; then
    echo "Usage:"
    echo "  $0 <experiment>          # Run single experiment"
    echo "  $0 --phase1              # Run all Phase 1 channel tests"
    echo ""
    echo "Examples:"
    echo "  $0 train/finetune/lmnb1_channel_test_c3"
    echo "  GPU=2 $0 train/finetune/lmnb1_channel_test_c3"
    exit 1
fi

run_experiment "$1"
```

**Step 2: Make executable and commit**

```bash
chmod +x benchmark/scripts/run_finetune.sh
git add benchmark/scripts/run_finetune.sh
git commit -m "Add launcher script for LMNB1 fine-tuning experiments"
```

---

### Task 6: Verify Hydra config resolution (dry run)

**Step 1: Run a dry-run to check config composes correctly**

```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark
CYTODL_CONFIG_PATH=/allen/aics/rep_learn/MegaSeg/benchmark/configs \
PROJECT_ROOT=/allen/aics/rep_learn/MegaSeg/benchmark \
  python -m cyto_dl.train \
  experiment=train/finetune/lmnb1_channel_test_c3 \
  --cfg job 2>&1 | head -100
```

Expected: Hydra prints the resolved config showing `C: 3` for source readers and `C: 0` for target readers, `weights_only: true`, correct checkpoint path, correct data path, batch_size 32, etc.

If `--cfg job` is not supported, try:

```bash
CYTODL_CONFIG_PATH=/allen/aics/rep_learn/MegaSeg/benchmark/configs \
PROJECT_ROOT=/allen/aics/rep_learn/MegaSeg/benchmark \
  python -m cyto_dl.train \
  experiment=train/finetune/lmnb1_channel_test_c3 \
  --info config 2>&1 | head -100
```

**Step 2: Verify key resolved values**

Check the output for:
- `raw_channel: 3`
- Source reader `C: 3` (not 0)
- Target reader `C: 0`
- `checkpoint.ckpt_path` points to `epoch_650.ckpt`
- `checkpoint.weights_only: true`
- `data.path` points to LMNB1 directory
- `data.batch_size: 32`
- `expanduser: false` on all LoadImaged entries

If any values are wrong, fix the configs and re-verify before proceeding.

**Step 3: Commit any fixes**

---

### Task 7: Launch Phase 1 channel tests

**Step 1: Run the 3 channel tests**

Option A — Sequential (one GPU):
```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark
bash scripts/run_finetune.sh --phase1
```

Option B — Parallel (3 GPUs, recommended since we have 4 A100s):
```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark
GPU=0 bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c1 &
GPU=1 bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c3 &
GPU=2 bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c6 &
wait
```

**Step 2: Monitor training**

Check `benchmark/outputs/` for training logs. Look for:
- Val loss decreasing (good channel) vs flat/high (wrong channel)
- Early stopping trigger on bad channels
- CSV logger output for epoch-by-epoch metrics

**Step 3: Compare results and pick winner**

The correct channel should show dramatically better val loss / Dice than the others.

---

### Task 8: Create Phase 2 experiment configs (after channel test)

Once the correct channel (likely C=3) is confirmed, create 7 L-shaped experiment configs. Each is a copy of the channel test winner with `raw_channel` set to the winning value, and two axes varied:

**Files to create:**
- `benchmark/configs/experiment/train/finetune/lmnb1_data100_converged.yaml` — subsample: null, max_epochs: 700
- `benchmark/configs/experiment/train/finetune/lmnb1_data100_10epochs.yaml` — subsample: null, max_epochs: 10
- `benchmark/configs/experiment/train/finetune/lmnb1_data100_50epochs.yaml` — subsample: null, max_epochs: 50
- `benchmark/configs/experiment/train/finetune/lmnb1_data100_100epochs.yaml` — subsample: null, max_epochs: 100
- `benchmark/configs/experiment/train/finetune/lmnb1_data10_converged.yaml` — subsample: 0.1, max_epochs: 700
- `benchmark/configs/experiment/train/finetune/lmnb1_data20_converged.yaml` — subsample: 0.2, max_epochs: 700
- `benchmark/configs/experiment/train/finetune/lmnb1_data50_converged.yaml` — subsample: 0.5, max_epochs: 700

Template (only `raw_channel`, `subsample`, `max_epochs`, names differ per variant):

```yaml
# @package _global_
defaults:
  - override /data: local/segmentation_s3.yaml
  - override /model: local/segmentation_ms25_float32.yaml
  - override /callbacks: default.yaml
  - override /trainer: ddp.yaml
  - override /logger: csv.yaml

tags: ["finetune", "lmnb1"]
seed: 12345

raw_channel: 3  # confirmed from Phase 1

checkpoint:
  ckpt_path: /allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/logs/train/runs/megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/2024-07-15_00-39-47/checkpoints/epoch_650.ckpt
  weights_only: true
  strict: true

experiment_name: lmnb1_data100_converged  # varies
run_name: lmnb1_ft_data100_converged      # varies
source_col: raw
target_col: seg
spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 700    # varies: 10, 50, 100, or 700
  devices: [0]

data:
  path: /allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/
  cache_dir: /allen/aics/rep_learn/MegaSeg/benchmark/cache/
  subsample:          # varies: null, 0.1, 0.2, 0.5
  batch_size: 32
  num_workers: 12
  _aux:
    patch_shape: [32, 128, 128]
```

**Commit:**

```bash
git add benchmark/configs/experiment/train/finetune/lmnb1_data*.yaml
git commit -m "Add Phase 2 L-shaped experiment configs for LMNB1 fine-tuning"
```

---

### Task 9: Launch Phase 2 and evaluate

**Step 1: Run all 7 experiments**

Use the launcher script or run in parallel across GPUs. The short-epoch runs (10, 50, 100 epochs) finish fast; the converged runs take longer.

**Step 2: Evaluate results**

Use the existing eval pipeline (`scripts/evaluate_benchmark.py`) or the finetune eval configs from PR #9 (adapted for new checkpoint paths).

**Step 3: Commit results and push**

```bash
git push -u origin finetune-lmnb1-s3
```
