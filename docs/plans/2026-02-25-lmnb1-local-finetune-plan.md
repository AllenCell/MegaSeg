# LMNB1 Local Fine-Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pre-download LMNB1 S3 data to local NFS, run quick channel validation (C=1 vs C=3), then execute all 7 L-shaped fine-tuning experiments with the correct channel.

**Architecture:** Download S3 files with `aws s3 cp`, generate local-path CSVs from existing S3 CSVs, parameterize `segmentation.yaml` for channel selection, run training via `run_finetune.sh`.

**Tech Stack:** aws CLI, Python/pandas (CSV generation), Hydra/cyto-dl (training), 4x A100 GPUs

---

### Task 1: Clean up failed S3 run artifacts

**Files:**
- Delete: `benchmark/cache/tmp0tluuram/`, `benchmark/cache/tmpl8brcx2_/`
- Delete: `benchmark/outputs/train/runs/lmnb1_channel_test_c1/`
- Delete: `benchmark/outputs/train/runs/lmnb1_channel_test_c3/`
- Delete: `benchmark/outputs/train/runs/lmnb1_channel_test_c6/`

**Step 1: Remove failed cache and output directories**

```bash
rm -rf benchmark/cache/tmp*
rm -rf benchmark/outputs/train/runs/lmnb1_channel_test_*
```

**Step 2: Verify cleanup**

```bash
ls benchmark/cache/
# Expected: .DS_Store, test/, train/, val/ (no tmp* dirs)
ls benchmark/outputs/train/runs/
# Expected: empty or non-existent
```

---

### Task 2: Download S3 files to local NFS

**Files:**
- Create: `data/variance-100x/LMNB1/raw/` (directory for raw .ome.tiff files)
- Create: `data/variance-100x/LMNB1/seg/` (directory for seg .tiff files)

The S3 CSVs contain URLs like:
- Raw: `s3://allencell/aics/hipsc_single_cell_image_dataset/fov_path/<hash>_<name>.ome.tiff`
- Seg: `s3://allencell/aics/hipsc_single_cell_image_dataset/struct_seg_path/<hash>_<name>_struct_segmentation.tiff`

**Step 1: Extract unique S3 URLs from all CSVs**

```bash
cd /allen/aics/rep_learn/MegaSeg
# Extract all unique raw and seg URLs
tail -n +2 data/variance-100x/LMNB1/*_s3_cytodl.csv | cut -d',' -f1 | sort -u > /tmp/lmnb1_raw_urls.txt
tail -n +2 data/variance-100x/LMNB1/*_s3_cytodl.csv | cut -d',' -f2 | sort -u > /tmp/lmnb1_seg_urls.txt
wc -l /tmp/lmnb1_raw_urls.txt /tmp/lmnb1_seg_urls.txt
# Expected: ~267 raw + ~267 seg unique URLs (some may be shared across splits)
```

**Step 2: Download raw files**

```bash
mkdir -p data/variance-100x/LMNB1/raw
# Download in parallel (8 concurrent)
cat /tmp/lmnb1_raw_urls.txt | xargs -P 8 -I {} aws s3 cp --no-sign-request {} data/variance-100x/LMNB1/raw/
```

Expected: ~30-60 min depending on bandwidth. Each file ~500MB.

**Step 3: Download seg files**

```bash
mkdir -p data/variance-100x/LMNB1/seg
cat /tmp/lmnb1_seg_urls.txt | xargs -P 8 -I {} aws s3 cp --no-sign-request {} data/variance-100x/LMNB1/seg/
```

Expected: Faster — seg files are smaller (single-channel masks).

**Step 4: Verify downloads**

```bash
# Count downloaded files
ls data/variance-100x/LMNB1/raw/*.ome.tiff | wc -l
ls data/variance-100x/LMNB1/seg/*.tiff | wc -l
# Should match URL counts from Step 1
# Spot-check a file
ls -lh data/variance-100x/LMNB1/raw/ | head -3
```

---

### Task 3: Generate local-path CSVs

**Files:**
- Create: `data/variance-100x/LMNB1/train_lmnb1_local_cytodl.csv`
- Create: `data/variance-100x/LMNB1/val_lmnb1_local_cytodl.csv`
- Create: `data/variance-100x/LMNB1/test_lmnb1_local_cytodl.csv`

**Step 1: Generate CSVs with Python**

```python
import pandas as pd
import os

data_dir = "/allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1"

for split in ["train", "val", "test"]:
    s3_csv = os.path.join(data_dir, f"{split}_lmnb1_s3_cytodl.csv")
    local_csv = os.path.join(data_dir, f"{split}_lmnb1_local_cytodl.csv")

    df = pd.read_csv(s3_csv)

    # Extract filename from S3 URL (everything after last /)
    df["raw"] = df["raw"].apply(lambda u: os.path.join(data_dir, "raw", u.split("/")[-1]))
    df["seg"] = df["seg"].apply(lambda u: os.path.join(data_dir, "seg", u.split("/")[-1]))

    df.to_csv(local_csv, index=False)
    print(f"{split}: {len(df)} rows -> {local_csv}")
```

**Step 2: Verify CSVs are correct**

```bash
head -2 data/variance-100x/LMNB1/train_lmnb1_local_cytodl.csv
# Expected: raw,seg headers with /allen/aics/.../raw/<file>.ome.tiff paths
# Verify all referenced files exist:
python -c "
import pandas as pd, os
for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/variance-100x/LMNB1/{split}_lmnb1_local_cytodl.csv')
    missing_raw = [f for f in df['raw'] if not os.path.exists(f)]
    missing_seg = [f for f in df['seg'] if not os.path.exists(f)]
    print(f'{split}: {len(missing_raw)} missing raw, {len(missing_seg)} missing seg')
"
# Expected: 0 missing for all splits
```

---

### Task 4: Parameterize segmentation.yaml for channel selection

**Files:**
- Modify: `benchmark/configs/data/local/segmentation.yaml` (lines 27, 83, 119, 141)

Change the 4 source-reader `C: 0` lines (train, test, predict, valid sections) to `C: ${raw_channel}`.
Keep all target/seg reader lines as `C: 0` (seg masks are always channel 0).

The source reader appears at these locations (train line 27, test line 83, predict line 119, valid line 141):
```yaml
          - _target_: cyto_dl.image.io.MonaiBioReader
            dimension_order_out: ...
            C: 0           # <-- change to C: ${raw_channel}
```

The target reader (lines 36, 92, 150) stays `C: 0`.

**Step 1: Apply edits**

Use `edit` to change each of the 4 source `C: 0` to `C: ${raw_channel}`.

**Step 2: Verify no existing experiments break**

The existing eval configs all set `raw_channel` via their experiment YAMLs. Check that:
- `segmentation.yaml` has exactly 4 occurrences of `${raw_channel}` (source) and 3 of `C: 0` (target)
- Existing experiment configs already define `raw_channel` or will need updating

**Important:** If existing eval experiments don't define `raw_channel`, this change will break them. In that case, create `segmentation_local.yaml` as a copy instead of modifying the shared file.

```bash
# Check if any existing experiment configs already set raw_channel
grep -r "raw_channel" benchmark/configs/experiment/ --include="*.yaml"
```

If only the new finetune configs set it, we must NOT modify `segmentation.yaml` — instead create `segmentation_local.yaml`.

---

### Task 5: Create channel test experiment configs (local data)

**Files:**
- Modify or Create: `benchmark/configs/experiment/train/finetune/lmnb1_channel_test_c1.yaml`
- Modify or Create: `benchmark/configs/experiment/train/finetune/lmnb1_channel_test_c3.yaml`

Update the existing channel test configs to:
1. Use the local data config (either modified `segmentation.yaml` or new `segmentation_local.yaml`)
2. Point `data.path` to the local CSV directory
3. Set `max_epochs: 10` (quick validation only)
4. Set `raw_channel: 1` or `raw_channel: 3`

Example config (`lmnb1_channel_test_c3.yaml`):
```yaml
# @package _global_
defaults:
  - override /data: local/segmentation_local.yaml  # or segmentation.yaml if safe
  - override /model: local/segmentation_ms25_float32.yaml
  - override /callbacks: default.yaml
  - override /trainer: ddp.yaml
  - override /logger: csv.yaml

tags: ["finetune", "lmnb1", "channel_test"]
seed: 12345
raw_channel: 3

checkpoint:
  ckpt_path: /allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/logs/train/runs/megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/2024-07-15_00-39-47/checkpoints/epoch_650.ckpt
  weights_only: true
  strict: true

experiment_name: lmnb1_channel_test_c3
run_name: lmnb1_ft_channel_test_c3
source_col: raw
target_col: seg
spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 10
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

**Step 1: Update c1 and c3 configs**

**Step 2: Dry-run config resolution**

```bash
CYTODL_CONFIG_PATH="$(pwd)/configs" PROJECT_ROOT="$(pwd)" \
  uv run python -m cyto_dl.train --cfg job -- \
  experiment=train/finetune/lmnb1_channel_test_c3
```

Verify: `raw_channel: 3`, `max_epochs: 10`, `data.path` points to local dir, source reader `C: 3`.

---

### Task 6: Launch quick channel validation (C=1 vs C=3, 10 epochs)

**Step 1: Launch both on separate GPUs**

```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark
nohup bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c1 > /tmp/finetune_c1.log 2>&1 &
GPU=1 nohup bash scripts/run_finetune.sh train/finetune/lmnb1_channel_test_c3 > /tmp/finetune_c3.log 2>&1 &
```

**Step 2: Monitor for sanity check completion (~2-5 min with local data)**

```bash
grep -i "epoch\|train_loss\|val_loss\|sanity" /tmp/finetune_c1.log /tmp/finetune_c3.log
```

**Step 3: Wait for completion (~10-20 min) and compare losses**

```bash
# Check CSV logger output
cat benchmark/outputs/train/runs/lmnb1_channel_test_c1/csv_logs/*/metrics.csv
cat benchmark/outputs/train/runs/lmnb1_channel_test_c3/csv_logs/*/metrics.csv
```

The channel with significantly lower val loss is the correct one (expect C=3).

---

### Task 7: Create Phase 2 experiment configs (7 L-shaped experiments)

**Files:** Create 7 experiment YAML files in `benchmark/configs/experiment/train/finetune/`:

| Config file | raw_channel | max_epochs | subsample |
|-------------|-------------|------------|-----------|
| `lmnb1_data100_converged.yaml` | (winner) | 700 | null |
| `lmnb1_data100_10epochs.yaml` | (winner) | 10 | null |
| `lmnb1_data100_50epochs.yaml` | (winner) | 50 | null |
| `lmnb1_data100_100epochs.yaml` | (winner) | 100 | null |
| `lmnb1_data10_converged.yaml` | (winner) | 700 | 0.1 |
| `lmnb1_data20_converged.yaml` | (winner) | 700 | 0.2 |
| `lmnb1_data50_converged.yaml` | (winner) | 700 | 0.5 |

All share the same base structure as the channel test config, with:
- `data.path: /allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/`
- The winning `raw_channel` value
- Unique `experiment_name` and `run_name`

**Step 1: Create all 7 configs**

**Step 2: Dry-run one config to verify**

---

### Task 8: Launch Phase 2 experiments

With 4 GPUs available, run up to 4 experiments in parallel. Prioritize the short ones first:

**Batch 1 (4 GPUs):** data100_10epochs, data100_50epochs, data100_100epochs, data10_converged

**Batch 2 (4 GPUs):** data20_converged, data50_converged, data100_converged (this one is long)

```bash
# Batch 1
nohup GPU=0 bash scripts/run_finetune.sh train/finetune/lmnb1_data100_10epochs > /tmp/ft_10ep.log 2>&1 &
nohup GPU=1 bash scripts/run_finetune.sh train/finetune/lmnb1_data100_50epochs > /tmp/ft_50ep.log 2>&1 &
nohup GPU=2 bash scripts/run_finetune.sh train/finetune/lmnb1_data100_100epochs > /tmp/ft_100ep.log 2>&1 &
nohup GPU=3 bash scripts/run_finetune.sh train/finetune/lmnb1_data10_converged > /tmp/ft_d10.log 2>&1 &
```

**Step 1: Launch batch 1**

**Step 2: Monitor completion, launch batch 2 as GPUs free up**

**Step 3: Verify all 7 experiments completed successfully**

---

### Task 9: Commit all changes

**Step 1: Stage and commit**

```bash
git add -A
git add -f data/variance-100x/LMNB1/*_local_cytodl.csv  # if in .gitignore
git commit -m "Add local data pipeline for LMNB1 fine-tuning

- Pre-download S3 files to NFS for fast training
- Create local-path CSVs (train/val/test_lmnb1_local_cytodl.csv)
- Parameterize segmentation config for channel selection
- Add 7 L-shaped experiment configs
- Add quick channel validation configs (C=1 vs C=3)"
```

**Note:** Do NOT commit the downloaded .ome.tiff and .tiff data files — only the CSVs and configs. Verify `.gitignore` excludes `data/variance-100x/LMNB1/raw/` and `data/variance-100x/LMNB1/seg/`.
