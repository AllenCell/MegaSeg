# LMNB1 S3 Fine-Tuning Re-Run Design

**Date:** 2026-02-24
**Branch:** `finetune-lmnb1-s3` (off `fix-expanduser-url-stripping`)
**Status:** Approved

## Problem

The original LMNB1 fine-tuning produced terrible Dice scores (0.01-0.08) because `MonaiBioReader` used `C: 0`, which selects a blank/noise channel from the 7-channel `.ome.tiff` files. The original 13-structure MegaSeg model trained on pre-extracted single-channel TIFFs where `C: 0` correctly returned the only channel. For LMNB1's multi-channel files, we need to select the correct channel explicitly.

### Channel layout (7-channel .ome.tiff)

| Index | Channel | Signal Level |
|-------|---------|-------------|
| 0 | blank | ~413 mean (noise) |
| 1 | membrane | ~952 mean |
| 2 | blank | ~402 mean |
| 3 | structure (LMNB1/lamin) | ~540 mean |
| 4 | blank | ~400 mean |
| 5 | nuclear dye | ~430 mean |
| 6 | brightfield | ~43921 mean |

## Approach

Create a new parameterized data config (`segmentation_s3.yaml`) where the source channel `C` is a variable (`${raw_channel}`), paired with experiment YAMLs that set this variable. This avoids modifying the shared `segmentation.yaml` used by 100+ eval configs.

## Two-Phase Execution

### Phase 1: Channel Test

Fine-tune from the base MegaSeg checkpoint with three channel candidates (C=1 membrane, C=3 structure, C=6 brightfield) on 100% data to convergence. Compare val loss and early Dice metrics to identify the correct channel. Early stopping (patience 100) lets bad channels fail fast.

### Phase 2: L-Shaped Experiment Grid

Once the correct channel is known, run all 7 experiment variants:

| # | Label | Data % | Max Epochs |
|---|-------|--------|------------|
| 1 | data100_converged | 100% | 700 |
| 2 | data100_10epochs | 100% | 10 |
| 3 | data100_50epochs | 100% | 50 |
| 4 | data100_100epochs | 100% | 100 |
| 5 | data10_converged | 10% | 700 |
| 6 | data20_converged | 20% | 700 |
| 7 | data50_converged | 50% | 700 |

All fine-tune from the base checkpoint (`weights_only: true`), not from the old (broken) fine-tune checkpoints.

## File Layout

```
benchmark/
├── configs/
│   ├── train.yaml                                  # NEW top-level train entry point
│   ├── data/local/
│   │   └── segmentation_s3.yaml                    # NEW parameterized data config
│   └── experiment/train/finetune/
│       ├── lmnb1_channel_test_c1.yaml              # Phase 1
│       ├── lmnb1_channel_test_c3.yaml              # Phase 1
│       ├── lmnb1_channel_test_c6.yaml              # Phase 1
│       ├── lmnb1_data100_converged.yaml            # Phase 2 (added after channel test)
│       ├── lmnb1_data100_10epochs.yaml             # Phase 2
│       ├── lmnb1_data100_50epochs.yaml             # Phase 2
│       ├── lmnb1_data100_100epochs.yaml            # Phase 2
│       ├── lmnb1_data10_converged.yaml             # Phase 2
│       ├── lmnb1_data20_converged.yaml             # Phase 2
│       └── lmnb1_data50_converged.yaml             # Phase 2
└── scripts/
    └── run_finetune.sh                             # Launcher script
```

## Config Details

### `segmentation_s3.yaml`

Copy of `segmentation.yaml` with:
- Source reader `C: 0` changed to `C: ${raw_channel}` (4 places, one per split)
- Target reader `C: 0` unchanged (seg files are single-channel)
- `expanduser: false` retained on all `LoadImaged` entries

### Experiment Config Template

```yaml
# @package _global_
defaults:
  - override /data: local/segmentation_s3.yaml
  - override /model: local/segmentation_ms25.yaml
  - override /callbacks: default.yaml
  - override /trainer: ddp.yaml
  - override /logger: csv.yaml

tags: ["finetune", "lmnb1"]
seed: 12345
raw_channel: 3                    # set per experiment

checkpoint:
  ckpt_path: <base MegaSeg epoch_650.ckpt>
  weights_only: true
  strict: true

experiment_name: <variant name>
run_name: <variant name>
source_col: raw
target_col: seg
spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 700                 # varies per experiment
  devices: [0]

data:
  path: /allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/
  cache_dir: /allen/aics/rep_learn/MegaSeg/benchmark/cache/
  subsample:                      # null for 100%, 0.1/0.2/0.5 for subsets
  batch_size: 32
  num_workers: 12
  _aux:
    patch_shape: [32, 128, 128]
```

### Training Entry Point

```bash
CYTODL_CONFIG_PATH=/allen/aics/rep_learn/MegaSeg/benchmark/configs \
  python -m cyto_dl.train experiment=train/finetune/lmnb1_channel_test_c3
```

### `train.yaml`

Mirrors cyto-dl's default `train.yaml` with the same defaults structure (data, model, callbacks, logger, trainer, paths, extras, hydra, experiment).

## Key Decisions

1. **`weights_only: true`** — resets optimizer/scheduler/epoch for clean fine-tuning
2. **Base checkpoint only** — old fine-tune checkpoints trained on wrong channel are discarded
3. **Same hyperparameters** — batch_size 32, lr 1e-4, patch [32,128,128], Adam, ExponentialLR gamma 0.995
4. **S3 paths directly** — no pre-download; `bioio` + `fsspec`/`s3fs` handle S3 natively
5. **Separate branch** — `finetune-lmnb1-s3` keeps this work isolated from PR #13
