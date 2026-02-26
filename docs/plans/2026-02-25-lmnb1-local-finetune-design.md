# LMNB1 Local Fine-Tuning Design (Revised)

**Date:** 2026-02-25
**Branch:** `finetune-lmnb1-s3`
**Status:** Approved
**Supersedes:** 2026-02-24-lmnb1-s3-finetune-design.md (S3 approach abandoned due to performance)

## Problem

The S3-based training approach (previous design) is impractical:

- BioImage S3 reads take ~10-15s per file just for metadata
- PersistentDataset caching: ~300MB per sample, first epoch takes 5-6 hours
- Sanity check alone requires ~35 min to cache 40 val files
- Multiple experiments multiply the redundant S3 downloads

**Root cause:** S3 download latency, not a cyto-dl version issue (MonaiBioReader is identical between v0.4.5 and v0.6.2).

**Additional finding:** C=6 (brightfield) is eliminated — some files have only 4 channels, causing IndexError.

## Approach

Pre-download all S3 files to local NFS, create new CSVs with local paths, and reuse the existing `segmentation.yaml` data config (parameterized for channel selection).

### 1. Data Download

- Download all raw `.ome.tiff` and seg `.tiff` files from S3 to local NFS
- Location: `/allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/raw/` and `.../seg/`
- Use `aws s3 cp --no-sign-request` for anonymous access
- ~267 files (160 train + 40 val + 67 test, raw + seg), estimated ~130GB total

### 2. New Local CSVs

- Create: `train_lmnb1_local_cytodl.csv`, `val_lmnb1_local_cytodl.csv`, `test_lmnb1_local_cytodl.csv`
- Same `raw,seg` column format, NFS paths instead of S3 URLs
- Existing `*_s3_cytodl.csv` files remain untouched

### 3. Config Reuse

- Use existing `segmentation.yaml` (no fs_kwargs needed for local files)
- Parameterize source channel: `C: ${raw_channel}` (currently hardcoded `C: 0` in 4 places)
- Target/seg channel stays `C: 0` (single-channel seg masks)
- Keep `segmentation_s3.yaml` for potential future S3 use

### 4. Experiment Plan

**Quick validation (10 epochs each):**
- C=1 (membrane) vs C=3 (structure fluorescence) on `data100_10epochs` config
- Compare train/val loss to confirm correct channel

**Full matrix (7 L-shaped experiments) with winning channel:**

| # | Label | Data % | Max Epochs | subsample |
|---|-------|--------|------------|-----------|
| 1 | data100_converged | 100% | 700 | null |
| 2 | data100_10epochs | 100% | 10 | null |
| 3 | data100_50epochs | 100% | 50 | null |
| 4 | data100_100epochs | 100% | 100 | null |
| 5 | data10_converged | 10% | 700 | 0.1 |
| 6 | data20_converged | 20% | 700 | 0.2 |
| 7 | data50_converged | 50% | 700 | 0.5 |

### 5. Cleanup

- Remove failed S3 run cache dirs (`cache/tmp*`)
- Remove failed S3 output dirs (`outputs/train/runs/lmnb1_channel_test_*`)
