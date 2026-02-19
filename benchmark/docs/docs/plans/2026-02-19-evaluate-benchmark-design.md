# Evaluate Benchmark Script Design

**Date:** 2026-02-19
**Status:** Approved

## Overview

Single script (`scripts/evaluate_benchmark.py`) with two subcommands that separates expensive metric computation from cheap report generation.

- **`compute`** — reads prediction and GT TIFF pairs, computes 5 metrics (Dice, IoU, Precision, Recall, Accuracy), writes a master per-sample CSV
- **`report`** — reads the per-sample CSV, generates 5 comparison tables

## Architecture: Single Script, Two Phases

Chosen over alternatives (modular package, per-type scripts) for simplicity. The expensive I/O phase produces a cached CSV; the report phase can be re-run instantly without recomputing.

Expected file size: ~500-700 lines.

## Prediction Discovery & GT Matching

Jobs are explicitly enumerated from known constants (same `STRUCTURES` and `BACKBONES` lists as `run_eval.py`), not discovered from the filesystem. Each job is a tuple:

```
(eval_type, dtype, backbone, model_struct, test_struct, test_set)
```

Path resolution rules:

| Eval type | Pred dir | Test CSV | GT source |
|---|---|---|---|
| `individual` | `outputs/.../individual/ms13-{S}/seg/` | `{S}/test/test.csv` | `seg` column |
| `individual_test_new` | `outputs/.../individual_test_new/ms13-{S}/seg/` | `test_new/test_new.csv` | `seg` column |
| `megaseg/{arch}` | `outputs/.../megaseg/{arch}/{S}/seg/` | `{S}/test/test.csv` | `seg` column |
| `megaseg/{arch}/test_new` | `outputs/.../megaseg/{arch}/test_new/seg/` | `test_new/test_new.csv` | `seg` column |
| `cross_eval` | `outputs/.../cross_eval/ms13-{M}/tested_on_{T}/seg/` | `{T}/test/test.csv` | `seg` column |

Prediction-to-GT pairing per CSV row:
1. Extract raw filename stem from the CSV `raw` column path
2. Look for `{stem}.tif` in prediction dir (with `.tiff` fallback)
3. GT path comes directly from the CSV `seg` column

## Performance

Uses `multiprocessing.Pool` with configurable worker count (default: `min(cpu_count(), 16)`). Each worker processes one (prediction, GT) pair independently. Work is I/O-bound (network filesystem reads of 3D TIFF files).

Total scale: ~34,000 prediction-GT pairs across all eval types and structures.

## Output Structure

### Per-sample CSV (`results/{dtype}/per_sample_metrics.csv`)

| Column | Example | Notes |
|---|---|---|
| `eval_type` | `individual` | one of 4 types |
| `dtype` | `uint8` | |
| `backbone` | `dynunet` or empty | only for megaseg |
| `model_struct` | `ACTN1` | structure model was trained on |
| `test_struct` | `ACTN1` | structure being tested on |
| `test_set` | `standard` or `test_new` | which test dataset |
| `sample_id` | `ACTN1_10917` | filename stem |
| `dice` | `0.8234` | |
| `iou` | `0.6995` | |
| `precision` | `0.8501` | |
| `recall` | `0.7982` | |
| `accuracy` | `0.9912` | |

### Report outputs (also in `results/{dtype}/`)

1. **`summary_by_structure.csv`** — mean +/- std per (eval_type, backbone, structure)
2. **`comparison_individual_vs_megaseg.csv`** — per-structure: individual vs each megaseg backbone, with deltas
3. **`comparison_backbones.csv`** — all 4 megaseg backbones compared per structure
4. **`cross_eval_heatmap.csv`** — 14x14 matrix (rows=model, cols=test), values=mean metric
5. **`test_new_comparison.csv`** — individual vs all megaseg backbones on held-out test_new set

All tables also printed to terminal during the `report` phase.

## CLI Interface

```bash
# Compute metrics (with optional filters)
python scripts/evaluate_benchmark.py compute \
    --project-root /allen/aics/rep_learn/megaseg-benchmark \
    --dtype uint8 \
    --workers 16 \
    [--eval-type individual]
    [--backbone dynunet]
    [--structure ACTN1]

# Generate reports (with optional metric selection)
python scripts/evaluate_benchmark.py report \
    --project-root /allen/aics/rep_learn/megaseg-benchmark \
    --dtype uint8 \
    [--metric dice]
```

- `--project-root` defaults to `$PROJECT_ROOT` env var or cwd
- `compute` is incremental — loads existing CSV, skips computed jobs, appends new results
- `--metric` on `report` selects which metric for comparison tables (default: dice)

## Error Handling

1. **Missing prediction files** — warn, record NaN metrics, continue. Sample counts in reports reveal missing data.
2. **Empty masks** — both empty = 1.0 (agreement), one empty = 0.0 (disagreement). Matches reference script.
3. **test_new structure extraction** — structure name parsed from filename prefix. Includes out-of-distribution structures (CETN2, ACTB, etc.).
4. **File extension fallback** — try `.tif`, fall back to `.tiff`.
5. **Incremental compute** — match on `(eval_type, dtype, backbone, model_struct, test_struct, sample_id)` to skip existing rows.

## Dependencies

- `numpy`, `pandas`, `tifffile` (already in `acm-eval` conda env)
- `tqdm` for progress bars
- Standard library: `argparse`, `multiprocessing`, `pathlib`, `logging`

## Key functions to reuse from reference script

From `/allen/aics/lumenoid/inner_outer_seg/megaseg_exp/evaluate_segmentation.py`:
- `compute_metrics(gt, pred)` — Dice, IoU, Precision, Recall, Accuracy
- `binarize_prediction(pred, model_type)` — uint8 >= 128, float32 >= 0.5
- `binarize_gt(gt)` — GT mask > 0
- `derive_prediction_path()` — .tif/.tiff extension fallback logic
