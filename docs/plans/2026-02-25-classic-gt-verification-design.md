# Classic GT Verification Design

**Date:** 2026-02-25
**Branch:** `verify-classic-gt`
**Issue:** [#3 — Verify classic segmentation GT reproducibility](https://github.com/AllenCell/MegaSeg/issues/3)

## Motivation

MegaSeg's ground truth (GT) segmentation masks were generated 2-3 years ago using
classic `aicssegmentation` workflows. Since then, key dependencies have been
upgraded: scikit-image 0.20 -> 0.26, numpy 1.24 -> 2.4, scipy 1.10 -> 1.17.

Before relying on these GT masks for model evaluation, we need to verify that
re-running the same classic workflows today produces equivalent outputs, or
quantify any discrepancies introduced by library version changes.

## Approach

### Phase 1: Existing Outputs vs. test_new.csv GT

Compare the 13 structures that have pre-computed classic workflow outputs in
`ms_classic/{STRUCT}/` against GT masks from `test_new.csv`. This phase requires
no workflow re-execution — it only loads and compares existing files.

- **Data source:** `data/standardized_test/test_new.csv` (50 images, 25 structures, 2 per structure)
- **Comparison:** `ms_classic/{STRUCT}/{ID}_classic_seg.tiff` vs `{ID}_GT.tiff`
- **Metric:** Dice coefficient (binarized with `arr > 0`)

### Phase 2: Fresh Workflow Runs on Per-Structure CSVs

Re-run classic workflows on a sample of images from per-structure CSVs in
`data/variance-100x/` (3 images per split: train, val, test = 9 per structure)
and compare fresh segmentation output against stored GT seg masks.

- **Data source:** Per-structure CSVs at `data/variance-100x/{STRUCT}/{split}.csv`
- **Workflow source:** `ms_classic/classic_segmentation_{STRUCT}.py` (dynamically imported)
- **Available structures:** 10 of 17 have per-structure CSVs

### Source Patching for Library Compatibility

Four classic workflow scripts have breaking incompatibilities with current library
versions. The verification script applies source-level patches at import time:

| Script | Issue | Patch |
|--------|-------|-------|
| LMNB1 | `from skimage.morphology import watershed` removed in 0.26 | Redirect to `skimage.segmentation.watershed` |
| CETN2 | `peak_local_max` returns Nx3 coords instead of boolean mask | Wrapper converts coords to boolean mask |
| SLC25A17 | Same `peak_local_max` change as CETN2 | Same wrapper |
| TOMM20 | `main()` broken; `numpy.ndarray.newbyteorder` removed in numpy 2.x | Import only the `Workflow_tomm20` function (still fails at import due to top-level numpy call) |

### GT Value Encoding

Two different encodings exist across data sources — the script normalizes both:

- `test_new.csv` GT (`_GT.tiff`): uint8, values {0, 1}
- Per-structure seg (`_seg.tiff`): uint8, values {0, 255}
- Classic workflow outputs: uint8, values {0, 255}
- Normalization: `(arr > 0)` binarization before Dice computation

## Results Summary

### Phase 1: Existing Outputs vs test_new.csv GT (24 comparisons)

| Tier | Structures | Dice Range |
|------|-----------|------------|
| Near-perfect (>0.99) | NPM1 (~1.00), CTNNB1 (0.999), LAMP1 (0.999), NUP153 (0.999), RAB5A (0.998) | 0.997-1.000 |
| Moderate drift (0.93-0.97) | SON (0.995), TUBA1B (0.927-1.00), MYH10 (0.960), ST6GAL1 (0.937), SEC61B (0.930-0.936) | 0.926-0.995 |
| Large discrepancy (<0.70) | ACTN1 (0.600-0.699), FBL (0.609-0.674) | 0.600-0.699 |

SEC61B_DUAL had no matching test_new.csv images.

### Phase 2: Fresh Runs on Per-Structure CSVs (81 comparisons, 9 of 10 structures)

| Structure | N | Mean Dice | Min | Max | Std | Notes |
|-----------|---|-----------|-----|-----|-----|-------|
| CTNNB1 | 9 | 0.999 | 0.999 | 1.000 | 0.000 | Perfect match |
| NPM1 | 9 | 0.997 | 0.977 | 1.000 | 0.008 | Near-perfect (1 outlier at 0.977) |
| RAB5A | 9 | 0.998 | 0.997 | 0.998 | 0.001 | Near-perfect |
| SON | 9 | 0.995 | 0.995 | 0.996 | 0.001 | Near-perfect |
| NUP153 | 9 | 0.995 | 0.987 | 0.998 | 0.003 | Near-perfect |
| MYH10 | 9 | 0.961 | 0.949 | 0.976 | 0.009 | Consistent moderate drift |
| SEC61B | 9 | 0.925 | 0.758 | 0.997 | 0.079 | Highly variable |
| ST6GAL1 | 9 | 0.905 | 0.867 | 0.999 | 0.054 | Variable |
| ACTN1 | 9 | 0.889 | 0.666 | 0.999 | 0.165 | Bimodal: most ~0.998, 3 of 9 ~0.67 |
| TOMM20 | - | - | - | - | - | Failed: numpy 2.x `newbyteorder` removal |

**Overall:** 81 images, mean Dice 0.963, min 0.666, max 1.000.

### Classification of Discrepancy Sources

1. **No meaningful drift** (Dice > 0.99): CTNNB1, NPM1, RAB5A, SON, NUP153, LAMP1
   - Simple threshold/morphology pipelines — deterministic across versions

2. **Moderate systematic drift** (Dice 0.93-0.97): MYH10, TUBA1B
   - Likely from floating-point changes in Gaussian filtering or morphological operations

3. **Large/variable drift** (Dice < 0.93): ACTN1, FBL, SEC61B, ST6GAL1
   - ACTN1: Bimodal — `vesselness3D` + `edge_preserving_smoothing_3d` are version-sensitive
   - FBL: Only Phase 1 data (no per-struct CSVs); consistently ~0.64
   - SEC61B: High variance across images; some perfect, some 0.76
   - ST6GAL1: Moderate variance; sensitive to parameter interactions

4. **Broken** (cannot run): TOMM20 (numpy 2.x incompatibility)

## Files

| File | Description |
|------|-------------|
| `benchmark/scripts/verify_classic_gt.py` | Verification script (Phase 1 + Phase 2) |
| `benchmark/results/classic_gt_verification.csv` | Phase 1 results (24 rows) |
| `benchmark/results/classic_gt_verification_phase2.csv` | Phase 2 results (81 rows) |

## Implications

- **5 structures are safe:** Their GT can be trusted as equivalent to current workflow output (CTNNB1, NPM1, RAB5A, SON, NUP153).
- **2 structures have minor drift:** MYH10 and TUBA1B GT is close but not exact; benchmark results should note ±3-7% Dice uncertainty from GT generation.
- **4 structures have significant GT uncertainty:** ACTN1, FBL, SEC61B, ST6GAL1 GT masks may differ materially from what current workflows produce. Evaluation against this GT should be interpreted cautiously.
- **1 structure is untestable:** TOMM20 workflow cannot run on current numpy without deeper fixes.
- **5 structures lack per-structure CSVs:** CETN2, FBL, LAMP1, SLC25A17, TUBA1B could not be tested in Phase 2.
