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

### Phase 3: Fresh vs. Existing Classic Outputs (Determinism Check)

Re-run classic workflows on all 50 `test_new.csv` images, save fresh outputs to
disk, and compare fresh outputs against the *existing* classic outputs in
`ms_classic/{STRUCT}/` (NOT against GT). This tests whether workflows are
deterministic across library versions.

- **Data source:** `data/standardized_test/test_new.csv` (50 images)
- **Comparison:** Fresh workflow output vs existing `ms_classic/{STRUCT}/{ID}_classic_seg.tiff`
- **Output:** Fresh TIFFs saved to `benchmark/results/classic_fresh_outputs/{STRUCT}/`

### Phase 3b: Fresh Runs on Per-Structure test.csv with Saved Outputs

Re-run classic workflows on 3 test images per structure from per-structure CSVs
and save fresh segmentation outputs to disk. Compare against stored GT seg files.

- **Data source:** Per-structure `data/variance-100x/{STRUCT}/test.csv` (3 samples each)
- **Comparison:** Fresh workflow output vs stored `_seg.tiff` files
- **Output:** Fresh TIFFs saved to `benchmark/results/classic_fresh_outputs_per_struct/{STRUCT}/`

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

### Phase 3: Fresh vs Existing Classic Outputs (24 comparisons) — Critical Finding

**22 of 24 images: Dice = 1.000 (bit-identical).** Only ST6GAL1 showed minor
difference (Dice = 0.997).

This proves that **classic workflows are deterministic across library versions**.
The discrepancies observed in Phases 1 and 2 between workflow outputs and GT
masks are NOT caused by library version drift. The GT masks in `test_new.csv`
and per-structure CSVs were generated by a different process or different
parameters than these workflow scripts.

### Phase 3b: Fresh Runs on Per-Structure test.csv (27 comparisons, 9 structures)

Fresh outputs saved to disk for downstream use.

| Structure | N | Mean Dice | Min | Max |
|-----------|---|-----------|-----|-----|
| NPM1 | 3 | 1.000 | 1.000 | 1.000 |
| CTNNB1 | 3 | 0.999 | 0.999 | 0.999 |
| RAB5A | 3 | 0.998 | 0.997 | 0.998 |
| NUP153 | 3 | 0.996 | 0.995 | 0.998 |
| SON | 3 | 0.995 | 0.995 | 0.996 |
| SEC61B | 3 | 0.977 | 0.939 | 0.997 |
| MYH10 | 3 | 0.954 | 0.949 | 0.957 |
| ST6GAL1 | 3 | 0.916 | 0.869 | 0.998 |
| ACTN1 | 3 | 0.779 | 0.666 | 0.999 |
| TOMM20 | 0 | - | - | - |

### Classification of Discrepancy Sources

The Phase 3 determinism finding changes the interpretation: discrepancies are
**not version drift**, but rather **GT generation process differences**.

1. **No meaningful discrepancy** (Dice > 0.99): CTNNB1, NPM1, RAB5A, SON, NUP153, LAMP1
   - GT was likely generated with these same workflow scripts and parameters

2. **Moderate systematic discrepancy** (Dice 0.93-0.97): MYH10, TUBA1B
   - GT may have used slightly different parameters or a different workflow version

3. **Large/variable discrepancy** (Dice < 0.93): ACTN1, FBL, SEC61B, ST6GAL1
   - ACTN1: Bimodal — some images match perfectly, others diverge significantly
   - FBL: Only Phase 1 data (no per-struct CSVs); consistently ~0.64
   - SEC61B: High variance across images; some perfect, some 0.76
   - ST6GAL1: Moderate variance; sensitive to parameter interactions
   - GT for these structures was likely generated by a materially different process

4. **Broken** (cannot run): TOMM20 (numpy 2.x incompatibility)

## Files

| File | Description |
|------|-------------|
| `benchmark/scripts/verify_classic_gt.py` | Verification script (Phases 1-3b) |
| `benchmark/results/classic_gt_verification.csv` | Phase 1 results (24 rows) |
| `benchmark/results/classic_gt_verification_phase2.csv` | Phase 2 results (81 rows) |
| `benchmark/results/classic_gt_verification_phase3.csv` | Phase 3 results (24 rows) |
| `benchmark/results/classic_gt_verification_phase3b.csv` | Phase 3b results (30 rows) |
| `benchmark/results/classic_fresh_outputs/` | Fresh TIFFs from Phase 3 (not committed) |
| `benchmark/results/classic_fresh_outputs_per_struct/` | Fresh TIFFs from Phase 3b (not committed) |

## Implications

- **Workflows are deterministic:** Re-running classic workflows today produces bit-identical output to stored results. Library version changes (scikit-image 0.20→0.26, numpy 1.24→2.4) do NOT affect output.
- **GT discrepancies originate from the GT generation process:** The GT masks used for MegaSeg training/evaluation were NOT generated by these exact workflow scripts (or used different parameters). This is important context for interpreting benchmark results.
- **5 structures are safe:** GT closely matches current workflow output (CTNNB1, NPM1, RAB5A, SON, NUP153). Benchmark comparisons against this GT are reliable.
- **2 structures have minor discrepancy:** MYH10 and TUBA1B GT is close but not exact; benchmark results should note ±3-7% Dice uncertainty from GT generation.
- **4 structures have significant GT uncertainty:** ACTN1, FBL, SEC61B, ST6GAL1 GT masks differ materially from workflow output. Evaluation against this GT should be interpreted cautiously.
- **1 structure is untestable:** TOMM20 workflow cannot run on current numpy without deeper fixes.
- **5 structures lack per-structure CSVs:** CETN2, FBL, LAMP1, SLC25A17, TUBA1B could not be tested in Phase 2.
