# megaseg-benchmark

Cross-structure segmentation evaluation benchmark comparing MegaSeg (multi-structure) models against individual (specialized) models across 14 biological structures. Also includes LMNB1 fine-tuning and from-scratch training experiments, classic segmentation GT verification, and drug perturbation evaluation.

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules git@github.com:AllenCell/MegaSeg.git
cd MegaSeg/benchmark

# 2. Install dependencies (one-time)
uv sync

# 3. Run a single evaluation
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1
```

## Directory Structure

```
MegaSeg/
├── data/
│   ├── variance-100x/              # Per-structure CSVs (at repo root)
│   │   ├── ACTN1/test.csv          # 14 benchmark structures + LMNB1
│   │   ├── ...
│   │   └── LMNB1/                  # Used for fine-tuning experiments
│   ├── standardized_test/          # Held-out test_new dataset
│   │   └── test_new.csv
│   ├── caroline_data/              # NPM1 Airyscan image manifest
│   └── new_data/                   # Additional TIFF pairs (5 structures, 125 pairs)
└── benchmark/
    ├── checkpoints/
    │   ├── individual/             # 14 single-structure DynUNet checkpoints
    │   ├── megaseg/                # 4 multi-structure backbone checkpoints
    │   └── README.md               # Checkpoint provenance mapping
    ├── configs/
    │   ├── paths/default.yaml      # Portable path config (uses PROJECT_ROOT)
    │   ├── data/local/             # Data loading configs
    │   │   ├── segmentation.yaml
    │   │   ├── segmentation_finetune.yaml   # Fine-tune data config (channel_name support)
    │   │   └── segmentation_s3.yaml         # S3 data config
    │   ├── model/local/            # Model architecture configs (8 variants)
    │   ├── callbacks/              # Callback configs (default, eval)
    │   ├── trainer/                # Trainer configs (gpu, ddp)
    │   ├── logger/                 # Logger config
    │   ├── hydra/                  # Hydra config
    │   ├── extras/                 # Extra utilities config
    │   ├── train.yaml              # Top-level train entry point
    │   ├── eval.yaml               # Top-level eval entry point
    │   └── experiment/
    │       ├── train/
    │       │   ├── finetune/       # 20 LMNB1 fine-tuning configs
    │       │   └── scratch/        # 16 LMNB1 from-scratch configs
    │       ├── eval/
    │       │   ├── finetune/       # 17 LMNB1 fine-tune eval configs
    │       │   ├── scratch/        # 16 LMNB1 scratch eval configs
    │       │   ├── float32/        # 102 float32 benchmark eval configs
    │       │   │   ├── individual/          # 14 configs
    │       │   │   ├── individual_test_new/ # 14 configs
    │       │   │   ├── megaseg/             # 60 configs (4 backbones x 15)
    │       │   │   └── cross_eval/          # 14 parameterized configs
    │       │   └── uint8/          # 112 uint8 benchmark eval configs
    │       │       ├── individual/          # 14 + drug_perturbation/ (4)
    │       │       ├── individual_test_new/ # 14 configs
    │       │       ├── megaseg/             # 60 + drug_perturbation/ (6)
    │       │       └── cross_eval/          # 14 parameterized configs
    │       └── finetune/lmnb1/     # 7 legacy fine-tune predict configs
    ├── cyto-dl/                    # Git submodule (AllenCell/cyto-dl)
    ├── results/
    │   ├── uint8/                  # Core benchmark metrics and figures
    │   │   ├── per_sample_metrics.csv
    │   │   ├── megaseg_dynunet_per_image_dice.csv
    │   │   ├── figures/            # 5 metrics x 6 plot types (PNG + PDF)
    │   │   └── filtered/           # Filtered results (bottom-10 removed)
    │   │       ├── excluded_samples.csv
    │   │       ├── per_sample_metrics.csv
    │   │       └── figures/
    │   ├── finetune/lmnb1/         # LMNB1 fine-tuning results
    │   │   ├── per_sample_metrics.csv
    │   │   ├── summary.csv
    │   │   └── figures/            # Dice heatmap + per-experiment comparisons
    │   ├── scratch/lmnb1/          # LMNB1 from-scratch results
    │   │   ├── per_sample_metrics.csv
    │   │   ├── summary.csv
    │   │   └── figures/
    │   ├── comparison/lmnb1/       # Side-by-side fine-tune vs scratch
    │   │   └── figures/            # delta_pretraining_benefit, side_by_side
    │   ├── drug_perturbation/      # Drug perturbation figures
    │   │   └── figures/            # 6 per-structure PDFs/PNGs + overview grid
    │   ├── classic_gt_verification.csv          # Phase 1 (24 images)
    │   ├── classic_gt_verification_phase2.csv   # Phase 2 (81 images)
    │   ├── classic_gt_verification_phase3.csv   # Phase 3 (24 images)
    │   ├── classic_gt_verification_phase3b.csv  # Phase 3b (30 images)
    │   ├── classic_fresh_outputs/               # Fresh classic workflow outputs
    │   └── classic_fresh_outputs_per_struct/     # Per-structure fresh outputs
    ├── scripts/
    │   ├── generate_configs.py         # Config generator (benchmark eval configs)
    │   ├── evaluate_benchmark.py       # Metrics computation, reporting, plotting, filtering
    │   ├── run_eval.py                 # Batch runner for individual/megaseg evals
    │   ├── run_cross_eval.py           # Batch runner for cross-eval jobs
    │   ├── experiment_matrix.csv       # Job definitions for cross-eval
    │   ├── test_evaluate_benchmark.py  # Tests (32 tests)
    │   ├── compute_lmnb1_dice.py       # Dice computation for LMNB1 experiments
    │   ├── visualize_finetune.py       # 4x4 Dice heatmap + GT/pred figures
    │   ├── visualize_comparison.py     # Side-by-side + delta heatmaps
    │   ├── run_finetune.py             # Batch predict runner for LMNB1 models
    │   ├── run_finetune.sh             # Fine-tuning launch script
    │   ├── evaluate_finetune.py        # Fine-tuning metrics pipeline
    │   ├── generate_lmnb1_data.py      # Copy LMNB1 data CSVs into benchmark layout
    │   ├── classic_workflows.py        # Shared classic segmentation module
    │   ├── run_classic.py              # Classic workflow pipeline CLI
    │   ├── verify_classic_gt.py        # GT reproducibility verification
    │   ├── run_classic_drug_perturbation.py  # Classic workflows on drug FOVs
    │   ├── visualize_drug_perturbation.py    # Drug perturbation comparison figures
    │   ├── sota_comparison.py          # SOTA method comparison aggregation
    │   ├── run_cellpose_zeroshot.py    # CellPose zero-shot evaluation
    │   ├── run_microsam_zeroshot.py    # SAM zero-shot evaluation
    │   ├── run_stardist_zeroshot.py    # StarDist zero-shot evaluation
    │   ├── run_otsu_baseline.py        # Otsu thresholding baseline
    │   ├── convert_to_nnunet.py        # Convert data to nnU-Net format
    │   ├── evaluate_nnunet.py          # Evaluate nnU-Net predictions
    │   └── rename_lmnb1_files.sh       # Strip S3 hash prefixes from filenames
    ├── outputs/                    # Evaluation outputs (gitignored)
    ├── cache/                      # Data cache (gitignored)
    └── pyproject.toml
```

## Evaluation Types

### 1. Individual Eval (14 configs)

Each individual (specialized) DynUNet model evaluated on its own structure's test set.

```bash
cd benchmark
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1
```

### 2. Individual Test-New (14 configs)

Each individual model evaluated on the held-out test_new dataset.

```bash
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual_test_new/ACTN1
```

### 3. MegaSeg Backbone Eval (60 configs)

Each of 4 multi-structure backbones evaluated on each of the 14 structures + test_new.

```bash
# DynUNet megaseg on ACTN1 test set
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/megaseg/dynunet/ACTN1

# Swin UNETR megaseg on test_new
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/megaseg/swin_unetr/test_new
```

### 4. Cross-Eval (14 parameterized configs, 196 jobs per dtype)

Each individual model evaluated on every other structure's test set. Uses a parameterized `test_struct` variable.

```bash
# ACTN1 model evaluated on ATP2A2 test set
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/cross_eval/ACTN1_model/cross_eval \
  test_struct=ATP2A2
```

### Batch Cross-Eval Runner

```bash
# Preview all 196 jobs
uv run python scripts/run_cross_eval.py --dry-run --gpus 0,1,2,3

# Launch all pending
uv run python scripts/run_cross_eval.py --gpus 0,1,2,3

# Check status
uv run python scripts/run_cross_eval.py --status
```

## Quantitative Evaluation

Compute metrics (Dice, IoU, Precision, Recall, Accuracy) across all completed evaluations:

```bash
cd benchmark

# Compute metrics for all uint8 evaluations
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py compute --dtype uint8 --workers 16

# Print summary report
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py report --dtype uint8 --metric dice

# Generate all plots (bar charts, heatmaps, radar, violin, rank, scatter, clustermap)
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py plot --dtype uint8
```

Results are saved to `results/{dtype}/` with CSVs and a `figures/` subdirectory.

**Note on plots:** NPM1 is excluded from the active benchmark pipeline. The current `STRUCTURES` list in `evaluate_benchmark.py` contains 14 structures total (AAVS1 included, NPM1 excluded). Plots use the `viridis` colormap for perceptual uniformity and colorblind accessibility. A hierarchically clustered clustermap (Ward linkage) is also generated for the cross-eval matrix.

### Per-Image Metrics Manifest

A standalone CSV with per-image Dice, IoU, precision, recall, and accuracy for all test images evaluated by the megaseg-dynunet model is available at:

```
results/uint8/megaseg_dynunet_per_image_dice.csv
```

This covers the 13 MegaSeg benchmark structures across 57 test images each (741 total, bottom-10 low-signal samples removed per structure) and is useful for sharing with collaborators without requiring the full pipeline.

## Low-Signal Sample Filtering

The `filter` subcommand in `evaluate_benchmark.py` identifies and removes the bottom-N lowest-scoring samples per structure, then regenerates all reports and plots on the filtered dataset.

**How it works:**
- For each active benchmark structure in the MegaSeg reference set, the bottom N samples by Dice (from the megaseg/dynunet/standard reference) are identified
- These samples are removed across **all** eval types (individual, megaseg, cross_eval) for consistency
- An `excluded_samples.csv` manifest is saved alongside the regenerated metrics and plots

**Usage:**

```bash
# Remove bottom 10 samples per structure (140 total)
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py filter --dtype uint8 --bottom-n 10

# Customize reference metric and backbone
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py filter \
  --dtype uint8 --bottom-n 10 --metric dice --reference-backbone dynunet
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--bottom-n` | `10` | Number of lowest-scoring samples to exclude per structure |
| `--metric` | `dice` | Metric used to rank samples |
| `--reference-backbone` | `dynunet` | MegaSeg backbone used as scoring reference |
| `--plot-metric` | `dice` | Metric used in regenerated plots |

**Output:** All filtered artifacts are saved to `results/{dtype}/filtered/`:

| File | Description |
|------|-------------|
| `excluded_samples.csv` | Manifest of 140 excluded samples with their Dice scores |
| `per_sample_metrics.csv` | Filtered version of the main metrics CSV |
| Summary CSVs | Comparison tables, heatmap, etc. (regenerated) |
| `figures/` | Regenerated plots for all 5 metrics |

With `--bottom-n 10`, the default filtering removes 140 samples total, reducing from 18,050 to 15,390 metric rows (breakdown: -140 individual, -560 megaseg across 4 backbones, -1960 cross_eval across 14 model structures).

## LMNB1 Fine-Tuning Experiments

A complete pipeline for LMNB1 (Lamin B1) fine-tuning, measuring how data fraction and training duration affect segmentation quality when fine-tuning a pretrained MegaSeg model.

### Experiment Design

A 4x4 grid varying **data fraction** (10%, 20%, 50%, 100%) and **max training epochs** (10, 50, 100, converged/700), plus a baseline (no fine-tuning). Total: 16 fine-tuned models + 1 baseline = 17 experiments.

### Results (Mean Dice, n=57 test samples)

|  | 10 epochs | 50 epochs | 100 epochs | Converged |
|---|---|---|---|---|
| **100% data** | 0.906 | 0.944 | 0.948 | **0.957** |
| **50% data** | 0.877 | 0.931 | 0.944 | 0.951 |
| **20% data** | 0.703 | 0.897 | 0.920 | 0.944 |
| **10% data** | 0.699 | 0.893 | 0.913 | 0.941 |

Baseline (no fine-tuning): **0.312**

**Key findings:**
- Fine-tuning improves Dice from 0.312 to 0.906-0.957
- Training epochs dominate over data fraction: the biggest jump is 10 to 50 epochs regardless of data size
- Even 10% of data (16 samples) trained to convergence reaches 0.941, only 0.016 below the best result
- Diminishing returns beyond 50 epochs for all data fractions (50 to converged adds ~0.03-0.05)

### Training

Training configs are in `configs/experiment/train/finetune/` (16 grid configs + 3 channel test configs + 1 sanity check).

```bash
cd benchmark

# Run a single fine-tuning experiment
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD \
uv run python -m cyto_dl.train \
  experiment=train/finetune/lmnb1_data100_converged \
  trainer.devices=[0]

# Or use the batch launcher
bash scripts/run_finetune.sh
```

### Evaluation

Eval configs are in `configs/experiment/eval/finetune/` (16 fine-tuned + 1 baseline = 17 configs).

```bash
# Run a single fine-tune eval
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/finetune/lmnb1_data100_converged

# Compute Dice scores for all experiments
uv run python scripts/compute_lmnb1_dice.py --mode finetune

# Generate 4x4 Dice heatmap and per-experiment comparison figures
uv run python scripts/visualize_finetune.py --mode finetune
```

### Technical Detail: channel_name

LMNB1 `.ome.tiff` files have inconsistent channel layouts: 4-channel files have the structure (EGFP) at C=1, while 7-channel files have it at C=3. The data config uses `channel_name: EGFP` to dynamically resolve the correct channel index per file via `MonaiBioReader`, rather than hardcoding a channel index. This `channel_name` parameter was added to the cyto-dl submodule's `MonaiBioReader`.

### Output Files

| Path | Description |
|------|-------------|
| `results/finetune/lmnb1/per_sample_metrics.csv` | Per-sample Dice for all 17 experiments (1,139 rows) |
| `results/finetune/lmnb1/summary.csv` | Summary table with mean Dice per experiment (17 rows) |
| `results/finetune/lmnb1/figures/dice_heatmap.{pdf,png}` | 4x4 Dice heatmap |
| `results/finetune/lmnb1/figures/{experiment}.{pdf,png}` | GT vs prediction comparison per experiment |

## LMNB1 From-Scratch Training

Companion to the fine-tuning experiments: trains the same DynUNet architecture from random initialization (no MegaSeg pretrained checkpoint) to quantify the pretraining benefit.

### From-Scratch Results (Mean Dice, n=57)

|  | 10 epochs | 50 epochs | 100 epochs | Converged |
|---|---|---|---|---|
| **100% data** | 0.528 | 0.836 | 0.893 | 0.934 |
| **50% data** | 0.258 | 0.783 | 0.851 | 0.899 |
| **20% data** | 0.096 | 0.466 | 0.696 | 0.828 |
| **10% data** | 0.101 | 0.461 | 0.691 | 0.801 |

### Pretraining Benefit (Fine-Tune Dice minus Scratch Dice)

|  | 10 epochs | 50 epochs | 100 epochs | Converged |
|---|---|---|---|---|
| **100% data** | +0.377 | +0.108 | +0.055 | +0.023 |
| **50% data** | +0.619 | +0.147 | +0.093 | +0.052 |
| **20% data** | +0.608 | +0.432 | +0.224 | +0.116 |
| **10% data** | +0.597 | +0.432 | +0.222 | +0.140 |

**Key takeaway:** MegaSeg pretraining provides +0.02 to +0.62 Dice improvement. The benefit is largest with limited data and training budget (bottom-left of the grid) and smallest when both data and compute are abundant (top-right). Even at convergence with 100% data, fine-tuning still outperforms scratch by +0.023 Dice.

### Running From-Scratch Experiments

Training configs: `configs/experiment/train/scratch/` (16 configs).
Eval configs: `configs/experiment/eval/scratch/` (16 configs).

```bash
cd benchmark

# Train from scratch
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD \
uv run python -m cyto_dl.train \
  experiment=train/scratch/lmnb1_data100_converged \
  trainer.devices=[0]

# Compute Dice (scratch mode)
uv run python scripts/compute_lmnb1_dice.py --mode scratch

# Generate scratch heatmap
uv run python scripts/visualize_finetune.py --mode scratch

# Generate comparison figures (side-by-side + delta)
uv run python scripts/visualize_comparison.py
```

### Output Files

| Path | Description |
|------|-------------|
| `results/scratch/lmnb1/summary.csv` | Summary Dice table (16 rows) |
| `results/scratch/lmnb1/figures/dice_heatmap.{pdf,png}` | 4x4 scratch Dice heatmap |
| `results/comparison/lmnb1/figures/side_by_side.{pdf,png}` | Fine-tune vs scratch heatmaps |
| `results/comparison/lmnb1/figures/delta_pretraining_benefit.{pdf,png}` | Pretraining benefit heatmap |

## Classic Segmentation GT Verification

Verifies whether classic `aicssegmentation` workflows reproduce the ground truth (GT) masks used for MegaSeg training and evaluation, given library version upgrades (skimage 0.20 to 0.26, numpy 1.24 to 2.4, scipy 1.10 to 1.17).

### Critical Finding

**Classic workflows are deterministic across library versions.** Phase 3 re-ran workflows on all test_new.csv images and compared fresh output against existing stored outputs: 22 of 24 images produced bit-identical results (Dice = 1.000). Only ST6GAL1 showed a minor difference (0.997).

The discrepancies between workflow output and GT masks are **not caused by library version drift**. The GT masks were generated by a different process than these workflow scripts: per-structure CSVs use AICS production pipeline segmentations (read from `struct_seg_path` in `allcells_beta.csv`), not classic workflow outputs.

### Structure Reliability Tiers

| Tier | Structures | Dice Range | Implication |
|------|-----------|------------|-------------|
| Near-perfect (>0.99) | CTNNB1, NPM1, RAB5A, SON, NUP153, LAMP1, TOMM20 | 0.987-1.000 | GT is reliable |
| Moderate discrepancy (~0.95) | MYH10, TUBA1B | 0.927-0.976 | GT usable with ~3-7% Dice uncertainty |
| Significant discrepancy | ACTN1, FBL, SEC61B, ST6GAL1 | 0.600-0.997 | GT differs materially from workflow output |

### Source Patches

The shared module (`classic_workflows.py`) applies import-time source patches for 4 broken workflows under modern library versions:

| Structure | Issue | Patch |
|-----------|-------|-------|
| LMNB1 | `watershed` moved from `skimage.morphology` to `skimage.segmentation` | Import redirect |
| CETN2 | `peak_local_max` returns coordinates instead of boolean mask | Adapter function |
| SLC25A17 | Same `peak_local_max` change as CETN2 | Adapter function |
| TOMM20 | Module-level notebook code triggers numpy 2.x incompatibility | AST-based source stripping |

### Classic Pipeline CLI (`run_classic.py`)

Runs classic segmentation workflows on arbitrary input data:

```bash
PYTHON=/path/to/msseg/env/bin/python

# List available structures (17 total)
$PYTHON benchmark/scripts/run_classic.py --list

# Run specific structures on a CSV with 'raw' column
$PYTHON benchmark/scripts/run_classic.py \
    --structures ACTN1 NPM1 --input data.csv --output-dir ./out/

# Run all structures on a directory of TIFFs
$PYTHON benchmark/scripts/run_classic.py \
    --input /path/to/tiffs/ --output-dir ./out/

# Preview what would run
$PYTHON benchmark/scripts/run_classic.py \
    --input data.csv --output-dir ./out/ --dry-run
```

**Note:** Classic workflows require the `msseg` conda environment with `aicssegmentation` and its dependencies. This is separate from the benchmark `uv` environment.

### Verification Phases

| Phase | What it tests | Images | Result |
|-------|--------------|--------|--------|
| Phase 1 | Existing classic outputs vs test_new GT | 24 | Identified GT provenance discrepancies |
| Phase 2 | Fresh workflow runs on per-structure CSVs | 81 (9 structures x 9) | Established per-structure Dice baselines |
| Phase 3 | Fresh vs existing classic outputs (determinism) | 24 | 22/24 bit-identical (Dice = 1.000) |
| Phase 3b | Fresh runs with saved outputs on per-structure test CSVs | 30 (10 structures x 3) | Confirmed determinism, includes TOMM20 fix |

### Output Files

| Path | Description |
|------|-------------|
| `results/classic_gt_verification.csv` | Phase 1 results (24 rows) |
| `results/classic_gt_verification_phase2.csv` | Phase 2 results (81 rows) |
| `results/classic_gt_verification_phase3.csv` | Phase 3 results (24 rows) |
| `results/classic_gt_verification_phase3b.csv` | Phase 3b results (30 rows) |
| `results/classic_fresh_outputs/` | Fresh workflow outputs for test_new images |
| `results/classic_fresh_outputs_per_struct/` | Fresh outputs for per-structure test images |

## Drug Perturbation Evaluation

Qualitative evaluation of segmentation on **108 drug-perturbed cell FOVs** (6 structures, 10 drug-structure combinations) across three model types. No ground truth segmentation exists for these FOVs, so this is a visual comparison only.

### Models Evaluated

| Model | Structures | Notes |
|-------|-----------|-------|
| **MegaSeg** (dynunet) | ACTB, NUP153, ST6GAL1, TJP1, TOMM20, TUBA1B | Single checkpoint for all |
| **Individual** | NUP153, ST6GAL1, TJP1, TOMM20 | No ACTB/TUBA1B checkpoints available |
| **Classic** (aicssegmentation) | ACTB, NUP153, ST6GAL1, TOMM20, TUBA1B | No TJP1 workflow exists |

### Drug Conditions

| Structure | Drugs | FOVs |
|-----------|-------|------|
| ACTB | Jasplakinolide, Latrunculin A | 15 |
| NUP153 | Thapsigargin | 12 |
| ST6GAL1 | Brefeldin A, Monensin | 21 |
| TJP1 | Staurosporine | 6 |
| TOMM20 | Monensin, Rotenone | 21 |
| TUBA1B | Nocodazole, Paclitaxel | 21 |

### Running Drug Perturbation Evaluation

```bash
cd benchmark

# Run MegaSeg on drug perturbation FOVs
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/megaseg/dynunet/drug_perturbation/ACTB

# Run individual model on drug perturbation FOVs
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/drug_perturbation/NUP153

# Run classic workflows on drug perturbation FOVs (requires msseg env)
$PYTHON scripts/run_classic_drug_perturbation.py

# Generate comparison figures
uv run python scripts/visualize_drug_perturbation.py
```

### Eval Configs

- `configs/experiment/eval/uint8/megaseg/dynunet/drug_perturbation/` - 6 configs (one per structure)
- `configs/experiment/eval/uint8/individual/drug_perturbation/` - 4 configs (structures with available checkpoints)

### Visualization

Each per-structure figure shows max-intensity projections with segmentation overlays:
- **Rows:** FOVs grouped by drug condition (drug-treated + DMSO control)
- **Columns:** Raw | Classic | MegaSeg | Individual (where available)
- **Colors:** Orange = classic, Green = megaseg, Blue = individual

Output: `results/drug_perturbation/figures/` (6 per-structure comparison PDFs/PNGs + 1 overview grid)

## Running Your Own Experiments

### Which config do I use?

| Choice | Options |
|--------|---------|
| **Output dtype** | `float32` (continuous [0.0, 1.0]) or `uint8` (quantized [0, 255]) |
| **Model type** | `individual` (specialized DynUNet) or `megaseg` (multi-structure) |
| **Test set** | One of 14 structures, `test_new`, or cross-eval (model x all 14) |

Config path pattern: `experiment=eval/{dtype}/{eval_type}/...`

### Running a single eval

```bash
cd MegaSeg/benchmark

CUDA_VISIBLE_DEVICES=0 \
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD \
uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1
```

Environment variables:
- `CUDA_VISIBLE_DEVICES`: GPU selection (0-3)
- `CYTODL_CONFIG_PATH`: path to benchmark configs (required; point to `benchmark/configs`)
- `PROJECT_ROOT`: root for portable path resolution (defaults to cwd)

### Where do results go?

```
outputs/eval/{dtype}/{eval_type}/...
  seg/          # predicted segmentation TIFFs
  tags.log      # experiment tags
```

### Common Hydra overrides

```bash
# Use a different GPU
CUDA_VISIBLE_DEVICES=1 CYTODL_CONFIG_PATH=$PWD/configs PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1 trainer.devices=[0]

# Change batch size
CUDA_VISIBLE_DEVICES=0 CYTODL_CONFIG_PATH=$PWD/configs PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1 data.batch_size=2

# Change number of data-loading workers
CUDA_VISIBLE_DEVICES=0 CYTODL_CONFIG_PATH=$PWD/configs PROJECT_ROOT=$PWD uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1 data.num_workers=8
```

### Adding New Experiments

This section covers three use cases, from simplest to most involved:

1. [Evaluate existing models on your own test data](#use-case-1-evaluate-existing-models-on-new-test-data) - just bring a `test.csv`
2. [Add a new structure to the full benchmark](#use-case-2-add-a-new-structure-to-the-full-benchmark) - train/val/test data + checkpoint
3. [Add a new backbone architecture](#use-case-3-add-a-new-backbone-architecture) - new model config + checkpoint

#### Data Format

All data CSVs must follow this schema:

| Column | Description |
|--------|-------------|
| *(unnamed index)* | Numeric row index (first column, no header) |
| `raw` | Absolute path to raw microscopy TIFF |
| `seg` | Absolute path to segmentation ground truth TIFF |

Example `test.csv`:
```csv
,raw,seg
0,/allen/aics/assay-dev/users/Suraj/megaSeg/14structs_rand/test/ACTN1_11269.tiff,/allen/aics/assay-dev/users/Suraj/megaSeg/14structs_rand/test/ACTN1_11269_seg.tiff
1,/allen/aics/assay-dev/users/Suraj/megaSeg/14structs_rand/test/ACTN1_11854.tiff,/allen/aics/assay-dev/users/Suraj/megaSeg/14structs_rand/test/ACTN1_11854_seg.tiff
```

**Requirements:**
- Paths must be absolute (start with `/allen/aics/...`)
- TIFF files must be accessible on the Allen filesystem
- For evaluation-only experiments, only `test.csv` is needed
- For training experiments, provide `train.csv`, `val.csv`, and `test.csv`
- Place CSVs in a directory under `data/` at the repo root (e.g., `data/my_structure/`)

---

#### Use Case 1: Evaluate Existing Models on New Test Data

**The simplest case.** You have a `test.csv` and want to evaluate one of the existing model checkpoints on it. No registry updates or config regeneration needed.

**What you need:**
- A `test.csv` file in the format above
- The TIFF files accessible on the Allen filesystem
- The benchmark environment set up (`uv sync`)

**Step 1 - Create a data directory:**

```bash
cd MegaSeg
mkdir -p data/my_experiment
cp /path/to/your/test.csv data/my_experiment/test.csv
```

**Step 2 - Create an experiment config:**

Copy an existing config as a starting point:

```bash
cd benchmark
cp configs/experiment/eval/uint8/individual/ACTN1.yaml \
   configs/experiment/eval/uint8/individual/MY_EXPERIMENT.yaml
```

**Step 3 - Edit the config:**

Open `configs/experiment/eval/uint8/individual/MY_EXPERIMENT.yaml` and update the
highlighted fields. Here is the full config with comments on what to change:

```yaml
# @package _global_
defaults:
  - override /data: local/segmentation.yaml
  - override /model: local/segmentation_ms25_uint8.yaml     # model architecture (see Backbones table)
  - override /callbacks: eval.yaml
  - override /trainer: gpu.yaml
  - override /logger: csv.yaml

tags: ["my_experiment", "uint8"]                             # CHANGE: descriptive tags
seed: 12345

checkpoint:
  ckpt_path: ${paths.checkpoint_dir}/individual/ACTN1.ckpt   # CHANGE: path to the checkpoint you want to evaluate

experiment_name: my_experiment                               # CHANGE: unique experiment name
run_name: dynUNet_my_experiment                              # CHANGE: unique run name
source_col: raw
target_col: seg
spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 1000
  devices: [0]

model:
  save_dir: ${paths.log_dir}/eval/uint8/my_experiment        # CHANGE: output directory

paths:
  output_dir: ${paths.log_dir}/eval/uint8/my_experiment      # CHANGE: must match model.save_dir

data:
  path: ${paths.data_dir}/my_experiment/                     # CHANGE: path to your data directory
  cache_dir: ${paths.cache_dir}
  subsample:
  batch_size: 1
  num_workers: 1
  _aux:
    patch_shape: [32, 128, 128]

callbacks:
  saving:
    _target_: cyto_dl.callbacks.ImageSaver
    save_dir: ${paths.log_dir}/eval/uint8/my_experiment      # CHANGE: must match model.save_dir
    stages: ["test", "predict"]
```

**Fields you must change** (5 fields, marked `# CHANGE` above):

| Field | What to set |
|-------|-------------|
| `checkpoint.ckpt_path` | Path to the model checkpoint. Use `${paths.checkpoint_dir}/individual/{STRUCT}.ckpt` for individual models or `${paths.checkpoint_dir}/megaseg/{backbone}.ckpt` for MegaSeg models. |
| `experiment_name` / `run_name` | Unique identifiers for this experiment. Used in log directory names. |
| `model.save_dir`, `paths.output_dir`, `callbacks.saving.save_dir` | All three must point to the same output directory. Use `${paths.log_dir}/eval/uint8/{your_name}`. |
| `data.path` | Path to your data directory (the one containing `test.csv`). Use `${paths.data_dir}/{your_dir}/` to reference directories under `data/` at the repo root. |

**Optional fields you might change:**

| Field | Default | Notes |
|-------|---------|-------|
| `override /model` | `local/segmentation_ms25_uint8.yaml` | Change to a different model architecture (see [Backbones table](#megaseg-backbones)) |
| `override /trainer` | `gpu.yaml` | Use `ddp.yaml` for multi-GPU (AttentionUNet, SwinUNETR, UNet backbones require this) |
| `data.batch_size` | `1` | Increase if GPU memory allows |
| `data._aux.patch_shape` | `[32, 128, 128]` | Match the model's expected input patch size |

**Step 4 - Run the evaluation:**

```bash
cd MegaSeg/benchmark

CUDA_VISIBLE_DEVICES=0 \
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD \
uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/MY_EXPERIMENT
```

**Step 5 - Check results:**

Predictions are saved as TIFF files in the output directory:
```bash
ls outputs/eval/uint8/my_experiment/seg/
# Should contain one .tif file per image in your test.csv
```

To compute metrics (Dice, IoU, etc.) on these predictions, see [Quantitative Evaluation](#quantitative-evaluation).

---

#### Use Case 2: Add a New Structure to the Full Benchmark

You have a complete dataset (train/val/test CSVs) and a trained individual model checkpoint, and want the structure included in the automated benchmark pipeline (batch evaluation, metrics, and plots).

**What you need:**
- `train.csv`, `val.csv`, `test.csv` in the [data format](#data-format) above
- A trained individual model checkpoint (`.ckpt` file)

**Step 1 - Place data:**

```bash
mkdir -p data/variance-100x/MY_STRUCT
cp train.csv val.csv test.csv data/variance-100x/MY_STRUCT/
```

**Step 2 - Place checkpoint:**

```bash
# Create a symlink to the checkpoint (recommended) or copy it
ln -s /path/to/original/checkpoint.ckpt benchmark/checkpoints/individual/MY_STRUCT.ckpt
```

Update `benchmark/checkpoints/README.md` with the checkpoint's provenance (original path, training epoch, etc.).

**Step 3 - Update registries in 3 files:**

> **This is the critical step.** Three scripts maintain their own lists of structures. All three must be updated or the pipeline will silently skip the new structure.

Add `"MY_STRUCT"` to the `STRUCTURES` list in each file:

| File | Line | Constant |
|------|------|----------|
| `scripts/generate_configs.py` | ~21 | `STRUCTURES = [...]` |
| `scripts/run_eval.py` | ~68 | `STRUCTURES = [...]` |
| `scripts/evaluate_benchmark.py` | ~44 | `STRUCTURES = [...]` |

**Step 4 - Regenerate configs:**

```bash
cd benchmark
uv run python scripts/generate_configs.py
```

This creates new experiment configs for the structure:
- `configs/experiment/eval/{uint8,float32}/individual/MY_STRUCT.yaml`
- `configs/experiment/eval/{uint8,float32}/individual_test_new/MY_STRUCT.yaml`
- `configs/experiment/eval/{uint8,float32}/megaseg/{dynunet,attention_unet,swin_unetr,unet}/MY_STRUCT.yaml`
- `configs/experiment/eval/{uint8,float32}/cross_eval/MY_STRUCT_model/cross_eval.yaml`
- Updated `scripts/experiment_matrix.csv`

**Step 5 - Verify configs were generated:**

```bash
ls configs/experiment/eval/uint8/individual/MY_STRUCT.yaml
ls configs/experiment/eval/uint8/megaseg/dynunet/MY_STRUCT.yaml
# Both should exist
```

**Step 6 - Run evaluations:**

```bash
# Run all eval types for the new structure
uv run python scripts/run_eval.py --dtype uint8 --struct MY_STRUCT --gpus 0,1,2,3

# Or run just the individual eval
uv run python scripts/run_eval.py --dtype uint8 --type individual --struct MY_STRUCT
```

**Step 7 - Compute metrics:**

```bash
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py compute --dtype uint8
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py report --dtype uint8 --metric dice
```

---

#### Use Case 3: Add a New Backbone Architecture

You have a multi-structure model trained with a new architecture (e.g., nnU-Net) and want to add it as a MegaSeg backbone variant.

**What you need:**
- A trained multi-structure model checkpoint (`.ckpt` file)
- Knowledge of the model architecture (backbone class, parameters)

**Step 1 - Create model config files:**

Create one config per dtype in `configs/model/local/`. Use an existing config as a template:

```bash
cp configs/model/local/segmentation_ms25_uint8.yaml \
   configs/model/local/segmentation_ms25_mybackbone_uint8.yaml
cp configs/model/local/segmentation_ms25_float32.yaml \
   configs/model/local/segmentation_ms25_mybackbone_float32.yaml
```

Edit the `backbone:` section to match your architecture. For example, replacing DynUNet with a UNet:

```yaml
backbone:
  _target_: monai.networks.nets.UNet           # CHANGE: your backbone class
  spatial_dims: ${spatial_dims}
  in_channels: ${raw_im_channels}
  out_channels: 1
  channels: [16, 32, 64, 128, 256]             # CHANGE: architecture-specific params
  strides: [2, 2, 2, 2]
  num_res_units: 2
```

The rest of the config (optimizer, loss, task heads, inference args) can usually stay the same. The only difference between `uint8` and `float32` variants is the `rescale_dtype` in the postprocessing section (`numpy.uint8` vs `numpy.float32`).

**Step 2 - Place checkpoint:**

```bash
ln -s /path/to/checkpoint.ckpt benchmark/checkpoints/megaseg/mybackbone.ckpt
```

Update `benchmark/checkpoints/README.md`.

**Step 3 - Update registries in 3 files:**

| File | Line | What to add |
|------|------|-------------|
| `scripts/generate_configs.py` | ~40 | Add entry to `MEGASEG_BACKBONES` dict (see below) |
| `scripts/run_eval.py` | ~84 | Add `"mybackbone"` to `BACKBONES` list |
| `scripts/evaluate_benchmark.py` | ~61 | Add `"mybackbone"` to `BACKBONES` list |

For `generate_configs.py`, add to the `MEGASEG_BACKBONES` dict:

```python
MEGASEG_BACKBONES = {
    # ... existing entries ...
    "mybackbone": {
        "model_config_template": "local/segmentation_ms25_mybackbone_{dtype}.yaml",
        "trainer": "gpu.yaml",      # use "ddp.yaml" if the model needs multi-GPU
        "checkpoint": "megaseg/mybackbone.ckpt",
    },
}
```

Choose `gpu.yaml` (single GPU) or `ddp.yaml` (multi-GPU with DDP) based on model memory requirements. DynUNet fits on a single GPU; larger models like SwinUNETR need DDP.

**Step 4 - Regenerate configs and run:**

```bash
uv run python scripts/generate_configs.py
uv run python scripts/run_eval.py --dtype uint8 --type megaseg --backbone mybackbone --gpus 0,1,2,3
```

---

#### Registry Sync Reference

Three scripts maintain independent copies of the same constants. When adding structures or backbones, **all three files must be updated** or some pipeline steps will silently skip the new entries.

| Constant | `generate_configs.py` | `run_eval.py` | `evaluate_benchmark.py` |
|----------|----------------------|---------------|------------------------|
| `STRUCTURES` | Line ~21 | Line ~68 | Line ~44 |
| `BACKBONES` / `MEGASEG_BACKBONES` | Line ~40 (dict) | Line ~84 (list) | Line ~61 (list) |
| `DTYPES` | Line ~38 | Line ~85 | n/a |
| `EVAL_TYPES` | n/a | Line ~86 | Line ~65 |

**What happens if they're out of sync:**
- Missing from `generate_configs.py` - no experiment YAML configs are generated
- Missing from `run_eval.py` - batch runner skips the structure/backbone
- Missing from `evaluate_benchmark.py` - metrics computation and plots skip it

## Structures

| # | Structure | Protein |
|---|-----------|---------|
| 1 | ACTN1 | Alpha-actinin-1 |
| 2 | ATP2A2 | SERCA2 |
| 3 | CTNNB1 | Beta-catenin |
| 4 | DSP | Desmoplakin |
| 5 | GJA1 | Connexin-43 |
| 6 | MYH10 | Non-muscle myosin IIB |
| 7 | NPM1 | Nucleophosmin |
| 8 | NUP153 | Nucleoporin 153 |
| 9 | RAB5A | Rab5a |
| 10 | SEC61B | Sec61 translocon |
| 11 | SON | SON protein |
| 12 | ST6GAL1 | ST6 beta-galactosidase |
| 13 | TJP1 | ZO-1 |
| 14 | TOMM20 | TOM20 |

LMNB1 (Lamin B1) is additionally available in `data/variance-100x/LMNB1/` for fine-tuning experiments but is not part of the 14-structure benchmark evaluation.

## MegaSeg Backbones

Each backbone has both `float32` and `uint8` model configs in `configs/model/local/`.

| Backbone | Model Config | Trainer | Checkpoint |
|----------|-------------|---------|------------|
| DynUNet | segmentation_ms25_float32 | gpu | dynunet.ckpt |
| Attention UNet | segmentation_ms25_attn_unet_float32 | ddp | attention_unet.ckpt |
| Swin UNETR | segmentation_ms25_swin_unetr_float32 | ddp | swin_unetr.ckpt |
| UNet | segmentation_ms25_unet_float32 | ddp | unet.ckpt |

## Path Portability

All configs use Hydra `${paths.*}` interpolation. The root is resolved from:

1. `PROJECT_ROOT` environment variable (if set)
2. Current working directory (fallback)

Data is referenced via `${paths.data_dir}` which resolves to `${paths.root_dir}/../data` (repo root's `data/` directory).

This means you should run commands from within the `benchmark/` directory, or set `PROJECT_ROOT` to point to it.

**Note on `expanduser`:** All `LoadImaged` entries in the data configs have `expanduser: false` to prevent MONAI's default `pathlib.Path().expanduser()` from normalizing `://` to `:/` in S3/HTTPS URLs.

## Regenerating Configs

If you need to modify the config templates or add structures:

```bash
uv run python scripts/generate_configs.py
```

This regenerates all 204+ experiment configs and the experiment_matrix.csv.

## cyto-dl Version

Included as a git submodule pointing to [AllenCell/cyto-dl](https://github.com/AllenCell/cyto-dl).

```bash
# Initialize after cloning
git submodule update --init --recursive
```

The submodule includes a custom `channel_name` parameter added to `MonaiBioReader` for dynamic channel selection by name (used by LMNB1 fine-tuning configs).

## uint8 Benchmark Results Summary

| Eval Type | Samples | Mean Dice |
|---|---|---|
| individual | 741 | 0.7733 |
| megaseg | 741 | 0.9092 |
| cross_eval | 13,132 | 0.5062 |

All results use the filtered test set (57 samples per structure, bottom-10 low-signal samples removed). MegaSeg outperforms individual models on **12/13 structures**. Best backbone: DynUNet, followed by SwinUNETR, AttentionUNet, UNet.
