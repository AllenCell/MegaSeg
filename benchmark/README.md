# megaseg-benchmark

Cross-structure segmentation evaluation benchmark comparing MegaSeg (multi-structure) models against individual (specialized) models across 14 biological structures.

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules git@github.com:AllenCell/MegaSeg.git
cd MegaSeg

# 2. Activate environment
conda activate acm-eval

# 3. Run a single evaluation
cd benchmark
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1
```

## Directory Structure

```
MegaSeg/
├── data/
│   ├── variance-100x/           # Per-structure CSVs (at repo root)
│   │   ├── ACTN1/test.csv
│   │   └── ...
│   └── standardized_test/       # Held-out test_new dataset
│       └── test_new.csv
└── benchmark/
    ├── checkpoints/
    │   ├── individual/          # 14 single-structure DynUNet checkpoints
    │   │   ├── ACTN1.ckpt
    │   │   └── ...
    │   ├── megaseg/             # 4 multi-structure backbone checkpoints
    │   │   ├── dynunet.ckpt
    │   │   ├── attention_unet.ckpt
    │   │   ├── swin_unetr.ckpt
    │   │   └── unet.ckpt
    │   └── README.md            # Checkpoint provenance mapping
    ├── configs/
    │   ├── paths/default.yaml   # Portable path config (uses PROJECT_ROOT)
    │   ├── data/local/          # Data loading config
    │   ├── model/local/         # Model architecture configs (8 variants)
    │   ├── callbacks/           # Callback configs
    │   ├── trainer/             # Trainer configs (gpu, ddp)
    │   ├── logger/              # Logger config
    │   ├── hydra/               # Hydra config
    │   ├── extras/              # Extra utilities config
    │   ├── eval.yaml            # Top-level eval entry point
    │   └── experiment/eval/     # 204 experiment configs (102 per dtype)
    │       ├── float32/
    │       │   ├── individual/      # 14 configs
    │       │   ├── individual_test_new/  # 14 configs
    │       │   ├── megaseg/         # 60 configs (4 backbones x 15)
    │       │   └── cross_eval/      # 14 parameterized configs
    │       └── uint8/               # (same layout)
    ├── cyto-dl/                 # Git submodule (AllenCell/cyto-dl @ fe8fa7ac)
    ├── results/uint8/           # Computed metrics and figures
    │   ├── metrics_all.csv
    │   └── figures/
    ├── scripts/
    │   ├── generate_configs.py  # Config generator (regenerate all 204 configs)
    │   ├── evaluate_benchmark.py # Metrics computation and plotting
    │   ├── run_eval.py          # Batch runner for individual/megaseg evals
    │   ├── run_cross_eval.py    # Batch runner for cross-eval jobs
    │   ├── experiment_matrix.csv # Job definitions for cross-eval
    │   └── test_evaluate_benchmark.py # Tests (32 tests)
    ├── outputs/                 # Evaluation outputs (gitignored)
    ├── cache/                   # Data cache (gitignored)
    └── pyproject.toml
```

## Evaluation Types

### 1. Individual Eval (14 configs)

Each individual (specialized) DynUNet model evaluated on its own structure's test set.

```bash
cd benchmark
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1
```

### 2. Individual Test-New (14 configs)

Each individual model evaluated on the held-out test_new dataset.

```bash
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual_test_new/ACTN1
```

### 3. MegaSeg Backbone Eval (60 configs)

Each of 4 multi-structure backbones evaluated on each of the 14 structures + test_new.

```bash
# DynUNet megaseg on ACTN1 test set
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/megaseg/dynunet/ACTN1

# Swin UNETR megaseg on test_new
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/megaseg/swin_unetr/test_new
```

### 4. Cross-Eval (14 parameterized configs, 196 jobs per dtype)

Each individual model evaluated on every other structure's test set. Uses a parameterized `test_struct` variable.

```bash
# ACTN1 model evaluated on ATP2A2 test set
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/cross_eval/ACTN1_model/cross_eval \
  test_struct=ATP2A2
```

### Batch Cross-Eval Runner

```bash
# Preview all 196 jobs
python scripts/run_cross_eval.py --dry-run --gpus 0,1,2,3

# Launch all pending
python scripts/run_cross_eval.py --gpus 0,1,2,3

# Check status
python scripts/run_cross_eval.py --status
```

## Quantitative Evaluation

Compute metrics (Dice, IoU, Precision, Recall, Accuracy) across all completed evaluations:

```bash
cd benchmark

# Compute metrics for all uint8 evaluations
PROJECT_ROOT=$PWD python scripts/evaluate_benchmark.py compute --dtype uint8 --workers 16

# Print summary report
PROJECT_ROOT=$PWD python scripts/evaluate_benchmark.py report --dtype uint8 --metric dice

# Generate all plots (bar charts, heatmaps, radar, violin, rank, scatter)
PROJECT_ROOT=$PWD python scripts/evaluate_benchmark.py plot --dtype uint8
```

Results are saved to `results/{dtype}/` with CSVs and a `figures/` subdirectory.

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
conda activate acm-eval

CUDA_VISIBLE_DEVICES=0 \
PROJECT_ROOT=$PWD \
python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/ACTN1
```

Environment variables:
- `CUDA_VISIBLE_DEVICES`: GPU selection (0-3)
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
... experiment=eval/uint8/individual/ACTN1 trainer.devices=[1]

# Change batch size
... experiment=eval/uint8/individual/ACTN1 data.batch_size=2

# Change number of data-loading workers
... experiment=eval/uint8/individual/ACTN1 data.num_workers=8
```

### Adding a new structure or backbone

1. Edit registries in `scripts/generate_configs.py` (STRUCTURES list, MEGASEG_BACKBONES dict)
2. If adding backbone: create model configs for both dtypes in `configs/model/local/`
3. Add checkpoint and update `checkpoints/README.md`
4. Re-run: `python scripts/generate_configs.py`

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

## Regenerating Configs

If you need to modify the config templates or add structures:

```bash
python scripts/generate_configs.py
```

This regenerates all 204 experiment configs and the experiment_matrix.csv.

## cyto-dl Version

Included as a git submodule pointing to [AllenCell/cyto-dl](https://github.com/AllenCell/cyto-dl) at commit `fe8fa7ac` ("Bugfix/pin tifffile (#460)").

```bash
# Initialize after cloning
git submodule update --init --recursive
```

## uint8 Benchmark Results Summary

| Eval Type | Samples | Mean Dice |
|---|---|---|
| individual | 938 | 0.7671 |
| megaseg | 3,952 | 0.8482 |
| cross_eval | 13,132 | 0.5062 |
| individual_test_new | 28 | 0.7885 |

MegaSeg outperforms individual models on **13/14 structures** (only NPM1 is the exception). Best backbone: DynUNet (0.8684), followed by SwinUNETR (0.8601), AttentionUNet (0.8586), UNet (0.8329).
