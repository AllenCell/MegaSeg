# megaseg-benchmark

Cross-structure segmentation evaluation benchmark comparing MegaSeg (multi-structure) models against individual (specialized) models across 14 biological structures.

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

# Generate all plots (bar charts, heatmaps, radar, violin, rank, scatter)
PROJECT_ROOT=$PWD uv run python scripts/evaluate_benchmark.py plot --dtype uint8
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

1. [Evaluate existing models on your own test data](#use-case-1-evaluate-existing-models-on-new-test-data) — just bring a `test.csv`
2. [Add a new structure to the full benchmark](#use-case-2-add-a-new-structure-to-the-full-benchmark) — train/val/test data + checkpoint
3. [Add a new backbone architecture](#use-case-3-add-a-new-backbone-architecture) — new model config + checkpoint

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

**Step 1 — Create a data directory:**

```bash
cd MegaSeg
mkdir -p data/my_experiment
cp /path/to/your/test.csv data/my_experiment/test.csv
```

**Step 2 — Create an experiment config:**

Copy an existing config as a starting point:

```bash
cd benchmark
cp configs/experiment/eval/uint8/individual/ACTN1.yaml \
   configs/experiment/eval/uint8/individual/MY_EXPERIMENT.yaml
```

**Step 3 — Edit the config:**

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

**Step 4 — Run the evaluation:**

```bash
cd MegaSeg/benchmark

CUDA_VISIBLE_DEVICES=0 \
CYTODL_CONFIG_PATH=$PWD/configs \
PROJECT_ROOT=$PWD \
uv run python cyto-dl/cyto_dl/eval.py \
  experiment=eval/uint8/individual/MY_EXPERIMENT
```

**Step 5 — Check results:**

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

**Step 1 — Place data:**

```bash
mkdir -p data/variance-100x/MY_STRUCT
cp train.csv val.csv test.csv data/variance-100x/MY_STRUCT/
```

**Step 2 — Place checkpoint:**

```bash
# Create a symlink to the checkpoint (recommended) or copy it
ln -s /path/to/original/checkpoint.ckpt benchmark/checkpoints/individual/MY_STRUCT.ckpt
```

Update `benchmark/checkpoints/README.md` with the checkpoint's provenance (original path, training epoch, etc.).

**Step 3 — Update registries in 3 files:**

> **This is the critical step.** Three scripts maintain their own lists of structures. All three must be updated or the pipeline will silently skip the new structure.

Add `"MY_STRUCT"` to the `STRUCTURES` list in each file:

| File | Line | Constant |
|------|------|----------|
| `scripts/generate_configs.py` | ~21 | `STRUCTURES = [...]` |
| `scripts/run_eval.py` | ~68 | `STRUCTURES = [...]` |
| `scripts/evaluate_benchmark.py` | ~44 | `STRUCTURES = [...]` |

**Step 4 — Regenerate configs:**

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

**Step 5 — Verify configs were generated:**

```bash
ls configs/experiment/eval/uint8/individual/MY_STRUCT.yaml
ls configs/experiment/eval/uint8/megaseg/dynunet/MY_STRUCT.yaml
# Both should exist
```

**Step 6 — Run evaluations:**

```bash
# Run all eval types for the new structure
uv run python scripts/run_eval.py --dtype uint8 --struct MY_STRUCT --gpus 0,1,2,3

# Or run just the individual eval
uv run python scripts/run_eval.py --dtype uint8 --type individual --struct MY_STRUCT
```

**Step 7 — Compute metrics:**

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

**Step 1 — Create model config files:**

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

**Step 2 — Place checkpoint:**

```bash
ln -s /path/to/checkpoint.ckpt benchmark/checkpoints/megaseg/mybackbone.ckpt
```

Update `benchmark/checkpoints/README.md`.

**Step 3 — Update registries in 3 files:**

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

**Step 4 — Regenerate configs and run:**

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
| `DTYPES` | Line ~38 | Line ~85 | — |
| `EVAL_TYPES` | — | Line ~86 | Line ~65 |

**What happens if they're out of sync:**
- Missing from `generate_configs.py` → no experiment YAML configs are generated
- Missing from `run_eval.py` → batch runner skips the structure/backbone
- Missing from `evaluate_benchmark.py` → metrics computation and plots skip it

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
uv run python scripts/generate_configs.py
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
