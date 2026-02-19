# MegaSeg

Multi-structure segmentation for fluorescently-tagged cellular structures. This repository contains classical segmentation workflows, ML-based segmentation experiments, and a comprehensive evaluation benchmark comparing MegaSeg (multi-structure) models against individual (specialized) models.

## Repository Structure

```
MegaSeg/
├── segmenter-classic/       # Classical segmentation notebooks (18 notebooks)
├── segmenter-ml/            # ML training configs and experiment tracking
├── benchmark/               # Evaluation benchmark (14 structures, 4 backbones)
├── data/
│   ├── variance-100x/       # Per-structure CSVs (train/val/test splits)
│   └── standardized_test/   # Held-out test_new dataset
└── Results_Figure_Tracking/ # Figure tracking CSV
```

## Components

### segmenter-classic

Classical (non-ML) segmentation pipelines implemented as Jupyter notebooks. Covers structure-specific segmentation using traditional image processing methods.

### segmenter-ml

ML-based segmentation training configurations for individual and multi-structure models. Contains Hydra configs for training with [cyto-dl](https://github.com/AllenCell/cyto-dl).

### benchmark

Systematic evaluation comparing MegaSeg multi-structure models (DynUNet, AttentionUNet, SwinUNETR, UNet) against 14 individual specialized models across multiple evaluation paradigms. Includes config generation, batch evaluation runners, quantitative metrics computation, and visualization.

See [`benchmark/README.md`](benchmark/README.md) for full details.

## Data

Test set CSVs for 14 structures live in `data/variance-100x/{STRUCTURE}/` with `train.csv`, `val.csv`, and `test.csv` splits. The held-out evaluation dataset is at `data/standardized_test/test_new.csv`.

**Structures:** ACTN1, ATP2A2, CTNNB1, DSP, GJA1, MYH10, NPM1, NUP153, RAB5A, SEC61B, SON, ST6GAL1, TJP1, TOMM20

## Setup

```bash
# Clone with submodules (benchmark uses cyto-dl as a submodule)
git clone --recurse-submodules git@github.com:AllenCell/MegaSeg.git

# Or initialize submodules after cloning
git submodule update --init --recursive

# Activate environment
conda activate acm-eval
```
