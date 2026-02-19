# AllenCell/MegaSeg Repo Reorganization Design

**Date:** 2026-02-19
**Status:** Approved

## Goal

Merge the local `megaseg-benchmark` project into the existing private `AllenCell/MegaSeg` GitHub repo, reorganizing for clarity while preserving all existing content.

## Current State

### Remote (`AllenCell/MegaSeg`, `main` branch)
- `segmenter-classic/` — 18 Jupyter notebooks for classical segmentation
- `segmenter-ml/` — Training-oriented configs (15 experiment configs, model/data configs)
- `data/variance-100x/` — 15 structures (14 benchmark + LMNB1), flat CSV layout
- `data/standardized_test/test_new.csv` — Held-out test set
- `Results_Figure_Tracking/Base_Figure1.csv` — Figure tracking CSV
- 24 commits, 2 branches (`main`, `segmenter-ml`)

### Local (`megaseg-benchmark`, `master` branch, no remote)
- `configs/` — 204 eval experiment configs (Hydra-based)
- `scripts/` — Evaluation pipeline (evaluate_benchmark.py, run_eval.py, etc.)
- `results/uint8/` — Computed metrics (6 CSVs + 60 figures)
- `cyto-dl/` — Vendored copy at commit `fe8fa7ac` (402 MB, tracked)
- `checkpoints/` — 18 .ckpt files (gitignored) + README.md
- `data/megaseg_13structs/` — Same data as remote, different directory layout
- `docs/plans/` — 3 design documents
- 9 commits on `master`

## Decisions

| Decision | Choice |
|----------|--------|
| Existing remote content | Merge everything together |
| Benchmark location | `benchmark/` subdirectory |
| Data layout | Use remote's `data/variance-100x/` layout |
| Config paths | Update all 204 configs via `generate_configs.py` |
| cyto-dl | Git submodule → `AllenCell/cyto-dl` @ `fe8fa7ac` |
| Results | Commit CSVs and figures |
| Git history | Squash local 9 commits into 1 clean commit |
| License | None for now |

## Target Structure

```
AllenCell/MegaSeg/
├── README.md                          # Updated: overview of all three components
├── .gitignore                         # Merged gitignore rules
├── .gitmodules                        # cyto-dl submodule
│
├── data/                              # Preserved from remote
│   ├── variance-100x/                 # 15 structures (flat CSV layout)
│   │   ├── ACTN1/{test,train,val}.csv
│   │   ├── ...
│   │   └── TOMM20/{test,train,val}.csv
│   └── standardized_test/
│       └── test_new.csv
│
├── segmenter-classic/                 # Preserved from remote (18 notebooks)
│   └── classic_segmentation_*.ipynb
│
├── segmenter-ml/                      # Preserved from remote (training configs)
│   └── configs/...
│
├── Results_Figure_Tracking/           # Preserved from remote
│   └── Base_Figure1.csv
│
├── benchmark/                         # NEW: all benchmark work
│   ├── README.md                      # Benchmark-specific documentation
│   ├── pyproject.toml                 # Project metadata
│   ├── cyto-dl/                       # Git submodule → AllenCell/cyto-dl @ fe8fa7ac
│   ├── configs/                       # 204 eval experiment configs
│   │   ├── eval.yaml
│   │   ├── callbacks/
│   │   ├── data/local/
│   │   ├── experiment/eval/{uint8,float32}/
│   │   ├── extras/
│   │   ├── hydra/
│   │   ├── logger/
│   │   ├── model/local/
│   │   ├── paths/
│   │   └── trainer/
│   ├── checkpoints/                   # README.md only (.ckpt files gitignored)
│   │   └── README.md
│   ├── scripts/
│   │   ├── evaluate_benchmark.py      # Three-phase eval pipeline
│   │   ├── run_eval.py                # Batch eval runner
│   │   ├── generate_configs.py        # Config generator
│   │   ├── run_cross_eval.py          # Cross-eval runner
│   │   ├── test_evaluate_benchmark.py # Tests
│   │   └── experiment_matrix.csv      # Job matrix
│   ├── results/
│   │   └── uint8/                     # Committed metrics + plots
│   │       ├── per_sample_metrics.csv
│   │       ├── summary_by_structure.csv
│   │       ├── comparison_*.csv
│   │       ├── cross_eval_heatmap.csv
│   │       ├── test_new_comparison.csv
│   │       └── figures/{dice,iou,precision,recall,accuracy}/*.{png,pdf}
│   └── docs/
│       └── plans/                     # Design documents
│
└── (benchmark/outputs/, benchmark/cache/ — gitignored)
```

## Path Changes

### paths/default.yaml

**Before:**
```yaml
data_dir: ${paths.root_dir}/data
```

**After:**
```yaml
data_dir: ${paths.root_dir}/../data
```

### Config data references

**Before:** `${paths.data_dir}/megaseg_13structs/{STRUCT}/test/test.csv`
**After:** `${paths.data_dir}/variance-100x/{STRUCT}/test.csv`

**Before:** `${paths.data_dir}/megaseg_13structs/test_new/test_new.csv`
**After:** `${paths.data_dir}/standardized_test/test_new.csv`

### evaluate_benchmark.py path constants

Update `DATA_DIR`, `GT_CSV_PATTERN`, and `TEST_NEW_CSV` constants to match new layout.

### Eval command pattern

```bash
cd benchmark
PROJECT_ROOT=$PWD python cyto-dl/cyto_dl/eval.py experiment=eval/uint8/individual/ACTN1
```

## .gitignore (merged)

```gitignore
# Benchmark outputs (large)
benchmark/outputs/
benchmark/cache/

# Checkpoint weights
benchmark/checkpoints/*.ckpt
benchmark/checkpoints/**/*.ckpt

# Python
*.pyc
__pycache__/

# OS
.DS_Store

# Logs
benchmark/eval.log
benchmark/scripts/job_logs/

# Environment
.venv/
.pytest_cache/
```

## Execution Plan

1. Clone `AllenCell/MegaSeg` into a working directory
2. Create branch `benchmark-overhaul` from `main`
3. Create `benchmark/` directory
4. Copy files from local project (configs, scripts, results, docs, checkpoints/README.md, README.md, pyproject.toml)
5. Add cyto-dl as git submodule at `benchmark/cyto-dl/`
6. Update `benchmark/configs/paths/default.yaml` for new data layout
7. Regenerate experiment configs with updated data paths via `generate_configs.py`
8. Update `evaluate_benchmark.py` path constants
9. Update `run_eval.py` if it has hardcoded data paths
10. Remove local `data/megaseg_13structs/` from benchmark (using remote's data/)
11. Update `.gitignore` with merged rules
12. Write updated root `README.md`
13. Verify: run tests, check config resolution
14. Squash into single commit
15. Push branch, create PR

## Risks

- **Path resolution**: `${paths.root_dir}/../data` may not work with all Hydra versions. Fallback: use absolute path or environment variable.
- **Submodule friction**: Contributors must run `git submodule update --init` after cloning. Documented in README.
- **Data CSV format difference**: Remote's layout is flat (`{STRUCT}/test.csv`) vs local's nested (`{STRUCT}/test/test.csv`). All 204 configs need updating.
