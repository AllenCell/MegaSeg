# AllenCell/MegaSeg Repo Reorganization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge the local megaseg-benchmark project into the existing private AllenCell/MegaSeg GitHub repo as a `benchmark/` subdirectory, with cyto-dl as a git submodule and data paths updated to use the remote's `data/variance-100x/` layout.

**Architecture:** Clone remote repo → create branch → copy benchmark files into `benchmark/` → add cyto-dl submodule → update data paths in configs/scripts/tests → update .gitignore and README → squash commit → push → create PR.

**Tech Stack:** Git, GitHub CLI (`/allen/aics/users/ashwin.samudre/miniconda3/bin/gh`), Python, Hydra configs (YAML)

---

### Task 1: Clone Remote Repo and Create Branch

**Files:**
- Create: `/allen/aics/rep_learn/MegaSeg/` (fresh clone)

**Step 1: Clone the remote repo**

```bash
cd /allen/aics/rep_learn
git clone https://github.com/AllenCell/MegaSeg.git
```

**Step 2: Create the benchmark-overhaul branch**

```bash
cd /allen/aics/rep_learn/MegaSeg
git checkout -b benchmark-overhaul
```

**Step 3: Verify**

```bash
git branch  # Should show * benchmark-overhaul
ls          # Should show README.md, data/, segmenter-classic/, segmenter-ml/, Results_Figure_Tracking/
```

---

### Task 2: Create benchmark/ Directory and Copy Files

**Files:**
- Create: `benchmark/` directory structure

**Step 1: Create benchmark directory**

```bash
mkdir -p /allen/aics/rep_learn/MegaSeg/benchmark
```

**Step 2: Copy configs (all Hydra configs)**

```bash
cp -r /allen/aics/rep_learn/megaseg-benchmark/configs /allen/aics/rep_learn/MegaSeg/benchmark/configs
```

**Step 3: Copy scripts (excluding job_logs and __pycache__)**

```bash
mkdir -p /allen/aics/rep_learn/MegaSeg/benchmark/scripts
cp /allen/aics/rep_learn/megaseg-benchmark/scripts/evaluate_benchmark.py /allen/aics/rep_learn/MegaSeg/benchmark/scripts/
cp /allen/aics/rep_learn/megaseg-benchmark/scripts/run_eval.py /allen/aics/rep_learn/MegaSeg/benchmark/scripts/
cp /allen/aics/rep_learn/megaseg-benchmark/scripts/generate_configs.py /allen/aics/rep_learn/MegaSeg/benchmark/scripts/
cp /allen/aics/rep_learn/megaseg-benchmark/scripts/run_cross_eval.py /allen/aics/rep_learn/MegaSeg/benchmark/scripts/
cp /allen/aics/rep_learn/megaseg-benchmark/scripts/test_evaluate_benchmark.py /allen/aics/rep_learn/MegaSeg/benchmark/scripts/
cp /allen/aics/rep_learn/megaseg-benchmark/scripts/experiment_matrix.csv /allen/aics/rep_learn/MegaSeg/benchmark/scripts/
```

**Step 4: Copy results**

```bash
cp -r /allen/aics/rep_learn/megaseg-benchmark/results /allen/aics/rep_learn/MegaSeg/benchmark/results
```

**Step 5: Copy docs**

```bash
cp -r /allen/aics/rep_learn/megaseg-benchmark/docs /allen/aics/rep_learn/MegaSeg/benchmark/docs
```

**Step 6: Copy checkpoints README**

```bash
mkdir -p /allen/aics/rep_learn/MegaSeg/benchmark/checkpoints
cp /allen/aics/rep_learn/megaseg-benchmark/checkpoints/README.md /allen/aics/rep_learn/MegaSeg/benchmark/checkpoints/README.md
```

**Step 7: Copy project files**

```bash
cp /allen/aics/rep_learn/megaseg-benchmark/README.md /allen/aics/rep_learn/MegaSeg/benchmark/README.md
cp /allen/aics/rep_learn/megaseg-benchmark/pyproject.toml /allen/aics/rep_learn/MegaSeg/benchmark/pyproject.toml
```

**Step 8: Verify the structure**

```bash
find /allen/aics/rep_learn/MegaSeg/benchmark -type f | head -30
```

---

### Task 3: Add cyto-dl as Git Submodule

**Files:**
- Create: `.gitmodules`
- Create: `benchmark/cyto-dl/` (submodule)

**Step 1: Add the submodule**

```bash
cd /allen/aics/rep_learn/MegaSeg
git submodule add https://github.com/AllenCell/cyto-dl.git benchmark/cyto-dl
```

**Step 2: Pin to the correct commit**

```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark/cyto-dl
git checkout fe8fa7ac
cd /allen/aics/rep_learn/MegaSeg
git add benchmark/cyto-dl
```

**Step 3: Verify**

```bash
git submodule status  # Should show fe8fa7ac for benchmark/cyto-dl
cat .gitmodules       # Should reference AllenCell/cyto-dl
```

---

### Task 4: Update paths/default.yaml for New Data Layout

**Files:**
- Modify: `benchmark/configs/paths/default.yaml`

**Current content:**
```yaml
root_dir: ${oc.env:PROJECT_ROOT, './'}
data_dir: ${paths.root_dir}/data
log_dir: ${paths.root_dir}/outputs
checkpoint_dir: ${paths.root_dir}/checkpoints
cache_dir: ${paths.root_dir}/cache
```

**New content — change `data_dir` to point up from benchmark/ to repo root's data/:**
```yaml
root_dir: ${oc.env:PROJECT_ROOT, './'}
data_dir: ${paths.root_dir}/../data
log_dir: ${paths.root_dir}/outputs
checkpoint_dir: ${paths.root_dir}/checkpoints
cache_dir: ${paths.root_dir}/cache
```

**Verify:** The Hydra interpolation `${paths.root_dir}/../data` will resolve correctly when `PROJECT_ROOT` points to the `benchmark/` directory. This means `${paths.data_dir}/variance-100x/ACTN1/` will resolve to `/path/to/MegaSeg/data/variance-100x/ACTN1/`.

---

### Task 5: Update generate_configs.py Data Path Templates

**Files:**
- Modify: `benchmark/scripts/generate_configs.py` (5 changes)

**Changes (all in template strings):**

1. **Line 101** (`individual_eval_config`):
   - Old: `path: ${{paths.data_dir}}/megaseg_13structs/{struct}/test/`
   - New: `path: ${{paths.data_dir}}/variance-100x/{struct}/`

2. **Line 151** (`individual_test_new_config`):
   - Old: `path: ${{paths.data_dir}}/megaseg_13structs/test_new/`
   - New: `path: ${{paths.data_dir}}/standardized_test/`

3. **Line 203** (`megaseg_eval_config`):
   - Old: `path: ${{paths.data_dir}}/megaseg_13structs/{struct}/test/`
   - New: `path: ${{paths.data_dir}}/variance-100x/{struct}/`

4. **Line 255** (`megaseg_test_new_config`):
   - Old: `path: ${{paths.data_dir}}/megaseg_13structs/test_new/`
   - New: `path: ${{paths.data_dir}}/standardized_test/`

5. **Line 307** (`cross_eval_config`):
   - Old: `path: ${{paths.data_dir}}/megaseg_13structs/${{test_struct}}/test/`
   - New: `path: ${{paths.data_dir}}/variance-100x/${{test_struct}}/`

---

### Task 6: Regenerate All 204 Experiment Configs

**Files:**
- Modify: All 204 YAML files under `benchmark/configs/experiment/eval/`

**Step 1: Run the config generator**

```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark
python scripts/generate_configs.py
```

Expected output:
```
Generated 204 experiment configs:
  individual: 28
  individual_test_new: 28
  megaseg: 120
  cross_eval: 28
Generated experiment_matrix.csv with 392 rows
```

**Step 2: Verify a sample config has updated path**

```bash
grep "path:" benchmark/configs/experiment/eval/uint8/individual/ACTN1.yaml
# Should show: path: ${paths.data_dir}/variance-100x/ACTN1/
```

---

### Task 7: Update evaluate_benchmark.py Path Constants

**Files:**
- Modify: `benchmark/scripts/evaluate_benchmark.py` (lines 264-271)

**Current `resolve_test_csv` function:**
```python
def resolve_test_csv(job: EvalJob, project_root: Path) -> Path:
    """Resolve the test CSV path for a given job."""
    data_root = project_root / "data" / "megaseg_13structs"

    if job.test_set == "test_new":
        return data_root / "test_new" / "test_new.csv"
    else:
        return data_root / job.test_struct / "test" / "test.csv"
```

**Updated `resolve_test_csv` function:**
```python
def resolve_test_csv(job: EvalJob, project_root: Path) -> Path:
    """Resolve the test CSV path for a given job."""
    data_root = project_root / ".." / "data"

    if job.test_set == "test_new":
        return data_root / "standardized_test" / "test_new.csv"
    else:
        return data_root / "variance-100x" / job.test_struct / "test.csv"
```

Note: `project_root` is the `benchmark/` directory (from `PROJECT_ROOT` env var). Going up one level with `..` reaches the repo root where `data/` lives.

---

### Task 8: Update Test Assertions

**Files:**
- Modify: `benchmark/scripts/test_evaluate_benchmark.py` (lines 186, 196)

**Change 1 (line 186):**
- Old: `== root / "data/megaseg_13structs/ACTN1/test/test.csv"`
- New: `== root / "../data/variance-100x/ACTN1/test.csv"`

**Change 2 (line 196):**
- Old: `== root / "data/megaseg_13structs/test_new/test_new.csv"`
- New: `== root / "../data/standardized_test/test_new.csv"`

**Run tests:**
```bash
cd /allen/aics/rep_learn/MegaSeg/benchmark
python -m pytest scripts/test_evaluate_benchmark.py -v
```

Expected: All 32 tests pass.

---

### Task 9: Update .gitignore

**Files:**
- Modify: `/allen/aics/rep_learn/MegaSeg/.gitignore`

**New `.gitignore` (replace existing):**
```gitignore
# Benchmark outputs (large, generated at runtime)
benchmark/outputs/
benchmark/cache/

# Checkpoint weights (large binary files)
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

# Environment / IDE
.venv/
.pytest_cache/
uv.lock
```

---

### Task 10: Update Root README.md

**Files:**
- Modify: `/allen/aics/rep_learn/MegaSeg/README.md`

Replace the existing short README with a comprehensive one that covers all three components:

1. **Overview** — what MegaSeg is, what this repo contains
2. **Repository Structure** — top-level directory overview
3. **segmenter-classic/** — brief description, link to notebooks
4. **segmenter-ml/** — brief description, training config overview
5. **benchmark/** — detailed section with:
   - Quick start (PROJECT_ROOT, conda activate, eval command)
   - Key results summary (MegaSeg +0.08 dice over individual models)
   - Link to benchmark/README.md for full details
6. **Data** — description of the data/ directory structure

---

### Task 11: Update benchmark/README.md

**Files:**
- Modify: `/allen/aics/rep_learn/MegaSeg/benchmark/README.md`

Update the benchmark README for the new layout:
- Change directory structure diagram (remove `data/megaseg_13structs/`, show `../data/variance-100x/`)
- Update quick start commands to reflect the new repo structure
- Update cyto-dl reference to mention it's a submodule (run `git submodule update --init` after cloning)
- Update any references to `megaseg_13structs` path pattern

---

### Task 12: Commit, Squash, Push, and Create PR

**Step 1: Stage all changes**

```bash
cd /allen/aics/rep_learn/MegaSeg
git add -A
git status  # Review what's staged
```

**Step 2: Create a single commit**

```bash
git commit -m "feat: add benchmark evaluation framework for MegaSeg vs individual models

Add comprehensive benchmark infrastructure comparing MegaSeg (multi-structure)
models against individual (per-structure) models across 14 biological structures.

Includes:
- 204 Hydra experiment configs for 4 eval types (individual, individual_test_new,
  megaseg with 4 backbones, cross_eval)
- Evaluation pipeline with metrics computation (Dice, IoU, Precision, Recall, Accuracy)
- Batch job runner with multi-GPU support
- Visualization plots (6 types × 5 metrics)
- Complete uint8 results: MegaSeg DynUNet achieves 0.8684 mean Dice vs
  0.7671 for individual models (+0.1013 improvement)
- cyto-dl pinned at commit fe8fa7ac as git submodule"
```

**Step 3: Push the branch**

```bash
cd /allen/aics/rep_learn/MegaSeg
git push -u origin benchmark-overhaul
```

**Step 4: Create PR**

```bash
/allen/aics/users/ashwin.samudre/miniconda3/bin/gh pr create \
  --title "feat: add benchmark evaluation framework" \
  --body "$(cat <<'EOF'
## Summary

- Add `benchmark/` directory with complete evaluation framework comparing MegaSeg (multi-structure) models vs individual (per-structure) models across 14 biological structures
- 204 Hydra experiment configs, evaluation scripts, batch job runner, visualization pipeline
- Complete uint8 results committed: 18,050 samples evaluated, MegaSeg DynUNet achieves 0.8684 mean Dice vs 0.7671 for individual (+13.2% improvement)
- cyto-dl added as git submodule (pinned at commit fe8fa7ac)
- Data paths updated to use existing `data/variance-100x/` layout

## Key Results
- MegaSeg outperforms individual models on 13/14 structures
- Best backbone: DynUNet (0.8684), followed by SwinUNETR (0.8601), AttentionUNet (0.8586), UNet (0.8329)
- Cross-eval confirms individual models don't generalize (diagonal 0.77 vs off-diagonal 0.49)
EOF
)"
```

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `benchmark/` (entire dir) | Create | Copy from megaseg-benchmark |
| `benchmark/cyto-dl/` | Create | Git submodule (replaces vendored copy) |
| `.gitmodules` | Create | Submodule config |
| `benchmark/configs/paths/default.yaml` | Modify | `data_dir` → `${paths.root_dir}/../data` |
| `benchmark/scripts/generate_configs.py` | Modify | 5 path template changes |
| 204 YAML configs | Regenerate | Via `generate_configs.py` |
| `benchmark/scripts/evaluate_benchmark.py` | Modify | `resolve_test_csv()` path logic |
| `benchmark/scripts/test_evaluate_benchmark.py` | Modify | 2 test assertion paths |
| `.gitignore` | Modify | Merged rules for benchmark |
| `README.md` (root) | Modify | Comprehensive overview |
| `benchmark/README.md` | Modify | Updated for new layout |
