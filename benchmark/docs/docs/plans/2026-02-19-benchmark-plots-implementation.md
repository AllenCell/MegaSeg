# Benchmark Visualization Plots — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `plot` subcommand to `scripts/evaluate_benchmark.py` that generates 6 publication-quality visualization plots for all 5 metrics in both PDF and PNG formats.

**Architecture:** Extend the existing two-phase pipeline (compute → report) with a third phase (plot). All 6 plot functions read from `per_sample_metrics.csv`, filter/aggregate internally, and save to `results/{dtype}/figures/{metric}/`. Uses seaborn + matplotlib hybrid approach.

**Tech Stack:** matplotlib 3.10.3, seaborn 0.13.2, pandas, numpy (all already installed in `acm-eval` conda env)

---

### Task 1: Add plot infrastructure (style setup, save helper, CLI subcommand)

**Files:**
- Modify: `scripts/evaluate_benchmark.py` (add imports, style setup, save helper, CLI wiring)
- Test: `scripts/test_evaluate_benchmark.py` (add basic plot tests)

**Step 1: Add matplotlib/seaborn imports and style constants**

Add after line 27 (`from tqdm import tqdm`) in evaluate_benchmark.py:

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
```

Add after `EVAL_TYPES` list (line 56):

```python
# Plot configuration
MODEL_COLORS = {
    "individual": "#7f7f7f",       # gray
    "megaseg_dynunet": "#1f77b4",  # blue
    "megaseg_attention_unet": "#ff7f0e",  # orange
    "megaseg_swin_unetr": "#2ca02c",     # green
    "megaseg_unet": "#d62728",           # red
}

BACKBONE_COLORS = {
    "dynunet": "#1f77b4",
    "attention_unet": "#ff7f0e",
    "swin_unetr": "#2ca02c",
    "unet": "#d62728",
}

BACKBONE_LABELS = {
    "dynunet": "DynUNet",
    "attention_unet": "AttentionUNet",
    "swin_unetr": "SwinUNETR",
    "unet": "UNet",
}
```

**Step 2: Add setup_plot_style() and save_figure() helper functions**

Add before the CLI section (before `parse_args`, around line 740):

```python
# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def setup_plot_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    log.info("Saved %s.{pdf,png} to %s", name, outdir)
```

**Step 3: Add the `plot` subcommand to CLI**

In `parse_args()`, add after the `p_report` block:

```python
    # -- plot --
    p_plot = sub.add_parser("plot", help="Generate visualization plots from CSV")
    p_plot.add_argument("--dtype", required=True, choices=["uint8", "float32"])
    p_plot.add_argument("--metric", choices=METRIC_COLS, default=None,
                        help="Generate plots for this metric only (default: all)")
    p_plot.add_argument("--outdir", type=Path, default=None,
                        help="Output directory (default: results/{dtype}/figures)")
```

In `main()`, add after the `elif args.command == "report":` block:

```python
    elif args.command == "plot":
        plot_all(
            project_root=args.project_root,
            dtype=args.dtype,
            metric=args.metric,
            outdir=args.outdir,
        )
```

**Step 4: Add stub plot_all() function**

```python
def plot_all(
    project_root: Path,
    dtype: str,
    metric: str | None = None,
    outdir: Path | None = None,
) -> None:
    """Generate all visualization plots."""
    df = _load_results(project_root, dtype)
    if outdir is None:
        outdir = project_root / "results" / dtype / "figures"

    setup_plot_style()

    metrics = [metric] if metric else METRIC_COLS
    for m in metrics:
        metric_dir = outdir / m
        log.info("Generating plots for metric=%s in %s", m, metric_dir)
        # Plot functions will be added in subsequent tasks
```

**Step 5: Add basic test for CLI wiring**

Add to `scripts/test_evaluate_benchmark.py`:

```python
def test_parse_args_plot(monkeypatch):
    """Verify plot subcommand parses correctly."""
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_benchmark.py", "plot", "--dtype", "uint8", "--metric", "dice"],
    )
    args = eb.parse_args()
    assert args.command == "plot"
    assert args.dtype == "uint8"
    assert args.metric == "dice"


def test_parse_args_plot_all_metrics(monkeypatch):
    """Verify plot subcommand with no --metric defaults to None (all metrics)."""
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_benchmark.py", "plot", "--dtype", "uint8"],
    )
    args = eb.parse_args()
    assert args.command == "plot"
    assert args.metric is None
```

**Step 6: Run tests**

Run: `conda run -n acm-eval python -m pytest scripts/test_evaluate_benchmark.py -v`
Expected: All tests pass (existing 30 + 2 new = 32)

**Step 7: Commit**

```bash
git add scripts/evaluate_benchmark.py scripts/test_evaluate_benchmark.py
git commit -m "feat: add plot subcommand infrastructure (style, save, CLI)"
```

---

### Task 2: Implement cross-eval heatmap (Plot 1)

**Files:**
- Modify: `scripts/evaluate_benchmark.py`

**Step 1: Add plot_cross_eval_heatmap function**

Add after `save_figure`:

```python
def plot_cross_eval_heatmap(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 1: 14x14 cross-evaluation heatmap."""
    mask = df["eval_type"] == "cross_eval"
    subset = df[mask]
    if subset.empty:
        log.warning("No cross_eval data; skipping heatmap")
        return

    means = subset.groupby(["model_struct", "test_struct"])[metric].mean().reset_index()
    heatmap_data = means.pivot(index="model_struct", columns="test_struct", values=metric)
    heatmap_data = heatmap_data.reindex(index=STRUCTURES, columns=STRUCTURES)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric.capitalize(), "shrink": 0.8},
        ax=ax,
    )

    # Highlight diagonal
    for i in range(len(STRUCTURES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

    ax.set_title(f"{metric.upper()} — Cross-Evaluation: Individual Models", fontsize=14, pad=12)
    ax.set_xlabel("Tested On", fontsize=12)
    ax.set_ylabel("Trained On", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_figure(fig, outdir, "cross_eval_heatmap")
```

**Step 2: Wire into plot_all**

Add to `plot_all` inside the metrics loop:

```python
        plot_cross_eval_heatmap(df, m, metric_dir)
```

**Step 3: Test manually**

Run: `conda run -n acm-eval bash -c 'PROJECT_ROOT=/allen/aics/rep_learn/megaseg-benchmark python /allen/aics/rep_learn/megaseg-benchmark/scripts/evaluate_benchmark.py plot --dtype uint8 --metric dice'`

Verify: `ls results/uint8/figures/dice/cross_eval_heatmap.{pdf,png}` — both files exist.

**Step 4: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: add cross-eval heatmap plot"
```

---

### Task 3: Implement individual vs MegaSeg bar chart (Plot 2)

**Files:**
- Modify: `scripts/evaluate_benchmark.py`

**Step 1: Add plot_individual_vs_megaseg function**

```python
def plot_individual_vs_megaseg(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 2: Grouped bar chart — individual vs 4 MegaSeg backbones per structure."""
    mask = df["eval_type"].isin(["individual", "megaseg"]) & (df["test_set"] == "standard")
    subset = df[mask].copy()
    if subset.empty:
        log.warning("No individual/megaseg data; skipping bar chart")
        return

    subset["model_label"] = subset.apply(
        lambda r: "individual" if r["eval_type"] == "individual" else f"megaseg_{r['backbone']}",
        axis=1,
    )

    stats = subset.groupby(["test_struct", "model_label"])[metric].agg(["mean", "std"]).reset_index()

    model_order = ["individual", "megaseg_dynunet", "megaseg_attention_unet", "megaseg_swin_unetr", "megaseg_unet"]
    model_labels = {"individual": "Individual", **{f"megaseg_{k}": v for k, v in BACKBONE_LABELS.items()}}

    fig, ax = plt.subplots(figsize=(14, 6))
    n_structs = len(STRUCTURES)
    n_models = len(model_order)
    bar_width = 0.15
    x = np.arange(n_structs)

    for i, model in enumerate(model_order):
        model_data = stats[stats["model_label"] == model].set_index("test_struct")
        means = [model_data.loc[s, "mean"] if s in model_data.index else 0 for s in STRUCTURES]
        stds = [model_data.loc[s, "std"] if s in model_data.index else 0 for s in STRUCTURES]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            label=model_labels.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.upper()} — Individual vs MegaSeg by Structure", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(STRUCTURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "individual_vs_megaseg")
```

**Step 2: Wire into plot_all**

```python
        plot_individual_vs_megaseg(df, m, metric_dir)
```

**Step 3: Test manually, verify output files exist**

**Step 4: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: add individual vs MegaSeg bar chart plot"
```

---

### Task 4: Implement backbone comparison chart (Plot 3)

**Files:**
- Modify: `scripts/evaluate_benchmark.py`

**Step 1: Add plot_backbone_comparison function**

```python
def plot_backbone_comparison(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 3: Grouped bar chart — 4 MegaSeg backbones per structure."""
    mask = (df["eval_type"] == "megaseg") & (df["test_set"] == "standard")
    subset = df[mask]
    if subset.empty:
        log.warning("No megaseg data; skipping backbone comparison")
        return

    stats = subset.groupby(["test_struct", "backbone"])[metric].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    n_structs = len(STRUCTURES)
    n_bb = len(BACKBONES)
    bar_width = 0.18
    x = np.arange(n_structs)

    for i, bb in enumerate(BACKBONES):
        bb_data = stats[stats["backbone"] == bb].set_index("test_struct")
        means = [bb_data.loc[s, "mean"] if s in bb_data.index else 0 for s in STRUCTURES]
        stds = [bb_data.loc[s, "std"] if s in bb_data.index else 0 for s in STRUCTURES]
        offset = (i - n_bb / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            label=BACKBONE_LABELS.get(bb, bb),
            color=BACKBONE_COLORS.get(bb, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.upper()} — MegaSeg Backbone Comparison", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(STRUCTURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "backbone_comparison")
```

**Step 2: Wire into plot_all**

**Step 3: Test manually, verify output files exist**

**Step 4: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: add backbone comparison chart plot"
```

---

### Task 5: Implement delta improvement chart (Plot 4)

**Files:**
- Modify: `scripts/evaluate_benchmark.py`

**Step 1: Add plot_delta_improvement function**

```python
def plot_delta_improvement(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 4: Horizontal bar chart — MegaSeg improvement over individual, 2x2 grid."""
    mask_ind = (df["eval_type"] == "individual") & (df["test_set"] == "standard")
    mask_mega = (df["eval_type"] == "megaseg") & (df["test_set"] == "standard")
    ind_means = df[mask_ind].groupby("test_struct")[metric].mean()
    mega_means = df[mask_mega].groupby(["backbone", "test_struct"])[metric].mean()

    if ind_means.empty or mega_means.empty:
        log.warning("Insufficient data for delta improvement chart")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, bb in enumerate(BACKBONES):
        ax = axes[idx]
        if bb in mega_means.index:
            bb_means = mega_means.loc[bb]
        else:
            ax.set_visible(False)
            continue

        # Compute deltas and sort
        common = sorted(set(bb_means.index) & set(ind_means.index))
        deltas = pd.Series(
            {s: bb_means[s] - ind_means[s] for s in common}
        ).sort_values()

        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas.values]
        y_pos = np.arange(len(deltas))

        ax.barh(y_pos, deltas.values, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(deltas.index, fontsize=9)
        ax.set_xlabel(f"Δ {metric.capitalize()}", fontsize=10)
        ax.set_title(BACKBONE_LABELS.get(bb, bb), fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

        # Annotate values
        for j, (val, name) in enumerate(zip(deltas.values, deltas.index)):
            ha = "left" if val >= 0 else "right"
            offset = 0.005 if val >= 0 else -0.005
            ax.text(val + offset, j, f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    fig.suptitle(
        f"{metric.upper()} — MegaSeg Improvement Over Individual",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, outdir, "delta_improvement")
```

**Step 2: Wire into plot_all**

**Step 3: Test manually, verify output files exist**

**Step 4: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: add delta improvement chart plot"
```

---

### Task 6: Implement test_new generalization chart (Plot 5)

**Files:**
- Modify: `scripts/evaluate_benchmark.py`

**Step 1: Add plot_test_new_generalization function**

```python
def plot_test_new_generalization(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 5: Grouped bar chart — individual vs MegaSeg on held-out test_new structures."""
    mask = df["test_set"] == "test_new"
    subset = df[mask].copy()
    if subset.empty:
        log.warning("No test_new data; skipping generalization chart")
        return

    subset["model_label"] = subset.apply(
        lambda r: "individual" if r["eval_type"] == "individual_test_new"
        else f"megaseg_{r['backbone']}",
        axis=1,
    )

    stats = subset.groupby(["test_struct", "model_label"])[metric].agg(["mean", "std"]).reset_index()
    structures = sorted(stats["test_struct"].unique())

    model_order = ["individual", "megaseg_dynunet", "megaseg_attention_unet", "megaseg_swin_unetr", "megaseg_unet"]
    model_labels = {"individual": "Individual", **{f"megaseg_{k}": v for k, v in BACKBONE_LABELS.items()}}
    # Filter to models actually present in data
    present_models = [m for m in model_order if m in stats["model_label"].values]

    fig, ax = plt.subplots(figsize=(max(10, len(structures) * 0.8), 6))
    n_structs = len(structures)
    n_models = len(present_models)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(n_structs)

    for i, model in enumerate(present_models):
        model_data = stats[stats["model_label"] == model].set_index("test_struct")
        means = [model_data.loc[s, "mean"] if s in model_data.index else 0 for s in structures]
        stds = [model_data.loc[s, "std"] if s in model_data.index else 0 for s in structures]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=2,
            label=model_labels.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure (held-out)", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.upper()} — Generalization to Novel Structures (test_new)", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "test_new_generalization")
```

**Step 2: Wire into plot_all**

**Step 3: Test manually, verify output files exist**

**Step 4: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: add test_new generalization chart plot"
```

---

### Task 7: Implement summary overview chart (Plot 6)

**Files:**
- Modify: `scripts/evaluate_benchmark.py`

**Step 1: Add plot_summary_overview function**

```python
def plot_summary_overview(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 6: Overall mean metric by eval type — high-level comparison."""
    categories = []

    # Individual
    ind = df[(df["eval_type"] == "individual") & (df["test_set"] == "standard")]
    if not ind.empty:
        categories.append(("Individual", ind[metric].mean(), ind[metric].std(), MODEL_COLORS["individual"]))

    # MegaSeg backbones
    for bb in BACKBONES:
        mega = df[(df["eval_type"] == "megaseg") & (df["backbone"] == bb) & (df["test_set"] == "standard")]
        if not mega.empty:
            label = f"MegaSeg\n{BACKBONE_LABELS.get(bb, bb)}"
            categories.append((label, mega[metric].mean(), mega[metric].std(), BACKBONE_COLORS[bb]))

    # Cross-eval diagonal vs off-diagonal
    cross = df[df["eval_type"] == "cross_eval"]
    if not cross.empty:
        diag = cross[cross["model_struct"] == cross["test_struct"]]
        off_diag = cross[cross["model_struct"] != cross["test_struct"]]
        if not diag.empty:
            categories.append(("Cross-Eval\n(same struct)", diag[metric].mean(), diag[metric].std(), "#9467bd"))
        if not off_diag.empty:
            categories.append(("Cross-Eval\n(transfer)", off_diag[metric].mean(), off_diag[metric].std(), "#e377c2"))

    if not categories:
        log.warning("No data for summary overview")
        return

    labels, means, stds, colors = zip(*categories)

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.2), 5))
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.6)

    # Reference line at individual mean
    if ind_mean := (ind[metric].mean() if not ind.empty else None):
        ax.axhline(y=ind_mean, color=MODEL_COLORS["individual"], linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Individual baseline ({ind_mean:.3f})")

    # Annotate bar values
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean_val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model / Evaluation Type", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.upper()} — Summary Overview", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "summary_overview")
```

**Step 2: Wire into plot_all**

**Step 3: Test manually, verify output files exist**

**Step 4: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: add summary overview chart plot"
```

---

### Task 8: Complete plot_all wiring, run full generation, verify all outputs

**Files:**
- Modify: `scripts/evaluate_benchmark.py` (finalize plot_all)

**Step 1: Finalize plot_all with all 6 plot calls and conditional execution**

The final `plot_all` function should be:

```python
def plot_all(
    project_root: Path,
    dtype: str,
    metric: str | None = None,
    outdir: Path | None = None,
) -> None:
    """Generate all visualization plots."""
    df = _load_results(project_root, dtype)
    if outdir is None:
        outdir = project_root / "results" / dtype / "figures"

    setup_plot_style()

    metrics = [metric] if metric else METRIC_COLS
    for m in metrics:
        metric_dir = outdir / m
        log.info("Generating plots for metric=%s in %s", m, metric_dir)

        plot_cross_eval_heatmap(df, m, metric_dir)
        plot_individual_vs_megaseg(df, m, metric_dir)
        plot_backbone_comparison(df, m, metric_dir)
        plot_delta_improvement(df, m, metric_dir)
        plot_test_new_generalization(df, m, metric_dir)
        plot_summary_overview(df, m, metric_dir)

    log.info("All plots saved to %s", outdir)
```

**Step 2: Run full generation for all metrics**

Run: `conda run -n acm-eval bash -c 'PROJECT_ROOT=/allen/aics/rep_learn/megaseg-benchmark python /allen/aics/rep_learn/megaseg-benchmark/scripts/evaluate_benchmark.py plot --dtype uint8'`

**Step 3: Verify all 60 output files exist**

Run: `find results/uint8/figures -type f | sort | wc -l` — expect 60
Run: `find results/uint8/figures -type f -name "*.pdf" | wc -l` — expect 30
Run: `find results/uint8/figures -type f -name "*.png" | wc -l` — expect 30

**Step 4: Run all tests**

Run: `conda run -n acm-eval python -m pytest scripts/test_evaluate_benchmark.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "feat: complete benchmark visualization with all 6 plot types"
```

---

### Task 9: Update script docstring and verify everything

**Files:**
- Modify: `scripts/evaluate_benchmark.py` (update module docstring)

**Step 1: Update the module docstring at the top of the file**

Replace the existing docstring (lines 2-11) with:

```python
"""
Quantitative evaluation of MegaSeg benchmark segmentation models.

Three-phase pipeline:
  compute — reads prediction/GT TIFF pairs, computes metrics, writes per-sample CSV
  report  — reads per-sample CSV, generates comparison tables
  plot    — reads per-sample CSV, generates publication-quality visualization plots

Usage:
    python scripts/evaluate_benchmark.py compute --dtype uint8 --workers 16
    python scripts/evaluate_benchmark.py report --dtype uint8 --metric dice
    python scripts/evaluate_benchmark.py plot --dtype uint8
    python scripts/evaluate_benchmark.py plot --dtype uint8 --metric dice
"""
```

**Step 2: Final commit**

```bash
git add scripts/evaluate_benchmark.py
git commit -m "docs: update evaluate_benchmark.py docstring for three-phase pipeline"
```
