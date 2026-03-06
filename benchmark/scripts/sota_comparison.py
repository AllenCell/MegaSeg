#!/usr/bin/env python
"""SOTA comparison: aggregate metrics from all methods and generate figures.

Combines per-sample metrics from:
  - MegaSeg (DynUNet) - from existing per_sample_metrics.csv
  - Individual (per-struct ML) - from existing per_sample_metrics.csv (eval_type=individual)
  - Classic (aicssegmentation) - from classic_cross_eval_metrics.csv
  - nnU-Net v2 - from nnunet_metrics.csv
  - CellPose v4 (zero-shot) - from cellpose_zeroshot_metrics.csv
  - SAM vit_b (zero-shot) - from sam_zeroshot_metrics.csv
  - Otsu (thresholding baseline) - from otsu_baseline_metrics.csv
  - StarDist 3D (zero-shot) - from stardist_zeroshot_metrics.csv

Outputs:
  - results/uint8/sota_comparison.csv (unified per-sample metrics)
  - results/uint8/sota_summary.csv (mean +/- std per method per structure)
  - results/uint8/figures/sota/ (comparison figures)

Usage:
    python scripts/sota_comparison.py
    python scripts/sota_comparison.py --skip-figures
    python scripts/sota_comparison.py --methods megaseg nnunet cellpose sam otsu stardist
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRUCTURES = [
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]

METRIC_COLS = ["dice", "iou", "precision", "recall", "accuracy"]

PROJECT_ROOT = Path("/allen/aics/rep_learn/MegaSeg/benchmark")
RESULTS_DIR = PROJECT_ROOT / "results" / "uint8"

# Method display names and colors
METHOD_CONFIG = {
    "megaseg": {
        "label": "MegaSeg (DynUNet)",
        "color": "#1f77b4",
        "short": "MegaSeg",
    },
    "classic": {
        "label": "Classic (aicssegmentation)",
        "color": "#7f7f7f",
        "short": "Classic",
    },
    "individual": {
        "label": "Individual (per-struct ML)",
        "color": "#bcbd22",
        "short": "Individual",
    },
    "nnunet": {
        "label": "nnU-Net v2",
        "color": "#2ca02c",
        "short": "nnU-Net",
    },
    "cellpose": {
        "label": "CellPose v4 (zero-shot)",
        "color": "#ff7f0e",
        "short": "CellPose",
    },
    "sam": {
        "label": "SAM vit_b (zero-shot)",
        "color": "#d62728",
        "short": "SAM",
    },
    "otsu": {
        "label": "Otsu (thresholding)",
        "color": "#9467bd",
        "short": "Otsu",
    },
    "stardist": {
        "label": "StarDist 3D (zero-shot)",
        "color": "#8c564b",
        "short": "StarDist",
    },
}

# Default method ordering for plots (trained methods first, then zero-shot/classical)
METHOD_ORDER = [
    "megaseg",
    "nnunet",
    "individual",
    "classic",
    "cellpose",
    "sam",
    "stardist",
    "otsu",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_megaseg_metrics() -> pd.DataFrame | None:
    """Load MegaSeg DynUNet metrics from per_sample_metrics.csv.

    Filters to: eval_type=megaseg, backbone=dynunet, test_set=standard,
    and only rows where model_struct == test_struct (same-structure eval).
    """
    csv_path = RESULTS_DIR / "per_sample_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # MegaSeg DynUNet, standard test, same-structure
    mask = (
        (df["eval_type"] == "megaseg")
        & (df["backbone"] == "dynunet")
        & (df["test_set"] == "standard")
        & (df["model_struct"] == df["test_struct"])
    )
    subset = df[mask].copy()

    if subset.empty:
        print("  WARNING: No MegaSeg DynUNet results found")
        return None

    # Normalize columns
    subset = subset.rename(columns={"test_struct": "structure"})
    subset["method"] = "megaseg"
    return subset[["method", "structure", "sample_id"] + METRIC_COLS]


def load_classic_metrics() -> pd.DataFrame | None:
    """Load classic aicssegmentation cross-eval metrics.

    Reads from classic_cross_eval_metrics.csv (produced by
    run_classic_cross_eval.py). Uses same-structure rows only
    (model_struct == test_struct) for the SOTA comparison.
    """
    csv_path = RESULTS_DIR / "classic_cross_eval_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found (run run_classic_cross_eval.py first)")
        return None

    df = pd.read_csv(csv_path)

    # Same-structure only for SOTA comparison
    mask = df["model_struct"] == df["test_struct"]
    subset = df[mask].copy()

    if subset.empty:
        print("  WARNING: No classic same-structure results found")
        return None

    subset = subset.rename(columns={"test_struct": "structure"})
    subset["method"] = "classic"
    return subset[["method", "structure", "sample_id"] + METRIC_COLS]


def load_individual_metrics() -> pd.DataFrame | None:
    """Load per-structure individually-trained ML model metrics.

    Filters to: eval_type=individual, test_set=standard,
    same-structure (model_struct == test_struct).

    Note: these are per-structure trained ML models, NOT classic
    aicssegmentation workflows.
    """
    csv_path = RESULTS_DIR / "per_sample_metrics.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    mask = (
        (df["eval_type"] == "individual")
        & (df["test_set"] == "standard")
        & (df["model_struct"] == df["test_struct"])
    )
    subset = df[mask].copy()

    if subset.empty:
        print("  WARNING: No individual model results found")
        return None

    subset = subset.rename(columns={"test_struct": "structure"})
    subset["method"] = "individual"
    return subset[["method", "structure", "sample_id"] + METRIC_COLS]


def load_nnunet_metrics() -> pd.DataFrame | None:
    """Load nnU-Net metrics from nnunet_metrics.csv."""
    csv_path = RESULTS_DIR / "nnunet_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found (nnU-Net not yet complete?)")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df = df.rename(columns={"test_struct": "structure"})
    df["method"] = "nnunet"
    return df[["method", "structure", "sample_id"] + METRIC_COLS]


def load_cellpose_metrics() -> pd.DataFrame | None:
    """Load CellPose zero-shot metrics from cellpose_zeroshot_metrics.csv."""
    csv_path = RESULTS_DIR / "cellpose_zeroshot_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found (CellPose not yet complete?)")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df = df.rename(columns={"test_struct": "structure"})
    df["method"] = "cellpose"
    return df[["method", "structure", "sample_id"] + METRIC_COLS]


def load_sam_metrics() -> pd.DataFrame | None:
    """Load SAM zero-shot metrics from sam_zeroshot_metrics.csv."""
    csv_path = RESULTS_DIR / "sam_zeroshot_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found (SAM not yet complete?)")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df = df.rename(columns={"test_struct": "structure"})
    df["method"] = "sam"
    return df[["method", "structure", "sample_id"] + METRIC_COLS]


def load_otsu_metrics() -> pd.DataFrame | None:
    """Load Otsu thresholding baseline metrics from otsu_baseline_metrics.csv."""
    csv_path = RESULTS_DIR / "otsu_baseline_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found (Otsu not yet complete?)")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df = df.rename(columns={"test_struct": "structure"})
    df["method"] = "otsu"
    return df[["method", "structure", "sample_id"] + METRIC_COLS]


def load_stardist_metrics() -> pd.DataFrame | None:
    """Load StarDist 3D zero-shot metrics from stardist_zeroshot_metrics.csv."""
    csv_path = RESULTS_DIR / "stardist_zeroshot_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found (StarDist not yet complete?)")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    df = df.rename(columns={"test_struct": "structure"})
    df["method"] = "stardist"
    return df[["method", "structure", "sample_id"] + METRIC_COLS]


LOADERS = {
    "megaseg": load_megaseg_metrics,
    "individual": load_individual_metrics,
    "classic": load_classic_metrics,
    "nnunet": load_nnunet_metrics,
    "cellpose": load_cellpose_metrics,
    "sam": load_sam_metrics,
    "otsu": load_otsu_metrics,
    "stardist": load_stardist_metrics,
}


def load_all_metrics(methods: list[str] | None = None) -> pd.DataFrame:
    """Load and combine metrics from all available methods."""
    if methods is None:
        methods = list(LOADERS.keys())

    dfs = []
    for method in methods:
        if method not in LOADERS:
            print(f"  WARNING: Unknown method '{method}', skipping")
            continue
        print(f"Loading {method}...")
        df = LOADERS[method]()
        if df is not None and not df.empty:
            dfs.append(df)
            print(f"  Loaded {len(df)} samples")
        else:
            print(f"  No data available")

    if not dfs:
        print("ERROR: No metrics loaded from any method")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std per method per structure."""
    records = []

    for method in df["method"].unique():
        method_df = df[df["method"] == method]

        # Per structure
        for struct in STRUCTURES:
            struct_df = method_df[method_df["structure"] == struct]
            if struct_df.empty:
                continue
            record = {
                "method": method,
                "structure": struct,
                "n_samples": len(struct_df),
            }
            for metric in METRIC_COLS:
                record[metric] = struct_df[metric].mean()
                record[f"{metric}_std"] = struct_df[metric].std()
            records.append(record)

        # Overall
        record = {
            "method": method,
            "structure": "OVERALL",
            "n_samples": len(method_df),
        }
        for metric in METRIC_COLS:
            record[metric] = method_df[metric].mean()
            record[f"{metric}_std"] = method_df[metric].std()
        records.append(record)

    return pd.DataFrame(records)


def print_comparison_table(summary: pd.DataFrame, metric: str = "dice") -> None:
    """Print a formatted comparison table to stdout."""
    # Pivot: structures as rows, methods as columns
    overall = summary[summary["structure"] == "OVERALL"]
    per_struct = summary[summary["structure"] != "OVERALL"]

    # Create pivot
    pivot = per_struct.pivot(index="structure", columns="method", values=metric)

    # Reorder columns by METHOD_ORDER
    present = [m for m in METHOD_ORDER if m in pivot.columns]
    pivot = pivot[present]

    # Add OVERALL row
    overall_row = {}
    for method in present:
        row = overall[overall["method"] == method]
        if not row.empty:
            overall_row[method] = row[metric].values[0]
    pivot.loc["OVERALL"] = overall_row

    # Format with labels
    pivot.columns = [METHOD_CONFIG.get(m, {}).get("short", m) for m in present]

    print(f"\n{'=' * 80}")
    print(f"SOTA Comparison - {metric.upper()} (mean per structure)")
    print(f"{'=' * 80}")
    print(pivot.round(4).to_string())
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def setup_plot_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.{{pdf,png}} to {outdir}")


def plot_sota_bar_chart(
    df: pd.DataFrame, metric: str, outdir: Path, methods: list[str] | None = None
) -> None:
    """Grouped bar chart: methods x structures."""
    if methods is None:
        methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    stats = (
        df[df["method"].isin(methods)]
        .groupby(["structure", "method"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    n_structs = len(STRUCTURES)
    n_methods = len(methods)
    bar_width = 0.8 / max(n_methods, 1)
    x = np.arange(n_structs)

    for i, method in enumerate(methods):
        method_data = stats[stats["method"] == method].set_index("structure")
        means = [
            method_data.loc[s, "mean"] if s in method_data.index else 0
            for s in STRUCTURES
        ]
        stds = [
            method_data.loc[s, "std"] if s in method_data.index else 0
            for s in STRUCTURES
        ]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        cfg = METHOD_CONFIG.get(method, {})
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            label=cfg.get("label", method),
            color=cfg.get("color", "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(
        f"{metric.upper()} - SOTA Method Comparison by Structure",
        fontsize=14,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STRUCTURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, f"sota_comparison_{metric}")


def plot_sota_overall_bar(summary: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Bar chart: overall mean metric per method."""
    overall = summary[summary["structure"] == "OVERALL"].copy()
    if overall.empty:
        return

    # Order by METHOD_ORDER
    present = [m for m in METHOD_ORDER if m in overall["method"].values]
    overall = overall.set_index("method").loc[present].reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [METHOD_CONFIG.get(m, {}).get("color", "#333") for m in present]
    labels = [METHOD_CONFIG.get(m, {}).get("label", m) for m in present]
    means = overall[metric].values
    stds = overall[f"{metric}_std"].values

    x = np.arange(len(present))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.6)

    # Annotate values
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(
        f"Overall {metric.upper()} - SOTA Method Comparison",
        fontsize=14,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, f"sota_overall_{metric}")


def plot_sota_box(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Box plot: distribution of metric per method across all structures."""
    present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    subset = df[df["method"].isin(present)].copy()

    # Map to display labels for ordering
    subset["method_label"] = subset["method"].map(
        lambda m: METHOD_CONFIG.get(m, {}).get("label", m)
    )
    label_order = [METHOD_CONFIG.get(m, {}).get("label", m) for m in present]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {
        METHOD_CONFIG.get(m, {}).get("label", m): METHOD_CONFIG.get(m, {}).get(
            "color", "#333"
        )
        for m in present
    }

    sns.boxplot(
        data=subset,
        x="method_label",
        y=metric,
        order=label_order,
        palette=palette,
        ax=ax,
        width=0.6,
        fliersize=2,
    )

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(
        f"{metric.upper()} Distribution - SOTA Comparison", fontsize=14, pad=12
    )
    ax.set_ylim(-0.05, 1.05)
    plt.xticks(rotation=15, ha="right")

    save_figure(fig, outdir, f"sota_boxplot_{metric}")


def plot_sota_heatmap(summary: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Heatmap: methods (cols) x structures (rows), colored by metric value."""
    per_struct = summary[summary["structure"] != "OVERALL"]
    present = [m for m in METHOD_ORDER if m in per_struct["method"].unique()]

    pivot = per_struct.pivot(index="structure", columns="method", values=metric)
    pivot = pivot[present]
    pivot = pivot.reindex(STRUCTURES)
    pivot.columns = [METHOD_CONFIG.get(m, {}).get("short", m) for m in present]

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric.capitalize(), "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(f"{metric.upper()} - SOTA Comparison Heatmap", fontsize=14, pad=12)
    ax.set_ylabel("Structure", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    plt.yticks(rotation=0)

    save_figure(fig, outdir, f"sota_heatmap_{metric}")


def generate_all_figures(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    """Generate all SOTA comparison figures."""
    setup_plot_style()

    for metric in ["dice", "iou"]:
        print(f"\nGenerating figures for {metric}...")
        metric_dir = outdir / metric
        plot_sota_bar_chart(df, metric, metric_dir)
        plot_sota_overall_bar(summary, metric, metric_dir)
        plot_sota_box(df, metric, metric_dir)
        plot_sota_heatmap(summary, metric, metric_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SOTA segmentation comparison")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to include (default: all available)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(RESULTS_DIR / "figures" / "sota"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)

    # Load all metrics
    print("=" * 60)
    print("Loading metrics from all methods")
    print("=" * 60)
    df = load_all_metrics(args.methods)
    print(f"\nTotal: {len(df)} samples across {df['method'].nunique()} methods")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Structures: {sorted(df['structure'].unique())}")

    # Save unified per-sample CSV
    csv_path = output_dir / "sota_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved unified metrics to {csv_path}")

    # Compute and save summary
    summary = compute_summary(df)
    summary_path = output_dir / "sota_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print comparison tables
    for metric in ["dice", "iou"]:
        print_comparison_table(summary, metric)

    # Generate figures
    if not args.skip_figures:
        print("\n" + "=" * 60)
        print("Generating comparison figures")
        print("=" * 60)
        generate_all_figures(df, summary, figure_dir)
        print(f"\nAll figures saved to {figure_dir}")
    else:
        print("\nSkipping figure generation (--skip-figures)")

    print("\nDone.")


if __name__ == "__main__":
    main()
