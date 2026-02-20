#!/usr/bin/env python
"""Evaluate fine-tuning experiment results.

Computes per-image Dice and IoU metrics for each of the 7 LMNB1
fine-tuning experiments (L-shaped design) and generates summary
tables and plots.

Usage:
    uv run python scripts/evaluate_finetune.py compute --workers 8
    uv run python scripts/evaluate_finetune.py report
    uv run python scripts/evaluate_finetune.py plot
"""

import argparse
import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# L-shaped experiment design:
# - Vary epochs with 100% data: converged, 10, 50, 100 epochs
# - Vary data fraction with convergence: 10%, 20%, 50%, 100%
ALL_EXPERIMENTS = [
    "data100_converged",
    "data100_10epochs",
    "data100_50epochs",
    "data100_100epochs",
    "data10_converged",
    "data20_converged",
    "data50_converged",
]

METRIC_COLS = ["dice", "iou"]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute Dice and IoU from two binary masks."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp / dice_denom) if dice_denom > 0 else 1.0

    iou_denom = tp + fp + fn
    iou = (tp / iou_denom) if iou_denom > 0 else 1.0

    return {"dice": dice, "iou": iou}


def binarize_prediction(pred: np.ndarray) -> np.ndarray:
    """Binarize uint8 prediction (threshold 128)."""
    if pred.dtype == np.uint8:
        return pred >= 128
    return pred >= 0.5


def binarize_gt(gt: np.ndarray) -> np.ndarray:
    """Binarize ground truth mask."""
    return gt > 0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve_pred_dir(experiment: str) -> Path:
    """Resolve the prediction directory for an experiment."""
    return PROJECT_ROOT / "outputs" / "finetune" / "lmnb1" / experiment / "seg"


def get_test_csv() -> Path:
    """Get the LMNB1 test CSV path."""
    return PROJECT_ROOT / "data" / "lmnb1" / "test_lmnb1.csv"


def derive_prediction_path(raw_filename: str, pred_dir: Path) -> Path:
    """Map a raw filename to its prediction path, handling .tif/.tiff."""
    name = Path(raw_filename).name
    exact = pred_dir / name
    if exact.is_file():
        return exact

    stem = exact.stem
    ext = exact.suffix.lower()
    if ext == ".tif":
        alt = pred_dir / (stem + ".tiff")
    elif ext == ".tiff":
        alt = pred_dir / (stem + ".tif")
    else:
        return exact

    return alt if alt.is_file() else exact


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------
def _evaluate_sample(args: tuple) -> dict:
    """Evaluate one prediction/GT pair."""
    pred_path_str, gt_path_str, metadata = args
    pred_path = Path(pred_path_str)
    gt_path = Path(gt_path_str)

    nan_result = {**metadata, "dice": np.nan, "iou": np.nan}

    try:
        gt_img = tifffile.imread(str(gt_path))
    except Exception as e:
        log.warning("Failed to load GT %s: %s", gt_path, e)
        return nan_result

    if not pred_path.is_file():
        log.warning("Prediction not found: %s", pred_path)
        return nan_result

    try:
        pred_img = tifffile.imread(str(pred_path))
    except Exception as e:
        log.warning("Failed to load prediction %s: %s", pred_path, e)
        return nan_result

    gt_mask = binarize_gt(gt_img)
    pred_mask = binarize_prediction(pred_img)

    if gt_mask.shape != pred_mask.shape:
        log.warning("Shape mismatch: GT %s vs Pred %s", gt_mask.shape, pred_mask.shape)
        return nan_result

    metrics = compute_metrics(gt_mask, pred_mask)
    return {**metadata, **metrics}


# ---------------------------------------------------------------------------
# Experiment metadata parsing
# ---------------------------------------------------------------------------
def parse_experiment_label(exp: str) -> dict:
    """Parse experiment label into data_pct and epoch_cfg."""
    if "_converged" in exp:
        data_pct_str = exp.replace("_converged", "").replace("data", "")
        return {"data_pct": int(data_pct_str), "epoch_cfg": "converged"}
    else:
        # data100_10epochs -> data_pct=100, epoch_cfg=10epochs
        parts = exp.split("_", 1)
        data_pct = int(parts[0].replace("data", ""))
        return {"data_pct": data_pct, "epoch_cfg": parts[1]}


# ---------------------------------------------------------------------------
# Compute phase
# ---------------------------------------------------------------------------
def compute_all(workers: int, experiments: list[str] | None = None) -> None:
    """Compute metrics for all experiments."""
    results_dir = PROJECT_ROOT / "results" / "finetune"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "per_sample_metrics.csv"

    test_csv = get_test_csv()
    if not test_csv.is_file():
        print(f"ERROR: Test CSV not found: {test_csv}", file=sys.stderr)
        print("Run: uv run python scripts/generate_lmnb1_data.py", file=sys.stderr)
        sys.exit(1)

    test_df = pd.read_csv(test_csv)
    if "raw" not in test_df.columns or "seg" not in test_df.columns:
        print("ERROR: Test CSV must have 'raw' and 'seg' columns.", file=sys.stderr)
        print(f"Found columns: {list(test_df.columns)}", file=sys.stderr)
        sys.exit(1)

    exp_list = experiments or ALL_EXPERIMENTS

    # Build work items
    all_items = []
    skipped = []
    for exp in exp_list:
        pred_dir = resolve_pred_dir(exp)
        if not pred_dir.is_dir():
            skipped.append(exp)
            continue

        exp_meta = parse_experiment_label(exp)

        for _, row in test_df.iterrows():
            pred_file = derive_prediction_path(row["raw"], pred_dir)
            sample_id = Path(row["raw"]).stem
            metadata = {
                "experiment": exp,
                "data_pct": exp_meta["data_pct"],
                "epoch_cfg": exp_meta["epoch_cfg"],
                "sample_id": sample_id,
            }
            all_items.append((str(pred_file), str(row["seg"]), metadata))

    if skipped:
        print(f"Skipped {len(skipped)} experiments (no predictions):")
        for s in skipped:
            print(f"  {s}")
        print()

    if not all_items:
        print("No work items. Are predict runs complete?")
        return

    print(
        f"Computing metrics for {len(all_items)} samples "
        f"across {len(exp_list) - len(skipped)} experiments ..."
    )

    with Pool(processes=min(workers, len(all_items))) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_evaluate_sample, all_items),
                total=len(all_items),
                desc="Evaluating",
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Print quick summary
    summary = df.groupby("experiment")[METRIC_COLS].mean()
    print(f"\n{summary.to_string()}")


# ---------------------------------------------------------------------------
# Report phase
# ---------------------------------------------------------------------------
def report() -> None:
    """Generate summary tables from per-sample metrics."""
    csv_path = PROJECT_ROOT / "results" / "finetune" / "per_sample_metrics.csv"
    if not csv_path.is_file():
        print(f"ERROR: Metrics not found: {csv_path}", file=sys.stderr)
        print(
            "Run: uv run python scripts/evaluate_finetune.py compute",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    results_dir = csv_path.parent

    # Summary by experiment
    summary = df.groupby("experiment")[METRIC_COLS].agg(["mean", "std", "count"])
    summary.to_csv(results_dir / "summary_stats.csv")
    print("Summary statistics by experiment:")
    print(summary.to_string())
    print(f"\nSaved to {results_dir / 'summary_stats.csv'}")

    # Epoch comparison (100% data): converged vs 10/50/100 epochs
    epoch_exps = [e for e in ALL_EXPERIMENTS if e.startswith("data100_")]
    epoch_df = df[df["experiment"].isin(epoch_exps)]
    if not epoch_df.empty:
        epoch_summary = epoch_df.groupby("experiment")[METRIC_COLS].agg(["mean", "std"])
        epoch_summary.to_csv(results_dir / "epoch_comparison.csv")
        print("\n\nEpoch comparison (100% data):")
        print(epoch_summary.to_string())

    # Data fraction comparison (converged): 10% vs 20% vs 50% vs 100%
    data_exps = [e for e in ALL_EXPERIMENTS if e.endswith("_converged")]
    data_df = df[df["experiment"].isin(data_exps)]
    if not data_df.empty:
        data_summary = data_df.groupby("experiment")[METRIC_COLS].agg(["mean", "std"])
        data_summary.to_csv(results_dir / "data_fraction_comparison.csv")
        print("\n\nData fraction comparison (converged):")
        print(data_summary.to_string())


# ---------------------------------------------------------------------------
# Plot phase
# ---------------------------------------------------------------------------
def plot() -> None:
    """Generate visualizations from per-sample metrics."""
    csv_path = PROJECT_ROOT / "results" / "finetune" / "per_sample_metrics.csv"
    if not csv_path.is_file():
        print("ERROR: Metrics not found. Run compute first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    results_dir = csv_path.parent

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Bar chart: mean Dice by experiment (all 7)
    fig, ax = plt.subplots(figsize=(12, 6))
    exp_order = [e for e in ALL_EXPERIMENTS if e in df["experiment"].unique()]
    summary = df.groupby("experiment")["dice"].agg(["mean", "std"]).reindex(exp_order)
    bars = ax.bar(
        range(len(summary)),
        summary["mean"],
        yerr=summary["std"],
        capsize=4,
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("LMNB1 Fine-Tuning: Dice by Experiment")
    ax.set_ylim(0, 1)
    # Add value labels on bars
    for bar, val in zip(bars, summary["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(results_dir / "bar_dice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart to {results_dir / 'bar_dice.png'}")

    # 2. Box plots: Dice distribution per experiment
    fig, ax = plt.subplots(figsize=(14, 6))
    order = [o for o in ALL_EXPERIMENTS if o in df["experiment"].unique()]
    sns.boxplot(data=df, x="experiment", y="dice", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Dice Score Distribution by Experiment")
    ax.set_ylabel("Dice Score")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(results_dir / "boxplot_dice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplot to {results_dir / 'boxplot_dice.png'}")

    # 3. Line plot: Epoch effect (100% data)
    epoch_exps = {
        "data100_10epochs": 10,
        "data100_50epochs": 50,
        "data100_100epochs": 100,
        "data100_converged": 1000,  # placeholder x-position for converged
    }
    epoch_df = df[df["experiment"].isin(epoch_exps.keys())].copy()
    if not epoch_df.empty:
        epoch_df["epochs"] = epoch_df["experiment"].map(epoch_exps)
        epoch_summary = (
            epoch_df.groupby("epochs")["dice"].agg(["mean", "std"]).reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            epoch_summary["epochs"],
            epoch_summary["mean"],
            yerr=epoch_summary["std"],
            fmt="o-",
            linewidth=2,
            capsize=5,
            markersize=8,
            color="steelblue",
        )
        # Mark the converged point specially
        converged_row = epoch_summary[epoch_summary["epochs"] == 1000]
        if not converged_row.empty:
            ax.plot(
                1000,
                converged_row["mean"].values[0],
                "*",
                markersize=15,
                color="red",
                zorder=5,
                label="converged",
            )
            ax.legend()
        ax.set_xlabel("Fine-Tuning Epochs")
        ax.set_ylabel("Mean Dice Score")
        ax.set_title("Effect of Fine-Tuning Duration (100% Data)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(results_dir / "epoch_effect.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved epoch effect plot to {results_dir / 'epoch_effect.png'}")

    # 4. Line plot: Data fraction effect (converged)
    data_exps = {
        "data10_converged": 10,
        "data20_converged": 20,
        "data50_converged": 50,
        "data100_converged": 100,
    }
    data_df = df[df["experiment"].isin(data_exps.keys())].copy()
    if not data_df.empty:
        data_df["data_pct"] = data_df["experiment"].map(data_exps)
        data_summary = (
            data_df.groupby("data_pct")["dice"].agg(["mean", "std"]).reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            data_summary["data_pct"],
            data_summary["mean"],
            yerr=data_summary["std"],
            fmt="s-",
            linewidth=2,
            capsize=5,
            markersize=8,
            color="darkorange",
        )
        ax.set_xlabel("Training Data Fraction (%)")
        ax.set_ylabel("Mean Dice Score")
        ax.set_title("Effect of Training Data Size (Converged Models)")
        ax.set_ylim(0, 1)
        ax.set_xticks([10, 20, 50, 100])
        plt.tight_layout()
        fig.savefig(
            results_dir / "data_fraction_effect.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved data fraction plot to {results_dir / 'data_fraction_effect.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuning experiments.")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    p_compute = subparsers.add_parser("compute", help="Compute metrics")
    p_compute.add_argument("--workers", type=int, default=min(8, cpu_count()))
    p_compute.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment labels",
    )

    subparsers.add_parser("report", help="Generate summary tables")
    subparsers.add_parser("plot", help="Generate visualizations")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if args.phase == "compute":
        exp_filter = None
        if args.experiments:
            exp_filter = [e.strip() for e in args.experiments.split(",")]
        compute_all(workers=args.workers, experiments=exp_filter)
    elif args.phase == "report":
        report()
    elif args.phase == "plot":
        plot()


if __name__ == "__main__":
    main()
