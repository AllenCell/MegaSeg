#!/usr/bin/env python
"""Evaluate fine-tuning experiment results.

Computes per-image Dice and IoU metrics for each of the 16 LMNB1
fine-tuning experiments and generates summary tables/plots.

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

DATA_FRACTIONS = [100, 50, 20, 10]
EPOCH_CONFIGS = ["epochs10", "epochs50", "epochs100", "earlystop"]
ALL_EXPERIMENTS = [f"data{pct}_{ec}" for pct in DATA_FRACTIONS for ec in EPOCH_CONFIGS]

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
    return PROJECT_ROOT / "data" / "lmnb1" / "test.csv"


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
    exp_list = experiments or ALL_EXPERIMENTS

    # Build work items
    all_items = []
    skipped = []
    for exp in exp_list:
        pred_dir = resolve_pred_dir(exp)
        if not pred_dir.is_dir():
            skipped.append(exp)
            continue

        # Parse experiment metadata
        parts = exp.split("_", 1)  # data100_epochs10 -> data100, epochs10
        data_pct = int(parts[0].replace("data", ""))
        epoch_cfg = parts[1]

        for _, row in test_df.iterrows():
            pred_file = derive_prediction_path(row["raw"], pred_dir)
            sample_id = Path(row["raw"]).stem
            metadata = {
                "experiment": exp,
                "data_pct": data_pct,
                "epoch_cfg": epoch_cfg,
                "sample_id": sample_id,
            }
            all_items.append((str(pred_file), str(row["seg"]), metadata))

    if skipped:
        print(f"Skipped {len(skipped)} experiments (no predictions):")
        for s in skipped:
            print(f"  {s}")
        print()

    if not all_items:
        print("No work items. Are training runs complete?")
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
            "Run: uv run python scripts/evaluate_finetune.py compute", file=sys.stderr
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)
    results_dir = csv_path.parent

    # Summary by experiment
    summary = df.groupby(["data_pct", "epoch_cfg"])[METRIC_COLS].agg(["mean", "std"])
    summary.to_csv(results_dir / "summary_stats.csv")
    print("Summary statistics:")
    print(summary.to_string())
    print(f"\nSaved to {results_dir / 'summary_stats.csv'}")

    # Pivot table: data_pct rows x epoch_cfg columns
    for metric in METRIC_COLS:
        pivot = df.pivot_table(
            values=metric, index="data_pct", columns="epoch_cfg", aggfunc="mean"
        )
        # Reorder columns
        col_order = [c for c in EPOCH_CONFIGS if c in pivot.columns]
        pivot = pivot[col_order]
        pivot = pivot.sort_index(ascending=False)
        pivot.to_csv(results_dir / f"pivot_{metric}.csv")
        print(f"\n{metric.upper()} by data fraction and epoch config:")
        print(pivot.to_string(float_format="%.4f"))


# ---------------------------------------------------------------------------
# Plot phase
# ---------------------------------------------------------------------------
def plot() -> None:
    """Generate visualizations from per-sample metrics."""
    csv_path = PROJECT_ROOT / "results" / "finetune" / "per_sample_metrics.csv"
    if not csv_path.is_file():
        print(f"ERROR: Metrics not found. Run compute first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    results_dir = csv_path.parent

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Heatmap: mean Dice by data fraction x epoch config
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, METRIC_COLS):
        pivot = df.pivot_table(
            values=metric, index="data_pct", columns="epoch_cfg", aggfunc="mean"
        )
        col_order = [c for c in EPOCH_CONFIGS if c in pivot.columns]
        pivot = pivot[col_order].sort_index(ascending=False)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=0.5,
        )
        ax.set_title(f"Mean {metric.upper()}")
        ax.set_xlabel("Epoch Config")
        ax.set_ylabel("Data Fraction (%)")

    plt.tight_layout()
    fig.savefig(results_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {results_dir / 'heatmap.png'}")

    # 2. Box plots: Dice distribution per experiment
    fig, ax = plt.subplots(figsize=(16, 6))
    order = [f"data{p}_{e}" for p in DATA_FRACTIONS for e in EPOCH_CONFIGS]
    order = [o for o in order if o in df["experiment"].unique()]
    sns.boxplot(data=df, x="experiment", y="dice", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Dice Score Distribution by Experiment")
    ax.set_ylabel("Dice Score")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(results_dir / "boxplot_dice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplot to {results_dir / 'boxplot_dice.png'}")

    # 3. Line plot: mean Dice vs epochs, grouped by data fraction
    fig, ax = plt.subplots(figsize=(10, 6))
    epoch_means = df[df["epoch_cfg"] != "earlystop"].copy()
    epoch_means["epochs"] = (
        epoch_means["epoch_cfg"].str.replace("epochs", "").astype(int)
    )
    summary = epoch_means.groupby(["data_pct", "epochs"])["dice"].mean().reset_index()

    for pct in DATA_FRACTIONS:
        subset = summary[summary["data_pct"] == pct].sort_values("epochs")
        ax.plot(
            subset["epochs"], subset["dice"], "o-", label=f"{pct}% data", linewidth=2
        )

    # Add early-stop points as stars
    es_data = df[df["epoch_cfg"] == "earlystop"].groupby("data_pct")["dice"].mean()
    for pct in DATA_FRACTIONS:
        if pct in es_data.index:
            ax.plot(1000, es_data[pct], "*", markersize=12)

    ax.set_xlabel("Max Epochs")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Fine-tuning Convergence by Data Fraction")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    fig.savefig(results_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence plot to {results_dir / 'convergence.png'}")


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
