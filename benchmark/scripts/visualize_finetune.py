#!/usr/bin/env python
"""
Generate figures for LMNB1 fine-tuning experiments.

1. Per-experiment GT vs prediction comparison figures (3 rows x 2 cols).
2. A 4x4 heatmap of mean Dice scores (data fraction x epochs).

Usage:
    uv run python scripts/visualize_finetune.py
    uv run python scripts/visualize_finetune.py --output-dir results/finetune/lmnb1/figures
    uv run python scripts/visualize_finetune.py --heatmap-only
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # benchmark/
DATA_ROOT = PROJECT_ROOT.parent / "data"

PRED_BASE = PROJECT_ROOT / "outputs" / "eval" / "finetune" / "lmnb1"
GT_DIR = DATA_ROOT / "variance-100x" / "LMNB1" / "seg"
RESULTS_DIR = PROJECT_ROOT / "results" / "finetune" / "lmnb1"

EXPERIMENTS = [
    "base_megaseg",
    # 100% data
    "data100_10epochs",
    "data100_50epochs",
    "data100_100epochs",
    "data100_converged",
    # 50% data
    "data50_10epochs",
    "data50_50epochs",
    "data50_100epochs",
    "data50_converged",
    # 20% data
    "data20_10epochs",
    "data20_50epochs",
    "data20_100epochs",
    "data20_converged",
    # 10% data
    "data10_10epochs",
    "data10_50epochs",
    "data10_100epochs",
    "data10_converged",
]

# Labels for display
EXP_LABELS = {
    "base_megaseg": "Baseline (no fine-tuning)",
    "data100_10epochs": "100% data, 10 epochs",
    "data100_50epochs": "100% data, 50 epochs",
    "data100_100epochs": "100% data, 100 epochs",
    "data100_converged": "100% data, converged",
    "data50_10epochs": "50% data, 10 epochs",
    "data50_50epochs": "50% data, 50 epochs",
    "data50_100epochs": "50% data, 100 epochs",
    "data50_converged": "50% data, converged",
    "data20_10epochs": "20% data, 10 epochs",
    "data20_50epochs": "20% data, 50 epochs",
    "data20_100epochs": "20% data, 100 epochs",
    "data20_converged": "20% data, converged",
    "data10_10epochs": "10% data, 10 epochs",
    "data10_50epochs": "10% data, 50 epochs",
    "data10_100epochs": "10% data, 100 epochs",
    "data10_converged": "10% data, converged",
}

# Grid axes for heatmap
DATA_FRACTIONS = ["100", "50", "20", "10"]  # rows (top=100%, bottom=10%)
EPOCH_CONFIGS = ["10epochs", "50epochs", "100epochs", "converged"]  # columns

# 3 representative samples: low, median, high Dice (from data100_converged)
SAMPLES = [
    {
        "stem": "3500000896_100X_20170515_F04_P48",
        "pred_name": "3500000896_100X_20170515_F04_P48.ome.tif",
        "gt_name": "3500000896_100X_20170515_F04_P48_struct_segmentation.tiff",
        "label": "Low Dice",
    },
    {
        "stem": "3500000938_100X_20170526_3-Scene-08-P22-E07",
        "pred_name": "3500000938_100X_20170526_3-Scene-08-P22-E07.ome.tif",
        "gt_name": "3500000938_100X_20170526_3-Scene-08-P22-E07_struct_segmentation.tiff",
        "label": "Median Dice",
    },
    {
        "stem": "3500000926_100X_20170523_1-Scene-02-P2-E04",
        "pred_name": "3500000926_100X_20170523_1-Scene-02-P2-E04.ome.tif",
        "gt_name": "3500000926_100X_20170523_1-Scene-02-P2-E04_struct_segmentation.tiff",
        "label": "High Dice",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice coefficient from two boolean masks."""
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


def load_and_project(path: Path, is_gt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load a 3D TIFF and return (max-Z-projection, binary mask).

    Returns the projection as float [0, 1] for display and the binary mask
    for Dice computation.
    """
    img = tifffile.imread(str(path))
    if is_gt:
        mask = img > 0
        proj = mask.max(axis=0).astype(np.float32)
    else:
        if np.issubdtype(img.dtype, np.floating):
            mask = img >= 0.5
            proj = img.max(axis=0)
        else:
            mask = img >= 128
            proj = (img.astype(np.float32) / 255.0).max(axis=0)
    return proj, mask


def make_figure(experiment: str, output_dir: Path) -> None:
    """Create a 3-row x 2-col figure for one experiment."""
    exp_label = EXP_LABELS.get(experiment, experiment)
    pred_dir = PRED_BASE / experiment / "seg"

    fig, axes = plt.subplots(
        len(SAMPLES), 2, figsize=(8, 3.5 * len(SAMPLES)), squeeze=False
    )
    fig.suptitle(f"LMNB1 — {exp_label}", fontsize=14, fontweight="bold", y=0.98)

    for row, sample in enumerate(SAMPLES):
        pred_path = pred_dir / sample["pred_name"]
        gt_path = GT_DIR / sample["gt_name"]

        if not pred_path.exists():
            print(f"  WARNING: missing pred {pred_path}")
            continue
        if not gt_path.exists():
            print(f"  WARNING: missing GT {gt_path}")
            continue

        gt_proj, gt_mask = load_and_project(gt_path, is_gt=True)
        pred_proj, pred_mask = load_and_project(pred_path, is_gt=False)
        dice = compute_dice(gt_mask, pred_mask)

        # GT
        ax_gt = axes[row, 0]
        ax_gt.imshow(gt_proj, cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title(f"GT — {sample['label']}", fontsize=10)
        ax_gt.axis("off")

        # Prediction
        ax_pred = axes[row, 1]
        ax_pred.imshow(pred_proj, cmap="gray", vmin=0, vmax=1)
        ax_pred.set_title(f"Pred — Dice={dice:.3f}", fontsize=10)
        ax_pred.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save PNG and PDF
    for ext in ("png", "pdf"):
        out = output_dir / f"{experiment}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {experiment}.png / .pdf")


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def make_heatmap(output_dir: Path, summary_csv: Path | None = None) -> None:
    """Create a 4x4 heatmap of mean Dice scores (data fraction x epochs).

    Rows: data fractions (100%, 50%, 20%, 10%)
    Columns: epoch configs (10, 50, 100, converged)
    Also annotates with the base_megaseg score.
    """
    csv_path = summary_csv or RESULTS_DIR / "summary.csv"
    if not csv_path.is_file():
        print(f"  ERROR: summary.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Build the 4x4 grid
    grid = np.full((len(DATA_FRACTIONS), len(EPOCH_CONFIGS)), np.nan)
    for i, frac in enumerate(DATA_FRACTIONS):
        for j, epoch_cfg in enumerate(EPOCH_CONFIGS):
            exp_name = f"data{frac}_{epoch_cfg}"
            row = df[df["experiment"] == exp_name]
            if not row.empty:
                grid[i, j] = row["dice_mean"].iloc[0]

    # Get base_megaseg score for annotation
    base_row = df[df["experiment"] == "base_megaseg"]
    base_dice = base_row["dice_mean"].iloc[0] if not base_row.empty else None

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(grid, cmap="YlOrRd", vmin=0.25, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                continue
            # White text for dark cells, black for light
            color = "white" if val < 0.75 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=color,
            )

    # Axis labels
    ax.set_xticks(range(len(EPOCH_CONFIGS)))
    ax.set_xticklabels(
        ["10 epochs", "50 epochs", "100 epochs", "Converged"], fontsize=11
    )
    ax.set_yticks(range(len(DATA_FRACTIONS)))
    ax.set_yticklabels(["100%", "50%", "20%", "10%"], fontsize=11)

    ax.set_xlabel("Training Epochs", fontsize=12, labelpad=8)
    ax.set_ylabel("Training Data Fraction", fontsize=12, labelpad=8)

    title = "LMNB1 Fine-Tuning: Mean Dice Score (n=67 test samples)"
    if base_dice is not None:
        title += f"\nBaseline (no fine-tuning): {base_dice:.3f}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Dice Score", fontsize=11)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"dice_heatmap.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dice_heatmap.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize LMNB1 fine-tuning results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "finetune" / "lmnb1" / "figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS,
        help="Experiments to visualize",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Only generate the heatmap (skip per-experiment comparison figures)",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip heatmap generation",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to summary.csv (default: results/finetune/lmnb1/summary.csv)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.heatmap_only:
        print(f"Saving figures to {args.output_dir}")
        print(f"Samples: {[s['label'] for s in SAMPLES]}")
        print()

        for exp in args.experiments:
            print(f"Processing {exp}...")
            make_figure(exp, args.output_dir)

        print(f"\n{len(args.experiments)} comparison figures saved.")

    if not args.skip_heatmap:
        print("\nGenerating heatmap...")
        make_heatmap(args.output_dir, summary_csv=args.summary_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
