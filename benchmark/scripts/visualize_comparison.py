#!/usr/bin/env python
"""
Generate comparison figures: fine-tuning vs from-scratch training.

Produces:
  1. Side-by-side heatmaps (finetune | scratch)
  2. Delta heatmap showing pretraining benefit (finetune - scratch)

Usage:
    uv run python scripts/visualize_comparison.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # benchmark/

FINETUNE_SUMMARY = PROJECT_ROOT / "results" / "finetune" / "lmnb1" / "summary.csv"
SCRATCH_SUMMARY = PROJECT_ROOT / "results" / "scratch" / "lmnb1" / "summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison" / "lmnb1" / "figures"

DATA_FRACTIONS = ["100", "50", "20", "10"]  # rows
EPOCH_CONFIGS = ["10epochs", "50epochs", "100epochs", "converged"]  # columns
EPOCH_LABELS = ["10 epochs", "50 epochs", "100 epochs", "Converged"]
FRAC_LABELS = ["100%", "50%", "20%", "10%"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_grid(summary_csv: Path) -> np.ndarray:
    """Load summary CSV and build a 4x4 grid of mean Dice scores."""
    df = pd.read_csv(summary_csv)
    grid = np.full((len(DATA_FRACTIONS), len(EPOCH_CONFIGS)), np.nan)
    for i, frac in enumerate(DATA_FRACTIONS):
        for j, epoch_cfg in enumerate(EPOCH_CONFIGS):
            exp_name = f"data{frac}_{epoch_cfg}"
            row = df[df["experiment"] == exp_name]
            if not row.empty:
                grid[i, j] = row["dice_mean"].iloc[0]
    return grid


def annotate_grid(ax, grid, fontsize=13, threshold=0.75, fmt=".3f"):
    """Add text annotations to heatmap cells."""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < threshold else "black"
            ax.text(
                j,
                i,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=color,
            )


# ---------------------------------------------------------------------------
# Figure 1: Side-by-side heatmaps
# ---------------------------------------------------------------------------


def make_side_by_side(
    finetune_grid: np.ndarray,
    scratch_grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Create side-by-side heatmaps for finetune vs scratch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin, vmax = 0.05, 1.0

    # Fine-tuning
    im1 = ax1.imshow(finetune_grid, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    annotate_grid(ax1, finetune_grid)
    ax1.set_xticks(range(len(EPOCH_CONFIGS)))
    ax1.set_xticklabels(EPOCH_LABELS, fontsize=10)
    ax1.set_yticks(range(len(DATA_FRACTIONS)))
    ax1.set_yticklabels(FRAC_LABELS, fontsize=10)
    ax1.set_xlabel("Training Epochs", fontsize=11)
    ax1.set_ylabel("Training Data Fraction", fontsize=11)
    ax1.set_title("Fine-Tuning (MegaSeg pretrained)", fontsize=12, fontweight="bold")

    # Scratch
    im2 = ax2.imshow(scratch_grid, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    annotate_grid(ax2, scratch_grid)
    ax2.set_xticks(range(len(EPOCH_CONFIGS)))
    ax2.set_xticklabels(EPOCH_LABELS, fontsize=10)
    ax2.set_yticks(range(len(DATA_FRACTIONS)))
    ax2.set_yticklabels(FRAC_LABELS, fontsize=10)
    ax2.set_xlabel("Training Epochs", fontsize=11)
    ax2.set_title("From Scratch (random init)", fontsize=12, fontweight="bold")

    # Shared colorbar
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02, label="Mean Dice Score")

    fig.suptitle(
        "LMNB1 Segmentation: Pretrained vs From-Scratch (n=67 test samples)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"side_by_side.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved side_by_side.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 2: Delta heatmap (pretraining benefit)
# ---------------------------------------------------------------------------


def make_delta_heatmap(
    finetune_grid: np.ndarray,
    scratch_grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Create a delta heatmap showing the pretraining benefit."""
    delta = finetune_grid - scratch_grid

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(delta, cmap="viridis", vmin=-0.05, vmax=0.65, aspect="auto")

    # Annotate with both delta and absolute values
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            d = delta[i, j]
            ft = finetune_grid[i, j]
            sc = scratch_grid[i, j]
            if np.isnan(d):
                continue
            color = "white" if d < 0.2 else "black"
            ax.text(
                j,
                i,
                f"+{d:.3f}\n({ft:.3f} vs {sc:.3f})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(len(EPOCH_CONFIGS)))
    ax.set_xticklabels(EPOCH_LABELS, fontsize=11)
    ax.set_yticks(range(len(DATA_FRACTIONS)))
    ax.set_yticklabels(FRAC_LABELS, fontsize=11)
    ax.set_xlabel("Training Epochs", fontsize=12, labelpad=8)
    ax.set_ylabel("Training Data Fraction", fontsize=12, labelpad=8)
    ax.set_title(
        "Pretraining Benefit: Fine-Tune minus From-Scratch Dice\n(n=67 test samples)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Dice Improvement (fine-tune − scratch)", fontsize=11)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"delta_pretraining_benefit.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("  Saved delta_pretraining_benefit.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare fine-tuning vs from-scratch LMNB1 results"
    )
    parser.add_argument(
        "--finetune-summary",
        type=Path,
        default=FINETUNE_SUMMARY,
        help="Path to finetune summary.csv",
    )
    parser.add_argument(
        "--scratch-summary",
        type=Path,
        default=SCRATCH_SUMMARY,
        help="Path to scratch summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    # Validate inputs
    for p, label in [
        (args.finetune_summary, "finetune"),
        (args.scratch_summary, "scratch"),
    ]:
        if not p.is_file():
            print(f"ERROR: {label} summary not found: {p}")
            return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    finetune_grid = load_grid(args.finetune_summary)
    scratch_grid = load_grid(args.scratch_summary)

    print(f"Finetune grid:\n{finetune_grid}\n")
    print(f"Scratch grid:\n{scratch_grid}\n")
    print(f"Delta (finetune - scratch):\n{finetune_grid - scratch_grid}\n")

    print("Generating side-by-side heatmaps...")
    make_side_by_side(finetune_grid, scratch_grid, args.output_dir)

    print("Generating delta heatmap...")
    make_delta_heatmap(finetune_grid, scratch_grid, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
