#!/usr/bin/env python
"""
Generate figures for fine-tuning / from-scratch experiments.

1. Per-experiment GT vs prediction comparison figures (3 rows x 2 cols).
2. A 4x4 heatmap of mean Dice scores (data fraction x epochs).

Supports multiple structures via --structure (lmnb1, lamp1, etc.).

Usage:
    uv run python scripts/visualize_finetune.py --structure lmnb1 --mode finetune
    uv run python scripts/visualize_finetune.py --structure lamp1 --mode finetune
    uv run python scripts/visualize_finetune.py --structure lmnb1 --mode scratch
    uv run python scripts/visualize_finetune.py --mode finetune --heatmap-only
"""

from __future__ import annotations

import argparse
import csv
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

# Grid experiments (shared between modes and structures)
GRID_EXPERIMENTS = [
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

# --- Structure-specific configuration ---

# Hardcoded representative samples for LMNB1 (low, median, high Dice from
# data100_converged).
_LMNB1_SAMPLES = [
    {
        "pred_name": "3500000896_100X_20170515_F04_P48.ome.tif",
        "gt_name": "3500000896_100X_20170515_F04_P48_struct_segmentation.tiff",
        "label": "Low Dice",
    },
    {
        "pred_name": "3500000938_100X_20170526_3-Scene-08-P22-E07.ome.tif",
        "gt_name": "3500000938_100X_20170526_3-Scene-08-P22-E07_struct_segmentation.tiff",
        "label": "Median Dice",
    },
    {
        "pred_name": "3500000926_100X_20170523_1-Scene-02-P2-E04.ome.tif",
        "gt_name": "3500000926_100X_20170523_1-Scene-02-P2-E04_struct_segmentation.tiff",
        "label": "High Dice",
    },
]


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def _get_structure_config(structure: str, mode: str) -> dict:
    """Return paths, samples, and metadata for a given structure + mode."""
    struct_upper = structure.upper()  # e.g. LMNB1, LAMP1

    pred_base = PROJECT_ROOT / "outputs" / "eval" / mode / structure
    results_dir = PROJECT_ROOT / "results" / mode / structure
    figures_dir = results_dir / "figures"

    has_baseline = mode == "finetune" and (pred_base / "base_megaseg" / "seg").is_dir()
    experiments = (
        (["base_megaseg"] + GRID_EXPERIMENTS) if has_baseline else GRID_EXPERIMENTS
    )

    label = "Fine-Tuning" if mode == "finetune" else "From-Scratch Training"

    return {
        "pred_base": pred_base,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "experiments": experiments,
        "label": label,
        "has_baseline": has_baseline,
        "structure_name": struct_upper,
    }


def _get_gt_dir(structure: str) -> Path:
    """Return the ground-truth segmentation directory for a structure."""
    struct_upper = structure.upper()
    # LMNB1 uses a flat seg/ directory
    seg_dir = DATA_ROOT / "variance-100x" / struct_upper / "seg"
    if seg_dir.is_dir():
        return seg_dir
    # Other structures (e.g. LAMP1) store GT inside dataset/test/ with _seg suffix
    test_dir = DATA_ROOT / "variance-100x" / struct_upper / "dataset" / "test"
    if test_dir.is_dir():
        return test_dir
    raise FileNotFoundError(
        f"No GT directory found for {struct_upper} at {seg_dir} or {test_dir}"
    )


def _get_samples(
    structure: str, results_dir: Path, pred_base: Path, gt_dir: Path
) -> list[dict]:
    """Return 3 representative samples (low, median, high Dice).

    Uses hardcoded samples for LMNB1.  For other structures, selects
    automatically from per_sample_metrics.csv using data100_converged.
    """
    if structure == "lmnb1":
        return _LMNB1_SAMPLES

    # Auto-select from per_sample_metrics.csv
    metrics_csv = results_dir / "per_sample_metrics.csv"
    if not metrics_csv.is_file():
        print(f"  WARNING: {metrics_csv} not found; cannot auto-select samples")
        return []

    df = pd.read_csv(metrics_csv)
    # Prefer data100_converged, fall back to base_megaseg, then first available
    for candidate in ("data100_converged", "base_megaseg"):
        subset = df[df["experiment"] == candidate].sort_values("dice")
        if not subset.empty:
            break
    if subset.empty:
        # Last resort: use whatever experiment is available
        subset = df.sort_values("dice")
    if subset.empty:
        print(f"  WARNING: no experiment rows in {metrics_csv}")
        return []

    n = len(subset)
    picks = [
        (subset.iloc[0], "Low Dice"),
        (subset.iloc[n // 2], "Median Dice"),
        (subset.iloc[-1], "High Dice"),
    ]

    samples = []
    for row, label in picks:
        sample_id = row["sample_id"]  # e.g. LAMP1_8273
        # Derive prediction and GT file names from the sample_id.
        # Predictions: <sample_id>.tif  (in pred_base/<experiment>/seg/)
        pred_name = f"{sample_id}.tif"
        # GT: try <sample_id>_seg.tiff first, then <sample_id>_struct_segmentation.tiff
        gt_candidates = [
            f"{sample_id}_seg.tiff",
            f"{sample_id}_struct_segmentation.tiff",
        ]
        gt_name = None
        for candidate in gt_candidates:
            if (gt_dir / candidate).exists():
                gt_name = candidate
                break
        if gt_name is None:
            # Fallback: check if the raw test CSV has the path
            gt_name = _find_gt_name_from_csv(structure, sample_id)
        if gt_name is None:
            print(f"  WARNING: cannot find GT file for {sample_id}, skipping")
            continue
        samples.append({"pred_name": pred_name, "gt_name": gt_name, "label": label})

    return samples


def _find_gt_name_from_csv(structure: str, sample_id: str) -> str | None:
    """Look up the GT filename from the test CSV."""
    struct_upper = structure.upper()
    test_csv = DATA_ROOT / "variance-100x" / struct_upper / "local" / "test.csv"
    if not test_csv.is_file():
        return None
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg_path = row.get("seg", "")
            if sample_id in seg_path:
                return Path(seg_path).name
    return None


def _count_test_samples(structure: str) -> int:
    """Count the number of test samples from the test CSV."""
    struct_upper = structure.upper()
    local_dir = DATA_ROOT / "variance-100x" / struct_upper / "local"
    if not local_dir.is_dir():
        return 0
    # Try structure-specific CSV first, then generic test.csv
    candidates = [
        local_dir / f"test_{structure}_local_cytodl.csv",
        local_dir / "test.csv",
    ]
    for csv_path in candidates:
        if csv_path.is_file():
            with open(csv_path) as f:
                return sum(1 for _ in f) - 1  # subtract header
    # Fallback: glob for test*.csv
    test_csvs = sorted(local_dir.glob("test*.csv"))
    if test_csvs:
        with open(test_csvs[0]) as f:
            return sum(1 for _ in f) - 1
    return 0


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


def make_figure(
    experiment: str,
    output_dir: Path,
    pred_base: Path,
    gt_dir: Path,
    samples: list[dict],
    structure_name: str = "LMNB1",
) -> None:
    """Create a 3-row x 2-col figure for one experiment."""
    exp_label = EXP_LABELS.get(experiment, experiment)
    pred_dir = pred_base / experiment / "seg"

    fig, axes = plt.subplots(
        len(samples), 2, figsize=(8, 3.5 * len(samples)), squeeze=False
    )
    fig.suptitle(
        f"{structure_name} — {exp_label}", fontsize=14, fontweight="bold", y=0.98
    )

    for row, sample in enumerate(samples):
        pred_path = pred_dir / sample["pred_name"]
        gt_path = gt_dir / sample["gt_name"]

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


def make_heatmap(
    output_dir: Path,
    summary_csv: Path | None = None,
    results_dir: Path | None = None,
    label: str = "Fine-Tuning",
    has_baseline: bool = True,
    structure_name: str = "LMNB1",
    n_test: int = 0,
) -> None:
    """Create a 4x4 heatmap of mean Dice scores (data fraction x epochs).

    Rows: data fractions (100%, 50%, 20%, 10%)
    Columns: epoch configs (10, 50, 100, converged)
    Also annotates with the base_megaseg score if has_baseline=True.
    """
    csv_path = summary_csv or (results_dir / "summary.csv" if results_dir else None)
    if csv_path is None or not csv_path.is_file():
        print(f"  ERROR: summary.csv not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "n_samples" in df.columns:
        n_test = int(df["n_samples"].mode().iloc[0])

    # Build the 4x4 grid
    grid = np.full((len(DATA_FRACTIONS), len(EPOCH_CONFIGS)), np.nan)
    for i, frac in enumerate(DATA_FRACTIONS):
        for j, epoch_cfg in enumerate(EPOCH_CONFIGS):
            exp_name = f"data{frac}_{epoch_cfg}"
            row = df[df["experiment"] == exp_name]
            if not row.empty:
                grid[i, j] = row["dice_mean"].iloc[0]

    # Get base_megaseg score for annotation (only for finetune mode)
    base_dice = None
    if has_baseline:
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

    title = f"{structure_name} {label}: Mean Dice Score (n={n_test} test samples)"
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
    parser = argparse.ArgumentParser(
        description="Visualize experiment results (finetune or scratch)"
    )
    parser.add_argument(
        "--structure",
        default="lmnb1",
        help="Structure name, e.g. lmnb1, lamp1 (default: lmnb1)",
    )
    parser.add_argument(
        "--mode",
        choices=["finetune", "scratch"],
        default="finetune",
        help="Experiment mode: finetune (pretrained) or scratch (random init)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: mode-specific figures dir)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiments to visualize (default: all for the chosen mode)",
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
        help="Path to summary.csv (default: mode-specific results dir)",
    )
    args = parser.parse_args()

    structure = args.structure.lower()
    cfg = _get_structure_config(structure, args.mode)
    gt_dir = _get_gt_dir(structure) if not args.heatmap_only else None
    experiments = args.experiments or cfg["experiments"]
    output_dir = args.output_dir or cfg["figures_dir"]
    pred_base = cfg["pred_base"]
    results_dir = cfg["results_dir"]
    label = cfg["label"]
    has_baseline = cfg["has_baseline"]
    structure_name = cfg["structure_name"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Structure: {structure_name}")
    print(f"Mode: {args.mode} ({label})")
    if gt_dir:
        print(f"GT dir: {gt_dir}")

    if not args.heatmap_only:
        samples = _get_samples(structure, results_dir, pred_base, gt_dir)
        if not samples:
            print("  ERROR: no representative samples found; skipping figures")
        else:
            print(f"Saving figures to {output_dir}")
            print(f"Samples: {[s['label'] for s in samples]}")
            print()

            for exp in experiments:
                print(f"Processing {exp}...")
                make_figure(
                    exp,
                    output_dir,
                    pred_base=pred_base,
                    gt_dir=gt_dir,
                    samples=samples,
                    structure_name=structure_name,
                )

            print(f"\n{len(experiments)} comparison figures saved.")

    if not args.skip_heatmap:
        print("\nGenerating heatmap...")
        make_heatmap(
            output_dir,
            summary_csv=args.summary_csv,
            results_dir=results_dir,
            label=label,
            has_baseline=has_baseline,
            structure_name=structure_name,
            n_test=0,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
