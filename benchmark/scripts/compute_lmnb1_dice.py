#!/usr/bin/env python
"""
Compute Dice scores for LMNB1 experiments (test set only).

Supports two modes:
  - finetune: MegaSeg pretrained → fine-tuned (includes base_megaseg baseline)
  - scratch:  Random init → trained from scratch (no baseline)

Reads the test CSV, finds matching prediction TIFFs for each experiment,
binarizes (pred >= 0.5, GT > 0), computes Dice = 2*TP / (2*TP + FP + FN),
and outputs per-sample CSV + summary table.

Usage:
    uv run python scripts/compute_lmnb1_dice.py --mode finetune
    uv run python scripts/compute_lmnb1_dice.py --mode scratch
    uv run python scripts/compute_lmnb1_dice.py --mode scratch --experiments data100_converged
"""

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # benchmark/
DATA_ROOT = PROJECT_ROOT.parent / "data"

TEST_CSV = (
    DATA_ROOT / "variance-100x" / "LMNB1" / "local" / "test_lmnb1_local_cytodl.csv"
)

# Grid experiments (shared between finetune and scratch)
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

# Mode-specific settings
MODE_CONFIG = {
    "finetune": {
        "pred_base": PROJECT_ROOT / "outputs" / "eval" / "finetune" / "lmnb1",
        "results_dir": PROJECT_ROOT / "results" / "finetune" / "lmnb1",
        "experiments": ["base_megaseg"] + GRID_EXPERIMENTS,
        "label": "Fine-Tuning",
    },
    "scratch": {
        "pred_base": PROJECT_ROOT / "outputs" / "eval" / "scratch" / "lmnb1",
        "results_dir": PROJECT_ROOT / "results" / "scratch" / "lmnb1",
        "experiments": GRID_EXPERIMENTS,
        "label": "From-Scratch Training",
    },
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_dice(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute Dice and related metrics from two binary boolean masks."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp / dice_denom) if dice_denom > 0 else 1.0

    iou_denom = tp + fp + fn
    iou = (tp / iou_denom) if iou_denom > 0 else 1.0

    prec_denom = tp + fp
    precision = (tp / prec_denom) if prec_denom > 0 else 1.0

    rec_denom = tp + fn
    recall = (tp / rec_denom) if rec_denom > 0 else 1.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _evaluate_sample(args: tuple) -> dict:
    """Evaluate one pred/GT pair. Designed for multiprocessing."""
    pred_path_str, gt_path_str, experiment, sample_id = args

    nan_result = {
        "experiment": experiment,
        "sample_id": sample_id,
        "dice": np.nan,
        "iou": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "error": "",
    }

    pred_path = Path(pred_path_str)
    gt_path = Path(gt_path_str)

    # Load GT
    if not gt_path.is_file():
        nan_result["error"] = f"GT not found: {gt_path}"
        return nan_result
    try:
        gt_img = tifffile.imread(str(gt_path))
    except Exception as e:
        nan_result["error"] = f"GT load failed: {e}"
        return nan_result

    # Load prediction
    if not pred_path.is_file():
        nan_result["error"] = f"Pred not found: {pred_path}"
        return nan_result
    try:
        pred_img = tifffile.imread(str(pred_path))
    except Exception as e:
        nan_result["error"] = f"Pred load failed: {e}"
        return nan_result

    # Binarize
    gt_mask = gt_img > 0
    if np.issubdtype(pred_img.dtype, np.floating):
        pred_mask = pred_img >= 0.5
    else:
        pred_mask = pred_img >= 128

    # Shape check
    if gt_mask.shape != pred_mask.shape:
        nan_result["error"] = (
            f"Shape mismatch: GT {gt_mask.shape} vs Pred {pred_mask.shape}"
        )
        return nan_result

    # Compute
    metrics = compute_dice(gt_mask, pred_mask)
    return {
        "experiment": experiment,
        "sample_id": sample_id,
        **metrics,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def derive_pred_path(raw_path: str, pred_seg_dir: Path) -> Path:
    """Map raw image path to prediction path.

    Raw: .../raw/3500000876_100X_...ome.tiff
    Pred: .../seg/3500000876_100X_...ome.tif  (note: .tif not .tiff)
    """
    name = Path(raw_path).name  # e.g. 3500000876_100X_...ome.tiff
    # Try exact match first
    exact = pred_seg_dir / name
    if exact.is_file():
        return exact

    # Try swapping .tiff -> .tif
    stem = Path(name).stem  # loses last extension
    if name.endswith(".ome.tiff"):
        alt_name = name[:-5] + ".tif"  # .ome.tiff -> .ome.tif
    elif name.endswith(".tiff"):
        alt_name = name[:-5] + ".tif"
    elif name.endswith(".tif"):
        alt_name = name[:-4] + ".tiff"
    else:
        return exact

    alt = pred_seg_dir / alt_name
    if alt.is_file():
        return alt

    return exact  # caller handles missing


def compute_for_experiments(
    experiments: list[str],
    pred_base: Path,
    workers: int = 8,
) -> pd.DataFrame:
    """Compute Dice for the given experiments on the test set."""
    # Load test CSV
    if not TEST_CSV.is_file():
        print(f"ERROR: Test CSV not found: {TEST_CSV}", file=sys.stderr)
        sys.exit(1)

    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(test_df)} test samples from {TEST_CSV}")

    # Build work items
    all_items = []
    for exp in experiments:
        pred_dir = pred_base / exp / "seg"
        if not pred_dir.is_dir():
            print(f"WARNING: Pred dir missing for {exp}: {pred_dir}")
            continue

        n_preds = len(list(pred_dir.glob("*.tif*")))
        print(f"  {exp}: {n_preds} prediction files in {pred_dir}")

        for _, row in test_df.iterrows():
            raw_path = str(row["raw"])
            gt_path = str(row["seg"])
            sample_id = Path(raw_path).stem

            pred_path = derive_pred_path(raw_path, pred_dir)
            all_items.append((str(pred_path), gt_path, exp, sample_id))

    if not all_items:
        print("No work items — nothing to compute.")
        return pd.DataFrame()

    print(
        f"\nComputing metrics for {len(all_items)} sample-experiment pairs "
        f"using {workers} workers..."
    )

    # Run with multiprocessing
    results = []
    with Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_evaluate_sample, all_items, chunksize=4),
            total=len(all_items),
            desc="Computing Dice",
        ):
            results.append(result)

    df = pd.DataFrame(results)

    # Report errors
    errors = df[df["error"] != ""]
    if len(errors) > 0:
        print(f"\nWARNING: {len(errors)} errors encountered:")
        for _, row in errors.head(10).iterrows():
            print(f"  [{row['experiment']}] {row['sample_id']}: {row['error']}")

    return df


def print_summary(df: pd.DataFrame, label: str = "Fine-Tuning") -> pd.DataFrame:
    """Print summary table of mean Dice per experiment."""
    if df.empty:
        return pd.DataFrame()

    valid = df[df["error"] == ""]

    print(f"\n{'=' * 80}")
    print(f"LMNB1 {label} Results — Dice Score (Test Set, n=57)")
    print(f"{'=' * 80}")

    summary_rows = []
    for exp in sorted(valid["experiment"].unique()):
        exp_data = valid[valid["experiment"] == exp]
        mean_dice = exp_data["dice"].mean()
        std_dice = exp_data["dice"].std()
        mean_iou = exp_data["iou"].mean()
        mean_prec = exp_data["precision"].mean()
        mean_rec = exp_data["recall"].mean()
        n = len(exp_data)
        summary_rows.append(
            {
                "experiment": exp,
                "n_samples": n,
                "dice_mean": mean_dice,
                "dice_std": std_dice,
                "iou_mean": mean_iou,
                "precision_mean": mean_prec,
                "recall_mean": mean_rec,
            }
        )

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format="%.4f"))
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute LMNB1 Dice scores (finetune or scratch)"
    )
    parser.add_argument(
        "--mode",
        choices=["finetune", "scratch"],
        default="finetune",
        help="Experiment mode: finetune (pretrained) or scratch (random init)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiments to evaluate (default: all for the chosen mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(cpu_count(), 16),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs (default: mode-specific results dir)",
    )
    args = parser.parse_args()

    cfg = MODE_CONFIG[args.mode]
    experiments = args.experiments or cfg["experiments"]
    output_dir = args.output_dir or cfg["results_dir"]
    pred_base = cfg["pred_base"]
    label = cfg["label"]

    print(f"Mode: {args.mode} ({label})")
    print(f"Pred base: {pred_base}")
    print(f"Output dir: {output_dir}")
    print()

    # Compute
    df = compute_for_experiments(experiments, pred_base=pred_base, workers=args.workers)
    if df.empty:
        return

    # Save per-sample CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_metrics.csv"
    df.to_csv(per_sample_path, index=False)
    print(f"\nPer-sample metrics saved to {per_sample_path}")

    # Print and save summary
    summary = print_summary(df, label=label)
    if not summary.empty:
        summary_path = output_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
