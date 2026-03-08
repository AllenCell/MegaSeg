#!/usr/bin/env python
"""Otsu thresholding baseline on MegaSeg test set.

Applies Otsu's method to each 3D test image (global threshold per image).
No training, no GPU - pure classical baseline.

Outputs binarized predictions and per-sample metrics
(Dice, IoU, Precision, Recall, Accuracy).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

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

DATA_CSV_DIR = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x")


def otsu_threshold(image: np.ndarray) -> float:
    """Compute Otsu's threshold for a grayscale image.

    Implements Otsu's method directly with numpy (no skimage dependency).
    Works on the full 3D volume as a flat histogram.
    """
    # Flatten and work with uint16 histogram
    flat = image.ravel().astype(np.float64)

    # Build histogram - use 256 bins for speed (downsample from uint16)
    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Total weight
    total = hist.sum()
    if total == 0:
        return 0.0

    # Class probabilities and means
    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg

    mean_bg = np.cumsum(hist * bin_centers)
    # Avoid division by zero
    mask_bg = weight_bg > 0
    mask_fg = weight_fg > 0

    mean_bg_safe = np.where(mask_bg, mean_bg / np.maximum(weight_bg, 1), 0)
    mean_fg_safe = np.where(
        mask_fg,
        (mean_bg[-1] - mean_bg) / np.maximum(weight_fg, 1),
        0,
    )

    # Between-class variance
    variance = weight_bg * weight_fg * (mean_bg_safe - mean_fg_safe) ** 2

    # Find the threshold that maximizes between-class variance
    idx = np.argmax(variance)
    return bin_centers[idx]


def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray):
    """Compute Dice, IoU, Precision, Recall, Accuracy for binary masks."""
    pred = pred_binary.astype(bool).ravel()
    gt = gt_binary.astype(bool).ravel()

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def main():
    parser = argparse.ArgumentParser(description="Otsu thresholding baseline")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/uint8/otsu_baseline",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/results/uint8/otsu_baseline_metrics.csv",
        help="Output CSV for per-sample metrics",
    )
    parser.add_argument(
        "--structures", nargs="+", default=None, help="Specific structures to run"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save prediction TIFFs",
    )
    args = parser.parse_args()

    structs = args.structures if args.structures else STRUCTURES
    out_dir = Path(args.output_dir)
    all_metrics = []

    for struct in structs:
        print(f"\n{'=' * 60}")
        print(f"Processing {struct}")
        print(f"{'=' * 60}")

        csv_path = DATA_CSV_DIR / struct / "test.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        pred_dir = out_dir / struct
        pred_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():
            raw_path = row["raw"]
            seg_path = row["seg"]
            sample_id = Path(raw_path).stem

            pred_path = pred_dir / f"{sample_id}_pred.tiff"

            # Check if already computed
            if pred_path.exists():
                pred_binary = tifffile.imread(str(pred_path))
                gt = tifffile.imread(seg_path)
                gt_binary = (gt > 0).astype(np.uint8)
                metrics = compute_metrics(pred_binary, gt_binary)
                metrics.update(
                    {
                        "eval_type": "otsu_baseline",
                        "test_struct": struct,
                        "sample_id": sample_id,
                    }
                )
                all_metrics.append(metrics)
                print(
                    f"  [{idx + 1}/{len(df)}] {sample_id}: cached "
                    f"(Dice={metrics['dice']:.4f})"
                )
                continue

            t0 = time.time()

            # Load raw image (Z, Y, X) uint16
            img = tifffile.imread(raw_path)

            # Compute Otsu threshold on the full 3D volume
            thresh = otsu_threshold(img)
            pred_binary = (img > thresh).astype(np.uint8)

            # Save prediction
            if args.save_predictions:
                tifffile.imwrite(str(pred_path), pred_binary)

            # Load GT and compute metrics
            gt = tifffile.imread(seg_path)
            gt_binary = (gt > 0).astype(np.uint8)

            metrics = compute_metrics(pred_binary, gt_binary)
            elapsed = time.time() - t0

            metrics.update(
                {
                    "eval_type": "otsu_baseline",
                    "test_struct": struct,
                    "sample_id": sample_id,
                }
            )
            all_metrics.append(metrics)

            print(
                f"  [{idx + 1}/{len(df)}] {sample_id}: Dice={metrics['dice']:.4f}, "
                f"IoU={metrics['iou']:.4f}, thresh={thresh:.1f} ({elapsed:.1f}s)"
            )

    # Save metrics
    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(results_path, index=False)
    print(f"\nSaved {len(metrics_df)} results to {results_path}")

    # Print summary
    if len(metrics_df) > 0:
        print(f"\nOverall mean Dice: {metrics_df['dice'].mean():.4f}")
        print("Per-structure mean Dice:")
        for struct, group in metrics_df.groupby("test_struct"):
            print(
                f"  {struct}: {group['dice'].mean():.4f} +/- {group['dice'].std():.4f}"
            )


if __name__ == "__main__":
    main()
