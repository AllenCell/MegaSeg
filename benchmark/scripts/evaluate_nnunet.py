#!/usr/bin/env python
"""Evaluate nnU-Net predictions against MegaSeg test set ground truth.

nnU-Net outputs NIfTI predictions in imagesTs naming convention.
This script:
  1. Maps each NIfTI prediction back to its structure + sample_id
  2. Loads the corresponding GT segmentation TIFF
  3. Computes Dice, IoU, Precision, Recall, Accuracy
  4. Writes per-sample metrics CSV (same format as cellpose/sam scripts)

Usage:
    python scripts/evaluate_nnunet.py \
        --pred-dir nnUNet_results/Dataset001_MegaSeg/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
        --output-csv results/uint8/nnunet_metrics.csv
"""

import argparse
import re
import sys
from pathlib import Path

import nibabel as nib
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


def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> dict:
    """Compute Dice, IoU, Precision, Recall, Accuracy for binary masks."""
    pred = pred_binary.astype(bool).ravel()
    gt = gt_binary.astype(bool).ravel()

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))

    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp / dice_denom) if dice_denom > 0 else 1.0

    iou_denom = tp + fp + fn
    iou = (tp / iou_denom) if iou_denom > 0 else 1.0

    prec_denom = tp + fp
    precision = (tp / prec_denom) if prec_denom > 0 else 1.0

    rec_denom = tp + fn
    recall = (tp / rec_denom) if rec_denom > 0 else 1.0

    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total) if total > 0 else 1.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def extract_structure(sample_id: str) -> str | None:
    """Extract structure name from a sample ID like 'ACTN1_11347'.

    Pattern: {STRUCT}_{DIGITS}
    """
    match = re.match(r"^(.+)_(\d+)$", sample_id)
    if match:
        candidate = match.group(1)
        if candidate in STRUCTURES:
            return candidate
    return None


def build_test_mapping() -> dict[str, tuple[str, str]]:
    """Build mapping from sample_id to (structure, gt_path).

    Reads test CSVs for all structures. The convert_to_nnunet.py script
    uses the original sample name (e.g., ACTN1_11269) as the case ID,
    so prediction files are named like ACTN1_11269.nii.gz.
    """
    mapping = {}
    for struct in STRUCTURES:
        csv_path = DATA_CSV_DIR / struct / "test.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sample_id = Path(row["raw"]).stem
            gt_path = row["seg"]
            mapping[sample_id] = (struct, gt_path)

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net predictions")
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing nnU-Net prediction NIfTIs",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/results/uint8/nnunet_metrics.csv",
        help="Output CSV for per-sample metrics",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        print(f"ERROR: Prediction directory not found: {pred_dir}")
        sys.exit(1)

    # Build sample_id -> (struct, gt_path) mapping from test CSVs
    print("Building test mapping from CSVs...")
    mapping = build_test_mapping()
    print(f"  Mapped {len(mapping)} test cases")

    # Find all prediction files
    # convert_to_nnunet.py uses original names: {STRUCT}_{ID}.nii.gz
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        pred_files = sorted(pred_dir.glob("*.nii"))
    print(f"  Found {len(pred_files)} prediction files in {pred_dir}")

    if not pred_files:
        print("ERROR: No prediction files found")
        sys.exit(1)

    all_metrics = []
    missing = 0

    for pred_file in pred_files:
        # Extract sample_id from filename: e.g. ACTN1_11347.nii.gz -> ACTN1_11347
        sample_id = pred_file.name.replace(".nii.gz", "").replace(".nii", "")

        if sample_id not in mapping:
            missing += 1
            continue

        struct, gt_path = mapping[sample_id]

        # Load prediction (NIfTI)
        try:
            pred_nii = nib.load(str(pred_file))
            pred_data = pred_nii.get_fdata()
            pred_binary = (pred_data > 0).astype(np.uint8)
        except Exception as e:
            print(f"  ERROR loading prediction {pred_file}: {e}")
            continue

        # Load GT (TIFF)
        try:
            gt = tifffile.imread(gt_path)
            gt_binary = (gt > 0).astype(np.uint8)
        except Exception as e:
            print(f"  ERROR loading GT {gt_path}: {e}")
            continue

        # Shape check - nnU-Net may have transposed axes
        if pred_binary.shape != gt_binary.shape:
            # Try common transpositions
            for axes in [(2, 1, 0), (0, 2, 1)]:
                transposed = np.transpose(pred_binary, axes)
                if transposed.shape == gt_binary.shape:
                    pred_binary = transposed
                    break
            if pred_binary.shape != gt_binary.shape:
                print(
                    f"  WARNING: Shape mismatch for {sample_id}: "
                    f"pred {pred_binary.shape} vs gt {gt_binary.shape}"
                )
                continue

        metrics = compute_metrics(pred_binary, gt_binary)
        metrics.update(
            {
                "eval_type": "nnunet",
                "model_type": "3d_fullres_fold0",
                "test_struct": struct,
                "sample_id": sample_id,
            }
        )
        all_metrics.append(metrics)

    # Save metrics
    results_path = Path(args.output_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(results_path, index=False)
    print(f"\nSaved {len(metrics_df)} results to {results_path}")

    if missing > 0:
        print(f"  ({missing} predictions had no mapping - likely train/val cases)")

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
