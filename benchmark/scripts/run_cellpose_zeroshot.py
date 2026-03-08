#!/usr/bin/env python
"""Zero-shot CellPose evaluation on MegaSeg test set.

Runs CellPose (v4 cpsam model) on all 14 structures' test images.
Outputs binarized predictions (instance labels -> binary foreground mask)
and per-sample metrics (Dice, IoU, Precision, Recall, Accuracy).

Note: CellPose v4 replaced cyto/cyto2/cyto3 with a single 'cpsam' model.
The model_type argument is kept for logging but is ignored by the API.
"""

import argparse
import os
import sys
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
    parser = argparse.ArgumentParser(description="CellPose 3 zero-shot evaluation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/uint8/cellpose_zeroshot",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/results/uint8/cellpose_zeroshot_metrics.csv",
        help="Output CSV for per-sample metrics",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--model-type",
        type=str,
        default="cpsam",
        help="CellPose model type (v4: cpsam)",
    )
    parser.add_argument(
        "--do-3d", action="store_true", default=True, help="Run 3D segmentation"
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from cellpose.models import CellposeModel

    print(f"Loading CellPose model (v4 cpsam, requested: {args.model_type})")
    model = CellposeModel(gpu=True, use_bfloat16=False)

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
                # Load cached prediction to compute metrics
                pred_binary = tifffile.imread(str(pred_path))
                gt = tifffile.imread(seg_path)
                gt_binary = (gt > 0).astype(np.uint8)
                metrics = compute_metrics(pred_binary, gt_binary)
                metrics.update(
                    {
                        "eval_type": "cellpose_zeroshot",
                        "model_type": args.model_type,
                        "test_struct": struct,
                        "sample_id": sample_id,
                    }
                )
                all_metrics.append(metrics)
                print(
                    f"  [{idx + 1}/{len(df)}] {sample_id}: cached (Dice={metrics['dice']:.4f})"
                )
                continue

            t0 = time.time()

            # Load raw image
            img = tifffile.imread(raw_path)  # (Z, Y, X) uint16

            # Normalize to [0, 1] float32 for CellPose
            img_f = img.astype(np.float32)
            img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)

            # Run CellPose - single channel, 3D
            # z_axis=0 since our images are (Z, Y, X)
            try:
                masks, flows, styles = model.eval(
                    img_f,
                    z_axis=0,
                    do_3D=args.do_3d,
                    diameter=None,  # auto-estimate
                )
            except Exception as e:
                print(f"  [{idx + 1}/{len(df)}] {sample_id}: FAILED - {e}")
                metrics = {
                    "dice": 0.0,
                    "iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "accuracy": 0.0,
                    "eval_type": "cellpose_zeroshot",
                    "model_type": args.model_type,
                    "test_struct": struct,
                    "sample_id": sample_id,
                }
                all_metrics.append(metrics)
                continue

            # Binarize: any instance label > 0 = foreground
            pred_binary = (masks > 0).astype(np.uint8)

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
                    "eval_type": "cellpose_zeroshot",
                    "model_type": args.model_type,
                    "test_struct": struct,
                    "sample_id": sample_id,
                }
            )
            all_metrics.append(metrics)

            print(
                f"  [{idx + 1}/{len(df)}] {sample_id}: Dice={metrics['dice']:.4f}, "
                f"IoU={metrics['iou']:.4f} ({elapsed:.1f}s)"
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
        print(f"Per-structure mean Dice:")
        for struct, group in metrics_df.groupby("test_struct"):
            print(
                f"  {struct}: {group['dice'].mean():.4f} +/- {group['dice'].std():.4f}"
            )


if __name__ == "__main__":
    main()
