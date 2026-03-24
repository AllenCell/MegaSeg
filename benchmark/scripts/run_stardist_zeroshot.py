#!/usr/bin/env python
"""Zero-shot StarDist evaluation on MegaSeg test set.

Runs StarDist 3D (pretrained model) on all 14 structures' test images.
Outputs binarized predictions (instance labels -> binary foreground mask)
and per-sample metrics (Dice, IoU, Precision, Recall, Accuracy).

StarDist is an instance segmentation method that detects star-convex polygons.
The pretrained 3D model was trained on fluorescence microscopy data.
We binarize the output (any instance > 0 = foreground) for comparison.
"""

import argparse
import os
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


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] float32 for StarDist input."""
    img_f = img.astype(np.float32)
    pmin, pmax = np.percentile(img_f, (1, 99.8))
    if pmax - pmin > 0:
        img_f = (img_f - pmin) / (pmax - pmin)
    img_f = np.clip(img_f, 0, 1)
    return img_f


def main():
    parser = argparse.ArgumentParser(description="StarDist 3D zero-shot evaluation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/uint8/stardist_zeroshot",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/results/uint8/stardist_zeroshot_metrics.csv",
        help="Output CSV for per-sample metrics",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
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

    # Set GPU before importing TF/StarDist
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Limit TF GPU memory growth
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    from stardist.models import StarDist3D

    print("Loading StarDist 3D pretrained model...")
    # StarDist3D.from_pretrained() downloads the pretrained 3D fluorescence model
    model = StarDist3D.from_pretrained("3D_demo")
    print(f"Model loaded: {model.name}")

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
                        "eval_type": "stardist_zeroshot",
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

            # Normalize for StarDist
            img_norm = normalize_image(img)

            # Run StarDist 3D prediction
            try:
                labels, details = model.predict_instances(
                    img_norm,
                    n_tiles=model._guess_n_tiles(img_norm),
                    verbose=False,
                )
            except Exception as e:
                print(f"  [{idx + 1}/{len(df)}] {sample_id}: FAILED - {e}")
                metrics = {
                    "dice": 0.0,
                    "iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "accuracy": 0.0,
                    "eval_type": "stardist_zeroshot",
                    "test_struct": struct,
                    "sample_id": sample_id,
                }
                all_metrics.append(metrics)
                continue

            # Binarize: any instance label > 0 = foreground
            pred_binary = (labels > 0).astype(np.uint8)

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
                    "eval_type": "stardist_zeroshot",
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
        print("Per-structure mean Dice:")
        for struct, group in metrics_df.groupby("test_struct"):
            print(
                f"  {struct}: {group['dice'].mean():.4f} +/- {group['dice'].std():.4f}"
            )


if __name__ == "__main__":
    main()
