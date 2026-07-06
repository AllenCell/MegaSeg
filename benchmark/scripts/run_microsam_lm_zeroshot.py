#!/usr/bin/env python
"""Zero-shot MicroSAM evaluation on MegaSeg test set.

Runs MicroSAM with the vit_b_lm model (fine-tuned on light microscopy)
on all structures' test images using per-slice 2D automatic mask generation.
Outputs binarized predictions and per-sample metrics.

This mirrors the vanilla SAM evaluation (run_microsam_zeroshot.py) but uses
the microscopy-specific model for a fairer comparison.

Usage:
    python scripts/run_microsam_lm_zeroshot.py --gpu 0
    python scripts/run_microsam_lm_zeroshot.py --gpu 0 --structures ACTN1 TOMM20
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
    "ACTN1", "ATP2A2", "CTNNB1", "DSP", "GJA1", "MYH10",
    "NUP153", "RAB5A", "SEC61B", "SON", "ST6GAL1", "TJP1", "TOMM20",
]

DATA_CSV_DIR = Path("/allen/aics/rep_learn/megaseg/data/variance-100x")
BENCHMARK_DIR = Path("/allen/aics/rep_learn/megaseg/benchmark")
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "outputs" / "eval" / "uint8" / "microsam_lm_zeroshot"
DEFAULT_METRICS_CSV = BENCHMARK_DIR / "results" / "uint8" / "microsam_lm_zeroshot_metrics.csv"

MODEL_TYPE = "vit_b_lm"


def compute_metrics(pred, gt):
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    tp = intersection
    fp = np.logical_and(pred_bool, ~gt_bool).sum()
    fn = np.logical_and(~pred_bool, gt_bool).sum()
    tn = np.logical_and(~pred_bool, ~gt_bool).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (union + 1e-8)
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


def segment_slice_2d(amg, image_slice):
    """Run AMG on a single 2D slice, return binary mask."""
    amg.initialize(image_slice)
    masks = amg.generate(
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )
    if isinstance(masks, np.ndarray):
        return (masks > 0).astype(np.uint8)
    if isinstance(masks, list):
        combined = np.zeros(image_slice.shape[:2], dtype=np.uint8)
        for m in masks:
            if isinstance(m, dict) and "segmentation" in m:
                combined[m["segmentation"]] = 1
            elif isinstance(m, np.ndarray):
                combined[m > 0] = 1
        return combined
    return np.zeros(image_slice.shape[:2], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="MicroSAM (vit_b_lm) zero-shot evaluation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--structures", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from micro_sam.util import get_sam_model
    from micro_sam.instance_segmentation import AutomaticMaskGenerator

    print(f"Loading MicroSAM model ({MODEL_TYPE})...")
    predictor = get_sam_model(model_type=MODEL_TYPE, device="cuda")
    amg = AutomaticMaskGenerator(predictor, points_per_side=32)
    print("Model loaded.")

    structs = args.structures if args.structures else STRUCTURES
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else DEFAULT_METRICS_CSV
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

            if pred_path.exists():
                pred_binary = tifffile.imread(str(pred_path))
                gt = tifffile.imread(seg_path)
                gt_binary = (gt > 0).astype(np.uint8)
                metrics = compute_metrics(pred_binary, gt_binary)
                metrics.update({
                    "eval_type": "microsam_lm_zeroshot",
                    "model_type": MODEL_TYPE,
                    "test_struct": struct,
                    "sample_id": sample_id,
                })
                all_metrics.append(metrics)
                print(f"  [{idx + 1}/{len(df)}] {sample_id}: cached (Dice={metrics['dice']:.4f})")
                continue

            t0 = time.time()

            img = tifffile.imread(raw_path)
            img_f = img.astype(np.float32)
            img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)

            # Per-slice 2D segmentation (same approach as vanilla SAM)
            combined = np.zeros(img.shape, dtype=np.uint8)
            for z in range(img.shape[0]):
                slice_2d = img_f[z]
                # MicroSAM expects (H, W) or (H, W, C) — single-channel grayscale
                mask = segment_slice_2d(amg, slice_2d)
                combined[z] = mask

            pred_binary = combined

            if args.save_predictions:
                tifffile.imwrite(str(pred_path), pred_binary)

            gt = tifffile.imread(seg_path)
            gt_binary = (gt > 0).astype(np.uint8)

            metrics = compute_metrics(pred_binary, gt_binary)
            elapsed = time.time() - t0

            metrics.update({
                "eval_type": "microsam_lm_zeroshot",
                "model_type": MODEL_TYPE,
                "test_struct": struct,
                "sample_id": sample_id,
            })
            all_metrics.append(metrics)

            print(
                f"  [{idx + 1}/{len(df)}] {sample_id}: "
                f"Dice={metrics['dice']:.4f} ({elapsed:.1f}s)"
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(str(metrics_csv), index=False)
    print(f"\nSaved {len(metrics_df)} metrics to {metrics_csv}")

    if len(metrics_df) > 0:
        print(f"\nOverall Dice: {metrics_df['dice'].mean():.4f} +/- {metrics_df['dice'].std():.4f}")
        print("\nPer-structure Dice:")
        for struct in structs:
            sub = metrics_df[metrics_df["test_struct"] == struct]
            if len(sub) > 0:
                print(f"  {struct:12s}: {sub['dice'].mean():.4f} +/- {sub['dice'].std():.4f} (n={len(sub)})")


if __name__ == "__main__":
    main()
