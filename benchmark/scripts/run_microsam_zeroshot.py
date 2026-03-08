#!/usr/bin/env python
"""Zero-shot SAM evaluation on MegaSeg test set.

Runs SAM (Segment Anything Model) with SamAutomaticMaskGenerator (AMG)
on all 14 structures' test images.  Processes each Z-slice independently
(2D AMG) and stacks results.

Uses the base segment-anything package (no micro-sam / elf dependency).
Model checkpoint is downloaded automatically if not present.

Outputs binarized predictions (instance labels -> binary foreground mask)
and per-sample metrics (Dice, IoU, Precision, Recall, Accuracy).
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

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

# SAM checkpoint URLs (from Meta)
SAM_CHECKPOINTS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


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


def normalize_slice(img_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to uint8 [0, 255] for SAM input."""
    img_f = img_slice.astype(np.float32)
    mn, mx = img_f.min(), img_f.max()
    if mx - mn > 0:
        img_f = (img_f - mn) / (mx - mn) * 255.0
    else:
        img_f = np.zeros_like(img_f)
    return img_f.astype(np.uint8)


def download_checkpoint(model_type: str, cache_dir: Path) -> Path:
    """Download SAM checkpoint if not cached."""
    url = SAM_CHECKPOINTS[model_type]
    filename = url.split("/")[-1]
    ckpt_path = cache_dir / filename

    if ckpt_path.exists():
        print(f"Checkpoint already cached: {ckpt_path}")
        return ckpt_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM {model_type} checkpoint to {ckpt_path} ...")
    urllib.request.urlretrieve(url, str(ckpt_path))
    print(f"Download complete: {ckpt_path}")
    return ckpt_path


def masks_to_binary(masks: list) -> np.ndarray:
    """Convert AMG output (list of mask dicts) to a single binary mask.

    Any pixel covered by at least one mask is foreground.
    """
    if not masks:
        return None
    h, w = masks[0]["segmentation"].shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        binary[m["segmentation"]] = 1
    return binary


def main():
    parser = argparse.ArgumentParser(description="SAM zero-shot evaluation (AMG)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/uint8/sam_zeroshot",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/results/uint8/sam_zeroshot_metrics.csv",
        help="Output CSV for per-sample metrics",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type (default: vit_b)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/envs/microsam/checkpoints",
        help="Directory to cache SAM checkpoints",
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
    # AMG parameters - use SAM defaults but allow tuning
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="AMG grid density (default: 32)",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.88,
        help="AMG predicted IoU threshold (default: 0.88)",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.95,
        help="AMG stability score threshold (default: 0.95)",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # Download/load checkpoint
    ckpt_path = download_checkpoint(args.model_type, Path(args.checkpoint_dir))

    print(f"Loading SAM model: {args.model_type}")
    sam = sam_model_registry[args.model_type](checkpoint=str(ckpt_path))
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )

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
                        "eval_type": "sam_zeroshot",
                        "model_type": args.model_type,
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
            img_3d = tifffile.imread(raw_path)
            nz, ny, nx = img_3d.shape

            # Process each Z-slice independently with AMG
            pred_3d = np.zeros((nz, ny, nx), dtype=np.uint8)

            for z in range(nz):
                img_slice = img_3d[z]  # (Y, X)

                # Normalize to uint8 and convert to RGB (SAM expects 3-channel)
                img_norm = normalize_slice(img_slice)
                img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)  # (H, W, 3)

                try:
                    masks = mask_generator.generate(img_rgb)
                    binary = masks_to_binary(masks)
                    if binary is not None:
                        pred_3d[z] = binary
                except Exception as e:
                    # Some slices might fail (e.g., completely dark)
                    if z == 0:
                        print(f"    Slice {z} failed: {e}")
                    pred_3d[z] = 0

            # Save prediction
            if args.save_predictions:
                tifffile.imwrite(str(pred_path), pred_3d)

            # Load GT and compute metrics
            gt = tifffile.imread(seg_path)
            gt_binary = (gt > 0).astype(np.uint8)

            metrics = compute_metrics(pred_3d, gt_binary)
            elapsed = time.time() - t0

            metrics.update(
                {
                    "eval_type": "sam_zeroshot",
                    "model_type": args.model_type,
                    "test_struct": struct,
                    "sample_id": sample_id,
                }
            )
            all_metrics.append(metrics)

            print(
                f"  [{idx + 1}/{len(df)}] {sample_id}: Dice={metrics['dice']:.4f}, "
                f"IoU={metrics['iou']:.4f} ({elapsed:.1f}s, {nz} slices)"
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
