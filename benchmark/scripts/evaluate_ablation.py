#!/usr/bin/env python
"""Evaluate ablation diversity models across all structures.

Tests each ablation model (2-struct, 4-struct, 7-struct, 14-struct) on all 14
individual structure test sets, computing Dice/HD95 per structure. Also checks
bottleneck activation health. Distinguishes "seen" (training) vs "held-out"
structures.

Usage:
    python scripts/evaluate_ablation.py --ablation 2
    python scripts/evaluate_ablation.py --ablation 4 7
    python scripts/evaluate_ablation.py --ablation 2 4 7 14
    python scripts/evaluate_ablation.py --all
    python scripts/evaluate_ablation.py --all --max-images 10   # Quick pass
    python scripts/evaluate_ablation.py --health-only --all     # Just bottleneck check
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data" / "variance-100x"

# DynUNet layers to check for health
LAYER_NAMES = [
    "input_block",
    "downsamples.0",
    "downsamples.1",
    "downsamples.2",
    "bottleneck",
    "upsamples.0",
    "upsamples.1",
    "upsamples.2",
    "upsamples.3",
    "output_block",
]

# All 14 structures evaluated
ALL_STRUCTURES = [
    "AAVS1",
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]

# Which structures each ablation was trained on
ABLATION_TRAINING_STRUCTS = {
    "2_structs": {"TOMM20", "TJP1"},
    "4_structs": {"ACTN1", "ATP2A2", "CTNNB1", "MYH10"},
    "7_structs": {"ACTN1", "ATP2A2", "CTNNB1", "MYH10", "RAB5A", "ST6GAL1", "TOMM20"},
    "14_structs": {
        "ACTN1",
        "ATP2A2",
        "CTNNB1",
        "DSP",
        "GJA1",
        "MYH10",
        "NUP153",
        "RAB5A",
        "SEC61B",
        "SON",
        "ST6GAL1",
        "TJP1",
        "TOMM20",
        "AAVS1",
    },
}

# MegaSeg pretrained was trained on 13 structures (all except AAVS1, LMNB1, NPM1)
MEGASEG_TRAINING_STRUCTS = {
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
}

# Checkpoint paths
ABLATION_CHECKPOINTS = {
    "2_structs": (
        "/allen/aics/rep_learn/MegaSeg/benchmark/outputs/train/runs/"
        "ablation_2_structs/dynunet_2structs_scratch/2026-03-16_23-11-50/"
        "checkpoints/epoch_527.ckpt"
    ),
    "4_structs": (
        "/allen/aics/rep_learn/MegaSeg/benchmark/outputs/train/runs/"
        "ablation_4_structs/dynunet_4structs_scratch/2026-03-06_00-14-10/"
        "checkpoints/epoch_677.ckpt"
    ),
    "7_structs": (
        "/allen/aics/rep_learn/MegaSeg/benchmark/outputs/train/runs/"
        "ablation_7_structs/dynunet_7structs_scratch/2026-03-06_00-14-12/"
        "checkpoints/epoch_683.ckpt"
    ),
    "14_structs": (
        "/allen/aics/rep_learn/MegaSeg/benchmark/outputs/train/runs/"
        "ablation_14_structs/dynunet_14structs_scratch/2026-03-06_00-14-14/"
        "checkpoints/epoch_400.ckpt"
    ),
}

# MegaSeg pretrained (13-struct) for comparison
MEGASEG_CKPT = (
    "/allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/logs/train/runs/"
    "megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/"
    "2024-07-15_00-39-47/checkpoints/epoch_650.ckpt"
)


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------


@dataclass
class ModelConfig:
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 2])
    kernel_size: List[int] = field(default_factory=lambda: [3, 3, 3, 3, 3])
    upsample_kernel_size: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout: float = 0.0
    res_block: bool = True


def load_dynunet(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load a DynUNet from a cyto-dl checkpoint."""
    from monai.networks.nets import DynUNet

    cfg = ModelConfig()
    model = DynUNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        strides=cfg.strides,
        kernel_size=cfg.kernel_size,
        upsample_kernel_size=cfg.upsample_kernel_size,
        dropout=cfg.dropout,
        res_block=cfg.res_block,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            clean[k.replace("backbone.", "", 1)] = v
    model.load_state_dict(clean, strict=True)
    model.eval()
    model.to(device)
    return model


# -----------------------------------------------------------------------
# Activation health check
# -----------------------------------------------------------------------


class ActivationExtractor:
    """Hook-based activation extractor."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks = []
        modules = dict(model.named_modules())
        for name in layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module, inp, output):
            self.activations[name] = output.detach()

        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def check_bottleneck_health(
    model: nn.Module,
    images: List[np.ndarray],
    device: str = "cuda",
    dead_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Check activation health across layers using a few sample images."""
    extractor = ActivationExtractor(model, LAYER_NAMES)
    patch_shape = (32, 128, 128)

    # Accumulate per-channel max activations
    channel_maxes: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for img in images:
            # Take a center patch
            z, y, x = img.shape
            pz, py, px = patch_shape
            zs = max(0, (z - pz) // 2)
            ys = max(0, (y - py) // 2)
            xs = max(0, (x - px) // 2)
            patch = img[zs : zs + pz, ys : ys + py, xs : xs + px]

            # Pad if needed
            actual = patch.shape
            if actual != patch_shape:
                padded = np.zeros(patch_shape, dtype=np.float32)
                padded[: actual[0], : actual[1], : actual[2]] = patch
                patch = padded

            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            extractor.clear()
            model(tensor)

            for name, act in extractor.activations.items():
                # act shape: (1, C, D, H, W)
                # Per-channel max across spatial dims
                cmax = act.abs().amax(dim=(0, 2, 3, 4)).cpu().numpy()
                if name not in channel_maxes:
                    channel_maxes[name] = cmax.copy()
                else:
                    channel_maxes[name] = np.maximum(channel_maxes[name], cmax)

    extractor.remove_hooks()

    rows = []
    for name in LAYER_NAMES:
        if name not in channel_maxes:
            continue
        cmax = channel_maxes[name]
        n_total = len(cmax)
        n_dead = int((cmax < dead_threshold).sum())
        rows.append(
            {
                "layer": name,
                "n_channels": n_total,
                "n_dead": n_dead,
                "pct_dead": 100.0 * n_dead / n_total if n_total > 0 else 0.0,
                "mean_max_act": float(cmax.mean()),
                "median_max_act": float(np.median(cmax)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Data loading (single-channel TIFFs for ablation structures)
# -----------------------------------------------------------------------


def load_test_data_for_structure(
    structure: str, n_images: Optional[int] = None
) -> List[Dict]:
    """Load test images + GT masks for a given structure."""
    import tifffile

    csv_path = DATA_DIR / structure / "test.csv"
    if not csv_path.exists():
        log.warning("Test CSV not found for %s: %s", structure, csv_path)
        return []

    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        raw_path = row["raw"]
        seg_path = row["seg"]

        if not Path(raw_path).exists():
            log.warning("Raw file missing: %s", raw_path)
            continue
        if not Path(seg_path).exists():
            log.warning("Seg file missing: %s", seg_path)
            continue

        try:
            raw = tifffile.imread(raw_path).astype(np.float32)
            seg = tifffile.imread(seg_path).astype(np.float32)

            # Normalize raw: zero mean, unit variance (matches training pipeline
            # which uses monai NormalizeIntensityd with channel_wise=True)
            mean = raw.mean()
            std = raw.std()
            if std > 0:
                raw = (raw - mean) / std
            else:
                raw = np.zeros_like(raw)

            # Binarize seg: values > 0.1 -> 1 (matches ThresholdIntensityd in pipeline)
            seg = (seg > 0.1).astype(np.float32)

            sample_id = Path(raw_path).stem
            samples.append({"sample_id": sample_id, "raw": raw, "gt_mask": seg})
        except Exception as e:
            log.warning("Error loading %s: %s", raw_path, e)
            continue

        if n_images and len(samples) >= n_images:
            break

    return samples


# -----------------------------------------------------------------------
# Sliding window inference
# -----------------------------------------------------------------------


def sliding_window_inference(
    model: nn.Module,
    volume: np.ndarray,
    device: str = "cuda",
    patch_shape: Tuple[int, int, int] = (32, 128, 128),
    overlap: float = 0.25,
) -> np.ndarray:
    """Run sliding window inference with overlap averaging."""
    z, y, x = volume.shape
    pz, py, px = patch_shape
    stride_z = max(1, int(pz * (1 - overlap)))
    stride_y = max(1, int(py * (1 - overlap)))
    stride_x = max(1, int(px * (1 - overlap)))

    pred_sum = np.zeros_like(volume, dtype=np.float32)
    count = np.zeros_like(volume, dtype=np.float32)

    z_starts = list(range(0, max(z - pz + 1, 1), stride_z))
    if z_starts[-1] + pz < z:
        z_starts.append(z - pz)
    y_starts = list(range(0, max(y - py + 1, 1), stride_y))
    if y_starts[-1] + py < y:
        y_starts.append(y - py)
    x_starts = list(range(0, max(x - px + 1, 1), stride_x))
    if x_starts[-1] + px < x:
        x_starts.append(x - px)

    with torch.no_grad():
        for zs in z_starts:
            for ys in y_starts:
                for xs in x_starts:
                    patch = volume[zs : zs + pz, ys : ys + py, xs : xs + px]
                    actual_shape = patch.shape
                    if actual_shape != patch_shape:
                        padded = np.zeros(patch_shape, dtype=np.float32)
                        padded[
                            : actual_shape[0], : actual_shape[1], : actual_shape[2]
                        ] = patch
                        patch = padded

                    tensor = (
                        torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    )
                    out = model(tensor)
                    pred = torch.sigmoid(out).squeeze().cpu().numpy()
                    pred = pred[: actual_shape[0], : actual_shape[1], : actual_shape[2]]

                    pred_sum[
                        zs : zs + actual_shape[0],
                        ys : ys + actual_shape[1],
                        xs : xs + actual_shape[2],
                    ] += pred
                    count[
                        zs : zs + actual_shape[0],
                        ys : ys + actual_shape[1],
                        xs : xs + actual_shape[2],
                    ] += 1.0

    count = np.maximum(count, 1.0)
    pred_avg = pred_sum / count
    return pred_avg >= 0.5


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------


def compute_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    denom = 2 * tp + np.logical_and(pred, ~gt).sum() + np.logical_and(~pred, gt).sum()
    return float(2 * tp / denom) if denom > 0 else 1.0


def compute_hausdorff_95(
    gt: np.ndarray, pred: np.ndarray, voxel_spacing=(1, 1, 1)
) -> float:
    """Compute 95th percentile Hausdorff distance."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return float("inf")

    struct = np.ones((3, 3, 3), dtype=bool)
    gt_surface = gt & ~binary_erosion(gt, structure=struct)
    pred_surface = pred & ~binary_erosion(pred, structure=struct)

    if not gt_surface.any() or not pred_surface.any():
        return float("inf")

    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt, sampling=voxel_spacing)

    dist_gt_to_pred = dt_pred[gt_surface]
    dist_pred_to_gt = dt_gt[pred_surface]

    all_dists = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
    return float(np.percentile(all_dists, 95))


# -----------------------------------------------------------------------
# Evaluation pipeline
# -----------------------------------------------------------------------


def evaluate_model_on_structure(
    model: nn.Module,
    structure: str,
    device: str = "cuda",
    max_images: Optional[int] = None,
) -> Dict:
    """Evaluate a model on a single structure's test set."""
    samples = load_test_data_for_structure(structure, n_images=max_images)
    if not samples:
        log.warning("No test data for %s", structure)
        return {
            "structure": structure,
            "n_samples": 0,
            "dice_mean": None,
            "hd95_mean": None,
        }

    per_sample = []
    for i, s in enumerate(samples):
        log.info("    [%d/%d] %s", i + 1, len(samples), s["sample_id"])
        pred_mask = sliding_window_inference(model, s["raw"], device=device)

        gt = s["gt_mask"]
        if pred_mask.shape != gt.shape:
            min_shape = tuple(min(a, b) for a, b in zip(pred_mask.shape, gt.shape))
            pred_mask = pred_mask[: min_shape[0], : min_shape[1], : min_shape[2]]
            gt = gt[: min_shape[0], : min_shape[1], : min_shape[2]]

        dice = compute_dice(gt, pred_mask)
        hd95 = compute_hausdorff_95(gt, pred_mask)
        per_sample.append(
            {
                "sample_id": s["sample_id"],
                "structure": structure,
                "dice": dice,
                "hausdorff_95": hd95,
            }
        )
        log.info("      Dice=%.4f  HD95=%.2f", dice, hd95)

    df = pd.DataFrame(per_sample)
    return {
        "structure": structure,
        "n_samples": len(per_sample),
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std()),
        "hd95_mean": float(df["hausdorff_95"].mean()),
        "hd95_std": float(df["hausdorff_95"].std()),
        "per_sample": per_sample,
    }


def evaluate_ablation(
    ablation: str,
    device: str = "cuda",
    max_images: Optional[int] = None,
    structures: Optional[List[str]] = None,
    health_only: bool = False,
) -> Dict:
    """Full evaluation of one ablation model across all structures."""
    ckpt_path = ABLATION_CHECKPOINTS.get(ablation)
    if not ckpt_path:
        log.error("No checkpoint for ablation %s", ablation)
        return {}
    if not Path(ckpt_path).exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        return {}

    training_structs = ABLATION_TRAINING_STRUCTS[ablation]
    eval_structures = structures or ALL_STRUCTURES

    log.info("=" * 60)
    log.info("Evaluating ablation: %s", ablation)
    log.info("Checkpoint: %s", ckpt_path)
    log.info("Training structures: %s", sorted(training_structs))
    log.info("Evaluating on: %s", eval_structures)
    log.info("=" * 60)

    t0 = time.time()
    model = load_dynunet(ckpt_path, device=device)
    log.info("Model loaded in %.1fs", time.time() - t0)

    result = {
        "ablation": ablation,
        "checkpoint": ckpt_path,
        "training_structures": sorted(training_structs),
    }

    # 1. Bottleneck health check
    log.info("Running bottleneck health check...")
    # Use first available structure for health check
    health_struct = eval_structures[0]
    health_images_data = load_test_data_for_structure(health_struct, n_images=5)
    health_images = [s["raw"] for s in health_images_data]
    if health_images:
        health_df = check_bottleneck_health(model, health_images, device=device)
        result["health"] = health_df.to_dict("records")
        bottleneck_row = health_df[health_df["layer"] == "bottleneck"]
        if not bottleneck_row.empty:
            result["bottleneck_pct_dead"] = float(bottleneck_row.iloc[0]["pct_dead"])
            log.info(
                "  Bottleneck dead channels: %.1f%%", result["bottleneck_pct_dead"]
            )
        log.info("\nHealth summary:\n%s\n", health_df.to_string(index=False))
    else:
        log.warning("No health images available")

    if health_only:
        del model
        torch.cuda.empty_cache()
        return result

    # 2. Per-structure evaluation
    all_per_sample = []
    struct_summaries = []

    for struct in eval_structures:
        seen = struct in training_structs
        label = "SEEN" if seen else "HELD-OUT"
        log.info("-" * 40)
        log.info("Structure: %s (%s)", struct, label)

        struct_result = evaluate_model_on_structure(
            model,
            struct,
            device=device,
            max_images=max_images,
        )
        struct_result["seen"] = seen
        struct_summaries.append(struct_result)

        if struct_result["per_sample"]:
            for ps in struct_result["per_sample"]:
                ps["seen"] = seen
            all_per_sample.extend(struct_result["per_sample"])

    result["structure_summaries"] = struct_summaries
    result["all_per_sample"] = all_per_sample

    # 3. Aggregate summaries
    if all_per_sample:
        full_df = pd.DataFrame(all_per_sample)

        # Overall
        result["overall"] = {
            "dice_mean": float(full_df["dice"].mean()),
            "dice_std": float(full_df["dice"].std()),
            "hd95_mean": float(full_df["hausdorff_95"].mean()),
            "hd95_std": float(full_df["hausdorff_95"].std()),
        }

        # Seen vs held-out
        seen_df = full_df[full_df["seen"]]
        heldout_df = full_df[~full_df["seen"]]

        if len(seen_df) > 0:
            result["seen_summary"] = {
                "dice_mean": float(seen_df["dice"].mean()),
                "dice_std": float(seen_df["dice"].std()),
                "hd95_mean": float(seen_df["hausdorff_95"].mean()),
                "hd95_std": float(seen_df["hausdorff_95"].std()),
                "n_structures": len([s for s in struct_summaries if s["seen"]]),
                "n_samples": len(seen_df),
            }
        if len(heldout_df) > 0:
            result["heldout_summary"] = {
                "dice_mean": float(heldout_df["dice"].mean()),
                "dice_std": float(heldout_df["dice"].std()),
                "hd95_mean": float(heldout_df["hausdorff_95"].mean()),
                "hd95_std": float(heldout_df["hausdorff_95"].std()),
                "n_structures": len([s for s in struct_summaries if not s["seen"]]),
                "n_samples": len(heldout_df),
            }

    elapsed = time.time() - t0
    result["eval_time_sec"] = elapsed
    log.info("=" * 60)
    log.info(
        "Ablation %s evaluation completed in %.1fs (%.1f min)",
        ablation,
        elapsed,
        elapsed / 60,
    )
    if "overall" in result:
        log.info(
            "  Overall: Dice=%.4f+-%.4f  HD95=%.2f+-%.2f",
            result["overall"]["dice_mean"],
            result["overall"]["dice_std"],
            result["overall"]["hd95_mean"],
            result["overall"]["hd95_std"],
        )
    if "seen_summary" in result:
        log.info(
            "  Seen: Dice=%.4f+-%.4f  HD95=%.2f+-%.2f",
            result["seen_summary"]["dice_mean"],
            result["seen_summary"]["dice_std"],
            result["seen_summary"]["hd95_mean"],
            result["seen_summary"]["hd95_std"],
        )
    if "heldout_summary" in result:
        log.info(
            "  Held-out: Dice=%.4f+-%.4f  HD95=%.2f+-%.2f",
            result["heldout_summary"]["dice_mean"],
            result["heldout_summary"]["dice_std"],
            result["heldout_summary"]["hd95_mean"],
            result["heldout_summary"]["hd95_std"],
        )

    del model
    torch.cuda.empty_cache()
    return result


def save_results(result: Dict, out_dir: Path):
    """Save evaluation results to disk."""
    ablation = result["ablation"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Health CSV
    if "health" in result:
        pd.DataFrame(result["health"]).to_csv(
            out_dir / f"health_{ablation}.csv", index=False
        )

    # Per-sample metrics CSV
    if "all_per_sample" in result:
        pd.DataFrame(result["all_per_sample"]).to_csv(
            out_dir / f"metrics_{ablation}_per_sample.csv", index=False
        )

    # Per-structure summary CSV
    if "structure_summaries" in result:
        summary_rows = []
        for s in result["structure_summaries"]:
            summary_rows.append(
                {
                    "structure": s["structure"],
                    "seen": s.get("seen", None),
                    "n_samples": s["n_samples"],
                    "dice_mean": s.get("dice_mean"),
                    "dice_std": s.get("dice_std"),
                    "hd95_mean": s.get("hd95_mean"),
                    "hd95_std": s.get("hd95_std"),
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            out_dir / f"metrics_{ablation}_per_structure.csv", index=False
        )

    # Full JSON
    with open(out_dir / f"results_{ablation}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("Results saved to %s", out_dir)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation diversity models")
    parser.add_argument(
        "--ablation",
        nargs="+",
        choices=["2", "4", "7", "14"],
        help="Ablation(s) to evaluate (2, 4, 7, 14)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluate all available ablations"
    )
    parser.add_argument(
        "--megaseg", action="store_true", help="Also evaluate MegaSeg pretrained"
    )
    parser.add_argument("--device", default="cuda:0", help="Device (default cuda:0)")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit to N images per structure"
    )
    parser.add_argument(
        "--health-only", action="store_true", help="Only run health check"
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=None,
        help="Evaluate only specific structures",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    if not args.ablation and not args.all and not args.megaseg:
        parser.error("Specify --ablation X, --all, or --megaseg")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / "ablation"
    )

    ablations_to_eval = []
    if args.all:
        for k, v in ABLATION_CHECKPOINTS.items():
            if v and Path(v).exists():
                ablations_to_eval.append(k)
            else:
                log.warning("Skipping %s (no checkpoint available)", k)
    elif args.ablation:
        for a in args.ablation:
            key = f"{a}_structs"
            if key in ABLATION_CHECKPOINTS and ABLATION_CHECKPOINTS[key]:
                ablations_to_eval.append(key)
            else:
                log.warning("No checkpoint for %s", key)

    all_results = []

    for ablation in ablations_to_eval:
        result = evaluate_ablation(
            ablation,
            device=args.device,
            max_images=args.max_images,
            structures=args.structures,
            health_only=args.health_only,
        )
        if result:
            save_results(result, out_dir)
            all_results.append(result)

    # Evaluate MegaSeg pretrained if requested
    if args.megaseg and not args.health_only:
        log.info("=" * 60)
        log.info("Evaluating MegaSeg pretrained (13-struct)")
        # Temporarily set as an ablation with correct 13 training structures
        # (MegaSeg-13 was trained on 13 structures, excluding AAVS1, LMNB1, NPM1)
        old_ckpts = ABLATION_CHECKPOINTS.copy()
        ABLATION_CHECKPOINTS["megaseg_pretrained"] = MEGASEG_CKPT
        ABLATION_TRAINING_STRUCTS["megaseg_pretrained"] = MEGASEG_TRAINING_STRUCTS

        result = evaluate_ablation(
            "megaseg_pretrained",
            device=args.device,
            max_images=args.max_images,
            structures=args.structures,
        )
        if result:
            save_results(result, out_dir)
            all_results.append(result)

    # Print final comparison table
    if len(all_results) > 1:
        log.info("\n" + "=" * 60)
        log.info("COMPARISON SUMMARY")
        log.info("=" * 60)
        for r in all_results:
            abl = r.get("ablation", "?")
            overall = r.get("overall", {})
            seen = r.get("seen_summary", {})
            held = r.get("heldout_summary", {})
            dead = r.get("bottleneck_pct_dead", "N/A")
            log.info(
                "  %s: Overall Dice=%.4f | Seen=%.4f | Held-out=%.4f | Dead=%.1f%%",
                abl,
                overall.get("dice_mean", 0),
                seen.get("dice_mean", 0),
                held.get("dice_mean", 0),
                dead if isinstance(dead, float) else 0,
            )


if __name__ == "__main__":
    main()
