#!/usr/bin/env python
"""Evaluate bottleneck surgery checkpoints.

Runs inference on LMNB1 test data, computes segmentation metrics
(Dice, Hausdorff-95, Surface Dice), and checks bottleneck activation health.

Usage:
    python scripts/evaluate_surgery.py --strategy a
    python scripts/evaluate_surgery.py --all
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

# Surgery strategy definitions
STRATEGIES = {
    "a": {
        "name": "reinit_bottleneck",
        "experiment": "lmnb1_surgery_a_50ep",
        "run_name": "lmnb1_surgery_a_reinit_bottleneck",
    },
    "b": {
        "name": "reinit_cascade",
        "experiment": "lmnb1_surgery_b_50ep",
        "run_name": "lmnb1_surgery_b_reinit_cascade",
    },
    "c": {
        "name": "reinit_deep",
        "experiment": "lmnb1_surgery_c_50ep",
        "run_name": "lmnb1_surgery_c_reinit_deep",
    },
    "d": {
        "name": "freeze_shallow",
        "experiment": "lmnb1_surgery_d_50ep",
        "run_name": "lmnb1_surgery_d_freeze_shallow",
    },
}

# Control baselines for comparison
CONTROLS = {
    "ft_100_converged": {
        "ckpt": "outputs/train/runs/lmnb1_data100_converged/lmnb1_ft_data100_converged/2026-02-25_15-35-24/checkpoints/epoch_270.ckpt",
        "label": "FT 100% converged (control)",
    },
    "megaseg_pretrained": {
        "ckpt": str(
            Path(__file__).resolve().parent.parent
            / "checkpoints"
            / "megaseg"
            / "dynunet.ckpt"
        ),
        "label": "MegaSeg pretrained (before surgery)",
    },
}

TEST_CSV = Path(
    "/allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/local/test_lmnb1_local_cytodl.csv"
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
    test_images: List[Tuple[str, torch.Tensor]],
    device: str = "cuda",
) -> pd.DataFrame:
    """Check activation health across all DynUNet layers.

    Returns DataFrame with per-layer dead channel counts.
    """
    extractor = ActivationExtractor(model, LAYER_NAMES)
    dead_threshold = 1e-6

    # Accumulate dead channel info across multiple images
    layer_dead_counts = {name: None for name in LAYER_NAMES}
    n_images = 0

    for sample_id, tensor in test_images:
        extractor.clear()
        with torch.no_grad():
            tensor = tensor.to(device)
            _ = model(tensor)

        for name in LAYER_NAMES:
            if name not in extractor.activations:
                continue
            act = extractor.activations[name].float()
            n_channels = act.shape[1]
            # Channel is dead if max absolute activation < threshold
            channel_max = act.abs().amax(dim=(0, *range(2, act.ndim)))  # [C]
            is_dead = (channel_max < dead_threshold).cpu().numpy()

            if layer_dead_counts[name] is None:
                layer_dead_counts[name] = np.zeros(n_channels, dtype=int)
            layer_dead_counts[name] += is_dead.astype(int)

        n_images += 1

    extractor.remove_hooks()

    rows = []
    for name in LAYER_NAMES:
        if layer_dead_counts[name] is None:
            continue
        counts = layer_dead_counts[name]
        n_ch = len(counts)
        # Channel is consistently dead if dead in ALL images
        n_dead = int((counts == n_images).sum())
        rows.append(
            {
                "layer": name,
                "n_channels": n_ch,
                "n_dead": n_dead,
                "pct_dead": round(100.0 * n_dead / n_ch, 1) if n_ch > 0 else 0,
                "n_images": n_images,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------


def _resolve_path(path: str) -> str:
    """Resolve a data path, stripping hash prefix if the file is not found.

    The test CSVs reference files as ``{64-hex-hash}_{fov}.ome.tiff`` but
    the actual files on disk may be stored without the hash prefix, i.e.
    ``{fov}.ome.tiff``.  Try the original path first; fall back to the
    hash-stripped variant.
    """
    p = Path(path)
    if p.exists():
        return path
    # Strip leading 64-char hex hash + underscore from the basename
    import re

    m = re.match(r"^[a-f0-9]{64}_(.+)$", p.name)
    if m:
        alt = p.parent / m.group(1)
        if alt.exists():
            return str(alt)
    raise FileNotFoundError(f"Cannot resolve path (tried hash-strip fallback): {path}")


def _get_egfp_channel_idx(path: str, channel_name: str = "EGFP") -> int:
    """Resolve EGFP channel index from OME-TIFF metadata.

    The channel index varies per file (e.g., index 3 in 7-channel files,
    index 1 in 4-channel files). The training pipeline uses BioIO with
    channel_name='EGFP' to resolve dynamically - we replicate that here
    using tifffile's OME metadata for speed.
    """
    import xml.etree.ElementTree as ET

    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.ome_metadata:
            raise ValueError(f"No OME metadata in {path}")
        root = ET.fromstring(tif.ome_metadata)
        # Try common OME namespace prefixes
        for ns_uri in [
            "http://www.openmicroscopy.org/Schemas/OME/2016-06",
            "http://www.openmicroscopy.org/Schemas/OME/2015-01",
        ]:
            channels = root.findall(f".//{{{ns_uri}}}Channel")
            if channels:
                break
        else:
            # Try without namespace
            channels = root.findall(".//Channel")

        for i, ch in enumerate(channels):
            if ch.attrib.get("Name") == channel_name:
                return i

    raise ValueError(f"Channel '{channel_name}' not found in {path}")


def _load_egfp_volume(path: str) -> np.ndarray:
    """Load EGFP channel from an OME-TIFF as a float32 ZYX volume."""
    import tifffile

    egfp_idx = _get_egfp_channel_idx(path)
    full_img = tifffile.imread(path)

    # Files are ZCYX (confirmed from OME axes metadata)
    if full_img.ndim == 4:
        img = full_img[:, egfp_idx, :, :].astype(np.float32)
    elif full_img.ndim == 3:
        img = full_img.astype(np.float32)
    else:
        raise ValueError(f"Unexpected ndim={full_img.ndim} for {path}")

    # Z-score normalization (matches training: NormalizeIntensityd channel_wise)
    mean_val = img.mean()
    std_val = img.std()
    if std_val > 0:
        img = (img - mean_val) / std_val

    return img


def load_test_images(
    csv_path: str,
    n_images: int = 5,
    patch_shape: Tuple[int, int, int] = (32, 128, 128),
) -> List[Tuple[str, torch.Tensor]]:
    """Load test images as patches for activation analysis."""
    df = pd.read_csv(csv_path)
    samples = df.head(n_images)

    images = []
    for _, row in samples.iterrows():
        path = _resolve_path(row["raw"])
        sample_id = Path(path).stem.split(".")[0]
        log.info("Loading: %s", sample_id)
        img = _load_egfp_volume(path)

        z, y, x = img.shape
        pz, py, px = patch_shape
        z0 = max(0, (z - pz) // 2)
        y0 = max(0, (y - py) // 2)
        x0 = max(0, (x - px) // 2)
        patch = img[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]

        if patch.shape != patch_shape:
            padded = np.zeros(patch_shape, dtype=np.float32)
            sz, sy, sx = patch.shape
            padded[:sz, :sy, :sx] = patch
            patch = padded

        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
        images.append((sample_id, tensor))

    return images


def load_full_test_data(
    csv_path: str,
) -> List[Dict]:
    """Load all test images and GT for full evaluation."""
    import tifffile

    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        raw_path = _resolve_path(row["raw"])
        seg_path = _resolve_path(row["seg"])
        sample_id = Path(raw_path).stem.split(".")[0]

        # Load raw with proper EGFP channel resolution
        img = _load_egfp_volume(raw_path)

        # Load GT
        gt = tifffile.imread(seg_path)
        gt_mask = gt > 0

        samples.append(
            {
                "sample_id": sample_id,
                "raw": img,
                "gt_mask": gt_mask,
            }
        )

    return samples


# -----------------------------------------------------------------------
# Sliding window inference
# -----------------------------------------------------------------------


def sliding_window_inference(
    model: nn.Module,
    volume: np.ndarray,
    patch_shape: Tuple[int, int, int] = (32, 128, 128),
    overlap: float = 0.25,
    device: str = "cuda",
) -> np.ndarray:
    """Run sliding window inference on a 3D volume.

    Args:
        model: DynUNet model in eval mode
        volume: (Z, Y, X) float32 normalized volume
        patch_shape: patch size matching training
        overlap: fraction of overlap between patches
        device: cuda or cpu

    Returns:
        pred_mask: (Z, Y, X) binary prediction
    """
    z, y, x = volume.shape
    pz, py, px = patch_shape
    stride_z = max(1, int(pz * (1 - overlap)))
    stride_y = max(1, int(py * (1 - overlap)))
    stride_x = max(1, int(px * (1 - overlap)))

    # Output accumulator
    pred_sum = np.zeros_like(volume, dtype=np.float32)
    count = np.zeros_like(volume, dtype=np.float32)

    # Generate patch positions
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
                    # Pad if needed
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

                    # Crop pred back to actual shape if we padded
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

    # Average overlapping predictions
    count = np.maximum(count, 1.0)
    pred_avg = pred_sum / count
    pred_mask = pred_avg >= 0.5
    return pred_mask


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
    from scipy.ndimage import distance_transform_edt

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return float("inf")

    # Distance from GT surface to nearest pred surface
    gt_border = gt ^ np.zeros_like(gt)  # placeholder
    pred_border = pred ^ np.zeros_like(pred)

    # Use distance transforms for efficiency
    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt, sampling=voxel_spacing)

    # Surface voxels (boundary)
    from scipy.ndimage import binary_erosion

    struct = np.ones((3, 3, 3), dtype=bool)
    gt_surface = gt & ~binary_erosion(gt, structure=struct)
    pred_surface = pred & ~binary_erosion(pred, structure=struct)

    if not gt_surface.any() or not pred_surface.any():
        return float("inf")

    # Distances from GT surface to nearest pred voxel and vice versa
    dist_gt_to_pred = dt_pred[gt_surface]
    dist_pred_to_gt = dt_gt[pred_surface]

    all_dists = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
    return float(np.percentile(all_dists, 95))


def compute_surface_dice(
    gt: np.ndarray, pred: np.ndarray, tolerance: float = 1.0, voxel_spacing=(1, 1, 1)
) -> float:
    """Compute Surface Dice (normalized surface distance) at given tolerance."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0

    struct = np.ones((3, 3, 3), dtype=bool)
    gt_surface = gt & ~binary_erosion(gt, structure=struct)
    pred_surface = pred & ~binary_erosion(pred, structure=struct)

    if not gt_surface.any() or not pred_surface.any():
        return 0.0

    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt, sampling=voxel_spacing)

    gt_to_pred = dt_pred[gt_surface]
    pred_to_gt = dt_gt[pred_surface]

    n_gt_close = (gt_to_pred <= tolerance).sum()
    n_pred_close = (pred_to_gt <= tolerance).sum()

    n_gt_surface = gt_surface.sum()
    n_pred_surface = pred_surface.sum()

    return float((n_gt_close + n_pred_close) / (n_gt_surface + n_pred_surface))


# -----------------------------------------------------------------------
# Evaluation pipeline
# -----------------------------------------------------------------------


def find_checkpoint(strategy_key: str) -> Optional[str]:
    """Find the best or last checkpoint for a surgery strategy."""
    info = STRATEGIES[strategy_key]
    exp_dir = PROJECT_ROOT / "outputs" / "train" / "runs" / info["experiment"]
    if not exp_dir.exists():
        return None

    # Find the run directory (latest timestamp)
    run_dirs = sorted(exp_dir.glob(f"{info['run_name']}/*"))
    if not run_dirs:
        return None

    ckpt_dir = run_dirs[-1] / "checkpoints"
    if not ckpt_dir.exists():
        return None

    # Prefer best checkpoint, fall back to last
    ckpts = list(ckpt_dir.glob("epoch_*.ckpt"))
    if ckpts:
        # Best = non-last checkpoint (saved by ModelCheckpoint monitor)
        return str(ckpts[0])

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)

    return None


def evaluate_checkpoint(
    ckpt_path: str,
    label: str,
    device: str = "cuda",
    n_health_images: int = 5,
    max_eval_images: Optional[int] = None,
) -> Dict:
    """Full evaluation: metrics + health check."""
    log.info("=== Evaluating: %s ===", label)
    log.info("Checkpoint: %s", ckpt_path)

    t0 = time.time()
    model = load_dynunet(ckpt_path, device=device)
    log.info("Model loaded in %.1fs", time.time() - t0)

    results = {"label": label, "checkpoint": ckpt_path}

    # 1. Bottleneck health check (quick, 5 images)
    log.info("Running bottleneck health check...")
    health_images = load_test_images(str(TEST_CSV), n_images=n_health_images)
    health_df = check_bottleneck_health(model, health_images, device=device)
    results["health"] = health_df.to_dict("records")

    bottleneck_row = health_df[health_df["layer"] == "bottleneck"]
    if not bottleneck_row.empty:
        pct_dead = bottleneck_row.iloc[0]["pct_dead"]
        log.info("  Bottleneck dead channels: %.1f%%", pct_dead)
        results["bottleneck_pct_dead"] = pct_dead

    # 2. Full segmentation evaluation
    log.info("Running segmentation evaluation on test set...")
    samples = load_full_test_data(str(TEST_CSV))
    if max_eval_images:
        samples = samples[:max_eval_images]

    per_sample = []
    for i, s in enumerate(samples):
        log.info("  [%d/%d] %s", i + 1, len(samples), s["sample_id"])
        pred_mask = sliding_window_inference(model, s["raw"], device=device)

        # Handle shape mismatch between prediction and GT
        gt = s["gt_mask"]
        if pred_mask.shape != gt.shape:
            # Crop/pad to match
            min_shape = tuple(min(a, b) for a, b in zip(pred_mask.shape, gt.shape))
            pred_mask = pred_mask[: min_shape[0], : min_shape[1], : min_shape[2]]
            gt = gt[: min_shape[0], : min_shape[1], : min_shape[2]]

        dice = compute_dice(gt, pred_mask)
        hd95 = compute_hausdorff_95(gt, pred_mask)
        sd = compute_surface_dice(gt, pred_mask)

        per_sample.append(
            {
                "sample_id": s["sample_id"],
                "dice": dice,
                "hausdorff_95": hd95,
                "surface_dice": sd,
            }
        )
        log.info("    Dice=%.4f  HD95=%.2f  SurfDice=%.4f", dice, hd95, sd)

    results["per_sample"] = per_sample
    metrics_df = pd.DataFrame(per_sample)
    results["summary"] = {
        "n_samples": len(per_sample),
        "dice_mean": float(metrics_df["dice"].mean()),
        "dice_std": float(metrics_df["dice"].std()),
        "hausdorff_95_mean": float(metrics_df["hausdorff_95"].mean()),
        "hausdorff_95_std": float(metrics_df["hausdorff_95"].std()),
        "surface_dice_mean": float(metrics_df["surface_dice"].mean()),
        "surface_dice_std": float(metrics_df["surface_dice"].std()),
    }

    elapsed = time.time() - t0
    results["eval_time_sec"] = elapsed
    log.info("Evaluation completed in %.1fs", elapsed)
    log.info(
        "  SUMMARY: Dice=%.4f+-%.4f  HD95=%.2f+-%.2f  SurfDice=%.4f+-%.4f",
        results["summary"]["dice_mean"],
        results["summary"]["dice_std"],
        results["summary"]["hausdorff_95_mean"],
        results["summary"]["hausdorff_95_std"],
        results["summary"]["surface_dice_mean"],
        results["summary"]["surface_dice_std"],
    )

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate surgery checkpoints")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        nargs="+",
        help="One or more strategies to evaluate (e.g. --strategy b c d)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluate all available strategies"
    )
    parser.add_argument(
        "--controls", action="store_true", help="Also evaluate control baselines"
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Device (default cuda:1 to avoid training GPU)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit eval to N images (for quick testing)",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only run health check, skip full eval",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    if not args.strategy and not args.all:
        parser.error("Specify --strategy X or --all")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "results" / "surgery"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies_to_eval = list(STRATEGIES.keys()) if args.all else (args.strategy or [])
    all_results = []

    for sk in strategies_to_eval:
        ckpt = find_checkpoint(sk)
        if not ckpt:
            log.warning("No checkpoint found for strategy %s, skipping", sk)
            continue

        label = f"Surgery {sk.upper()}: {STRATEGIES[sk]['name']}"

        if args.health_only:
            log.info("=== Health check only: %s ===", label)
            model = load_dynunet(ckpt, device=args.device)
            health_images = load_test_images(str(TEST_CSV), n_images=5)
            health_df = check_bottleneck_health(
                model, health_images, device=args.device
            )
            log.info("\n%s", health_df.to_string(index=False))
            health_df.to_csv(out_dir / f"health_strategy_{sk}.csv", index=False)
            del model
            torch.cuda.empty_cache()
            continue

        result = evaluate_checkpoint(
            ckpt, label, device=args.device, max_eval_images=args.max_images
        )
        all_results.append(result)

        # Save per-strategy results
        pd.DataFrame(result["per_sample"]).to_csv(
            out_dir / f"metrics_strategy_{sk}.csv", index=False
        )
        pd.DataFrame(result["health"]).to_csv(
            out_dir / f"health_strategy_{sk}.csv", index=False
        )

    # Evaluate controls if requested
    if args.controls and not args.health_only:
        for ck, cinfo in CONTROLS.items():
            ckpt = cinfo["ckpt"]
            if not ckpt.startswith("/"):
                ckpt = str(PROJECT_ROOT / ckpt)
            if not Path(ckpt).exists():
                log.warning("Control checkpoint not found: %s", ckpt)
                continue

            result = evaluate_checkpoint(
                ckpt,
                cinfo["label"],
                device=args.device,
                max_eval_images=args.max_images,
            )
            all_results.append(result)
            pd.DataFrame(result["per_sample"]).to_csv(
                out_dir / f"metrics_{ck}.csv", index=False
            )
            pd.DataFrame(result["health"]).to_csv(
                out_dir / f"health_{ck}.csv", index=False
            )

    # Save summary
    if all_results:
        summary_rows = []
        for r in all_results:
            row = {"label": r["label"], **r["summary"]}
            if "bottleneck_pct_dead" in r:
                row["bottleneck_pct_dead"] = r["bottleneck_pct_dead"]
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "surgery_summary.csv", index=False)
        log.info("\n=== SUMMARY ===")
        log.info("\n%s", summary_df.to_string(index=False))

        # Also save as JSON for easy loading
        with open(out_dir / "surgery_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
