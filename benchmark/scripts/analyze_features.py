"""
Feature Space Analysis for MegaSeg DynUNet Models

Analyzes weight statistics, activation distributions, and feature space structure
across multiple DynUNet checkpoints (pretrained, fine-tuned, scratch-trained).

Addresses GitHub issue #23: Feature Analysis / Interpretability

Usage:
    # Phase 1A: Weight statistics only (no GPU needed)
    uv run python scripts/analyze_features.py --phase weights

    # Phase 1B: Activation statistics (needs GPU + test images)
    uv run python scripts/analyze_features.py --phase activations

    # Phase 2: Feature visualization (t-SNE, PCA, channel activity)
    uv run python scripts/analyze_features.py --phase visualize

    # All phases
    uv run python scripts/analyze_features.py --phase all

    # Custom checkpoint set
    uv run python scripts/analyze_features.py --phase all --checkpoints custom.json
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # benchmark/
RESULTS_DIR = REPO_ROOT / "results" / "feature_analysis"

# DynUNet layers in forward-pass order
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

# Short display names for plots
LAYER_SHORT = [
    "input",
    "enc0",
    "enc1",
    "enc2",
    "bottleneck",
    "dec0",
    "dec1",
    "dec2",
    "dec3",
    "output",
]

# Expected output channels per layer
LAYER_CHANNELS = [32, 64, 128, 256, 320, 256, 128, 64, 32, 1]

# Default 4 backbone checkpoints
DEFAULT_CHECKPOINTS = {
    "megaseg_pretrained": {
        "path": (
            "/allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/"
            "logs/train/runs/megaseg_b64_deep_new_13-STRUCTS/"
            "dynUNet_st6_base_cytodl/2024-07-15_00-39-47/checkpoints/"
            "epoch_650.ckpt"
        ),
        "label": "MegaSeg pretrained",
        "color": "#e74c3c",
        "dice": 0.298,
    },
    "finetune_100pct_converged": {
        "path": (
            str(
                REPO_ROOT
                / "outputs"
                / "train"
                / "runs"
                / "lmnb1_data100_converged"
                / "lmnb1_ft_data100_converged"
                / "2026-02-25_15-35-24"
                / "checkpoints"
                / "epoch_270.ckpt"
            )
        ),
        "label": "FT 100% converged",
        "color": "#2ecc71",
        "dice": 0.958,
    },
    "scratch_100pct_converged": {
        "path": (
            str(
                REPO_ROOT
                / "outputs"
                / "train"
                / "runs"
                / "lmnb1_scratch_data100_converged"
                / "lmnb1_scratch_data100_converged"
                / "2026-03-03_16-48-33"
                / "checkpoints"
                / "epoch_411.ckpt"
            )
        ),
        "label": "Scratch 100% converged",
        "color": "#3498db",
        "dice": 0.935,
    },
    "finetune_10pct_10ep": {
        "path": (
            str(
                REPO_ROOT
                / "outputs"
                / "train"
                / "runs"
                / "lmnb1_data10_10epochs"
                / "lmnb1_ft_data10_10epochs"
                / "2026-03-02_10-34-45"
                / "checkpoints"
                / "epoch_009.ckpt"
            )
        ),
        "label": "FT 10% 10ep",
        "color": "#f39c12",
        "dice": 0.699,
    },
}

# Test data CSV for activation analysis
TEST_CSV = (
    "/allen/aics/rep_learn/MegaSeg/data/variance-100x/"
    "LMNB1/local_renamed/test/test_lmnb1_local_cytodl.csv"
)


@dataclass
class ModelConfig:
    """DynUNet configuration - must match training config."""

    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2, 2])
    kernel_size: List[int] = field(default_factory=lambda: [3, 3, 3, 3, 3])
    upsample_kernel_size: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout: float = 0.0
    res_block: bool = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_dynunet(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load a DynUNet from a cyto-dl checkpoint.

    The checkpoint stores weights under 'backbone.*' keys.
    We strip that prefix and load into a bare DynUNet.
    """
    cfg = ModelConfig()
    from monai.networks.nets import DynUNet

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
    # Strip 'backbone.' prefix from cyto-dl MultiTaskIm2Im wrapper
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            clean[k.replace("backbone.", "", 1)] = v
    model.load_state_dict(clean, strict=True)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Phase 1A: Weight Statistics
# ---------------------------------------------------------------------------


def compute_weight_stats(model: nn.Module) -> pd.DataFrame:
    """Compute per-layer weight statistics for a DynUNet model.

    Returns DataFrame with columns:
        layer, param_name, shape, n_params, mean, std,
        abs_mean, pct_near_zero, max_abs, min_abs
    """
    rows = []
    for layer_name in LAYER_NAMES:
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            continue
        for pname, param in module.named_parameters():
            w = param.detach().float().cpu()
            n = w.numel()
            near_zero = (w.abs() < 1e-6).sum().item()
            rows.append(
                {
                    "layer": layer_name,
                    "param": pname,
                    "shape": str(list(w.shape)),
                    "n_params": n,
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                    "abs_mean": w.abs().mean().item(),
                    "pct_near_zero": 100.0 * near_zero / n if n > 0 else 0.0,
                    "max_abs": w.abs().max().item(),
                    "min_abs": w.abs().min().item(),
                }
            )
    return pd.DataFrame(rows)


def compute_channel_health(model: nn.Module) -> pd.DataFrame:
    """Check per-channel health of conv layers.

    A channel is 'dead' if all its weights have abs < 1e-6.
    """
    rows = []
    for layer_name in LAYER_NAMES:
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            continue
        for pname, param in module.named_parameters():
            if "weight" not in pname or param.dim() < 4:
                continue
            w = param.detach().float().cpu()
            # Conv weight shape: [out_ch, in_ch, *kernel]
            out_channels = w.shape[0]
            for ch in range(out_channels):
                ch_w = w[ch]
                rows.append(
                    {
                        "layer": layer_name,
                        "param": pname,
                        "channel": ch,
                        "mean": ch_w.mean().item(),
                        "std": ch_w.std().item(),
                        "abs_mean": ch_w.abs().mean().item(),
                        "is_dead": bool(ch_w.abs().max().item() < 1e-6),
                        "max_abs": ch_w.abs().max().item(),
                    }
                )
    return pd.DataFrame(rows)


def run_weight_analysis(checkpoints: Dict, out_dir: Path) -> Dict:
    """Run Phase 1A: weight statistics across all checkpoints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    all_channel_health = {}

    for ckpt_key, ckpt_info in checkpoints.items():
        print(f"\n{'=' * 60}")
        print(f"Loading: {ckpt_info['label']}")
        print(f"  Path: {ckpt_info['path']}")
        print(f"{'=' * 60}")

        model = load_dynunet(ckpt_info["path"], device="cpu")
        stats_df = compute_weight_stats(model)
        ch_df = compute_channel_health(model)

        all_stats[ckpt_key] = stats_df
        all_channel_health[ckpt_key] = ch_df

        # Save per-checkpoint CSV
        stats_df.to_csv(out_dir / f"weight_stats_{ckpt_key}.csv", index=False)
        ch_df.to_csv(out_dir / f"channel_health_{ckpt_key}.csv", index=False)

        # Print summary
        print(f"\n  Layer-level summary:")
        for layer in LAYER_NAMES:
            layer_params = stats_df[stats_df["layer"] == layer]
            if layer_params.empty:
                continue
            conv_params = layer_params[layer_params["param"].str.contains("conv")]
            if conv_params.empty:
                continue
            total_pnz = conv_params["pct_near_zero"].mean()
            max_std = conv_params["std"].max()
            print(
                f"    {layer:20s}  avg_pct_near_zero={total_pnz:6.1f}%  "
                f"max_std={max_std:.6f}"
            )

        # Channel health summary
        dead_by_layer = (
            ch_df.groupby("layer")["is_dead"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "dead", "count": "total"})
        )
        dead_by_layer["pct_dead"] = (
            100.0 * dead_by_layer["dead"] / dead_by_layer["total"]
        )
        print(f"\n  Dead channels per layer:")
        for layer in LAYER_NAMES:
            if layer in dead_by_layer.index:
                row = dead_by_layer.loc[layer]
                if row["dead"] > 0:
                    print(
                        f"    {layer:20s}  {int(row['dead'])}/{int(row['total'])} "
                        f"({row['pct_dead']:.1f}%) DEAD"
                    )

        del model  # free memory

    # Build cross-checkpoint comparison
    _plot_weight_comparison(all_stats, all_channel_health, checkpoints, out_dir)

    return {"weight_stats": all_stats, "channel_health": all_channel_health}


def _plot_weight_comparison(
    all_stats: Dict,
    all_channel_health: Dict,
    checkpoints: Dict,
    out_dir: Path,
):
    """Generate comparison figures for weight analysis."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Weight std per layer per model (heatmap) ---
    ckpt_keys = list(checkpoints.keys())
    labels = [checkpoints[k]["label"] for k in ckpt_keys]

    # For conv weights, compute mean std per layer
    data_std = np.zeros((len(ckpt_keys), len(LAYER_NAMES)))
    data_pnz = np.zeros((len(ckpt_keys), len(LAYER_NAMES)))
    data_dead_pct = np.zeros((len(ckpt_keys), len(LAYER_NAMES)))

    for i, ck in enumerate(ckpt_keys):
        stats_df = all_stats[ck]
        ch_df = all_channel_health[ck]
        for j, layer in enumerate(LAYER_NAMES):
            lp = stats_df[
                (stats_df["layer"] == layer)
                & (stats_df["param"].str.contains("conv"))
                & (stats_df["param"].str.contains("weight"))
            ]
            if not lp.empty:
                data_std[i, j] = lp["std"].mean()
                data_pnz[i, j] = lp["pct_near_zero"].mean()

            lc = ch_df[ch_df["layer"] == layer]
            if not lc.empty:
                data_dead_pct[i, j] = 100.0 * lc["is_dead"].sum() / len(lc)

    # Heatmap: weight std
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for ax, data, title, cmap, fmt in [
        (axes[0], data_std, "Conv Weight Std by Layer", "YlOrRd", ".4f"),
        (axes[1], data_pnz, "% Near-Zero Conv Weights by Layer", "YlOrRd", ".1f"),
        (axes[2], data_dead_pct, "% Dead Channels by Layer", "Reds", ".1f"),
    ]:
        sns.heatmap(
            data,
            ax=ax,
            xticklabels=LAYER_SHORT,
            yticklabels=labels,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(fig_dir / "weight_stats_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'weight_stats_heatmap.png'}")

    # --- Figure 2: Per-layer weight distribution (violin/box) ---
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()

    for j, (layer, short) in enumerate(zip(LAYER_NAMES, LAYER_SHORT)):
        ax = axes[j]
        plot_data = []
        for ck in ckpt_keys:
            stats_df = all_stats[ck]
            lp = stats_df[
                (stats_df["layer"] == layer)
                & (stats_df["param"].str.contains("conv"))
                & (stats_df["param"].str.contains("weight"))
            ]
            if not lp.empty:
                plot_data.append(
                    {
                        "model": checkpoints[ck]["label"],
                        "std": lp["std"].mean(),
                        "abs_mean": lp["abs_mean"].mean(),
                    }
                )

        if plot_data:
            pdf = pd.DataFrame(plot_data)
            colors = [checkpoints[ck]["color"] for ck in ckpt_keys[: len(pdf)]]
            ax.bar(range(len(pdf)), pdf["std"], color=colors, alpha=0.8)
            ax.set_xticks(range(len(pdf)))
            ax.set_xticklabels(
                [p["model"] for p in plot_data], rotation=45, ha="right", fontsize=7
            )
        ax.set_title(short, fontweight="bold")
        ax.set_ylabel("Weight Std" if j % 5 == 0 else "")

    plt.suptitle(
        "Conv Weight Std per Layer Across Models",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "weight_std_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'weight_std_bars.png'}")

    # --- Figure 3: Dead channel comparison bar chart ---
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_width = 0.2
    x = np.arange(len(LAYER_NAMES))

    for i, ck in enumerate(ckpt_keys):
        ax.bar(
            x + i * bar_width,
            data_dead_pct[i],
            bar_width,
            label=checkpoints[ck]["label"],
            color=checkpoints[ck]["color"],
            alpha=0.85,
        )

    ax.set_xticks(x + bar_width * (len(ckpt_keys) - 1) / 2)
    ax.set_xticklabels(LAYER_SHORT)
    ax.set_ylabel("% Dead Channels")
    ax.set_title("Dead Channels per Layer Across Models", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(data_dead_pct.max() * 1.2, 5))

    plt.tight_layout()
    fig.savefig(fig_dir / "dead_channels_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'dead_channels_comparison.png'}")


# ---------------------------------------------------------------------------
# Phase 1B: Activation Statistics
# ---------------------------------------------------------------------------


class ActivationExtractor:
    """Hook-based activation extractor for DynUNet layers."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks = []
        modules = dict(model.named_modules())
        for name in layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
            else:
                print(f"  WARNING: layer '{name}' not found in model")

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            self.activations[name] = output.detach()

        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def load_test_images(
    csv_path: str,
    n_images: int = 5,
    patch_shape: Tuple[int, int, int] = (32, 128, 128),
    channel_name: str = "EGFP",
) -> List[Tuple[str, torch.Tensor]]:
    """Load test images as patches for activation analysis.

    The raw LMNB1 images are multi-channel (Z, C, Y, X) with shape ~(70, 7, 624, 924).
    We select the EGFP channel, apply z-score normalization (matching training pipeline),
    and extract a center patch of the specified size.

    Returns list of (sample_id, tensor) where tensor is [1, 1, Z, Y, X].
    """
    import tifffile

    df = pd.read_csv(csv_path)
    raw_col = [c for c in df.columns if c.lower() in ("raw", "source", "input")][0]
    samples = df.head(n_images)

    # Determine EGFP channel index using bioio
    egfp_channel_idx = None
    first_path = samples.iloc[0][raw_col]
    try:
        from bioio import BioImage

        bio_img = BioImage(first_path)
        if bio_img.channel_names:
            for idx, name in enumerate(bio_img.channel_names):
                if channel_name.lower() in name.lower():
                    egfp_channel_idx = idx
                    print(f"  Found channel '{name}' at index {idx}")
                    break
        if egfp_channel_idx is None:
            print(f"  Channel '{channel_name}' not found in {bio_img.channel_names}")
            print("  Falling back to channel index 3 (typical EGFP position)")
            egfp_channel_idx = 3
    except ImportError:
        print("  bioio not available, using channel index 3 for EGFP")
        egfp_channel_idx = 3

    images = []
    for _, row in samples.iterrows():
        path = row[raw_col]
        sample_id = Path(path).stem.split(".")[0]
        print(f"  Loading: {sample_id}")
        full_img = tifffile.imread(path)

        # Select EGFP channel from multi-channel volume
        if full_img.ndim == 4:
            # Determine channel axis: smaller dim is likely channels
            if full_img.shape[1] < full_img.shape[0]:
                # (Z, C, Y, X) format
                img = full_img[:, egfp_channel_idx, :, :].astype(np.float32)
            else:
                # (C, Z, Y, X) format
                img = full_img[egfp_channel_idx, :, :, :].astype(np.float32)
        elif full_img.ndim == 3:
            img = full_img.astype(np.float32)
        else:
            print(f"  WARNING: unexpected shape {full_img.shape}, skipping")
            continue

        # Z-score normalization (matches NormalizeIntensityd with channel_wise=True)
        mean_val = img.mean()
        std_val = img.std()
        if std_val > 0:
            img = (img - mean_val) / std_val

        # Extract center patch matching training patch_shape
        z, y, x = img.shape
        pz, py, px = patch_shape
        z0 = max(0, (z - pz) // 2)
        y0 = max(0, (y - py) // 2)
        x0 = max(0, (x - px) // 2)
        patch = img[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]

        # Pad if image is smaller than patch
        if patch.shape != patch_shape:
            padded = np.zeros(patch_shape, dtype=np.float32)
            sz, sy, sx = patch.shape
            padded[:sz, :sy, :sx] = patch
            patch = padded

        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]
        print(
            f"    Shape: {list(tensor.shape)}, "
            f"mean={patch.mean():.3f}, std={patch.std():.3f}"
        )
        images.append((sample_id, tensor))

    return images


def compute_activation_stats(
    activations: Dict[str, torch.Tensor],
) -> pd.DataFrame:
    """Compute statistics from a single forward pass's activations."""
    rows = []
    for layer_name, act in activations.items():
        # act shape: [B, C, *spatial]
        a = act.float()
        n_channels = a.shape[1]

        # Global stats
        rows.append(
            {
                "layer": layer_name,
                "stat_type": "global",
                "channel": -1,
                "mean": a.mean().item(),
                "std": a.std().item(),
                "abs_mean": a.abs().mean().item(),
                "pct_near_zero": 100.0 * (a.abs() < 1e-6).float().mean().item(),
                "max_abs": a.abs().max().item(),
                "min_val": a.min().item(),
                "max_val": a.max().item(),
                "spatial_shape": str(list(a.shape[2:])),
                "n_channels": n_channels,
            }
        )

        # Per-channel stats
        for ch in range(n_channels):
            ch_act = a[:, ch]
            rows.append(
                {
                    "layer": layer_name,
                    "stat_type": "channel",
                    "channel": ch,
                    "mean": ch_act.mean().item(),
                    "std": ch_act.std().item(),
                    "abs_mean": ch_act.abs().mean().item(),
                    "pct_near_zero": 100.0
                    * (ch_act.abs() < 1e-6).float().mean().item(),
                    "max_abs": ch_act.abs().max().item(),
                    "min_val": ch_act.min().item(),
                    "max_val": ch_act.max().item(),
                    "spatial_shape": str(list(ch_act.shape[1:])),
                    "n_channels": n_channels,
                }
            )

    return pd.DataFrame(rows)


def run_activation_analysis(
    checkpoints: Dict,
    out_dir: Path,
    n_images: int = 5,
    device: str = "cuda",
) -> Dict:
    """Run Phase 1B: activation statistics using forward hooks."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test images (once, shared across models)
    print("\nLoading test images...")
    images = load_test_images(TEST_CSV, n_images=n_images)
    print(f"Loaded {len(images)} images")

    # Hook layers (skip output_block - it's just 1 channel)
    hook_layers = LAYER_NAMES[:-1]  # all except output_block
    all_act_stats = {}

    for ckpt_key, ckpt_info in checkpoints.items():
        print(f"\n{'=' * 60}")
        print(f"Activation analysis: {ckpt_info['label']}")
        print(f"{'=' * 60}")

        model = load_dynunet(ckpt_info["path"], device=device)
        extractor = ActivationExtractor(model, hook_layers)

        image_stats = []
        for sample_id, img_tensor in images:
            print(f"  Forward pass: {sample_id} {list(img_tensor.shape)}")
            extractor.clear()
            with torch.no_grad():
                img_gpu = img_tensor.to(device)
                _ = model(img_gpu)
                del img_gpu

            stats_df = compute_activation_stats(extractor.activations)
            stats_df["sample_id"] = sample_id
            image_stats.append(stats_df)

        extractor.remove_hooks()

        # Combine stats across images
        combined = pd.concat(image_stats, ignore_index=True)
        all_act_stats[ckpt_key] = combined
        combined.to_csv(out_dir / f"activation_stats_{ckpt_key}.csv", index=False)

        # Print global summary
        global_stats = combined[combined["stat_type"] == "global"]
        avg_by_layer = global_stats.groupby("layer")[
            ["mean", "std", "abs_mean", "pct_near_zero"]
        ].mean()

        print(f"\n  Average activation stats across {len(images)} images:")
        for layer in hook_layers:
            if layer in avg_by_layer.index:
                r = avg_by_layer.loc[layer]
                status = "DEAD" if r["pct_near_zero"] > 99 else "OK"
                print(
                    f"    {layer:20s}  mean={r['mean']:+.4f}  "
                    f"std={r['std']:.4f}  "
                    f"pnz={r['pct_near_zero']:.1f}%  [{status}]"
                )

        del model
        torch.cuda.empty_cache()

    # Plot activation comparison
    _plot_activation_comparison(all_act_stats, checkpoints, out_dir)

    return {"activation_stats": all_act_stats}


def _plot_activation_comparison(
    all_act_stats: Dict,
    checkpoints: Dict,
    out_dir: Path,
):
    """Generate activation comparison figures."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_keys = list(checkpoints.keys())
    hook_layers = LAYER_NAMES[:-1]

    # --- Figure: Activation std heatmap ---
    data_act_std = np.zeros((len(ckpt_keys), len(hook_layers)))
    data_act_pnz = np.zeros((len(ckpt_keys), len(hook_layers)))

    for i, ck in enumerate(ckpt_keys):
        global_stats = all_act_stats[ck][all_act_stats[ck]["stat_type"] == "global"]
        avg = global_stats.groupby("layer")[["std", "pct_near_zero"]].mean()
        for j, layer in enumerate(hook_layers):
            if layer in avg.index:
                data_act_std[i, j] = avg.loc[layer, "std"]
                data_act_pnz[i, j] = avg.loc[layer, "pct_near_zero"]

    labels = [checkpoints[k]["label"] for k in ckpt_keys]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    sns.heatmap(
        data_act_std,
        ax=axes[0],
        xticklabels=LAYER_SHORT[:-1],
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
    )
    axes[0].set_title(
        "Activation Std by Layer (avg over test images)", fontsize=13, fontweight="bold"
    )

    sns.heatmap(
        data_act_pnz,
        ax=axes[1],
        xticklabels=LAYER_SHORT[:-1],
        yticklabels=labels,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
    )
    axes[1].set_title(
        "% Near-Zero Activations by Layer (avg over test images)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(fig_dir / "activation_stats_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'activation_stats_heatmap.png'}")

    # --- Figure: Per-channel activity profile per model ---
    # For bottleneck and encoder layers, show channel std distribution
    focus_layers = ["downsamples.1", "downsamples.2", "bottleneck", "upsamples.0"]
    focus_short = ["enc1", "enc2", "bottleneck", "dec0"]

    fig, axes = plt.subplots(
        len(focus_layers),
        len(ckpt_keys),
        figsize=(5 * len(ckpt_keys), 4 * len(focus_layers)),
    )
    if len(focus_layers) == 1:
        axes = axes[np.newaxis, :]

    for j_layer, (layer, short) in enumerate(zip(focus_layers, focus_short)):
        for i_model, ck in enumerate(ckpt_keys):
            ax = axes[j_layer, i_model]
            ch_stats = all_act_stats[ck][
                (all_act_stats[ck]["stat_type"] == "channel")
                & (all_act_stats[ck]["layer"] == layer)
            ]
            if ch_stats.empty:
                ax.set_visible(False)
                continue

            # Average std across images per channel
            avg_ch = ch_stats.groupby("channel")["std"].mean().sort_index()
            colors_arr = [
                "#e74c3c" if v < 1e-4 else checkpoints[ck]["color"]
                for v in avg_ch.values
            ]
            ax.bar(avg_ch.index, avg_ch.values, color=colors_arr, width=1.0, alpha=0.8)
            ax.set_xlabel("Channel")
            ax.set_ylabel("Activation Std")
            ax.set_title(f"{short} - {checkpoints[ck]['label']}", fontsize=10)
            # Mark dead channels
            n_dead = (avg_ch.values < 1e-4).sum()
            if n_dead > 0:
                ax.text(
                    0.02,
                    0.95,
                    f"{n_dead} dead ch",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="red",
                    va="top",
                )

    plt.suptitle(
        "Per-Channel Activation Std (red = dead)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "channel_activity_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'channel_activity_profiles.png'}")


# ---------------------------------------------------------------------------
# Phase 2: Feature Space Visualization
# ---------------------------------------------------------------------------


def extract_layer_features(
    model: nn.Module,
    images: List[Tuple[str, torch.Tensor]],
    layer_names: List[str],
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract spatially-averaged features at specified layers.

    Returns dict of layer_name -> array of shape [n_images, n_channels].
    """
    extractor = ActivationExtractor(model, layer_names)
    features = {name: [] for name in layer_names}

    for sample_id, img_tensor in images:
        extractor.clear()
        with torch.no_grad():
            img_gpu = img_tensor.to(device)
            _ = model(img_gpu)
            del img_gpu

        for name in layer_names:
            if name in extractor.activations:
                act = extractor.activations[name].float()
                # Global average pooling: [B, C, Z, Y, X] -> [B, C]
                feat = act.mean(dim=tuple(range(2, act.dim()))).cpu().numpy()
                features[name].append(feat[0])  # remove batch dim

    extractor.remove_hooks()

    return {name: np.array(feats) for name, feats in features.items() if feats}


def run_feature_visualization(
    checkpoints: Dict,
    out_dir: Path,
    n_images: int = 20,
    device: str = "cuda",
):
    """Run Phase 2: t-SNE and PCA visualization of feature spaces."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load more images for better visualization
    print("\nLoading test images for feature visualization...")
    images = load_test_images(TEST_CSV, n_images=n_images)
    print(f"Loaded {len(images)} images")

    # Focus on intermediate encoder + bottleneck layers
    viz_layers = ["downsamples.1", "downsamples.2", "bottleneck"]
    viz_short = ["enc1 (128ch)", "enc2 (256ch)", "bottleneck (320ch)"]

    # Extract features for each model
    all_features = {}
    for ckpt_key, ckpt_info in checkpoints.items():
        print(f"\nExtracting features: {ckpt_info['label']}")
        model = load_dynunet(ckpt_info["path"], device=device)
        features = extract_layer_features(model, images, viz_layers, device=device)
        all_features[ckpt_key] = features
        del model
        torch.cuda.empty_cache()

    ckpt_keys = list(checkpoints.keys())

    # --- Figure: t-SNE per layer ---
    fig, axes = plt.subplots(1, len(viz_layers), figsize=(7 * len(viz_layers), 6))
    if len(viz_layers) == 1:
        axes = [axes]

    for j, (layer, short) in enumerate(zip(viz_layers, viz_short)):
        ax = axes[j]
        combined = []
        model_labels = []
        for ck in ckpt_keys:
            feats = all_features[ck].get(layer)
            if feats is not None:
                combined.append(feats)
                model_labels.extend([checkpoints[ck]["label"]] * len(feats))

        if not combined:
            continue

        X = np.concatenate(combined, axis=0)

        # Remove zero-variance features
        var = X.var(axis=0)
        keep = var > 1e-10
        X_clean = X[:, keep]
        n_dropped = (~keep).sum()

        if X_clean.shape[1] < 2:
            ax.text(
                0.5,
                0.5,
                f"Only {X_clean.shape[1]} non-zero channels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(short)
            continue

        # PCA first to reduce dimensionality if needed
        n_components = min(50, X_clean.shape[1], X_clean.shape[0] - 1)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_clean)

        perplexity = min(30, len(X_pca) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X_pca)

        # Plot colored by model
        for ck in ckpt_keys:
            mask = [l == checkpoints[ck]["label"] for l in model_labels]
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=checkpoints[ck]["color"],
                label=f"{checkpoints[ck]['label']} (Dice={checkpoints[ck]['dice']:.3f})",
                alpha=0.7,
                s=40,
                edgecolors="white",
                linewidths=0.5,
            )

        title = short
        if n_dropped > 0:
            title += f"\n({n_dropped} zero-var channels dropped)"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if j == len(viz_layers) - 1:
            ax.legend(fontsize=8, loc="best")

    plt.suptitle(
        "t-SNE of Layer Features Across Models", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "tsne_features_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'tsne_features_by_model.png'}")

    # --- Figure: PCA variance explained per layer per model ---
    fig, axes = plt.subplots(1, len(viz_layers), figsize=(6 * len(viz_layers), 5))
    if len(viz_layers) == 1:
        axes = [axes]

    for j, (layer, short) in enumerate(zip(viz_layers, viz_short)):
        ax = axes[j]
        for ck in ckpt_keys:
            feats = all_features[ck].get(layer)
            if feats is None or feats.shape[0] < 3:
                continue
            # Remove zero-var
            var = feats.var(axis=0)
            feats_clean = feats[:, var > 1e-10]
            if feats_clean.shape[1] < 2:
                continue

            n_comp = min(feats_clean.shape[0] - 1, feats_clean.shape[1], 50)
            pca = PCA(n_components=n_comp)
            pca.fit(feats_clean)
            cumvar = np.cumsum(pca.explained_variance_ratio_)

            ax.plot(
                range(1, len(cumvar) + 1),
                cumvar,
                color=checkpoints[ck]["color"],
                label=checkpoints[ck]["label"],
                linewidth=2,
            )

        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance Explained")
        ax.set_title(short, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.suptitle(
        "PCA Variance Explained by Layer", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "pca_variance_explained.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'pca_variance_explained.png'}")

    # --- Figure: Effective dimensionality comparison ---
    # Count components needed for 95% variance
    dim_data = []
    for ck in ckpt_keys:
        for layer, short in zip(viz_layers, viz_short):
            feats = all_features[ck].get(layer)
            if feats is None or feats.shape[0] < 3:
                continue
            var = feats.var(axis=0)
            feats_clean = feats[:, var > 1e-10]
            if feats_clean.shape[1] < 2:
                dim_data.append(
                    {
                        "model": checkpoints[ck]["label"],
                        "layer": short,
                        "n_active_channels": int((var > 1e-10).sum()),
                        "total_channels": feats.shape[1],
                        "dim_95pct": 0,
                    }
                )
                continue

            n_comp = min(feats_clean.shape[0] - 1, feats_clean.shape[1], 50)
            pca = PCA(n_components=n_comp)
            pca.fit(feats_clean)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim_95 = np.searchsorted(cumvar, 0.95) + 1

            dim_data.append(
                {
                    "model": checkpoints[ck]["label"],
                    "layer": short,
                    "n_active_channels": int((var > 1e-10).sum()),
                    "total_channels": feats.shape[1],
                    "dim_95pct": int(dim_95),
                }
            )

    if dim_data:
        dim_df = pd.DataFrame(dim_data)
        dim_df.to_csv(out_dir / "effective_dimensionality.csv", index=False)
        print(f"Saved: {out_dir / 'effective_dimensionality.csv'}")

        # Print summary table
        print("\nEffective Dimensionality (components for 95% variance):")
        pivot = dim_df.pivot(index="model", columns="layer", values="dim_95pct")
        print(pivot.to_string())

        # Also print active channels
        print("\nActive Channels (non-zero variance):")
        pivot_ch = dim_df.pivot(
            index="model", columns="layer", values="n_active_channels"
        )
        print(pivot_ch.to_string())

    return {"features": all_features}


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def write_summary(out_dir: Path, checkpoints: Dict):
    """Write a text summary of all analysis results."""
    summary_path = out_dir / "analysis_summary.txt"
    lines = ["Feature Space Analysis Summary", "=" * 50, ""]

    # List models
    lines.append("Models analyzed:")
    for ck, info in checkpoints.items():
        lines.append(f"  {info['label']:30s}  Dice={info['dice']:.3f}")
    lines.append("")

    # Weight stats summary
    for ck, info in checkpoints.items():
        csv_path = out_dir / f"weight_stats_{ck}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            lines.append(f"\nWeight Stats: {info['label']}")
            lines.append("-" * 40)
            for layer in LAYER_NAMES:
                lp = df[
                    (df["layer"] == layer)
                    & (df["param"].str.contains("conv"))
                    & (df["param"].str.contains("weight"))
                ]
                if not lp.empty:
                    lines.append(
                        f"  {layer:20s}  std={lp['std'].mean():.6f}  "
                        f"pnz={lp['pct_near_zero'].mean():.1f}%"
                    )

    # Activation stats summary
    for ck, info in checkpoints.items():
        csv_path = out_dir / f"activation_stats_{ck}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            global_stats = df[df["stat_type"] == "global"]
            avg = global_stats.groupby("layer")[["std", "pct_near_zero"]].mean()
            lines.append(f"\nActivation Stats: {info['label']}")
            lines.append("-" * 40)
            for layer in LAYER_NAMES[:-1]:
                if layer in avg.index:
                    r = avg.loc[layer]
                    status = "DEAD" if r["pct_near_zero"] > 99 else "OK"
                    lines.append(
                        f"  {layer:20s}  std={r['std']:.4f}  "
                        f"pnz={r['pct_near_zero']:.1f}%  [{status}]"
                    )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved summary: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Feature space analysis for MegaSeg DynUNet models"
    )
    parser.add_argument(
        "--phase",
        choices=["weights", "activations", "visualize", "all"],
        default="all",
        help="Which analysis phase to run",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Path to JSON file with custom checkpoint definitions",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of test images for activation analysis (default: 5)",
    )
    parser.add_argument(
        "--n-viz-images",
        type=int,
        default=20,
        help="Number of test images for feature visualization (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (default: cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/feature_analysis)",
    )
    args = parser.parse_args()

    # Resolve checkpoints
    if args.checkpoints:
        with open(args.checkpoints) as f:
            checkpoints = json.load(f)
    else:
        checkpoints = DEFAULT_CHECKPOINTS

    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate checkpoint paths
    print("Validating checkpoint paths...")
    for ck, info in checkpoints.items():
        p = Path(info["path"])
        if not p.exists():
            print(f"  ERROR: checkpoint not found: {p}")
            sys.exit(1)
        print(f"  OK: {info['label']}")

    # Run phases
    if args.phase in ("weights", "all"):
        print("\n" + "=" * 70)
        print("PHASE 1A: WEIGHT STATISTICS")
        print("=" * 70)
        run_weight_analysis(checkpoints, out_dir)

    if args.phase in ("activations", "all"):
        print("\n" + "=" * 70)
        print("PHASE 1B: ACTIVATION STATISTICS")
        print("=" * 70)
        run_activation_analysis(
            checkpoints, out_dir, n_images=args.n_images, device=args.device
        )

    if args.phase in ("visualize", "all"):
        print("\n" + "=" * 70)
        print("PHASE 2: FEATURE SPACE VISUALIZATION")
        print("=" * 70)
        run_feature_visualization(
            checkpoints, out_dir, n_images=args.n_viz_images, device=args.device
        )

    # Write summary
    write_summary(out_dir, checkpoints)

    print("\n" + "=" * 70)
    print(f"All results saved to: {out_dir}")
    print(f"Figures saved to: {out_dir / 'figures'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
