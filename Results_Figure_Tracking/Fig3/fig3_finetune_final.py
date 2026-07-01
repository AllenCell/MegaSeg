"""
Fig 3 — Fine-tuning & from-scratch comparison for LMNB1 and LAMP1.

Generates:
  1. fig3_heatmaps.pdf — 2×2 Dice heatmap (LMNB1/LAMP1 × finetune/scratch)
  2. fig3_qualitative.pdf — MIP-based qualitative comparison with MegaSeg
     generalist prediction column

Reads 57-sample summary CSVs from benchmark/results/.

Usage:
    python fig3_finetune_final.py              # both figures
    python fig3_finetune_final.py --only heatmap
    python fig3_finetune_final.py --only qualitative
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════
BENCHMARK = Path("/allen/aics/rep_learn/megaseg/benchmark")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # megaseg/ or worktree root
RESULTS = REPO_ROOT / "benchmark" / "results"
DATA_ROOT = Path("/allen/aics/rep_learn/megaseg/data/variance-100x")
FIGURE_DIR = Path(__file__).parent

# Fine-tune prediction roots
LMNB1_PRED_ROOT = BENCHMARK / "outputs" / "eval" / "finetune" / "lmnb1"
LAMP1_PRED_ROOT = BENCHMARK / "outputs" / "eval" / "finetune" / "lamp1"

DATA_FRACTIONS = [100, 50, 20, 10]
EPOCH_CONFIGS = ["10epochs", "50epochs", "100epochs", "converged"]
EPOCH_DISPLAY = ["10 ep", "50 ep", "100 ep", "Conv."]
FRAC_DISPLAY = ["100%", "50%", "20%", "10%"]

# ══════════════════════════════════════════════════════════════
#  HEATMAPS — 2×2 panel
# ══════════════════════════════════════════════════════════════

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)


def load_summary_grid(summary_csv):
    """Load a 4×4 grid + base from a summary CSV."""
    df = pd.read_csv(summary_csv)
    grid = np.full((len(DATA_FRACTIONS), len(EPOCH_CONFIGS)), np.nan)
    std_grid = np.full_like(grid, np.nan)

    for i, frac in enumerate(DATA_FRACTIONS):
        for j, ep in enumerate(EPOCH_CONFIGS):
            row = df[df["experiment"] == f"data{frac}_{ep}"]
            if not row.empty:
                grid[i, j] = row["dice_mean"].iloc[0]
                std_grid[i, j] = row["dice_std"].iloc[0]

    base = df[df["experiment"] == "base_megaseg"]
    base_mean = base["dice_mean"].iloc[0] if not base.empty else None
    base_std = base["dice_std"].iloc[0] if not base.empty else None
    n = int(df["n_samples"].mode().iloc[0]) if "n_samples" in df.columns else 0

    return grid, std_grid, base_mean, base_std, n


def plot_single_heatmap(ax, grid, std_grid, base_mean, base_std, title, n,
                        show_ylabel=True, show_xlabel=True):
    """Plot one 4×4 heatmap on the given axes."""
    vmin = min(np.nanmin(grid), base_mean if base_mean else 1.0) - 0.02
    vmax = np.nanmax(grid) + 0.02
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax.imshow(grid, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    for i in range(4):
        for j in range(4):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="gray")
                continue
            std = std_grid[i, j]
            rgba = HEATMAP_CMAP(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if brightness < 0.55 else "black"
            ax.text(j, i, f"{val:.3f}\n±{std:.3f}",
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(EPOCH_DISPLAY if show_xlabel else [], fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(FRAC_DISPLAY if show_ylabel else [], fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Training Epochs", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Data Fraction", fontsize=10)

    subtitle = f"(n={n})" if n else ""
    ax.set_title(f"{title} {subtitle}", fontsize=11, fontweight="bold", pad=8)

    if base_mean is not None:
        ax.text(0.5, -0.15,
                f"Base MegaSeg (no finetuning): {base_mean:.3f} ± {base_std:.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#2C2C2C",
                          edgecolor="black", linewidth=1.5),
                color="#FFDD44")

    return norm


def build_heatmaps(out_path):
    """Build 2×2 heatmap figure."""
    panels = [
        ("LMNB1 Fine-Tuning", RESULTS / "finetune" / "lmnb1" / "summary.csv"),
        ("LMNB1 From-Scratch", RESULTS / "scratch" / "lmnb1" / "summary.csv"),
        ("LAMP1 Fine-Tuning", RESULTS / "finetune" / "lamp1" / "summary.csv"),
        ("LAMP1 From-Scratch", RESULTS / "scratch" / "lamp1" / "summary.csv"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    for idx, (title, csv_path) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        if not csv_path.exists():
            ax.text(0.5, 0.5, f"{title}\n(data not available)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue
        grid, std_grid, base_mean, base_std, n = load_summary_grid(csv_path)
        plot_single_heatmap(
            ax, grid, std_grid, base_mean, base_std, title, n,
            show_ylabel=(idx % 2 == 0),
            show_xlabel=(idx // 2 == 1),
        )

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  QUALITATIVE — MIP-based figure with MegaSeg generalist column
# ══════════════════════════════════════════════════════════════
CROP_SIZE = 320
PIXEL_SIZE_UM = 0.10833
SCALEBAR_UM = 10

# LMNB1 samples — largest base→converged improvement
LMNB1_STEMS = [
    "3500000896_100X_20170515_F05_P59",
    "3500000896_100X_20170515_E05_P17",
]
# LAMP1 samples — largest base→converged improvement
LAMP1_STEMS = [
    "LAMP1_150841",
    "LAMP1_150666",
]

EPOCH_CONDS = [
    "data100_10epochs",
    "data100_50epochs",
    "data100_100epochs",
    "data100_converged",
]
EPOCH_LABELS = ["10 ep", "50 ep", "100 ep", "Conv."]


def load_lmnb1_raw(stem):
    path = DATA_ROOT / "LMNB1" / "raw" / f"{stem}.ome.tiff"
    vol = tif.imread(str(path))
    ch = 3 if vol.shape[1] >= 7 else 1
    return vol[:, ch, :, :].astype(np.float32)


def load_lmnb1_gt(stem):
    path = DATA_ROOT / "LMNB1" / "seg" / f"{stem}_struct_segmentation.tiff"
    return (tif.imread(str(path)) > 0).astype(np.bool_)


def load_lmnb1_pred(condition, stem):
    path = LMNB1_PRED_ROOT / condition / "seg" / f"{stem}.ome.tif"
    vol = tif.imread(str(path))
    if vol.dtype == np.uint8:
        return (vol >= 128).astype(np.bool_)
    return (vol >= 0.5).astype(np.bool_)


def load_lamp1_raw(stem):
    path = DATA_ROOT / "LAMP1" / "dataset" / "test" / f"{stem}.tiff"
    vol = tif.imread(str(path))
    if vol.ndim == 4:
        return vol[:, 0, :, :].astype(np.float32)
    return vol.astype(np.float32)


def load_lamp1_gt(stem):
    path = DATA_ROOT / "LAMP1" / "dataset" / "test" / f"{stem}_seg.tiff"
    return (tif.imread(str(path)) > 0).astype(np.bool_)


def load_lamp1_pred(condition, stem):
    path = LAMP1_PRED_ROOT / condition / "seg" / f"{stem}.tif"
    if not path.exists():
        path = LAMP1_PRED_ROOT / condition / "seg" / f"{stem}.ome.tif"
    vol = tif.imread(str(path))
    if vol.dtype == np.uint8:
        return (vol >= 128).astype(np.bool_)
    return (vol >= 0.5).astype(np.bool_)


def mip(volume):
    return np.max(volume, axis=0)


def normalize_percentile(img, p_lo=1, p_hi=95):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return np.clip(img, 0, 1)


def smart_crop(shape_2d, crop_size, ref_mask=None):
    h, w = shape_2d
    crop_size = min(crop_size, h, w)
    if ref_mask is not None and ref_mask.sum() > 0:
        ys, xs = np.where(ref_mask > 0)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
    else:
        cy, cx = h // 2, w // 2
    y0 = max(0, min(cy - crop_size // 2, h - crop_size))
    x0 = max(0, min(cx - crop_size // 2, w - crop_size))
    return y0, x0, crop_size


def make_error_map(gt_mask, pred_mask):
    h, w = gt_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[gt_mask & pred_mask] = [0.85, 0.75, 0.25]
    rgb[~gt_mask & pred_mask] = [0.82, 0.35, 0.45]
    rgb[gt_mask & ~pred_mask] = [0.30, 0.55, 0.75]
    rgb[~gt_mask & ~pred_mask] = [0.12, 0.12, 0.12]
    return rgb


def add_scalebar(ax, crop_pixels):
    bar_px = SCALEBAR_UM / PIXEL_SIZE_UM
    margin = 8
    bar_y = crop_pixels - margin
    bar_x_end = crop_pixels - margin
    bar_x_start = bar_x_end - bar_px
    ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
            color="white", linewidth=2, solid_capstyle="butt")
    ax.text((bar_x_start + bar_x_end) / 2, bar_y - 5,
            f"{SCALEBAR_UM} µm", color="white", fontsize=5,
            ha="center", va="bottom", fontweight="bold")


def load_row_data(stem, conditions, load_raw_fn, load_gt_fn, load_pred_fn):
    """Load one row: raw, GT, MegaSeg generalist pred, then error maps for conditions."""
    raw_vol = load_raw_fn(stem)
    gt_vol = load_gt_fn(stem)

    raw_mip = normalize_percentile(mip(raw_vol))
    gt_mip = mip(gt_vol.astype(np.uint8)) > 0

    y0, x0, cs = smart_crop(gt_mip.shape, CROP_SIZE, ref_mask=gt_mip)

    def crop(img):
        return img[y0:y0 + cs, x0:x0 + cs]

    raw_crop = crop(raw_mip)
    gt_crop = crop(gt_mip)

    # MegaSeg generalist prediction (binary MIP)
    base_vol = load_pred_fn("base_megaseg", stem)
    base_mip = mip(base_vol.astype(np.uint8)) > 0
    base_crop = crop(base_mip).astype(np.float32)

    # Error maps for fine-tuning conditions
    error_maps = []
    for cond in conditions:
        try:
            pred_vol = load_pred_fn(cond, stem)
            pred_mip = mip(pred_vol.astype(np.uint8)) > 0
            error_maps.append(make_error_map(gt_crop, crop(pred_mip)))
        except Exception as e:
            print(f"    WARNING: {cond}/{stem}: {e}")
            error_maps.append(np.zeros((cs, cs, 3), dtype=np.float32))

    return raw_crop, gt_crop, base_crop, error_maps, cs


def build_qualitative(out_path):
    """4 rows × 8 columns: Raw | GT | MegaSeg (generalist) | 4 fine-tuning conditions."""
    # Columns: Raw, GT, MegaSeg, 10ep, 50ep, 100ep, Converged
    col_titles = ["Raw", "GT", "MegaSeg"] + EPOCH_LABELS
    n_rows = 4  # 2 LMNB1 + 2 LAMP1
    n_cols = len(col_titles)

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2 + 1.6), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)

    rows_spec = []
    for stem in LMNB1_STEMS:
        rows_spec.append((stem, EPOCH_CONDS, load_lmnb1_raw, load_lmnb1_gt, load_lmnb1_pred, "LMNB1"))
    for stem in LAMP1_STEMS:
        rows_spec.append((stem, EPOCH_CONDS, load_lamp1_raw, load_lamp1_gt, load_lamp1_pred, "LAMP1"))

    for row_idx, (stem, conditions, load_raw_fn, load_gt_fn, load_pred_fn, struct) in enumerate(rows_spec):
        print(f"  Loading row {row_idx} ({struct}): {stem}...")
        raw_crop, gt_crop, base_crop, error_maps, crop_px = load_row_data(
            stem, conditions, load_raw_fn, load_gt_fn, load_pred_fn)

        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if col_idx == 0:
                ax.imshow(raw_crop, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
                add_scalebar(ax, crop_px)
            elif col_idx == 1:
                ax.imshow(gt_crop.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            elif col_idx == 2:
                ax.imshow(base_crop, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            else:
                ax.imshow(error_maps[col_idx - 3],
                          interpolation="nearest", aspect="equal")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("#555555")

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9, fontweight="bold", pad=6)

    # Panel labels
    fig.text(0.01, 0.78, "LMNB1",
             fontsize=11, fontweight="bold", va="center", ha="left", rotation=90)
    fig.text(0.01, 0.32, "LAMP1",
             fontsize=11, fontweight="bold", va="center", ha="left", rotation=90)

    # Separator line between LMNB1 and LAMP1
    sep_y = (gs[1, 0].get_position(fig).y0 + gs[2, 0].get_position(fig).y1) / 2
    fig.add_artist(plt.Line2D(
        [0.04, 0.98], [sep_y, sep_y],
        transform=fig.transFigure,
        color="gray", linewidth=1, linestyle="--",
    ))

    # Error map legend
    legend_handles = [
        Patch(facecolor=(0.85, 0.75, 0.25), edgecolor="gray", label="True Positive"),
        Patch(facecolor=(0.82, 0.35, 0.45), edgecolor="gray", label="False Positive"),
        Patch(facecolor=(0.30, 0.55, 0.75), edgecolor="gray", label="False Negative"),
        Patch(facecolor=(0.12, 0.12, 0.12), edgecolor="gray", label="True Negative"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, edgecolor="black",
               bbox_to_anchor=(0.52, -0.01))

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Fig 3: fine-tuning & from-scratch")
    parser.add_argument("--only", choices=["heatmap", "qualitative"],
                        help="Generate only one figure")
    args = parser.parse_args()

    if args.only in (None, "heatmap"):
        print("\n[1] Heatmaps (2×2)")
        build_heatmaps(FIGURE_DIR / "fig3_heatmaps.pdf")

    if args.only in (None, "qualitative"):
        print("\n[2] Qualitative (MIP + MegaSeg generalist)")
        build_qualitative(FIGURE_DIR / "fig3_qualitative.pdf")

    print("\nDone!")


if __name__ == "__main__":
    main()
