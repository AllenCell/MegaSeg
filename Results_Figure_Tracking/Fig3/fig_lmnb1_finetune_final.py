"""
LMNB1 Finetuning — Final figures.

Generates both publication figures from a single script:
  1. fig_lmnb1_finetune_qualitative_v3.pdf  (MIP-based qualitative comparison)
  2. fig_lmnb1_finetune_heatmap_v2.pdf      (4×4 Dice heatmap)

The heatmap reads pre-computed metrics from lmnb1_finetune_metrics_v2.csv.
The qualitative figure loads raw/GT/prediction volumes directly.

Usage:
    python fig_lmnb1_finetune_final.py              # both figures
    python fig_lmnb1_finetune_final.py --only heatmap
    python fig_lmnb1_finetune_final.py --only qualitative
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile as tif
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

# Nature Methods-style typography: Arial, embedded TrueType
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5,
})

# ══════════════════════════════════════════════════════════════
# Shared paths & constants
# ══════════════════════════════════════════════════════════════
PRED_ROOT = "//allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/finetune/lmnb1"
GT_DIR = "//allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/seg"
RAW_DIR = "//allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/raw"
FIGURE_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════
#  QUALITATIVE v3 — MIP-based figure
# ══════════════════════════════════════════════════════════════
CROP_SIZE = 320
PIXEL_SIZE_UM = 0.10833  # μm per pixel
SCALEBAR_UM = 10  # 10 μm scale bar

# Pre-selected samples with largest base→converged improvement
PANEL_A_STEMS = [
    # 10ep=0.890 -> conv=0.954; fp_drop=+0.078, fn_drop=+0.049
    # (balanced FP/FN correction — both error types should be visually obvious)
    "3500000956_100X_20170606_1r-Scene-2-P5-E04",
    "3500000896_100X_20170515_E05_P17",                     # base 0.184 → conv 0.958
]
PANEL_B_STEMS = [
    "3500000953_100X_20170607_2c-Scene-3-P18-F07",          # base 0.243, 10%=0.687 → 100%=0.930
    "3500000884_100X_20170509_1-Scene-20-P60-F05",          # base 0.300, 10%=0.678 → 100%=0.921
]

EPOCH_CONDS = [
    "data100_10epochs",
    "data100_50epochs",
    "data100_100epochs",
    "data100_converged",
]
EPOCH_LABELS = ["10 ep", "50 ep", "100 ep", "Converged"]

DATA_CONDS = [
    "data10_10epochs",
    "data20_10epochs",
    "data50_10epochs",
    "data100_10epochs",
]
DATA_LABELS = ["10%", "20%", "50%", "100%"]


# ── I/O helpers ──────────────────────────────────────────────
def load_gt(stem):
    path = os.path.join(GT_DIR, f"{stem}_struct_segmentation.tiff")
    vol = tif.imread(path)
    return (vol > 0).astype(np.bool_)


def load_pred(condition, stem):
    path = os.path.join(PRED_ROOT, condition, "seg", f"{stem}.ome.tif")
    vol = tif.imread(path)
    if vol.dtype == np.uint8:
        return (vol >= 128).astype(np.bool_)
    return (vol >= 0.5).astype(np.bool_)


def load_raw(stem):
    path = os.path.join(RAW_DIR, f"{stem}.ome.tiff")
    vol = tif.imread(path)  # (Z, C, Y, X)
    ch = 3 if vol.shape[1] >= 7 else 1
    return vol[:, ch, :, :].astype(np.float32)


# ── MIP + normalization ─────────────────────────────────────
def mip(volume):
    return np.max(volume, axis=0)


def normalize_percentile(img, p_lo=1, p_hi=95):
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return np.clip(img, 0, 1)


# ── Crop + error map ────────────────────────────────────────
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
    """Muted, publication-friendly error map.

    Muted gold   = True Positive
    Soft rose    = False Positive
    Steel blue   = False Negative
    Dark gray    = True Negative
    """
    h, w = gt_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[gt_mask & pred_mask] = [0.85, 0.75, 0.25]       # TP — muted gold
    rgb[~gt_mask & pred_mask] = [0.82, 0.35, 0.45]      # FP — soft rose
    rgb[gt_mask & ~pred_mask] = [0.30, 0.55, 0.75]      # FN — steel blue
    rgb[~gt_mask & ~pred_mask] = [0.12, 0.12, 0.12]     # TN — dark gray
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
            f"{SCALEBAR_UM} \u00b5m", color="white", fontsize=6,
            ha="center", va="bottom", fontweight="bold")


# ── Load one row of MIP data ────────────────────────────────
def load_row_data(stem, conditions):
    raw_vol = load_raw(stem)
    gt_vol = load_gt(stem)

    raw_mip = normalize_percentile(mip(raw_vol))
    gt_mip = mip(gt_vol.astype(np.uint8)) > 0

    y0, x0, cs = smart_crop(gt_mip.shape, CROP_SIZE, ref_mask=gt_mip)

    def crop(img):
        return img[y0:y0 + cs, x0:x0 + cs]

    raw_crop = crop(raw_mip)
    gt_crop = crop(gt_mip)

    error_maps = []
    for cond in conditions:
        pred_vol = load_pred(cond, stem)
        pred_mip = mip(pred_vol.astype(np.uint8)) > 0
        error_maps.append(make_error_map(gt_crop, crop(pred_mip)))

    return raw_crop, gt_crop, error_maps, cs


# ── Build qualitative figure ────────────────────────────────
def build_qualitative(out_path, show_legend=True):
    """4 rows × 6 columns MIP qualitative figure (Base col removed)."""
    n_rows = 4
    n_cols = 6  # Raw + GT + 4 conditions
    col_titles_a = ["Raw", "GT"] + EPOCH_LABELS
    col_titles_b = ["Raw", "GT"] + DATA_LABELS

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5 + 1.6), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.04)

    panel_rows = []
    for stem in PANEL_A_STEMS:
        panel_rows.append((stem, EPOCH_CONDS, col_titles_a))
    for stem in PANEL_B_STEMS:
        panel_rows.append((stem, DATA_CONDS, col_titles_b))

    for row_idx, (stem, conditions, col_titles) in enumerate(panel_rows):
        print(f"  Loading row {row_idx}: {stem}...")
        raw_crop, gt_crop, error_maps, crop_px = load_row_data(stem, conditions)

        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if col_idx == 0:
                ax.imshow(raw_crop, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
                add_scalebar(ax, crop_px)
            elif col_idx == 1:
                ax.imshow(gt_crop.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            else:
                ax.imshow(error_maps[col_idx - 2],
                          interpolation="nearest", aspect="equal")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("#555555")

            if row_idx == 0:
                ax.set_title(col_titles_a[col_idx], fontsize=10,
                             fontweight="bold", pad=6)
            elif row_idx == 2:
                ax.set_title(col_titles_b[col_idx], fontsize=10,
                             fontweight="bold", pad=6)

    # Panel labels
    fig.text(0.01, 0.75, "A  Epoch effect\n(100% data)",
             fontsize=11, fontweight="bold", va="center", ha="left", rotation=90)
    fig.text(0.01, 0.30, "B  Data effect\n(10 epochs)",
             fontsize=11, fontweight="bold", va="center", ha="left", rotation=90)

    # Separator line
    sep_y = (gs[1, 0].get_position(fig).y0 + gs[2, 0].get_position(fig).y1) / 2
    fig.add_artist(plt.Line2D(
        [0.03, 0.98], [sep_y, sep_y],
        transform=fig.transFigure,
        color="gray", linewidth=1, linestyle="--",
    ))

    if show_legend:
        legend_handles = [
            Patch(facecolor=(0.85, 0.75, 0.25), edgecolor="gray", label="True Positive"),
            Patch(facecolor=(0.82, 0.35, 0.45), edgecolor="gray", label="False Positive"),
            Patch(facecolor=(0.30, 0.55, 0.75), edgecolor="gray", label="False Negative"),
            Patch(facecolor=(0.12, 0.12, 0.12), edgecolor="gray", label="True Negative"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=10, frameon=True, edgecolor="black",
                   bbox_to_anchor=(0.52, -0.01))

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  HEATMAP v2 — 4×4 Dice heatmap from pre-computed CSV
# ══════════════════════════════════════════════════════════════
HEATMAP_DATA_PCTS = [10, 20, 50, 100]
HEATMAP_EPOCH_LABELS = ["10epochs", "50epochs", "100epochs", "converged"]
HEATMAP_EPOCH_DISPLAY = ["10 ep", "50 ep", "100 ep", "Converged"]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)


def load_metrics_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["dice"] = float(r["dice"])
            rows.append(r)
    return rows


def build_heatmap_data(rows):
    """Build 4×4 matrix (data% rows × epoch cols). Also returns base stats."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r["dice"])

    row_labels = ["10%", "20%", "50%", "100%"]
    col_labels = HEATMAP_EPOCH_DISPLAY

    mean_mat = np.zeros((4, 4))
    std_mat = np.zeros((4, 4))

    for i, dp in enumerate(HEATMAP_DATA_PCTS):
        for j, ep in enumerate(HEATMAP_EPOCH_LABELS):
            cond = f"data{dp}_{ep}"
            if cond in groups:
                mean_mat[i, j] = np.mean(groups[cond])
                std_mat[i, j] = np.std(groups[cond])

    base_vals = groups.get("base_megaseg", [])
    base_mean = float(np.mean(base_vals)) if base_vals else 0.0
    base_std = float(np.std(base_vals)) if base_vals else 0.0

    return mean_mat, std_mat, row_labels, col_labels, base_mean, base_std


def build_heatmap(rows, out_path):
    """4×4 heatmap with base MegaSeg reference annotation."""
    mean_mat, std_mat, row_labels, col_labels, base_mean, base_std = \
        build_heatmap_data(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    vmin = min(np.min(mean_mat), base_mean) - 0.02
    vmax = np.max(mean_mat) + 0.02
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(mean_mat, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = mean_mat[i, j]
            std = std_mat[i, j]
            rgba = HEATMAP_CMAP(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if brightness < 0.55 else "black"
            ax.text(j, i, f"{val:.3f}\n\u00b1{std:.3f}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(4))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Data (%)", fontsize=12, fontweight="bold")
    ax.set_title("LMNB1 Finetuning \u2014 Mean Dice Score",
                 fontsize=13, fontweight="bold")

    # Base MegaSeg reference
    ax.text(
        0.5, -0.18,
        f"Base MegaSeg (no finetuning): Dice = {base_mean:.3f} \u00b1 {base_std:.3f}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="white",
                  edgecolor="black", linewidth=1.0, alpha=1.0),
        color="black",
    )

    # Colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=HEATMAP_CMAP),
                        ax=ax, shrink=0.85)
    cbar.set_label("Dice Score", fontsize=11)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate final LMNB1 finetuning figures")
    parser.add_argument(
        "--only", choices=["heatmap", "qualitative"],
        help="Generate only one figure (default: both)")
    args = parser.parse_args()

    qual_path = FIGURE_DIR / "fig_lmnb1_finetune_qualitative_v3.pdf"
    heatmap_path = FIGURE_DIR / "fig_lmnb1_finetune_heatmap_v2.pdf"
    metrics_csv = FIGURE_DIR / "lmnb1_finetune_metrics_v2.csv"

    do_qual = args.only in (None, "qualitative")
    do_heat = args.only in (None, "heatmap")

    if do_heat:
        print("\n[1/2] Heatmap")
        if not metrics_csv.exists():
            print(f"  ERROR: {metrics_csv} not found. "
                  "Run lmnb1_finetune_analysis_v2.py first to generate it.")
        else:
            rows = load_metrics_csv(str(metrics_csv))
            print(f"  Loaded {len(rows)} metric rows")
            build_heatmap(rows, heatmap_path)

    if do_qual:
        print("\n[2/2] Qualitative (MIP)")
        print(f"  Crop: {CROP_SIZE}px, Scale bar: {SCALEBAR_UM} \u00b5m")
        build_qualitative(qual_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
