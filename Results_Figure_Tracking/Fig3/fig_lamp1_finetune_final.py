"""
LAMP1 Finetuning — component figures for the combined Fig 3.

Generates the two LAMP1 panels that `fig_finetune_combined.py` consumes:
  1. fig_lamp1_finetune_heatmap_v3.pdf            (4x4 Dice heatmap)
  2. fig_lamp1_finetune_qualitative_nolegend.pdf  (MIP error-map comparison)

Data sources:
  - Heatmap:      lamp1_finetune_metrics.csv  (per-sample Dice; from AllenCell/
                  MegaSeg PR #40 -> benchmark/results/finetune/lamp1/
                  per_sample_metrics.csv).
  - Qualitative:  raw/GT from //allen .../variance-100x/LAMP1/dataset/test/
                  (LAMP1_<id>.tiff and LAMP1_<id>_seg.tiff); predictions from
                  .../eval/finetune/lamp1/<condition>/seg/LAMP1_<id>.tif.

Qualitative sample selection is derived from the metrics CSV (largest
base->converged and 10%->100% Dice improvement), so it is reproducible.

Usage:
    python fig_lamp1_finetune_final.py                 # both panels
    python fig_lamp1_finetune_final.py --only heatmap
    python fig_lamp1_finetune_final.py --only qualitative
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5,
})

# ══════════════════════════════════════════════════════════════
# Paths & constants
# ══════════════════════════════════════════════════════════════
PRED_ROOT = "//allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/finetune/lamp1"
DATA_TEST_DIR = "//allen/aics/rep_learn/MegaSeg/data/variance-100x/LAMP1/dataset/test"
FIGURE_DIR = Path(__file__).parent
METRICS_CSV = FIGURE_DIR / "lamp1_finetune_metrics.csv"

CROP_SIZE = 320
PIXEL_SIZE_UM = 0.10833  # μm per pixel
SCALEBAR_UM = 10

EPOCH_CONDS = ["data100_10epochs", "data100_50epochs",
               "data100_100epochs", "data100_converged"]
EPOCH_LABELS = ["10 ep", "50 ep", "100 ep", "Converged"]
DATA_CONDS = ["data10_10epochs", "data20_10epochs",
              "data50_10epochs", "data100_10epochs"]
DATA_LABELS = ["10%", "20%", "50%", "100%"]


# ── I/O helpers ──────────────────────────────────────────────
def load_gt(stem):
    vol = tif.imread(os.path.join(DATA_TEST_DIR, f"{stem}_seg.tiff"))
    return (vol > 0).astype(np.bool_)


def load_raw(stem):
    vol = tif.imread(os.path.join(DATA_TEST_DIR, f"{stem}.tiff"))
    return vol.astype(np.float32)  # (Z, Y, X), single channel


def load_pred(condition, stem):
    vol = tif.imread(os.path.join(PRED_ROOT, condition, "seg", f"{stem}.tif"))
    if vol.dtype == np.uint8:
        return (vol >= 128).astype(np.bool_)
    return (vol >= 0.5).astype(np.bool_)


# ── MIP / crop / error map ──────────────────────────────────
def mip(volume):
    return np.max(volume, axis=0)


def normalize_percentile(img, p_lo=1, p_hi=95):
    lo, hi = np.percentile(img, p_lo), np.percentile(img, p_hi)
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
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
    """Muted error map: gold=TP, rose=FP, steel blue=FN, dark gray=TN."""
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
    y = crop_pixels - margin
    x_end = crop_pixels - margin
    x_start = x_end - bar_px
    ax.plot([x_start, x_end], [y, y], color="white", linewidth=2, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2, y - 5, f"{SCALEBAR_UM} µm",
            color="white", fontsize=6, ha="center", va="bottom", fontweight="bold")


# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════
def load_metrics_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["dice"] = float(r["dice"])
            rows.append(r)
    return rows


HEATMAP_DATA_PCTS = [10, 20, 50, 100]
HEATMAP_EPOCH_CFG = ["10epochs", "50epochs", "100epochs", "converged"]
HEATMAP_EPOCH_DISPLAY = ["10 ep", "50 ep", "100 ep", "Converged"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)


def build_heatmap_data(rows):
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[(r["data_pct"], r["epoch_cfg"])].append(r["dice"])

    mean_mat = np.zeros((4, 4))
    std_mat = np.zeros((4, 4))
    for i, dp in enumerate(HEATMAP_DATA_PCTS):
        for j, ep in enumerate(HEATMAP_EPOCH_CFG):
            vals = by_cond.get((str(dp), ep), [])
            if vals:
                mean_mat[i, j] = np.mean(vals)
                std_mat[i, j] = np.std(vals)

    base = [r["dice"] for r in rows if r["experiment"] == "base_megaseg"]
    base_mean = float(np.mean(base)) if base else 0.0
    base_std = float(np.std(base)) if base else 0.0
    return mean_mat, std_mat, ["10%", "20%", "50%", "100%"], HEATMAP_EPOCH_DISPLAY, base_mean, base_std


def build_heatmap(rows, out_path):
    mean_mat, std_mat, row_labels, col_labels, base_mean, base_std = build_heatmap_data(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    vmin = min(np.min(mean_mat), base_mean) - 0.02
    vmax = np.max(mean_mat) + 0.02
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(mean_mat, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    for i in range(4):
        for j in range(4):
            val, std = mean_mat[i, j], std_mat[i, j]
            rgba = HEATMAP_CMAP(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if brightness < 0.55 else "black"
            ax.text(j, i, f"{val:.3f}\n±{std:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(4)); ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(4)); ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Data (%)", fontsize=12, fontweight="bold")
    ax.set_title("LAMP1 Finetuning — Mean Dice Score", fontsize=13, fontweight="bold")
    ax.text(0.5, -0.18,
            f"Base MegaSeg (no finetuning): Dice = {base_mean:.3f} ± {base_std:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="black", linewidth=1.0))
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), ax=ax, shrink=0.85)
    cbar.set_label("Dice Score", fontsize=11)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
# Qualitative
# ══════════════════════════════════════════════════════════════
def select_samples(rows):
    """Reproducibly pick 2 epoch-effect + 2 data-effect stems by improvement."""
    dice = defaultdict(dict)  # experiment -> {sample_id: dice}
    for r in rows:
        dice[r["experiment"]][r["sample_id"]] = r["dice"]

    def top2(cond_hi, cond_lo):
        common = set(dice[cond_hi]) & set(dice[cond_lo])
        ranked = sorted(common, key=lambda s: dice[cond_hi][s] - dice[cond_lo][s], reverse=True)
        return ranked[:2]

    panel_a = top2("data100_converged", "data100_10epochs")   # epoch effect (100% data)
    panel_b = top2("data100_10epochs", "data10_10epochs")     # data effect (10 epochs)
    return panel_a, panel_b


def load_row_data(stem, conditions):
    raw_mip = normalize_percentile(mip(load_raw(stem)))
    gt_mip = mip(load_gt(stem).astype(np.uint8)) > 0
    y0, x0, cs = smart_crop(gt_mip.shape, CROP_SIZE, ref_mask=gt_mip)

    def crop(img):
        return img[y0:y0 + cs, x0:x0 + cs]

    raw_crop, gt_crop = crop(raw_mip), crop(gt_mip)
    error_maps = []
    for cond in conditions:
        pred_mip = mip(load_pred(cond, stem).astype(np.uint8)) > 0
        error_maps.append(make_error_map(gt_crop, crop(pred_mip)))
    return raw_crop, gt_crop, error_maps, cs


def build_qualitative(rows, out_path, show_legend=False):
    panel_a_stems, panel_b_stems = select_samples(rows)
    print(f"  Panel A (epoch) stems: {panel_a_stems}")
    print(f"  Panel B (data)  stems: {panel_b_stems}")

    n_rows, n_cols = 4, 6
    col_titles_a = ["Raw", "GT"] + EPOCH_LABELS
    col_titles_b = ["Raw", "GT"] + DATA_LABELS

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5 + 1.6), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.04)

    panel_rows = ([(s, EPOCH_CONDS, col_titles_a) for s in panel_a_stems]
                  + [(s, DATA_CONDS, col_titles_b) for s in panel_b_stems])

    for row_idx, (stem, conditions, _ct) in enumerate(panel_rows):
        print(f"  Loading row {row_idx}: {stem}...")
        raw_crop, gt_crop, error_maps, crop_px = load_row_data(stem, conditions)
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if col_idx == 0:
                ax.imshow(raw_crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
                add_scalebar(ax, crop_px)
            elif col_idx == 1:
                ax.imshow(gt_crop.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            else:
                ax.imshow(error_maps[col_idx - 2], interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3); sp.set_color("#555555")
            if row_idx == 0:
                ax.set_title(col_titles_a[col_idx], fontsize=10, fontweight="bold", pad=6)
            elif row_idx == 2:
                ax.set_title(col_titles_b[col_idx], fontsize=10, fontweight="bold", pad=6)

    fig.text(0.01, 0.75, "A  Epoch effect\n(100% data)", fontsize=11,
             fontweight="bold", va="center", ha="left", rotation=90)
    fig.text(0.01, 0.30, "B  Data effect\n(10 epochs)", fontsize=11,
             fontweight="bold", va="center", ha="left", rotation=90)

    sep_y = (gs[1, 0].get_position(fig).y0 + gs[2, 0].get_position(fig).y1) / 2
    fig.add_artist(plt.Line2D([0.03, 0.98], [sep_y, sep_y], transform=fig.transFigure,
                              color="gray", linewidth=1, linestyle="--"))

    if show_legend:
        handles = [Patch(facecolor=c, edgecolor="gray", label=l) for c, l in [
            ((0.85, 0.75, 0.25), "True Positive"), ((0.82, 0.35, 0.45), "False Positive"),
            ((0.30, 0.55, 0.75), "False Negative"), ((0.12, 0.12, 0.12), "True Negative")]]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
                   frameon=True, edgecolor="black", bbox_to_anchor=(0.52, -0.01))

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate LAMP1 finetuning component figures")
    parser.add_argument("--only", choices=["heatmap", "qualitative"],
                        help="Generate only one figure (default: both)")
    args = parser.parse_args()

    rows = load_metrics_csv(str(METRICS_CSV))
    print(f"  Loaded {len(rows)} metric rows from {METRICS_CSV.name}")

    if args.only in (None, "heatmap"):
        print("\n[1/2] Heatmap")
        build_heatmap(rows, FIGURE_DIR / "fig_lamp1_finetune_heatmap_v3.pdf")

    if args.only in (None, "qualitative"):
        print("\n[2/2] Qualitative (MIP, nolegend)")
        build_qualitative(rows, FIGURE_DIR / "fig_lamp1_finetune_qualitative_nolegend.pdf",
                           show_legend=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
