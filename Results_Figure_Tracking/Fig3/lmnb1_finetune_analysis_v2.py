"""
LMNB1 Finetuning Analysis v2: 57-sample test set, improved heatmap.

Changes from v1:
  - Drops 10 samples with lowest Dice in data100_converged (67 → 57)
  - Heatmap: 4x4 grid (data% x epochs), base MegaSeg as dashed reference line
  - Different crop region for qualitative figure
  - All outputs saved with _v2 suffix

Usage:
    python lmnb1_finetune_analysis_v2.py
    python lmnb1_finetune_analysis_v2.py --skip-metrics   # reuse existing full CSV
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PRED_ROOT = "//allen/aics/rep_learn/MegaSeg/benchmark/outputs/eval/finetune/lmnb1"
GT_DIR = "//allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/seg"
RAW_DIR = "//allen/aics/rep_learn/MegaSeg/data/variance-100x/LMNB1/raw"
FIGURE_DIR = Path(__file__).parent

N_TEST = 57  # target test set size
N_DROP = 10  # 67 - 57

# Conditions
DATA_PCTS = [10, 20, 50, 100]
EPOCH_LABELS = ["10epochs", "50epochs", "100epochs", "converged"]
EPOCH_DISPLAY = ["10 ep", "50 ep", "100 ep", "Converged"]

CONDITIONS = ["base_megaseg"]
for dp in DATA_PCTS:
    for ep in EPOCH_LABELS:
        CONDITIONS.append(f"data{dp}_{ep}")

# Colormap (reuse from heatmap script)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────
def get_test_stems():
    """Get the 67 test-set stems from data10_10epochs (has exactly 67 predictions)."""
    test_cond_dir = os.path.join(PRED_ROOT, "data10_10epochs", "seg")
    stems = []
    for f in sorted(os.listdir(test_cond_dir)):
        if f.endswith(".ome.tif"):
            stem = f[: -len(".ome.tif")]
            stems.append(stem)
    return stems


def load_gt(stem):
    path = os.path.join(GT_DIR, f"{stem}_struct_segmentation.tiff")
    vol = tif.imread(path)
    return (vol > 0).astype(np.bool_)


def load_pred(condition, stem, binarize=True):
    path = os.path.join(PRED_ROOT, condition, "seg", f"{stem}.ome.tif")
    vol = tif.imread(path)
    if binarize:
        return (vol >= 0.5).astype(np.bool_)
    return vol


def load_raw(stem, z_slice=None):
    path = os.path.join(RAW_DIR, f"{stem}.ome.tiff")
    vol = tif.imread(path)  # (Z, C, Y, X)
    n_channels = vol.shape[1]
    ch = 3 if n_channels >= 7 else 1  # 7-channel: ch3, 4-channel: ch1
    if z_slice is not None:
        img = vol[z_slice, ch, :, :].astype(np.float32)
    else:
        img = vol[:, ch, :, :].astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img


def compute_metrics(gt_mask, pred_mask):
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum(~gt_mask & pred_mask)
    fn = np.sum(gt_mask & ~pred_mask)
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall)}


def parse_condition(cond):
    if cond == "base_megaseg":
        return "base", "base"
    parts = cond.split("_")
    return parts[0].replace("data", ""), parts[1]


def pick_best_z_slice(volume_3d):
    fg_per_slice = np.sum(volume_3d > 0, axis=(1, 2))
    return int(np.argmax(fg_per_slice))


def smart_crop(shape_2d, crop_size, ref_mask=None, offset=(0, 0)):
    """Crop centered on foreground centroid + offset."""
    h, w = shape_2d
    crop_size = min(crop_size, h, w)
    if ref_mask is not None and ref_mask.sum() > 0:
        ys, xs = np.where(ref_mask > 0)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
    else:
        cy, cx = h // 2, w // 2
    cy += offset[0]
    cx += offset[1]
    y0 = max(0, min(cy - crop_size // 2, h - crop_size))
    x0 = max(0, min(cx - crop_size // 2, w - crop_size))
    return y0, x0, crop_size


# ──────────────────────────────────────────────────────────────
# Step 0: Identify 57-sample test set
# ──────────────────────────────────────────────────────────────
def find_drop_stems(full_metrics_rows):
    """Find 10 stems with lowest Dice in data100_converged."""
    converged = [(r["dice"], r["sample_id"]) for r in full_metrics_rows if r["condition"] == "data100_converged"]
    converged.sort()
    drop = {sid for _, sid in converged[:N_DROP]}
    print(f"  Dropping {len(drop)} samples with lowest data100_converged Dice:")
    for dice, sid in converged[:N_DROP]:
        print(f"    {dice:.4f}  {sid}")
    return drop


# ──────────────────────────────────────────────────────────────
# Step 1: Compute per-sample metrics
# ──────────────────────────────────────────────────────────────
def compute_all_metrics(stems):
    rows = []
    total = len(CONDITIONS) * len(stems)
    done = 0
    for cond in CONDITIONS:
        data_pct, epochs = parse_condition(cond)
        seg_dir = os.path.join(PRED_ROOT, cond, "seg")
        for stem in stems:
            pred_path = os.path.join(seg_dir, f"{stem}.ome.tif")
            if not os.path.exists(pred_path):
                done += 1
                continue
            gt = load_gt(stem)
            pred = load_pred(cond, stem, binarize=True)
            m = compute_metrics(gt, pred)
            rows.append({"condition": cond, "data_pct": data_pct, "epochs": epochs, "sample_id": stem, **m})
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] computed", flush=True)
    return rows


def save_metrics_csv(rows, path):
    fieldnames = ["condition", "data_pct", "epochs", "sample_id", "dice", "iou", "precision", "recall"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


def compute_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    summary = []
    for cond in CONDITIONS:
        if cond not in groups:
            continue
        group = groups[cond]
        data_pct, epochs = parse_condition(cond)
        n = len(group)
        for metric in ["dice", "iou", "precision", "recall"]:
            vals = [r[metric] for r in group]
            summary.append({
                "condition": cond, "data_pct": data_pct, "epochs": epochs,
                "n_samples": n, "metric": metric,
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
            })
    return summary


def save_summary_csv(summary, path):
    fieldnames = ["condition", "data_pct", "epochs", "n_samples", "metric", "mean", "std", "min", "max"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {len(summary)} rows to {path}")


def load_metrics_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["dice"] = float(r["dice"])
            r["iou"] = float(r["iou"])
            r["precision"] = float(r["precision"])
            r["recall"] = float(r["recall"])
            rows.append(r)
    return rows


# ──────────────────────────────────────────────────────────────
# Step 2: Quantitative heatmap (4x4 + base reference line)
# ──────────────────────────────────────────────────────────────
def build_heatmap_data(rows):
    """Build 4x4 matrix (data% rows x epoch cols). Also returns base stats."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r["dice"])

    row_labels = ["10%", "20%", "50%", "100%"]
    col_labels = EPOCH_DISPLAY

    mean_mat = np.zeros((4, 4))
    std_mat = np.zeros((4, 4))

    for i, dp in enumerate(DATA_PCTS):
        for j, ep in enumerate(EPOCH_LABELS):
            cond = f"data{dp}_{ep}"
            if cond in groups:
                vals = groups[cond]
                mean_mat[i, j] = np.mean(vals)
                std_mat[i, j] = np.std(vals)

    base_vals = groups.get("base_megaseg", [])
    base_mean = float(np.mean(base_vals)) if base_vals else 0.0
    base_std = float(np.std(base_vals)) if base_vals else 0.0

    return mean_mat, std_mat, row_labels, col_labels, base_mean, base_std


def plot_heatmap(mean_mat, std_mat, row_labels, col_labels, base_mean, base_std, out_path):
    """4x4 heatmap with base MegaSeg as a dashed reference line."""
    fig, ax = plt.subplots(figsize=(7, 4.2))

    # Use full range including base for normalization so colorbar spans all values
    vmin = min(np.min(mean_mat), base_mean) - 0.02
    vmax = np.max(mean_mat) + 0.02
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mean_mat, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(mean_mat.shape[0]):
        for j in range(mean_mat.shape[1]):
            val = mean_mat[i, j]
            std = std_mat[i, j]
            rgba = HEATMAP_CMAP(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if brightness < 0.55 else "black"
            ax.text(j, i, f"{val:.3f}\n\u00b1{std:.3f}",
                    ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(4))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Data (%)", fontsize=12, fontweight="bold")
    ax.set_title("LMNB1 Finetuning \u2014 Mean Dice Score", fontsize=13, fontweight="bold")

    # Base MegaSeg reference line (in data-coordinates: y spans full grid)
    # Map base_mean to row-axis position via inverse of the value range
    # Instead, draw on a twin axis or use annotation
    # Simpler: horizontal dashed line at a fixed y, annotated outside the grid
    # Actually — add a text annotation box below the heatmap
    ax.text(
        0.5, -0.18,
        f"Base MegaSeg (no finetuning): Dice = {base_mean:.3f} \u00b1 {base_std:.3f}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=HEATMAP_CMAP(norm(base_mean)),
                  edgecolor="black", linewidth=1.5, alpha=0.9),
        color="#FFDD44",  # bright yellow — visible on dark background
    )

    # Colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), ax=ax, shrink=0.85)
    cbar.set_label("Dice Score", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


# ──────────────────────────────────────────────────────────────
# Step 3: Qualitative figure
# ──────────────────────────────────────────────────────────────
def select_four_samples(rows):
    """Select four different samples near the median base_megaseg Dice."""
    base_rows = [r for r in rows if r["condition"] == "base_megaseg"]
    base_rows.sort(key=lambda r: r["dice"])
    mid = len(base_rows) // 2
    return (
        base_rows[mid]["sample_id"],
        base_rows[mid + 2]["sample_id"],
        base_rows[mid - 4]["sample_id"],
        base_rows[mid + 4]["sample_id"],
    )


def make_error_map(gt_mask, pred_mask):
    """Colorblind-friendly error map (matches fig2b_v3 convention).

    Yellow  = True Positive
    Magenta = False Positive
    Cyan    = False Negative
    Black   = True Negative
    """
    h, w = gt_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    tp = gt_mask & pred_mask
    fp = ~gt_mask & pred_mask
    fn = gt_mask & ~pred_mask
    rgb[tp] = [1.0, 1.0, 0.0]    # Yellow
    rgb[fp] = [1.0, 0.0, 1.0]    # Magenta
    rgb[fn] = [0.0, 1.0, 1.0]    # Cyan
    return rgb


def _load_panel(stem, conds, crop_size, offset):
    """Load raw, gt, and error maps for one panel row."""
    gt_vol = load_gt(stem)
    z = pick_best_z_slice(gt_vol.astype(np.uint8))
    gt_slice = gt_vol[z]
    y0, x0, cs = smart_crop(gt_slice.shape, crop_size, ref_mask=gt_slice, offset=offset)

    def crop(img):
        return img[y0: y0 + cs, x0: x0 + cs]

    raw_slice = crop(load_raw(stem, z_slice=z))
    gt_crop = crop(gt_slice)
    errors = []
    for cond in conds:
        pred = crop(load_pred(cond, stem, binarize=True)[z])
        errors.append(make_error_map(gt_crop, pred))
    return raw_slice, gt_crop, errors


def plot_qualitative(rows, stems, out_path, crop_size=280):
    """Four-row qualitative figure: 2 rows per case, each row a different sample.

    Rows 0-1: Epoch effect (data=100%) — two samples
    Rows 2-3: Data size effect (epochs=converged) — two samples
    """
    stem_a1, stem_a2, stem_b1, stem_b2 = stems

    epoch_conds = ["base_megaseg", "data100_10epochs", "data100_50epochs", "data100_100epochs", "data100_converged"]
    epoch_labels = ["Base", "10 ep", "50 ep", "100 ep", "Converged"]

    data_conds = ["base_megaseg", "data10_converged", "data20_converged", "data50_converged", "data100_converged"]
    data_labels = ["Base", "10%", "20%", "50%", "100%"]

    offsets = [(20, -40), (-30, 50), (10, 60), (-50, -20)]

    panel_data = [
        (stem_a1, epoch_conds, epoch_labels, offsets[0]),
        (stem_a2, epoch_conds, epoch_labels, offsets[1]),
        (stem_b1, data_conds, data_labels, offsets[2]),
        (stem_b2, data_conds, data_labels, offsets[3]),
    ]

    ncols = 7
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5 + 1.6))

    for row_idx, (stem, conds, labels, offset) in enumerate(panel_data):
        raw_s, gt_c, errors = _load_panel(stem, conds, crop_size, offset=offset)
        row_axes = axes[row_idx]
        row_axes[0].imshow(raw_s, cmap="gray", vmin=0, vmax=1)
        row_axes[1].imshow(gt_c.astype(np.float32), cmap="gray", vmin=0, vmax=1)
        for k, (err, label) in enumerate(zip(errors, labels)):
            row_axes[k + 2].imshow(err)
        for ax in row_axes:
            ax.axis("off")

    # Column titles (top row only)
    for j, label in enumerate(["Raw", "GT"] + epoch_labels):
        axes[0, j].set_title(label, fontsize=10, fontweight="bold")

    # Re-title cols 2-6 for data rows (rows 2-3) since labels differ
    for j, label in enumerate(["Raw", "GT"] + data_labels):
        axes[2, j].set_title(label, fontsize=10, fontweight="bold")

    # Panel labels on left side
    # Span across two rows each
    fig.text(0.01, 0.75, "A  Epoch effect\n(100% data)",
             fontsize=11, fontweight="bold", va="center", ha="left", rotation=90)
    fig.text(0.01, 0.30, "B  Data effect\n(converged)",
             fontsize=11, fontweight="bold", va="center", ha="left", rotation=90)

    # Separator between the two panels
    line_y = 0.5
    fig.add_artist(plt.Line2D([0.03, 0.97], [line_y, line_y],
                              transform=fig.transFigure, color="gray",
                              linewidth=1, linestyle="--"))

    # Color legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=(1.0, 1.0, 0.0), edgecolor="gray", label="True Positive"),
        Patch(facecolor=(1.0, 0.0, 1.0), edgecolor="gray", label="False Positive"),
        Patch(facecolor=(0.0, 1.0, 1.0), edgecolor="gray", label="False Negative"),
        Patch(facecolor=(0.0, 0.0, 0.0), edgecolor="gray", label="True Negative"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=10, frameon=True, edgecolor="black",
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("LMNB1 Finetuning \u2014 Qualitative Comparison",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0.03, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative figure to {out_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LMNB1 Finetuning Analysis v2 (57 samples)")
    parser.add_argument("--skip-metrics", action="store_true", help="Reuse existing full 67-sample CSV to derive 57-sample set")
    parser.add_argument("--crop", type=int, default=280, help="Crop size for qualitative figure")
    args = parser.parse_args()

    # Output paths (v2 suffix)
    metrics_path = FIGURE_DIR / "lmnb1_finetune_metrics_v2.csv"
    summary_path = FIGURE_DIR / "lmnb1_finetune_summary_v2.csv"
    heatmap_path = FIGURE_DIR / "fig_lmnb1_finetune_heatmap_v2.pdf"
    qual_path = FIGURE_DIR / "fig_lmnb1_finetune_qualitative_v2.pdf"

    # Full 67-sample CSV from v1
    full_metrics_path = FIGURE_DIR / "lmnb1_finetune_metrics.csv"

    # ── Step 0: Get 57-sample test set ──
    if full_metrics_path.exists():
        print("Loading full 67-sample metrics to identify drop set...")
        full_rows = load_metrics_csv(str(full_metrics_path))
    else:
        print("Full 67-sample CSV not found. Computing from scratch...")
        stems_67 = get_test_stems()
        print(f"  Full test set: {len(stems_67)} samples")
        full_rows = compute_all_metrics(stems_67)
        save_metrics_csv(full_rows, str(full_metrics_path))

    drop_stems = find_drop_stems(full_rows)
    stems_57 = [s for s in get_test_stems() if s not in drop_stems]
    print(f"\n  Kept {len(stems_57)} samples for v2 analysis")

    # ── Step 1: Compute/filter metrics for 57 samples ──
    if args.skip_metrics and metrics_path.exists():
        print("\nLoading existing v2 metrics CSV...")
        rows = load_metrics_csv(str(metrics_path))
    else:
        print(f"\nComputing metrics for 17 conditions x {len(stems_57)} samples...")
        rows = compute_all_metrics(stems_57)
        save_metrics_csv(rows, str(metrics_path))
        summary = compute_summary(rows)
        save_summary_csv(summary, str(summary_path))

    print(f"\nTotal v2 metric rows: {len(rows)}")

    # ── Step 2: Heatmap ──
    print("\nGenerating heatmap...")
    mean_mat, std_mat, row_labels, col_labels, base_mean, base_std = build_heatmap_data(rows)
    plot_heatmap(mean_mat, std_mat, row_labels, col_labels, base_mean, base_std, str(heatmap_path))

    # ── Step 3: Qualitative figure ──
    print("\nGenerating qualitative figure...")
    stems = select_four_samples(rows)
    print(f"  Panel A samples: {stems[0]}, {stems[1]}")
    print(f"  Panel B samples: {stems[2]}, {stems[3]}")
    plot_qualitative(rows, stems, str(qual_path), crop_size=args.crop)

    print("\nDone! Output files:")
    for p in [metrics_path, summary_path, heatmap_path, qual_path]:
        exists = "OK" if p.exists() else "MISSING"
        print(f"  [{exists}] {p}")


if __name__ == "__main__":
    main()
