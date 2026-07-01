"""
Fig 3 (Finetuning) — single, config-driven generator.

Produces the combined figure in one shot from `fig_finetune_config.json`:

    a  LMNB1 heatmap      b  LAMP1 heatmap        (top row)
    c  LMNB1 qualitative  (full width)
    d  LAMP1 qualitative  (full width)
    + shared TP/FP/FN/TN legend

Everything is driven by the JSON: output name, render DPI, scale bar, the 4x4
heatmap grid, the error-map legend, and per-structure data paths + pinned
qualitative samples (stems, conditions, column labels, crop mode).

Inputs (in this folder): the two per-sample metric CSVs referenced by the
config. Raw / GT / prediction volumes are read from //allen. No intermediate
component PDFs are written, and there are no other scripts to run.

Usage:
    python fig_finetune.py
    python fig_finetune.py --config other_config.json
"""

import argparse
import csv
import io
import json
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
from matplotlib.image import imread
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5,
})

DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════
def mip(vol):
    return np.max(vol, axis=0)


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


def make_error_map(gt, pred, colors):
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[gt & pred] = colors["True Positive"]
    rgb[~gt & pred] = colors["False Positive"]
    rgb[gt & ~pred] = colors["False Negative"]
    rgb[~gt & ~pred] = colors["True Negative"]
    return rgb


def add_scalebar(ax, crop_px, scalebar_um, pixel_um):
    bar_px = scalebar_um / pixel_um
    margin = 8
    y = crop_px - margin
    x_end = crop_px - margin
    x_start = x_end - bar_px
    ax.plot([x_start, x_end], [y, y], color="white", linewidth=2, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2, y - 5, f"{scalebar_um} µm", color="white",
            fontsize=6, ha="center", va="bottom", fontweight="bold")


# ══════════════════════════════════════════════════════════════
# Data loading (config-driven, handles LMNB1 & LAMP1 layouts)
# ══════════════════════════════════════════════════════════════
def load_raw(dcfg, stem):
    vol = tif.imread(f"{dcfg['raw_dir']}/{dcfg['raw_template'].format(stem=stem)}")
    ch = dcfg.get("raw_channel")
    if ch is not None and vol.ndim == 4:      # (Z, C, Y, X) -> pick structure channel
        if isinstance(ch, dict):              # channel-count-dependent rule
            ch = ch["then"] if vol.shape[1] >= ch["if_channels_ge"] else ch["else"]
        vol = vol[:, ch, :, :]
    return vol.astype(np.float32)


def load_gt(dcfg, stem):
    vol = tif.imread(f"{dcfg['gt_dir']}/{dcfg['gt_template'].format(stem=stem)}")
    return (vol > 0).astype(np.bool_)


def load_pred(dcfg, cond, stem):
    vol = tif.imread(f"{dcfg['pred_root']}/{dcfg['pred_template'].format(cond=cond, stem=stem)}")
    if vol.dtype == np.uint8:
        return (vol >= 128).astype(np.bool_)
    return (vol >= 0.5).astype(np.bool_)


# ══════════════════════════════════════════════════════════════
# Heatmap panel
# ══════════════════════════════════════════════════════════════
def load_metrics(path, cond_col, dice_col):
    by_cond = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_cond[r[cond_col]].append(float(r[dice_col]))
    return by_cond


def heatmap_fig(struct, cfg):
    hm = cfg["heatmap"]
    by_cond = load_metrics(DIR / struct["metrics_csv"], struct["condition_col"], struct["dice_col"])
    dp, ec = hm["data_pcts"], hm["epoch_cfgs"]
    mean = np.zeros((len(dp), len(ec)))
    std = np.zeros((len(dp), len(ec)))
    for i, d in enumerate(dp):
        for j, e in enumerate(ec):
            vals = by_cond.get(f"data{d}_{e}", [])
            if vals:
                mean[i, j], std[i, j] = np.mean(vals), np.std(vals)
    base = by_cond.get(hm["base_condition"], [])
    base_mean = float(np.mean(base)) if base else 0.0
    base_std = float(np.std(base)) if base else 0.0

    cmap = LinearSegmentedColormap.from_list("hm", hm["cmap"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vmin, vmax = min(mean.min(), base_mean) - 0.02, mean.max() + 0.02
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.imshow(mean, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(dp)):
        for j in range(len(ec)):
            rgba = cmap(norm(mean[i, j]))
            b = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{mean[i, j]:.3f}\n±{std[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white" if b < 0.55 else "black")
    ax.set_xticks(range(len(ec))); ax.set_xticklabels(hm["epoch_labels"], fontsize=11)
    ax.set_yticks(range(len(dp))); ax.set_yticklabels([f"{d}%" for d in dp], fontsize=11)
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Data (%)", fontsize=12, fontweight="bold")
    ax.set_title(struct["heatmap_title"], fontsize=13, fontweight="bold")
    ax.text(0.5, -0.18, f"Base MegaSeg (no finetuning): Dice = {base_mean:.3f} ± {base_std:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.0))
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.85)
    cbar.set_label("Dice Score", fontsize=11)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Qualitative panel
# ══════════════════════════════════════════════════════════════
def load_row(struct, cfg, stem, conditions, crop_ref_cond, colors):
    dcfg = struct["data"]
    crop_px = cfg["crop_size_px"]
    raw = normalize_percentile(mip(load_raw(dcfg, stem)))
    gt = mip(load_gt(dcfg, stem).astype(np.uint8)) > 0
    ref = gt
    if crop_ref_cond:
        pred_ref = mip(load_pred(dcfg, crop_ref_cond, stem).astype(np.uint8)) > 0
        fn = gt & (~pred_ref)
        if fn.sum() >= 50:
            ref = fn
    y0, x0, cs = smart_crop(gt.shape, crop_px, ref_mask=ref)
    crop = lambda im: im[y0:y0 + cs, x0:x0 + cs]
    emaps = [make_error_map(crop(gt), crop(mip(load_pred(dcfg, c, stem).astype(np.uint8)) > 0), colors)
             for c in conditions]
    return crop(raw), crop(gt), emaps, cs


def qualitative_fig(struct, cfg, colors):
    q = struct["qualitative"]
    groups = [q["panel_A"], q["panel_B"]]
    rows = []
    for g in groups:
        for si, stem in enumerate(g["stems"]):
            rows.append((stem, g, si == 0))
    n_rows, n_cols = len(rows), 6

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5 + 1.6), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.04)

    group_first_row = {}
    for ri, (stem, g, first) in enumerate(rows):
        col_titles = ["Raw", "GT"] + g["column_labels"]
        print(f"    row {ri}: {stem}")
        raw, gt, emaps, cs = load_row(struct, cfg, stem, g["conditions"],
                                      g.get("crop_ref_cond"), colors)
        if first:
            group_first_row.setdefault(g["label"], ri)
        for ci in range(n_cols):
            ax = fig.add_subplot(gs[ri, ci])
            if ci == 0:
                ax.imshow(raw, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
                add_scalebar(ax, cs, cfg["scalebar_um"], cfg["pixel_size_um"])
            elif ci == 1:
                ax.imshow(gt.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            else:
                ax.imshow(emaps[ci - 2], interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3); sp.set_color("#555555")
            if first:
                ax.set_title(col_titles[ci], fontsize=10, fontweight="bold", pad=6)

    # Panel A/B labels on the left + dashed separator between the two groups
    fig.text(0.01, 0.75, groups[0]["label"], fontsize=11, fontweight="bold",
             va="center", ha="left", rotation=90)
    fig.text(0.01, 0.30, groups[1]["label"], fontsize=11, fontweight="bold",
             va="center", ha="left", rotation=90)
    split = len(groups[0]["stems"])
    sep_y = (gs[split - 1, 0].get_position(fig).y0 + gs[split, 0].get_position(fig).y1) / 2
    fig.add_artist(plt.Line2D([0.03, 0.98], [sep_y, sep_y], transform=fig.transFigure,
                              color="gray", linewidth=1, linestyle="--"))
    return fig


# ══════════════════════════════════════════════════════════════
# Assembly
# ══════════════════════════════════════════════════════════════
def fig_to_array(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return imread(buf)[:, :, :3]


def main():
    ap = argparse.ArgumentParser(description="Generate the combined finetuning figure")
    ap.add_argument("--config", default="fig_finetune_config.json")
    args = ap.parse_args()
    cfg = json.load(open(DIR / args.config, encoding="utf-8"))
    dpi = cfg["render_dpi"]
    colors = {e["label"]: tuple(e["rgb"]) for e in cfg["legend"]}
    structs = {s["name"]: s for s in cfg["structures"]}

    print("Rendering panels...")
    panels = {}
    for name, s in structs.items():
        print(f"  {name} heatmap"); panels[(name, "hm")] = fig_to_array(heatmap_fig(s, cfg), dpi)
        print(f"  {name} qualitative"); panels[(name, "q")] = fig_to_array(qualitative_fig(s, cfg, colors), dpi)

    # order structures by panel letter: a/c = first, b/d = second
    left = min(structs.values(), key=lambda s: s["panel_heatmap"])
    right = max(structs.values(), key=lambda s: s["panel_heatmap"])
    hm_l, hm_r = panels[(left["name"], "hm")], panels[(right["name"], "hm")]
    q_c, q_d = panels[(left["name"], "q")], panels[(right["name"], "q")]

    def asp(im):
        return im.shape[0] / im.shape[1]

    FIG_W = 10.0
    left_in, right_in = 0.30, 0.15
    content_w = FIG_W - left_in - right_in
    gap_hm, gap_v = 0.5, 0.30
    top_in, legend_h, bottom_in = 0.35, 0.45, 0.20
    hm_w = (content_w - gap_hm) / 2.0
    hm_h = hm_w * max(asp(hm_l), asp(hm_r))
    qc_h, qd_h = content_w * asp(q_c), content_w * asp(q_d)
    FIG_H = top_in + hm_h + gap_v + qc_h + gap_v + qd_h + gap_v + legend_h + bottom_in

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    def place(im, x_in, y_top, w_in, h_in):
        ax = fig.add_axes([x_in / FIG_W, 1 - (y_top + h_in) / FIG_H, w_in / FIG_W, h_in / FIG_H])
        ax.imshow(im, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def label(x_in, y_top, txt):
        fig.text(x_in / FIG_W, 1 - y_top / FIG_H, txt, fontsize=14, fontweight="bold",
                 va="top", ha="left")

    y = top_in
    place(hm_l, left_in, y, hm_w, hm_h)
    place(hm_r, left_in + hm_w + gap_hm, y, hm_w, hm_h)
    label(0.06, y, left["panel_heatmap"])
    label((left_in + hm_w + gap_hm) - 0.24, y, right["panel_heatmap"])
    y += hm_h + gap_v
    place(q_c, left_in, y, content_w, qc_h); label(0.06, y, left["panel_qualitative"]); y += qc_h + gap_v
    place(q_d, left_in, y, content_w, qd_h); label(0.06, y, right["panel_qualitative"]); y += qd_h + gap_v

    leg_ax = fig.add_axes([0.05, bottom_in / FIG_H * 0.4, 0.90, legend_h / FIG_H])
    leg_ax.axis("off")
    handles = [Patch(facecolor=tuple(e["rgb"]), edgecolor="black", label=e["label"]) for e in cfg["legend"]]
    leg_ax.legend(handles=handles, loc="center", ncol=4, fontsize=11, frameon=True, edgecolor="black")

    out = DIR / cfg["output"]
    fig.savefig(str(out), dpi=dpi, facecolor="white")
    fig.savefig(str(out.with_suffix(".png")), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
