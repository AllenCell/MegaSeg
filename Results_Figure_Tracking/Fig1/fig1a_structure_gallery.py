"""
Fig 1a — MegaSeg structure gallery (13 training structures).

Self-contained regeneration script. Reads the image list and all styling from
`fig1a_image_manifest.json` (same folder) and writes `fig1a_structure_gallery.pdf`.

Naming convention follows Viana et al., Nature 2023 (s41586-022-05563-7), Fig. 1a:
each panel is labelled "<structure> via <protein>" with a small cellular-compartment
colour square beside the name (6-bucket radial scheme, nucleus -> cell periphery).

Usage:
    python fig1a_structure_gallery.py
"""

import json
import os
from pathlib import Path

import numpy as np
import tifffile as tif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Nature-style rcParams (Arial, editable PDF text) ──────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).parent
MANIFEST = HERE / "fig1a_image_manifest.json"
OUT_PDF = HERE / "fig1a_structure_gallery.pdf"


# ── Image helpers ─────────────────────────────────────────────────
def mip(vol):
    return np.max(vol, axis=0)


def normalize(img, p_lo=2, p_hi=99.7, gamma=0.85):
    """Per-image percentile contrast stretch + gamma (gamma < 1 lifts dim signal)."""
    lo, hi = np.percentile(img, p_lo), np.percentile(img, p_hi)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, 0, 1)
    if gamma and gamma != 1.0:
        img = np.power(img, gamma)
    return img


def center_crop(img, size):
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def add_scalebar(ax, img_size, bar_um, pixel_um):
    """White scale bar + value label in the bottom-right corner.

    Drawn once for the whole gallery — all crops share the same pixel size.
    """
    import matplotlib.patheffects as pe
    bar_px = bar_um / pixel_um
    margin = img_size * 0.04
    y = img_size - margin
    x_end = img_size - margin
    x_start = x_end - bar_px
    stroke = [pe.withStroke(linewidth=1.4, foreground="black")]
    line, = ax.plot([x_start, x_end], [y, y], color="white", linewidth=1.6,
                    solid_capstyle="butt")
    line.set_path_effects(stroke)
    txt = ax.text((x_start + x_end) / 2, y - img_size * 0.03,
                  f"{bar_um:g} µm", color="white", ha="center", va="bottom",
                  fontsize=5.5, fontweight="bold")
    txt.set_path_effects(stroke)


# ── Build the gallery ─────────────────────────────────────────────
def main():
    with open(MANIFEST, encoding="utf-8") as f:
        m = json.load(f)

    structs = m["structures"]
    colors = m["compartment_colors"]
    labels = m["compartment_labels"]
    order = m["compartment_legend_order"]
    crop = m["crop_size_px"]
    px_um = m["pixel_size_um"]
    bar_um = m["scalebar_um"]
    r = m["render"]
    p_lo, p_hi, gamma = r["p_lo"], r["p_hi"], r["gamma"]

    n_cols, n_rows = 7, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 2.8), facecolor="white")
    fig.subplots_adjust(wspace=0.06, hspace=0.35, left=0.01, right=0.99,
                        top=0.85, bottom=0.08)

    for idx, s in enumerate(structs):
        ax = axes[idx // n_cols, idx % n_cols]
        try:
            vol = tif.imread(s["raw_path"]).astype(np.float32)
            img = center_crop(normalize(mip(vol), p_lo, p_hi, gamma), crop)
        except Exception as e:
            print(f"  {s['gene']}: ERROR {e}")
            img = np.zeros((crop, crop))

        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        if idx == len(structs) - 1:  # single shared, labelled scale bar
            add_scalebar(ax, crop, bar_um, px_um)

        # Compartment colour square + "<structure> via <protein>" label.
        ax.plot(0.03, 1.15, marker="s", markersize=4.2,
                color=colors[s["compartment"]], transform=ax.transAxes, clip_on=False)
        ax.text(0.14, 1.15, f"{s['structure']}\nvia {s['marker']}",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=4.8, fontweight="bold", linespacing=1.1)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        print(f"  {s['gene']}: {s['sample_stem']} OK")

    # Compartment legend in the empty cell (row 1, col 6).
    ax_leg = axes[1, 6]
    ax_leg.axis("off")
    handles = [Patch(facecolor=colors[k], label=labels[k], edgecolor="none")
               for k in order]
    ax_leg.legend(handles=handles, loc="center", ncol=1, fontsize=5,
                  frameon=False, handlelength=1.2, handletextpad=0.4,
                  labelspacing=0.6)

    fig.savefig(str(OUT_PDF), dpi=300, facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
