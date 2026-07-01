"""
Fig 3 (Finetuning) — combined figure assembler.

Stitches the four pre-rendered component PDFs into the final combined figure:

    a  LMNB1 heatmap      b  LAMP1 heatmap        (top row, side by side)
    c  LMNB1 qualitative  (full width)
    d  LAMP1 qualitative  (full width)
    + shared TP/FP/FN/TN legend

Outputs:
  - fig_finetune_combined_v6.pdf / .png
  - fig_finetune_combined_manifest.json   (records the component images used)

Components (must already exist in this folder):
  - fig_lmnb1_finetune_heatmap_v3.pdf          (from fig_lmnb1_finetune_final.py)
  - fig_lamp1_finetune_heatmap_v3.pdf
  - fig_lmnb1_finetune_qualitative_nolegend.pdf
  - fig_lamp1_finetune_qualitative_nolegend.pdf

This script only performs the final layout; it does not recompute any panel.

Dependencies: PyMuPDF (`pip install pymupdf`).

Usage:
    python fig_finetune_combined.py
"""

import json
from pathlib import Path

import numpy as np
import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DIR = Path(__file__).parent
OUT_PDF = DIR / "fig_finetune_combined_v6.pdf"
MANIFEST = DIR / "fig_finetune_combined_manifest.json"
RENDER_DPI = 300  # rasterization DPI for each component PDF

# Ordered a -> d. Each entry documents the panel, structure, and provenance.
COMPONENTS = [
    {"key": "heatmap_lmnb1", "panel": "a", "structure": "LMNB1", "type": "heatmap",
     "file": "fig_lmnb1_finetune_heatmap_v3.pdf",
     "source_script": "fig_lmnb1_finetune_final.py"},
    {"key": "heatmap_lamp1", "panel": "b", "structure": "LAMP1", "type": "heatmap",
     "file": "fig_lamp1_finetune_heatmap_v3.pdf",
     "source_script": "fig_lamp1_finetune_final.py (reads lamp1_finetune_metrics.csv)"},
    {"key": "qual_lmnb1", "panel": "c", "structure": "LMNB1", "type": "qualitative",
     "file": "fig_lmnb1_finetune_qualitative_nolegend.pdf",
     "source_script": "fig_lmnb1_finetune_final.py (build_qualitative show_legend=False)"},
    {"key": "qual_lamp1", "panel": "d", "structure": "LAMP1", "type": "qualitative",
     "file": "fig_lamp1_finetune_qualitative_nolegend.pdf",
     "source_script": "fig_lamp1_finetune_final.py (build_qualitative show_legend=False)"},
]

# Error-map colors (must match fig_lmnb1_finetune_final.make_error_map)
LEGEND = [
    ((0.85, 0.75, 0.25), "True Positive"),
    ((0.82, 0.35, 0.45), "False Positive"),
    ((0.30, 0.55, 0.75), "False Negative"),
    ((0.12, 0.12, 0.12), "True Negative"),
]


def render_pdf(path, dpi=RENDER_DPI):
    """Rasterize page 1 of a PDF to an (H, W, 3) uint8 RGB array."""
    doc = fitz.open(str(path))
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    return np.ascontiguousarray(img[:, :, :3])


def write_manifest(imgs):
    """Record which component images were composited into the figure."""
    components = []
    for c in COMPONENTS:
        im = imgs[c["key"]]
        components.append({
            "panel": c["panel"],
            "structure": c["structure"],
            "type": c["type"],
            "file": c["file"],
            "source_script": c["source_script"],
            "pixels_wh": [int(im.shape[1]), int(im.shape[0])],
        })
    manifest = {
        "figure": "Fig 3 (Finetuning) combined",
        "output": OUT_PDF.name,
        "assembler": Path(__file__).name,
        "render_dpi": RENDER_DPI,
        "layout": ("a=LMNB1 heatmap, b=LAMP1 heatmap (top row); "
                   "c=LMNB1 qualitative, d=LAMP1 qualitative (full width); "
                   "shared TP/FP/FN/TN legend at bottom"),
        "legend": [{"rgb": list(rgb), "label": lbl} for rgb, lbl in LEGEND],
        "components": components,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {MANIFEST}")


def main():
    imgs = {}
    for c in COMPONENTS:
        p = DIR / c["file"]
        if not p.exists():
            raise FileNotFoundError(f"Missing component: {p}")
        imgs[c["key"]] = render_pdf(p)
        print(f"  loaded {c['file']}: {imgs[c['key']].shape[1]}x{imgs[c['key']].shape[0]} px")

    def asp(im):  # height / width
        return im.shape[0] / im.shape[1]

    # ── Layout (inches) ───────────────────────────────────────────
    FIG_W = 10.0
    left_in, right_in = 0.30, 0.15
    content_w = FIG_W - left_in - right_in
    gap_hm = 0.5   # gap between the two heatmaps
    gap_v = 0.30   # vertical gap between blocks
    top_in = 0.35
    legend_h = 0.45
    bottom_in = 0.20

    hm_w = (content_w - gap_hm) / 2.0
    hm_h = hm_w * max(asp(imgs["heatmap_lmnb1"]), asp(imgs["heatmap_lamp1"]))
    q1_h = content_w * asp(imgs["qual_lmnb1"])
    q2_h = content_w * asp(imgs["qual_lamp1"])

    FIG_H = (top_in + hm_h + gap_v + q1_h + gap_v + q2_h
             + gap_v + legend_h + bottom_in)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    def place(im, x_in, y_top_in, w_in, h_in):
        ax = fig.add_axes([x_in / FIG_W,
                           1 - (y_top_in + h_in) / FIG_H,
                           w_in / FIG_W,
                           h_in / FIG_H])
        ax.imshow(im, aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def panel_label(x_in, y_top_in, txt):
        fig.text(x_in / FIG_W, 1 - y_top_in / FIG_H, txt,
                 fontsize=14, fontweight="bold", va="top", ha="left")

    y = top_in
    # a | b heatmaps
    place(imgs["heatmap_lmnb1"], left_in, y, hm_w, hm_h)
    place(imgs["heatmap_lamp1"], left_in + hm_w + gap_hm, y, hm_w, hm_h)
    panel_label(0.06, y, "a")
    panel_label((left_in + hm_w + gap_hm) - 0.24, y, "b")
    y += hm_h + gap_v

    # c LMNB1 qualitative
    place(imgs["qual_lmnb1"], left_in, y, content_w, q1_h)
    panel_label(0.06, y, "c")
    y += q1_h + gap_v

    # d LAMP1 qualitative
    place(imgs["qual_lamp1"], left_in, y, content_w, q2_h)
    panel_label(0.06, y, "d")
    y += q2_h + gap_v

    # shared legend
    leg_ax = fig.add_axes([0.05, bottom_in / FIG_H * 0.4, 0.90, legend_h / FIG_H])
    leg_ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor="black", label=l) for c, l in LEGEND]
    leg_ax.legend(handles=handles, loc="center", ncol=4, fontsize=11,
                  frameon=True, edgecolor="black")

    fig.savefig(str(OUT_PDF), dpi=RENDER_DPI, facecolor="white")
    fig.savefig(str(OUT_PDF.with_suffix(".png")), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {OUT_PDF}")
    print(f"  Saved: {OUT_PDF.with_suffix('.png')}")

    write_manifest(imgs)


if __name__ == "__main__":
    main()
