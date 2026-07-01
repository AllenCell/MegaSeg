"""
Fig 1c — Structure-agnostic training schematic + architecture comparison.

Self-contained regeneration script. All data are embedded below (no external
files), so it reproduces `fig1c_training_and_backbones.pdf` in this folder.

Design notes:
  - Left schematic uses a colorblind-safe muted palette (Paul Tol) to
    distinguish the abstract training flow; text color auto-picks dark/white
    for legibility on each box.
  - Right bar chart is monochrome (publication standard): grey bars with the
    MegaSeg backbone (DynUNet) emphasised in charcoal, since the backbones are
    already labelled on the axis and color would only decorate.

Usage:
    python fig1c_training_and_backbones.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Nature-style rcParams (Arial, editable PDF text) ──────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_PDF = Path(__file__).parent / "fig1c_training_and_backbones.pdf"

# Real mean Dice scores (13 training structures, excluding NPM1).
BACKBONES_DATA = {
    "DynUNet":   0.9092,
    "SwinUNETR": 0.9041,
    "Attn-UNet": 0.8918,
    "UNet":      0.8703,
}
SPECIALIST_MEAN = 0.7733


def readable_text(hexcolor):
    """Return a dark or white text color for legibility on a given fill."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b  # perceived brightness 0-255
    return "#1A1A1A" if luminance > 150 else "white"


def main():
    fig = plt.figure(figsize=(7.2, 2.8), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.25,
                           left=0.03, right=0.97, top=0.88, bottom=0.08)

    # ── Left: Structure-agnostic training concept ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Structure-agnostic training", fontsize=7,
                 fontweight="bold", pad=6)

    # Mixed-structure training patches (abstract A, B, C, ..., N).
    # Colorblind-safe muted palette (Paul Tol): teal / cyan / sand / rose.
    patch_rows = [("A", "#44AA99"), ("B", "#88CCEE"), ("C", "#DDCC77"),
                  ("...", None), ("N", "#CC6677")]
    for i, (lbl, c) in enumerate(patch_rows):
        y = 5.0 - i * 0.8
        if lbl == "...":
            ax.text(1.6, y, "...", ha="center", va="center",
                    fontsize=9, color="#999999")
            continue
        rect = FancyBboxPatch((0.2, y - 0.25), 2.8, 0.5,
                               boxstyle="round,pad=0.05",
                               facecolor=c, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(1.6, y, f"Structure {lbl}", ha="center", va="center",
                fontsize=5, fontweight="bold", color=readable_text(c))

    ax.text(1.6, 0.5, "Mixed batches\n(no structure label)", ha="center",
            va="center", fontsize=5, color="#555555", style="italic")

    # Arrow to model
    ax.annotate("", xy=(4.2, 2.8), xytext=(3.2, 2.8),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1))

    # Generic 3D segmentation backbone box
    rect = FancyBboxPatch((4.3, 1.2), 3.4, 3.2, boxstyle="round,pad=0.15",
                          facecolor="#4477AA", edgecolor="#2F5577", linewidth=1)
    ax.add_patch(rect)
    ax.text(6.0, 3.5, "Any 3D\nSegmentation\nBackbone", ha="center",
            va="center", fontsize=7, fontweight="bold", color="white",
            linespacing=1.2)

    # Swappable backbone labels inside box
    for i, bb in enumerate(["DynUNet", "SwinUNETR", "Attn-UNet", "UNet"]):
        y = 2.4 - i * 0.38
        rect_bb = FancyBboxPatch((4.7, y - 0.13), 2.6, 0.26,
                                 boxstyle="round,pad=0.03",
                                 facecolor="white", edgecolor="#D7E2EA",
                                 linewidth=0.4, alpha=0.9)
        ax.add_patch(rect_bb)
        ax.text(6.0, y, bb, ha="center", va="center",
                fontsize=4.5, color="#4F6E84", fontweight="bold")

    # Arrow to output
    ax.annotate("", xy=(8.8, 2.8), xytext=(7.8, 2.8),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1))

    # Shared feature space output
    rect = FancyBboxPatch((8.9, 1.6), 2.8, 2.4, boxstyle="round,pad=0.1",
                          facecolor="#E2ECE0", edgecolor="#86A578", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(10.3, 3.2, "Shared\nfeature space", ha="center", va="center",
            fontsize=6, fontweight="bold", color="#5A7A4C", linespacing=1.2)
    ax.text(10.3, 2.1, "Segments any\nmorphology", ha="center", va="center",
            fontsize=5, color="#7B9A6E", style="italic")

    # ── Right: Backbone comparison bar chart ──
    ax2 = fig.add_subplot(gs[0, 1])
    names = list(BACKBONES_DATA.keys())
    scores = list(BACKBONES_DATA.values())

    # Monochrome (publication standard): neutral grey bars, DynUNet in charcoal.
    bb_bar_colors = ["#404040" if n == "DynUNet" else "#BBBBBB" for n in names]

    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, scores, height=0.55, color=bb_bar_colors,
                    edgecolor="white", linewidth=0.5)

    # Specialist baseline line + label (horizontal, in the SwinUNETR/Attn-UNet gap)
    ax2.axvline(SPECIALIST_MEAN, color="#555555", linewidth=1.0,
                linestyle="--", zorder=0)
    ax2.text(SPECIALIST_MEAN + 0.004, 1.5,
             f"Specialist model mean ({SPECIALIST_MEAN:.3f})",
             ha="left", va="center", fontsize=7, color="#555555",
             fontweight="bold")

    # Score labels on bars
    for i, score in enumerate(scores):
        ax2.text(score + 0.005, i, f"{score:.3f}",
                 ha="left", va="center", fontsize=5.5, fontweight="bold")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=6, fontweight="bold")
    ax2.set_xlim(0.72, 0.94)
    ax2.set_xlabel("Mean Dice Score", fontsize=6.5, fontweight="bold")
    ax2.set_title("Architecture comparison", fontsize=7, fontweight="bold", pad=6)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=5.5)

    fig.savefig(str(OUT_PDF), dpi=300, facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
