"""
f2_new — Quantitative comparison figure.

Combines three panels:
  a) Individual vs MegaSeg backbone heatmap (existing fig_heatmap_models)
  b) Cross-evaluation generalization heatmap (existing fig_heatmap_crosseval)
  c) SOTA methods comparison heatmap (NEW — from sota_comparison_all_methods.csv)

All panels use the same blue sequential colormap and dendrogram styling.

Usage:
    python fig_f2_new.py
    python fig_f2_new.py --only sota     # just the SOTA panel
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, set_link_color_palette
from scipy.spatial.distance import pdist

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULTS = REPO_ROOT / "benchmark" / "results"
SOTA_CSV = RESULTS / "uint8" / "filtered" / "sota_comparison_all_methods.csv"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)
DEND_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"]

GENE_TO_STRUCTURE = {
    "ACTN1": "Actin bundles",
    "ATP2A2": "ER (SERCA)",
    "CTNNB1": "Adherens junctions",
    "DSP": "Desmosomes",
    "GJA1": "Gap junctions",
    "MYH10": "Actomyosin bundles",
    "NUP153": "Nuclear pores",
    "RAB5A": "Endosomes",
    "SEC61B": "ER (SEC61β)",
    "SON": "Nuclear speckles",
    "ST6GAL1": "Golgi",
    "TJP1": "Tight junctions",
    "TOMM20": "Mitochondria",
}

METHOD_LABELS = {
    "classic_wf": "Classic\n(aicsseg)",
    "megaseg": "MegaSeg\n(DynUNet)",
    "nnunet": "nnU-Net v2",
    "individual": "Individual\nDynUNet",
    "otsu": "Otsu",
    "stardist": "StarDist",
    "cellpose": "CellPose v4",
    "microsam_lm": "MicroSAM\n(vit_b_lm)",
    "sam": "SAM\n(vit_b)",
}

METHOD_ORDER = [
    "classic_wf", "megaseg", "nnunet", "individual",
    "otsu", "cellpose", "stardist", "microsam_lm", "sam",
]


def build_sota_heatmap(out_path):
    """SOTA methods × structures Dice heatmap with dendrogram."""
    df = pd.read_csv(SOTA_CSV)
    df = df[df["structure"] != "OVERALL"]

    structures = sorted(df["structure"].unique())
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    n_methods = len(methods)
    n_structs = len(structures)

    grid = np.full((n_methods, n_structs), np.nan)
    std_grid = np.full_like(grid, np.nan)
    for i, method in enumerate(methods):
        for j, struct in enumerate(structures):
            row = df[(df["method"] == method) & (df["structure"] == struct)]
            if not row.empty:
                grid[i, j] = row["dice_mean"].iloc[0]
                std_grid[i, j] = row["dice_std"].iloc[0]

    struct_labels = [GENE_TO_STRUCTURE.get(s, s) for s in structures]
    method_labels = [METHOD_LABELS.get(m, m) for m in methods]

    # Cluster columns (structures) by Dice profile
    valid_grid = np.nan_to_num(grid, nan=0.0)
    col_dist = pdist(valid_grid.T, metric="euclidean")
    col_link = linkage(col_dist, method="ward")
    col_order = leaves_list(col_link)

    # Keep method order fixed (trained → zero-shot, already meaningful)
    row_order = list(range(n_methods))

    grid_ordered = grid[np.ix_(row_order, col_order)]
    std_ordered = std_grid[np.ix_(row_order, col_order)]
    method_labels_ordered = [method_labels[i] for i in row_order]
    struct_labels_ordered = [struct_labels[i] for i in col_order]

    # Figure layout: column dendrogram on top, colorbar on right
    fig = plt.figure(figsize=(20, 9), facecolor="white")
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.0, 0.02],
        height_ratios=[0.15, 1.0],
        wspace=0.02, hspace=0.01,
    )

    # Column dendrogram (top)
    ax_dend_col = fig.add_subplot(gs[0, 0])
    set_link_color_palette(DEND_PALETTE)
    dendrogram(col_link, ax=ax_dend_col,
               above_threshold_color="#999999",
               color_threshold=0.7 * col_link[-1, 2],
               no_labels=True)
    set_link_color_palette(None)
    ax_dend_col.set_axis_off()

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax_heat.imshow(grid_ordered, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    for i in range(n_methods):
        for j in range(n_structs):
            val = grid_ordered[i, j]
            if np.isnan(val):
                ax_heat.text(j, i, "N/A", ha="center", va="center",
                             fontsize=7, color="gray")
                continue
            rgba = HEATMAP_CMAP(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if brightness < 0.55 else "black"
            std_val = std_ordered[i, j]
            label = f"{val:.2f}\n±{std_val:.2f}" if not np.isnan(std_val) else f"{val:.2f}"
            ax_heat.text(j, i, label,
                         ha="center", va="center",
                         fontsize=13, fontweight="bold", color=text_color)

    ax_heat.set_xticks(range(n_structs))
    ax_heat.set_xticklabels(struct_labels_ordered, fontsize=14, fontweight="bold",
                            rotation=45, ha="right")
    ax_heat.set_yticks(range(n_methods))
    ax_heat.set_yticklabels(method_labels_ordered, fontsize=14, fontweight="bold")

    # Colorbar
    cbar_ax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), cax=cbar_ax)
    cbar.set_label("Dice Score", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    ax_dend_col.set_title("SOTA Method Comparison: Mean Dice Score (n=57 per structure)",
                          fontsize=18, fontweight="bold", pad=12)

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="f2_new: quantitative comparison panels")
    parser.add_argument("--only", choices=["sota"], help="Generate only one panel")
    args = parser.parse_args()

    if args.only in (None, "sota"):
        print("\nGenerating SOTA heatmap...")
        build_sota_heatmap(SCRIPT_DIR / "fig_sota_heatmap.pdf")

    print("\nDone!")


if __name__ == "__main__":
    main()
