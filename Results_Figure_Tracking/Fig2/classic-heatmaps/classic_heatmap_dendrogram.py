"""
Classic Cross-Evaluation Heatmap + Dendrogram.

Reads the pre-aggregated 14x14 classic_cross_eval_heatmap.csv (mean Dice)
and generates publication-quality heatmaps with hierarchical clustering
dendrograms, matching the style of megaseg_heatmap_dendrogram.py.

Generates:
  1. fig_classic_crosseval.pdf         — 14x14 with dendrogram ordering
  2. fig_classic_crosseval_radial.pdf   — 14x14 with radial order (Viana et al. 2023)

Usage:
    python classic_heatmap_dendrogram.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, set_link_color_palette
from scipy.spatial.distance import pdist

# ──────────────────────────────────────────────────────────────
# Styling (matches megaseg_heatmap_dendrogram.py)
# ──────────────────────────────────────────────────────────────
HEATMAP_CMAP_BLUE = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)

# RdBu_r diverging: red (high) -> white (mid) -> blue (low)
# Best for cross-eval matrices — highlights generalization structure
HEATMAP_CMAP_RDBU = plt.get_cmap("RdBu_r")

# Default colormap
HEATMAP_CMAP = HEATMAP_CMAP_RDBU

DEND_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"]

# Radial order from Viana et al. Nature 2023 Fig 3d (nucleus -> cell periphery)
RADIAL_ORDER = [
    "SON", "NUP153", "SEC61B", "TOMM20", "RAB5A", "ST6GAL1",
    "CTNNB1", "ATP2A2", "ACTN1", "MYH10", "DSP", "GJA1", "TJP1",
]

# Structures to exclude from the heatmap
EXCLUDE_STRUCTS = {"NPM1"}

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / "classic_cross_eval_heatmap.csv"


# ──────────────────────────────────────────────────────────────
# Load CSV matrix
# ──────────────────────────────────────────────────────────────
def load_matrix(csv_path):
    """Read the 14x14 cross-eval CSV into a numpy matrix and label lists.

    Returns (mean_mat, row_labels, col_labels).
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        col_labels = [c for c in reader.fieldnames if c != "model_struct"]
        row_labels = []
        rows = []
        for r in reader:
            row_labels.append(r["model_struct"])
            rows.append([float(r[c]) for c in col_labels])

    mean_mat = np.array(rows)

    # Filter out excluded structures from both axes
    keep_row = [i for i, r in enumerate(row_labels) if r not in EXCLUDE_STRUCTS]
    keep_col = [j for j, c in enumerate(col_labels) if c not in EXCLUDE_STRUCTS]
    mean_mat = mean_mat[np.ix_(keep_row, keep_col)]
    row_labels = [row_labels[i] for i in keep_row]
    col_labels = [col_labels[j] for j in keep_col]

    return mean_mat, row_labels, col_labels


# ──────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────
def _do_linkage(data):
    clean = np.nan_to_num(data, nan=0.0)
    if clean.shape[0] < 2:
        return None, list(range(clean.shape[0]))
    Z = linkage(pdist(clean, metric="euclidean"), method="ward")
    return Z, list(leaves_list(Z))


def _draw_dendrogram(ax, Z, orientation, color_threshold=None):
    set_link_color_palette(DEND_PALETTE)
    if color_threshold is None:
        color_threshold = 0.7 * Z[-1, 2]
    dendrogram(
        Z, ax=ax, orientation=orientation,
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="#999999",
    )
    set_link_color_palette(None)
    ax.set_axis_off()


# ──────────────────────────────────────────────────────────────
# Heatmap figure
# ──────────────────────────────────────────────────────────────
def plot_heatmap_with_dendrograms(
    mean_mat, row_labels, col_labels,
    title, save_path, vmin=0.0, vmax=1.0,
    fontsize_annot=7, figsize=None,
    symmetric=False, fixed_order=None, cmap=None,
):
    """Heatmap with column dendrogram, matching megaseg_heatmap_dendrogram style."""
    cmap = cmap or HEATMAP_CMAP
    n_rows, n_cols = mean_mat.shape

    # Determine ordering
    if fixed_order is not None:
        row_order = [row_labels.index(s) for s in fixed_order if s in row_labels]
        col_order = [col_labels.index(s) for s in fixed_order if s in col_labels]
        Z_col = None
    elif symmetric:
        avg = np.nan_to_num(mean_mat, nan=0.0)
        sym = (avg + avg.T) / 2.0
        Z_shared, shared_order = _do_linkage(sym)
        Z_col = Z_shared
        row_order = shared_order
        col_order = shared_order
    else:
        Z_row, row_order = _do_linkage(mean_mat)
        Z_col, col_order = _do_linkage(mean_mat.T)

    # Reorder
    mean_ordered = mean_mat[row_order, :][:, col_order]
    row_labels_ordered = [row_labels[i] for i in row_order]
    col_labels_ordered = [col_labels[i] for i in col_order]

    # Figure layout
    if figsize is None:
        w = max(8, n_cols * 1.4 + 3)
        h = max(6, n_rows * 0.55 + 3)
        figsize = (w, h)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.0, 0.02],
        height_ratios=[0.10, 1.0],
        wspace=0.02, hspace=0.02,
    )

    # Column dendrogram
    ax_dend_col = fig.add_subplot(gs[0, 0])
    if Z_col is not None:
        _draw_dendrogram(ax_dend_col, Z_col, "top")
    else:
        ax_dend_col.set_axis_off()

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax_heat.imshow(mean_ordered, cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = mean_ordered[i, j]
            if np.isnan(val):
                continue
            rgba = cmap(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if brightness < 0.55 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=fontsize_annot, color=text_color, fontweight="bold")

    # Axis labels
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(col_labels_ordered, rotation=45, ha="right",
                            fontsize=10, fontweight="bold")
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels_ordered, fontsize=10, fontweight="bold")
    ax_heat.tick_params(axis="both", which="both", length=0)

    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 1])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label("Dice Score", fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print(f"Loading: {INPUT_CSV}")
    mean_mat, row_labels, col_labels = load_matrix(INPUT_CSV)
    print(f"  Matrix: {mean_mat.shape[0]} models x {mean_mat.shape[1]} structures")
    print(f"  Structures: {', '.join(col_labels)}")

    # Generate figures with both colormaps
    cmaps = [
        ("RdBu_r", HEATMAP_CMAP_RDBU, ""),
        ("Blue",    HEATMAP_CMAP_BLUE, "_blue"),
    ]

    for cmap_name, cmap_obj, suffix in cmaps:
        print(f"\n-- {cmap_name} colormap --")

        # Dendrogram-ordered
        print(f"  Generating dendrogram-ordered heatmap ({cmap_name})...")
        plot_heatmap_with_dendrograms(
            mean_mat, row_labels, col_labels,
            title="Classic Cross-Evaluation: Specialist Model Generalization (Dice Score)",
            save_path=str(SCRIPT_DIR / f"fig_classic_crosseval{suffix}.pdf"),
            fontsize_annot=7,
            figsize=(12, 10),
            symmetric=True,
            cmap=cmap_obj,
        )

        # Radial order
        print(f"  Generating radial-order heatmap ({cmap_name})...")
        plot_heatmap_with_dendrograms(
            mean_mat, row_labels, col_labels,
            title="Classic Cross-Evaluation: Radial Order (Viana et al. 2023)",
            save_path=str(SCRIPT_DIR / f"fig_classic_crosseval_radial{suffix}.pdf"),
            fontsize_annot=7,
            figsize=(12, 10),
            fixed_order=RADIAL_ORDER,
            cmap=cmap_obj,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
