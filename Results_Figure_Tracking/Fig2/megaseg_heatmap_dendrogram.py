"""
MegaSeg Heatmap + Dendrogram: Model Performance Comparison.

Generates two publication-quality figures:
  1. fig_heatmap_models.pdf  — 5 models x 13 structures (no NPM1)
  2. fig_heatmap_crosseval.pdf — 13x13 cross-eval (specialist models x test structures, no NPM1)

Both include hierarchical clustering dendrograms on rows and columns.

Also saves heatmap_summary_stats.csv with mean/std Dice for all (model, structure) pairs.

Usage:
    python megaseg_heatmap_dendrogram.py
"""

import csv
import math
import os
from collections import defaultdict
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
# Color palette inspired by Viana et al., Nature 2023, Fig. 3d
# ──────────────────────────────────────────────────────────────
# Single-color sequential heatmap: light to dark blue
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "light_to_dark_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)

# Dendrogram branch colors (6 distinct cluster colors)
DEND_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"]

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
INPUT_CSV = REPO_ROOT / "benchmark" / "results" / "uint8" / "per_sample_metrics.csv"

# 13 training structures (NPM1 excluded)
STRUCTURES_13 = [
    "ACTN1", "ATP2A2", "CTNNB1", "DSP", "GJA1", "MYH10",
    "NUP153", "RAB5A", "SEC61B", "SON", "ST6GAL1", "TJP1", "TOMM20",
]

# Radial order from Viana et al. Nature 2023 Fig 3d (nucleus -> cell periphery)
RADIAL_ORDER = [
    "SON", "NUP153", "SEC61B", "TOMM20", "RAB5A", "ST6GAL1",
    "CTNNB1", "ATP2A2", "ACTN1", "MYH10", "DSP", "GJA1", "TJP1",
]

# Gene -> structure ("organelle") name, Viana et al. Nature 2023, Fig. 1a
GENE_TO_STRUCTURE = {
    "ACTN1":   "Actin bundles",
    "ATP2A2":  "ER (SERCA)",
    "CTNNB1":  "Adherens junctions",
    "DSP":     "Desmosomes",
    "GJA1":    "Gap junctions",
    "MYH10":   "Actomyosin bundles",
    "NUP153":  "Nuclear pores",
    "RAB5A":   "Endosomes",
    "SEC61B":  "ER (SEC61β)",
    "SON":     "Nuclear speckles",
    "ST6GAL1": "Golgi",
    "TJP1":    "Tight junctions",
    "TOMM20":  "Mitochondria",
}

# Number of lowest Dice scores to drop per group (67 -> 57 samples)
DROP_LOWEST = 10

# Model display names and their (eval_type, backbone) keys
MODEL_CONFIGS = [
    ("Individual",           "individual", None),
    ("MegaSeg\nDynUNet",     "megaseg",    "dynunet"),
    ("MegaSeg\nAttentionUNet", "megaseg",  "attention_unet"),
    ("MegaSeg\nSwinUNETR",   "megaseg",    "swin_unetr"),
    ("MegaSeg\nUNet",        "megaseg",    "unet"),
]


# ──────────────────────────────────────────────────────────────
# Step 1: Load and aggregate per-sample Dice scores
# ──────────────────────────────────────────────────────────────
def load_per_sample_csv(csv_path):
    """Read per_sample_metrics.csv using csv module. Returns list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_dynunet_drop_samples(rows):
    """
    For each structure, find the DROP_LOWEST sample_ids with the lowest Dice
    in the megaseg/dynunet case. Returns dict: structure -> set of sample_ids to drop.
    """
    # Collect per-structure Dice for megaseg dynunet (matched: model_struct == test_struct)
    struct_samples = defaultdict(list)
    for row in rows:
        if row["eval_type"] == "megaseg" and row["backbone"] == "dynunet" and row["model_struct"] == row["test_struct"]:
            try:
                struct_samples[row["test_struct"]].append((float(row["dice"]), row["sample_id"]))
            except (ValueError, KeyError):
                continue

    drop_ids = {}
    for struct, samples in struct_samples.items():
        samples.sort(key=lambda x: x[0])  # sort by dice ascending
        drop_ids[struct] = {s[1] for s in samples[:DROP_LOWEST]}
        print(f"  {struct}: dropping {len(drop_ids[struct])} lowest-scoring samples")
    return drop_ids


def aggregate_dice(rows, drop_ids):
    """
    Aggregate per-sample Dice into mean/std per (eval_type, backbone, model_struct, test_struct).
    Drops sample_ids identified as lowest-10 from megaseg dynunet per structure.
    Returns dict: (eval_type, backbone, model_struct, test_struct) -> (mean, std, n).
    """
    buckets = defaultdict(list)
    dropped = 0
    for row in rows:
        # Drop samples by test_struct matching the dynunet drop list
        test_struct = row["test_struct"]
        if test_struct in drop_ids and row["sample_id"] in drop_ids[test_struct]:
            dropped += 1
            continue

        key = (row["eval_type"], row["backbone"], row["model_struct"], row["test_struct"])
        try:
            buckets[key].append(float(row["dice"]))
        except (ValueError, KeyError):
            continue

    print(f"  Dropped {dropped} per-sample rows total (lowest {DROP_LOWEST} dynunet samples per structure)")

    stats = {}
    for key, values in buckets.items():
        arr = np.array(values)
        stats[key] = (float(np.mean(arr)), float(np.std(arr)), len(arr))
    return stats


# ──────────────────────────────────────────────────────────────
# Step 2: Build matrices
# ──────────────────────────────────────────────────────────────
def build_model_comparison_matrix(stats):
    """
    Build 13-structure x 5-model matrix of mean Dice scores.
    Returns (mean_matrix, std_matrix, row_labels, col_labels).
    """
    n_structs = len(STRUCTURES_13)
    n_models = len(MODEL_CONFIGS)
    mean_mat = np.full((n_structs, n_models), np.nan)
    std_mat = np.full((n_structs, n_models), np.nan)

    for j, (display_name, eval_type, backbone) in enumerate(MODEL_CONFIGS):
        for i, struct in enumerate(STRUCTURES_13):
            # For individual: model_struct == test_struct == struct
            # For megaseg: model_struct == test_struct == struct
            key = (eval_type, backbone or "", struct, struct)
            if key in stats:
                mean_mat[i, j] = stats[key][0]
                std_mat[i, j] = stats[key][1]

    row_labels = list(STRUCTURES_13)
    col_labels = [cfg[0] for cfg in MODEL_CONFIGS]
    return mean_mat, std_mat, row_labels, col_labels


def build_crosseval_matrix(stats):
    """
    Build 13x13 cross-eval matrix: rows = specialist model, cols = test structure (no NPM1).
    Returns (mean_matrix, std_matrix, row_labels, col_labels).
    """
    n = len(STRUCTURES_13)
    mean_mat = np.full((n, n), np.nan)
    std_mat = np.full((n, n), np.nan)

    for i, model_struct in enumerate(STRUCTURES_13):
        for j, test_struct in enumerate(STRUCTURES_13):
            key = ("cross_eval", "", model_struct, test_struct)
            if key in stats:
                mean_mat[i, j] = stats[key][0]
                std_mat[i, j] = stats[key][1]

    return mean_mat, std_mat, list(STRUCTURES_13), list(STRUCTURES_13)


# ──────────────────────────────────────────────────────────────
# Step 3: Plotting helpers
# ──────────────────────────────────────────────────────────────
def _do_linkage(data):
    """Hierarchical clustering on rows of data. Returns (linkage_matrix, leaf_order)."""
    clean = np.nan_to_num(data, nan=0.0)
    if clean.shape[0] < 2:
        return None, list(range(clean.shape[0]))
    Z = linkage(pdist(clean, metric="euclidean"), method="ward")
    return Z, list(leaves_list(Z))


def _draw_dendrogram(ax, Z, orientation, color_threshold=None):
    """Draw a dendrogram with colored branches."""
    set_link_color_palette(DEND_PALETTE)
    if color_threshold is None:
        # Auto: use 70% of max distance to get coloured sub-clusters
        color_threshold = 0.7 * Z[-1, 2]
    dendrogram(
        Z, ax=ax, orientation=orientation,
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="#999999",
    )
    set_link_color_palette(None)  # reset
    ax.set_axis_off()


def plot_heatmap_with_dendrograms(
    mean_mat, std_mat, row_labels, col_labels,
    title, save_path, vmin=0.0, vmax=1.0,
    annotate_std=True, figsize=None, fontsize_annot=8,
    symmetric=False, row_cluster_col=None, fixed_order=None,
    label_map=None, xlabel=None, ylabel=None,
):
    """
    Plot heatmap with colored dendrograms on left (rows) and top (columns).
    Uses blue-white-red palette (Viana et al. Nature 2023 style).
    If symmetric=True, rows and columns share the same clustering order.
    """
    n_rows, n_cols = mean_mat.shape

    # Cluster / order
    if fixed_order is not None:
        # Use a predetermined order for both rows and columns
        # Map fixed_order labels to indices in the original label lists
        row_order = [row_labels.index(s) for s in fixed_order if s in row_labels]
        col_order = [col_labels.index(s) for s in fixed_order if s in col_labels]
        Z_row, Z_col = None, None
    elif symmetric:
        # Square matrix: cluster once, apply same order to both axes
        avg = np.nan_to_num(mean_mat, nan=0.0)
        sym = (avg + avg.T) / 2.0
        Z_shared, shared_order = _do_linkage(sym)
        Z_row, row_order = Z_shared, shared_order
        Z_col, col_order = Z_shared, shared_order
    elif row_cluster_col is not None:
        # Cluster rows using only the specified column index
        Z_row, row_order = _do_linkage(mean_mat[:, row_cluster_col:row_cluster_col+1])
        Z_col, col_order = _do_linkage(mean_mat.T)
    else:
        Z_row, row_order = _do_linkage(mean_mat)
        Z_col, col_order = _do_linkage(mean_mat.T)

    # Reorder
    mean_ordered = mean_mat[row_order, :][:, col_order]
    std_ordered = std_mat[row_order, :][:, col_order]
    row_labels_ordered = [row_labels[i] for i in row_order]
    col_labels_ordered = [col_labels[i] for i in col_order]

    # Optional gene -> structure ("organelle") name remap for tick labels
    if label_map is not None:
        row_labels_ordered = [label_map.get(l, l) for l in row_labels_ordered]
        col_labels_ordered = [label_map.get(l, l) for l in col_labels_ordered]

    # Figure layout
    if figsize is None:
        w = max(8, n_cols * 1.4 + 3)
        h = max(6, n_rows * 0.55 + 3)
        figsize = (w, h)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.0, 0.02],
        height_ratios=[0.15, 1.0],
        wspace=0.02, hspace=0.01,
    )

    # ── Column dendrogram (top only) ──
    ax_dend_col = fig.add_subplot(gs[0, 0])
    if Z_col is not None:
        _draw_dendrogram(ax_dend_col, Z_col, "top")
    else:
        ax_dend_col.set_axis_off()

    # ── Heatmap ──
    ax_heat = fig.add_subplot(gs[1, 0])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = HEATMAP_CMAP

    ax_heat.imshow(mean_ordered, cmap=cmap_obj, norm=norm,
                   aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = mean_ordered[i, j]
            if np.isnan(val):
                continue
            std_val = std_ordered[i, j]

            # Text color: white on dark (blue/red), black on light (white)
            rgba = cmap_obj(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if brightness < 0.55 else "black"

            if annotate_std and not np.isnan(std_val):
                txt = f"{val:.2f}\n\u00b1{std_val:.2f}"
            else:
                txt = f"{val:.2f}"

            ax_heat.text(j, i, txt, ha="center", va="center",
                        fontsize=fontsize_annot, color=text_color, fontweight="bold")

    # Axis labels
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(col_labels_ordered, rotation=45, ha="right",
                            fontsize=14, fontweight="bold")
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels_ordered, fontsize=14, fontweight="bold")
    ax_heat.tick_params(axis="both", which="both", length=0)

    # Axis titles (e.g. which axis is the trained vs the evaluation structure)
    if xlabel:
        ax_heat.set_xlabel(xlabel, fontsize=16, fontweight="bold", labelpad=10)
    if ylabel:
        ax_heat.set_ylabel(ylabel, fontsize=16, fontweight="bold", labelpad=10)

    # ── Colorbar ──
    ax_cbar = fig.add_subplot(gs[1, 1])
    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label("Dice Score", fontsize=14, fontweight="bold")
    cb.ax.tick_params(labelsize=12)

    # Title — placed on dendrogram axes to eliminate gap
    ax_dend_col.set_title(title, fontsize=18, fontweight="bold", pad=12)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    print(f"Saved: {save_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Step 4: Save summary CSV
# ──────────────────────────────────────────────────────────────
def save_summary_csv(stats, csv_path):
    """Save aggregated mean/std Dice for all (model, structure) combos."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_type", "backbone", "model_struct", "test_struct",
                         "mean_dice", "std_dice", "n_samples"])
        for (eval_type, backbone, model_struct, test_struct), (mean, std, n) in sorted(stats.items()):
            writer.writerow([eval_type, backbone, model_struct, test_struct,
                            f"{mean:.6f}", f"{std:.6f}", n])


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print(f"Loading: {INPUT_CSV}")
    rows = load_per_sample_csv(INPUT_CSV)
    print(f"  {len(rows)} per-sample rows loaded")

    # Filter to relevant eval types
    relevant = [r for r in rows if r["eval_type"] in ("individual", "megaseg", "cross_eval")
                and r["model_struct"] != "test_new"]
    print(f"  {len(relevant)} rows after filtering (individual/megaseg/cross_eval, no test_new)")

    print("\nIdentifying lowest-scoring dynunet samples to drop:")
    drop_ids = find_dynunet_drop_samples(relevant)

    stats = aggregate_dice(relevant, drop_ids)
    print(f"  {len(stats)} unique (eval_type, backbone, model_struct, test_struct) combos")

    # ── Figure 1: Model comparison heatmap (13 structs x 5 models) ──
    print("\nBuilding model comparison heatmap...")
    mean_mat, std_mat, row_labels, col_labels = build_model_comparison_matrix(stats)

    # Print matrix for verification
    print("\nModel comparison matrix (mean Dice):")
    header = "".rjust(10) + "  ".join(c.replace("\n", " ").ljust(12) for c in col_labels)
    print(header)
    for i, rl in enumerate(row_labels):
        vals = "  ".join(f"{mean_mat[i,j]:.4f}".ljust(12) if not np.isnan(mean_mat[i,j]) else "NaN".ljust(12)
                        for j in range(len(col_labels)))
        print(f"{rl:>10}  {vals}")

    plot_heatmap_with_dendrograms(
        mean_mat, std_mat, row_labels, col_labels,
        title="Dice Score: Individual vs. MegaSeg Models (13 Structures)",
        save_path=str(SCRIPT_DIR / "fig_heatmap_models.pdf"),
        annotate_std=True, fontsize_annot=11,
        figsize=(12, 11),
        row_cluster_col=0,  # cluster rows using Individual model scores only
        label_map=GENE_TO_STRUCTURE,  # remaps structure rows; model cols unchanged
        xlabel="Model",
        ylabel="Structure",
    )

    # ── Figure 2: Cross-eval heatmap (13x13, no NPM1) ──
    print("\nBuilding cross-eval heatmap (13x13, no NPM1)...")
    ce_mean, ce_std, ce_rows, ce_cols = build_crosseval_matrix(stats)

    plot_heatmap_with_dendrograms(
        ce_mean, ce_std, ce_rows, ce_cols,
        title="Cross-Evaluation: Specialist Model Generalization (Dice Score)",
        save_path=str(SCRIPT_DIR / "fig_heatmap_crosseval.pdf"),
        annotate_std=True, fontsize_annot=10,
        figsize=(16, 12),
        symmetric=True,
        label_map=GENE_TO_STRUCTURE,
        xlabel="Evaluation structure",
        ylabel="Specialist model (trained on structure)",
    )

    # ── Figure 3: Cross-eval with radial order (Viana et al. Nature 2023 Fig 3d) ──
    print("\nBuilding cross-eval heatmap with radial order (Viana et al.)...")
    plot_heatmap_with_dendrograms(
        ce_mean, ce_std, ce_rows, ce_cols,
        title="Cross-Evaluation: Radial Order (Viana et al. 2023)",
        save_path=str(SCRIPT_DIR / "fig_heatmap_crosseval_radial.pdf"),
        annotate_std=True, fontsize_annot=10,
        figsize=(16, 12),
        fixed_order=RADIAL_ORDER,
    )

    # ── Save summary CSV ──
    csv_out = SCRIPT_DIR / "heatmap_summary_stats.csv"
    save_summary_csv(stats, csv_out)
    print(f"\nSaved summary: {csv_out}")


if __name__ == "__main__":
    main()
