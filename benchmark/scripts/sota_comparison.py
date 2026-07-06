#!/usr/bin/env python
"""
SOTA Segmentation Comparison for MegaSeg benchmark.

Loads per-sample results from all methods, applies the standard low-signal
sample filter (bottom 10 per structure removed, 57 samples per structure),
restricts to the shared structure intersection across methods, then generates:
  1. Bar chart: overall Dice per method
  2. Grouped bar chart: per-structure Dice for all methods
  3. Heatmap: Dice[method x structure]
  4. Violin plots: per-sample Dice distributions
  5. Statistical tests: paired Wilcoxon signed-rank (MegaSeg vs. each method)
  6. Combined summary CSV

Usage:
    uv run python scripts/sota_comparison.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "uint8"
FIG_DIR = ROOT / "results" / "uint8" / "figures"

STRUCTURES = [
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]

CLASSIC_STRUCTURES = [
    "ACTN1",
    "CTNNB1",
    "MYH10",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TOMM20",
]


def get_comparison_structures(df: pd.DataFrame) -> list[str]:
    """Return structures present for every active method."""
    methods = _active_methods(df)
    if not methods:
        return STRUCTURES
    shared = set(STRUCTURES)
    for method, _, _ in methods:
        method_structs = set(df.loc[df["method"] == method, "structure"].dropna().unique())
        shared &= method_structs
    return [s for s in STRUCTURES if s in shared]


def describe_comparison(df: pd.DataFrame) -> str:
    """Build a subtitle from the actual shared structure/sample coverage."""
    structures = get_comparison_structures(df)
    if not structures:
        return "(no shared structures across methods)"
    counts = []
    for method, _, _ in _active_methods(df):
        sub = df[(df["method"] == method) & (df["structure"].isin(structures))]
        per_struct = sub.groupby("structure").size()
        if not per_struct.empty:
            counts.append(int(per_struct.min()))
    samples_per = min(counts) if counts else 0
    total = len(structures) * samples_per
    return f"({len(structures)} structures, {samples_per} test samples each = {total} total)"

# Method display order: (key, label, color)
# - Trained methods first (descending by expected Dice), then zero-shot
METHODS = [
    ("megaseg", "MegaSeg (DynUNet)", "#1f77b4"),
    ("nnunet", "nnU-Net v2", "#ff7f0e"),
    ("individual", "Individual DynUNet", "#2ca02c"),
    ("classic_wf", "Classic (aicsseg)", "#17becf"),
    ("otsu", "Otsu", "#7f7f7f"),
    ("stardist", "StarDist", "#9467bd"),
    ("cellpose", "CellPose v4", "#8c564b"),
    ("microsam_lm", "MicroSAM (vit_b_lm)", "#e377c2"),
    ("sam", "SAM (vit_b)", "#d62728"),
]

COLS = [
    "method",
    "structure",
    "sample_id",
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_megaseg() -> pd.DataFrame:
    df = pd.read_csv(RES / "megaseg_dynunet_per_image_dice.csv")
    df["method"] = "megaseg"
    df = df.rename(columns={"test_struct": "structure"})
    return df[COLS]


def load_individual() -> pd.DataFrame:
    df = pd.read_csv(RES / "per_sample_metrics.csv")
    df = df[df["eval_type"] == "individual"].copy()
    df["method"] = "individual"
    df = df.rename(columns={"test_struct": "structure"})
    return df[COLS]


def load_classic_workflow() -> pd.DataFrame | None:
    """Load classic aicssegmentation workflow results (10 of 14 structures)."""
    csv_path = RES / "classic_workflow_metrics.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping classic workflow method")
        return None
    df = pd.read_csv(csv_path)
    df["method"] = "classic_wf"
    df = df.rename(columns={"test_struct": "structure"})
    return df[COLS]


def _load_generic(path: Path, method: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping {method}")
        return None
    df = pd.read_csv(path)
    df["method"] = method
    df = df.rename(columns={"test_struct": "structure"})
    return df[COLS]


def load_all() -> pd.DataFrame:
    frames = [
        load_megaseg(),
        load_individual(),
        _load_generic(RES / "nnunet_metrics.csv", "nnunet"),
        _load_generic(RES / "cellpose_zeroshot_metrics.csv", "cellpose"),
        _load_generic(RES / "stardist_zeroshot_metrics.csv", "stardist"),
        _load_generic(RES / "sam_zeroshot_metrics.csv", "sam"),
        _load_generic(RES / "microsam_b_lm_zeroshot_metrics.csv", "microsam_lm"),
        _load_generic(RES / "otsu_baseline_metrics.csv", "otsu"),
        load_classic_workflow(),
    ]
    return pd.concat([f for f in frames if f is not None], ignore_index=True)


def _active_methods(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return only METHODS entries that have data in df."""
    present = set(df["method"].unique())
    return [(k, l, c) for k, l, c in METHODS if k in present]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-method-structure summary with mean/std Dice."""
    methods = _active_methods(df)
    comparison_structures = get_comparison_structures(df)
    rows = []
    for method, _, _ in methods:
        sub = df[df["method"] == method]
        sub = sub[sub["structure"].isin(comparison_structures)]
        for struct in comparison_structures:
            s = sub[sub["structure"] == struct]["dice"]
            rows.append(
                {
                    "method": method,
                    "structure": struct,
                    "n": len(s),
                    "dice_mean": s.mean() if len(s) > 0 else np.nan,
                    "dice_std": s.std() if len(s) > 0 else np.nan,
                }
            )
        overall = sub["dice"]
        rows.append(
            {
                "method": method,
                "structure": "OVERALL",
                "n": len(overall),
                "dice_mean": overall.mean(),
                "dice_std": overall.std(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------


def run_statistical_tests(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test: MegaSeg vs each other method.

    For each comparison method, we pair samples by (structure, sample_id).
    For each comparison method, we pair samples by (structure, sample_id).
    If a method covers fewer structures, its comparison is restricted to
    the paired samples available.

    Returns a DataFrame with test results and saves to CSV.
    """
    methods = _active_methods(df)
    comparison_structures = get_comparison_structures(df)
    df = df[df["structure"].isin(comparison_structures)].copy()
    megaseg_df = df[df["method"] == "megaseg"].set_index(["structure", "sample_id"])
    results = []

    for method_key, method_label, _ in methods:
        if method_key == "megaseg":
            continue

        other_df = df[df["method"] == method_key].set_index(["structure", "sample_id"])

        # Find paired samples (inner join on index)
        paired = megaseg_df[["dice"]].join(
            other_df[["dice"]], lsuffix="_megaseg", rsuffix="_other", how="inner"
        )

        if len(paired) < 5:
            print(f"  {method_label}: too few paired samples ({len(paired)}), skipping")
            continue

        diff = paired["dice_megaseg"] - paired["dice_other"]
        mean_diff = diff.mean()
        median_diff = diff.median()
        n_paired = len(paired)

        # Wilcoxon signed-rank test (two-sided)
        try:
            stat, p_value = stats.wilcoxon(
                paired["dice_megaseg"], paired["dice_other"], alternative="two-sided"
            )
        except ValueError:
            # All differences are zero
            stat, p_value = 0.0, 1.0

        # Also compute per-structure tests
        struct_results = []
        for struct in comparison_structures:
            ms = megaseg_df.loc[
                megaseg_df.index.get_level_values("structure") == struct, "dice"
            ]
            ot = other_df.loc[
                other_df.index.get_level_values("structure") == struct, "dice"
            ]
            # Align by sample_id
            ms_aligned = ms.droplevel("structure")
            ot_aligned = ot.droplevel("structure")
            common_ids = ms_aligned.index.intersection(ot_aligned.index)
            if len(common_ids) < 5:
                continue
            ms_vals = ms_aligned.loc[common_ids]
            ot_vals = ot_aligned.loc[common_ids]
            try:
                s_stat, s_p = stats.wilcoxon(ms_vals, ot_vals, alternative="two-sided")
            except ValueError:
                s_stat, s_p = 0.0, 1.0
            struct_results.append(
                {
                    "comparison": f"MegaSeg vs {method_label}",
                    "structure": struct,
                    "n_paired": len(common_ids),
                    "mean_diff": (ms_vals - ot_vals).mean(),
                    "wilcoxon_stat": s_stat,
                    "p_value": s_p,
                    "significant_005": s_p < 0.05,
                }
            )

        results.append(
            {
                "comparison": f"MegaSeg vs {method_label}",
                "structure": "OVERALL",
                "n_paired": n_paired,
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "wilcoxon_stat": stat,
                "p_value": p_value,
                "significant_005": p_value < 0.05,
            }
        )
        results.extend(struct_results)

    result_df = pd.DataFrame(results)
    out_csv = out_dir / "statistical_tests.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    # Print overall summary
    overall_results = result_df[result_df["structure"] == "OVERALL"]
    print("\n  Paired Wilcoxon signed-rank tests (MegaSeg vs each method, overall):")
    for _, row in overall_results.iterrows():
        sig = "*" if row["significant_005"] else ""
        print(
            f"    {row['comparison']:40s}: "
            f"n={row['n_paired']:4.0f}, "
            f"mean_diff={row['mean_diff']:+.4f}, "
            f"p={row['p_value']:.2e} {sig}"
        )

    return result_df


# ---------------------------------------------------------------------------
# Figure 1: Overall Dice bar chart
# ---------------------------------------------------------------------------


def plot_overall_bar(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: Path,
    subtitle: str | None = None,
) -> None:
    if subtitle is None:
        subtitle = describe_comparison(df)
    methods = _active_methods(df)
    overall = summary[summary["structure"] == "OVERALL"].copy()
    overall = overall.set_index("method").loc[[m for m, _, _ in methods]]

    # Determine expected n from the data (for annotation threshold)
    expected_n = int(overall["n"].mode().iloc[0]) if len(overall) else 0

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [label for _, label, _ in methods]
    colors = [color for _, _, color in methods]
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        overall["dice_mean"].values,
        yerr=overall["dice_std"].values,
        color=colors,
        width=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.2},
    )

    for bar, val, n in zip(bars, overall["dice_mean"].values, overall["n"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        # Show sample count below the bar label if different from expected
        if n != expected_n:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -0.04,
                f"(n={int(n)})",
                ha="center",
                va="top",
                fontsize=7,
                color="gray",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"SOTA Segmentation Comparison: Overall Dice\n{subtitle}",
        fontsize=12,
        fontweight="bold",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sota_overall_bar.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sota_overall_bar.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 2: Grouped bar chart (per structure x method)
# ---------------------------------------------------------------------------


def plot_per_structure_grouped(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: Path,
    subtitle: str | None = None,
) -> None:
    if subtitle is None:
        subtitle = describe_comparison(df)
    methods = _active_methods(df)
    comparison_structures = get_comparison_structures(df)
    struct_df = summary[summary["structure"] != "OVERALL"]
    n_structs = len(comparison_structures)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = np.arange(n_structs)

    fig, ax = plt.subplots(figsize=(22, 6))
    for i, (method, label, color) in enumerate(methods):
        vals = []
        for struct in comparison_structures:
            row = struct_df[
                (struct_df["method"] == method) & (struct_df["structure"] == struct)
            ]
            val = (
                row["dice_mean"].values[0]
                if len(row) and row["n"].values[0] > 0
                else np.nan
            )
            vals.append(val)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width=width * 0.9, color=color, label=label, alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_structures, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"SOTA Comparison: Per-Structure Dice\n{subtitle}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"sota_per_structure_grouped.{ext}", dpi=200, bbox_inches="tight"
        )
    plt.close(fig)
    print("  Saved sota_per_structure_grouped.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 3: Heatmap (method x structure)
# ---------------------------------------------------------------------------


def plot_heatmap(summary: pd.DataFrame, df: pd.DataFrame, out_dir: Path) -> None:
    methods = _active_methods(df)
    comparison_structures = get_comparison_structures(df)
    struct_df = summary[summary["structure"] != "OVERALL"]
    methods_keys = [m for m, _, _ in methods]
    methods_labels = [lbl for _, lbl, _ in methods]

    grid = np.full((len(methods_keys), len(comparison_structures)), np.nan)
    for i, method in enumerate(methods_keys):
        for j, struct in enumerate(comparison_structures):
            row = struct_df[
                (struct_df["method"] == method) & (struct_df["structure"] == struct)
            ]
            if len(row) and row["n"].values[0] > 0:
                grid[i, j] = row["dice_mean"].values[0]

    fig, ax = plt.subplots(figsize=(18, 6))
    masked_grid = np.ma.masked_invalid(grid)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked_grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(methods_keys)):
        for j in range(len(comparison_structures)):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="gray")
            else:
                color = "white" if val < 0.4 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(range(len(comparison_structures)))
    ax.set_xticklabels(comparison_structures, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods_keys)))
    ax.set_yticklabels(methods_labels, fontsize=10)
    ax.set_title(
        "SOTA Comparison Heatmap: Mean Dice per Method and Structure",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean Dice")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sota_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sota_heatmap.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 4: Violin plots (per-sample Dice distributions)
# ---------------------------------------------------------------------------


def plot_violin(df: pd.DataFrame, out_dir: Path) -> None:
    """Violin plot showing per-sample Dice distribution for each method."""
    methods = _active_methods(df)
    method_keys = [m for m, _, _ in methods]
    method_labels = [l for _, l, _ in methods]
    method_colors = [c for _, _, c in methods]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Gather data in method order
    data = []
    for key in method_keys:
        data.append(df[df["method"] == key]["dice"].values)

    positions = np.arange(len(methods))
    parts = ax.violinplot(
        data, positions=positions, showmeans=True, showmedians=True, showextrema=False
    )

    # Color each violin
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(method_colors[i])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("red")

    # Overlay individual points (jittered) for small datasets
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            positions[i] + jitter,
            vals,
            s=3,
            alpha=0.15,
            color=method_colors[i],
            zorder=1,
        )

    # Annotate mean values
    for i, vals in enumerate(data):
        mean_val = np.mean(vals)
        ax.text(
            positions[i],
            1.02,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(method_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Dice Score (per sample)", fontsize=12)
    ax.set_ylim(-0.05, 1.12)
    ax.set_title(
        "Per-Sample Dice Distributions by Method",
        fontsize=13,
        fontweight="bold",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Legend for mean/median lines
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", lw=1.5, label="Mean"),
        Line2D([0], [0], color="red", lw=1.5, label="Median"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sota_violin.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sota_violin.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 5: Per-structure violin (MegaSeg vs top methods)
# ---------------------------------------------------------------------------


def plot_per_structure_violin(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-structure violin plot for the top 4 methods (trained models)."""
    top_methods = [
        ("megaseg", "MegaSeg", "#1f77b4"),
        ("nnunet", "nnU-Net v2", "#ff7f0e"),
        ("individual", "Indiv. DynUNet", "#2ca02c"),
        ("classic_wf", "Classic (aicsseg)", "#17becf"),
    ]
    # Only include methods present in the data
    present = set(df["method"].unique())
    top_methods = [(k, l, c) for k, l, c in top_methods if k in present]

    comparison_structures = get_comparison_structures(df)
    n_structs = len(comparison_structures)
    n_methods = len(top_methods)

    n_cols = 7
    n_rows = int(np.ceil(max(n_structs, 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 5 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    for idx, struct in enumerate(comparison_structures):
        ax = axes_flat[idx]
        struct_data = []
        colors = []
        labels = []
        for key, label, color in top_methods:
            vals = df[(df["method"] == key) & (df["structure"] == struct)][
                "dice"
            ].values
            struct_data.append(vals if len(vals) > 0 else np.array([np.nan]))
            colors.append(color)
            labels.append(label)

        if any(len(d) > 1 for d in struct_data):
            # Filter out empty arrays for violin
            valid_data = []
            valid_pos = []
            valid_colors = []
            for i, d in enumerate(struct_data):
                if len(d) > 1 and not np.all(np.isnan(d)):
                    valid_data.append(d[~np.isnan(d)])
                    valid_pos.append(i)
                    valid_colors.append(colors[i])

            if valid_data:
                parts = ax.violinplot(
                    valid_data,
                    positions=valid_pos,
                    showmeans=True,
                    showmedians=False,
                    showextrema=False,
                )
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(valid_colors[i])
                    pc.set_alpha(0.7)
                parts["cmeans"].set_color("black")

        ax.set_title(struct, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels([l.split()[0] for l in labels], fontsize=7, rotation=30)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
        if idx % 7 == 0:
            ax.set_ylabel("Dice", fontsize=10)

    fig.suptitle(
        "Per-Structure Dice Distributions: Trained Methods",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"sota_per_structure_violin.{ext}", dpi=200, bbox_inches="tight"
        )
    plt.close(fig)
    print("  Saved sota_per_structure_violin.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 6: Method ranking per structure
# ---------------------------------------------------------------------------


def plot_structure_rankings(
    summary: pd.DataFrame, df: pd.DataFrame, out_dir: Path
) -> None:
    """Heatmap showing rank of each method per structure (1 = best)."""
    methods = _active_methods(df)
    struct_df = summary[(summary["structure"] != "OVERALL") & (summary["n"] > 0)]
    methods_keys = [m for m, _, _ in methods]
    methods_labels = [lbl for _, lbl, _ in methods]

    comparison_structures = get_comparison_structures(df)

    # Build grid of mean Dice
    grid = np.full((len(methods_keys), len(comparison_structures)), np.nan)
    for i, method in enumerate(methods_keys):
        for j, struct in enumerate(comparison_structures):
            row = struct_df[
                (struct_df["method"] == method) & (struct_df["structure"] == struct)
            ]
            if len(row) and row["n"].values[0] > 0:
                grid[i, j] = row["dice_mean"].values[0]

    # Rank per structure (column-wise), handling NaN
    ranks = np.full_like(grid, np.nan)
    for j in range(len(comparison_structures)):
        col = grid[:, j]
        valid_mask = ~np.isnan(col)
        if valid_mask.any():
            valid_vals = col[valid_mask]
            order = np.argsort(-valid_vals)
            for rank, idx in enumerate(order):
                orig_indices = np.where(valid_mask)[0]
                ranks[orig_indices[idx], j] = rank + 1

    fig, ax = plt.subplots(figsize=(18, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    masked_ranks = np.ma.masked_invalid(ranks)
    im = ax.imshow(
        masked_ranks, cmap=cmap, vmin=1, vmax=len(methods_keys), aspect="auto"
    )

    for i in range(len(methods_keys)):
        for j in range(len(comparison_structures)):
            val = ranks[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="gray")
            else:
                color = "white" if val <= 2 else "black"
                ax.text(
                    j,
                    i,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(range(len(comparison_structures)))
    ax.set_xticklabels(comparison_structures, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods_keys)))
    ax.set_yticklabels(methods_labels, fontsize=10)
    ax.set_title(
        "Method Ranking per Structure (1 = best Dice)",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Rank (1=best)")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"sota_rankings.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sota_rankings.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def filter_excluded_samples(df: pd.DataFrame, excluded_csv: Path) -> pd.DataFrame:
    """Remove samples listed in *excluded_csv* from *df*.

    The excluded CSV has columns ``test_struct, sample_id, dice``.
    We match on ``(structure, sample_id)`` in the combined DataFrame.
    """
    excluded = pd.read_csv(excluded_csv)
    excluded = excluded.rename(columns={"test_struct": "structure"})
    exclude_keys = set(zip(excluded["structure"], excluded["sample_id"]))

    before = len(df)
    mask = df.apply(
        lambda r: (r["structure"], r["sample_id"]) not in exclude_keys, axis=1
    )
    df_filtered = df[mask].copy()
    after = len(df_filtered)

    n_structs = excluded["structure"].nunique()
    n_per = len(excluded) // n_structs
    print(
        f"  Filtered: {before} -> {after} rows "
        f"(removed {before - after} across {len(df['method'].unique())} methods, "
        f"bottom {n_per} per structure)"
    )
    return df_filtered


def _run_pipeline(
    df: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
) -> None:
    """Run the full aggregation, stats, and figure pipeline on *df*."""
    comparison_structures = get_comparison_structures(df)
    df = df[df["structure"].isin(comparison_structures)].copy()
    subtitle = describe_comparison(df)

    methods = _active_methods(df)
    print(f"  Structures: {comparison_structures}")
    print(f"  Total rows: {len(df)}")
    for method, label, _ in methods:
        n = len(df[df["method"] == method])
        d = df[df["method"] == method]["dice"].mean()
        print(f"  {label}: {n} samples, Dice={d:.4f}")

    print("\nAggregating...")
    summary = aggregate(df)

    out_csv = out_dir / "sota_comparison_all_methods.csv"
    summary.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    print("\nRunning statistical tests...")
    run_statistical_tests(df, out_dir)

    print("\nGenerating figures...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_overall_bar(summary, df, fig_dir, subtitle=subtitle)
    plot_per_structure_grouped(summary, df, fig_dir, subtitle=subtitle)
    plot_heatmap(summary, df, fig_dir)
    plot_violin(df, fig_dir)
    plot_per_structure_violin(df, fig_dir)
    plot_structure_rankings(summary, df, fig_dir)

    print("\nOverall Dice summary:")
    overall = summary[summary["structure"] == "OVERALL"].set_index("method")
    for method, label, _ in methods:
        row = overall.loc[method]
        print(
            f"  {label:30s}: {row['dice_mean']:.4f} +/- {row['dice_std']:.4f}  (n={int(row['n'])})"
        )


def main():
    print("Loading per-sample results...")
    df = load_all()

    excluded_csv = RES / "filtered" / "excluded_samples.csv"
    if not excluded_csv.exists():
        print(f"\nERROR: {excluded_csv} not found. Run evaluate_benchmark.py filter first.")
        return

    print("\nApplying low-signal sample filter (bottom-10 per structure)...")
    df = filter_excluded_samples(df, excluded_csv)

    out_dir = RES / "filtered"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _run_pipeline(df, out_dir=out_dir, fig_dir=fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
