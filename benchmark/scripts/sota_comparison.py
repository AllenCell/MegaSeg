#!/usr/bin/env python
"""
SOTA Segmentation Comparison for MegaSeg benchmark.

Loads per-sample results from all 7 methods, aggregates by structure and
overall, then generates:
  1. Bar chart: overall Dice per method
  2. Grouped bar chart: per-structure Dice for all methods
  3. Heatmap: Dice[method x structure]
  4. Combined summary CSV

Usage:
    uv run python scripts/sota_comparison.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]

# method display order, label, color
METHODS = [
    ("megaseg", "MegaSeg (DynUNet)", "#1f77b4"),
    ("nnunet", "nnU-Net v2", "#ff7f0e"),
    ("classic", "Classic (aicsseg)", "#2ca02c"),
    ("stardist", "StarDist", "#9467bd"),
    ("cellpose", "CellPose v4", "#8c564b"),
    ("otsu", "Otsu", "#7f7f7f"),
    ("sam", "SAM (vit_b)", "#d62728"),
]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_megaseg() -> pd.DataFrame:
    df = pd.read_csv(RES / "megaseg_dynunet_per_image_dice.csv")
    df["method"] = "megaseg"
    df = df.rename(columns={"test_struct": "structure"})
    return df[
        [
            "method",
            "structure",
            "sample_id",
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy",
        ]
    ]


def load_classic() -> pd.DataFrame:
    df = pd.read_csv(RES / "per_sample_metrics.csv")
    df = df[df["eval_type"] == "individual"].copy()
    df["method"] = "classic"
    df = df.rename(columns={"test_struct": "structure"})
    return df[
        [
            "method",
            "structure",
            "sample_id",
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy",
        ]
    ]


def _load_generic(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = method
    df = df.rename(columns={"test_struct": "structure"})
    return df[
        [
            "method",
            "structure",
            "sample_id",
            "dice",
            "iou",
            "precision",
            "recall",
            "accuracy",
        ]
    ]


def load_all() -> pd.DataFrame:
    frames = [
        load_megaseg(),
        load_classic(),
        _load_generic(RES / "nnunet_metrics.csv", "nnunet"),
        _load_generic(RES / "cellpose_zeroshot_metrics.csv", "cellpose"),
        _load_generic(RES / "stardist_zeroshot_metrics.csv", "stardist"),
        _load_generic(RES / "sam_zeroshot_metrics.csv", "sam"),
        _load_generic(RES / "otsu_baseline_metrics.csv", "otsu"),
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-method-structure summary with mean/std Dice."""
    rows = []
    for method, _, _ in METHODS:
        sub = df[df["method"] == method]
        for struct in STRUCTURES:
            s = sub[sub["structure"] == struct]["dice"]
            rows.append(
                {
                    "method": method,
                    "structure": struct,
                    "n": len(s),
                    "dice_mean": s.mean(),
                    "dice_std": s.std(),
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
# Figure 1: Overall Dice bar chart
# ---------------------------------------------------------------------------


def plot_overall_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    overall = summary[summary["structure"] == "OVERALL"].copy()
    overall = overall.set_index("method").loc[[m for m, _, _ in METHODS]]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [label for _, label, _ in METHODS]
    colors = [color for _, _, color in METHODS]
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

    for bar, val in zip(bars, overall["dice_mean"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "SOTA Segmentation Comparison: Overall Dice (938 test samples)",
        fontsize=13,
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


def plot_per_structure_grouped(summary: pd.DataFrame, out_dir: Path) -> None:
    struct_df = summary[summary["structure"] != "OVERALL"]
    n_structs = len(STRUCTURES)
    n_methods = len(METHODS)
    width = 0.8 / n_methods
    x = np.arange(n_structs)

    fig, ax = plt.subplots(figsize=(20, 6))
    for i, (method, label, color) in enumerate(METHODS):
        vals = []
        for struct in STRUCTURES:
            row = struct_df[
                (struct_df["method"] == method) & (struct_df["structure"] == struct)
            ]
            vals.append(row["dice_mean"].values[0] if len(row) else np.nan)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width=width * 0.9, color=color, label=label, alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STRUCTURES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "SOTA Comparison: Per-Structure Dice (67 test samples each)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
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


def plot_heatmap(summary: pd.DataFrame, out_dir: Path) -> None:
    struct_df = summary[summary["structure"] != "OVERALL"]
    methods_keys = [m for m, _, _ in METHODS]
    methods_labels = [lbl for _, lbl, _ in METHODS]

    grid = np.zeros((len(methods_keys), len(STRUCTURES)))
    for i, method in enumerate(methods_keys):
        for j, struct in enumerate(STRUCTURES):
            row = struct_df[
                (struct_df["method"] == method) & (struct_df["structure"] == struct)
            ]
            grid[i, j] = row["dice_mean"].values[0] if len(row) else np.nan

    fig, ax = plt.subplots(figsize=(18, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(methods_keys)):
        for j in range(len(STRUCTURES)):
            val = grid[i, j]
            color = "white" if val < 0.35 or val > 0.8 else "black"
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

    ax.set_xticks(range(len(STRUCTURES)))
    ax.set_xticklabels(STRUCTURES, rotation=30, ha="right", fontsize=9)
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
# Main
# ---------------------------------------------------------------------------


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading per-sample results...")
    df = load_all()
    print(f"  Total rows: {len(df)}")
    for method, label, _ in METHODS:
        n = len(df[df["method"] == method])
        d = df[df["method"] == method]["dice"].mean()
        print(f"  {label}: {n} samples, Dice={d:.4f}")

    print("\nAggregating...")
    summary = aggregate(df)

    out_csv = RES / "sota_comparison_all_methods.csv"
    summary.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")

    print("\nGenerating figures...")
    plot_overall_bar(summary, FIG_DIR)
    plot_per_structure_grouped(summary, FIG_DIR)
    plot_heatmap(summary, FIG_DIR)

    print("\nOverall Dice summary:")
    overall = summary[summary["structure"] == "OVERALL"].set_index("method")
    for method, label, _ in METHODS:
        row = overall.loc[method]
        print(f"  {label:30s}: {row['dice_mean']:.4f} +/- {row['dice_std']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
