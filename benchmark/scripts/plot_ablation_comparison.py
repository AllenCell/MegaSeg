#!/usr/bin/env python
"""Generate comparison figures for ablation diversity analysis.

Creates:
1. Bar chart: Dice by structure, grouped by model (4/7/MegaSeg-13)
2. Seen vs held-out scatter/bar
3. Bottleneck health comparison
4. Overall Dice vs number of training structures
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ablation"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Training structure membership
TRAINING_STRUCTS = {
    "2_structs": {"TOMM20", "TJP1"},
    "4_structs": {"ACTN1", "ATP2A2", "CTNNB1", "MYH10"},
    "7_structs": {"ACTN1", "ATP2A2", "CTNNB1", "MYH10", "RAB5A", "ST6GAL1", "TOMM20"},
    "14_structs": {
        "AAVS1",
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
    },
    "megaseg_pretrained": {
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
    },
}

MODEL_LABELS = {
    "2_structs": "2-struct (scratch)",
    "4_structs": "4-struct (scratch)",
    "7_structs": "7-struct (scratch)",
    "14_structs": "14-struct (scratch)",
    "megaseg_pretrained": "MegaSeg-13 (pretrained)",
}

MODEL_COLORS = {
    "2_structs": "#55A868",
    "4_structs": "#4C72B0",
    "7_structs": "#DD8452",
    "14_structs": "#8172B2",
    "megaseg_pretrained": "#C44E52",
}

N_TRAIN_STRUCTS = {
    "2_structs": 2,
    "4_structs": 4,
    "7_structs": 7,
    "14_structs": 14,
    "megaseg_pretrained": 13,
}


def load_all_results():
    """Load per-structure metrics for all available models."""
    frames = {}
    for model_key in MODEL_LABELS:
        path = RESULTS_DIR / f"metrics_{model_key}_per_structure.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = model_key
            df["model_label"] = MODEL_LABELS[model_key]
            df["n_train"] = N_TRAIN_STRUCTS[model_key]
            frames[model_key] = df
    return frames


def load_all_health():
    """Load bottleneck health for all available models."""
    frames = {}
    for model_key in MODEL_LABELS:
        path = RESULTS_DIR / f"health_{model_key}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = model_key
            df["model_label"] = MODEL_LABELS[model_key]
            frames[model_key] = df
    return frames


def plot_dice_by_structure(frames):
    """Bar chart: Dice by structure, grouped by model."""
    all_df = pd.concat(frames.values(), ignore_index=True)
    structures = sorted(all_df["structure"].unique())
    models = [k for k in MODEL_LABELS if k in frames]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(structures))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        df = frames[model]
        vals = []
        errs = []
        for s in structures:
            row = df[df["structure"] == s]
            if len(row) > 0:
                vals.append(row["dice_mean"].values[0])
                errs.append(row["dice_std"].values[0])
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            capsize=2,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Structure", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title(
        "Segmentation Performance by Structure and Training Diversity", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    # Mark held-out structures with * for each model
    mark_models = [k for k in ["2_structs", "4_structs", "7_structs"] if k in frames]
    for i, s in enumerate(structures):
        for j, model in enumerate(mark_models):
            if s not in TRAINING_STRUCTS[model]:
                offset_x = (j - len(mark_models) / 2 + 0.5) * 0.2
                ax.annotate(
                    "*",
                    (i + offset_x, 0.02),
                    fontsize=8,
                    color=MODEL_COLORS[model],
                    ha="center",
                    fontweight="bold",
                )

    fig.tight_layout()
    out = FIGURES_DIR / "dice_by_structure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_seen_vs_heldout(frames):
    """Bar chart comparing seen vs held-out Dice for each model."""
    models = [
        k
        for k in [
            "2_structs",
            "4_structs",
            "7_structs",
            "14_structs",
            "megaseg_pretrained",
        ]
        if k in frames
    ]
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        df = frames[model]
        seen = df[df["seen"] == True]["dice_mean"]
        heldout = df[df["seen"] == False]["dice_mean"]

        data = [seen.values, heldout.values]
        labels = [f"Seen (n={len(seen)})", f"Held-out (n={len(heldout)})"]
        colors = ["#55A868", "#C44E52"]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for i, d in enumerate(data):
            jitter = np.random.normal(0, 0.03, len(d))
            ax.scatter(
                np.full_like(d, i + 1) + jitter,
                d,
                color="black",
                alpha=0.5,
                s=20,
                zorder=3,
            )

        ax.set_ylabel("Dice Score" if model == models[0] else "", fontsize=12)
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

        # Annotate means
        for i, d in enumerate(data):
            mean_val = np.mean(d)
            ax.annotate(
                f"mean={mean_val:.3f}",
                (i + 1, mean_val + 0.03),
                ha="center",
                fontsize=9,
                color="black",
            )

    fig.suptitle("Seen vs Held-out Generalization", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "seen_vs_heldout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_dice_vs_num_structures(frames):
    """Scatter: overall/seen/held-out Dice vs number of training structures."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_key, df in frames.items():
        n = N_TRAIN_STRUCTS[model_key]
        overall_mean = df["dice_mean"].mean()
        seen = df[df["seen"] == True]["dice_mean"]
        heldout = df[df["seen"] == False]["dice_mean"]

        ax.scatter(
            n,
            overall_mean,
            s=120,
            color=MODEL_COLORS[model_key],
            marker="o",
            zorder=3,
            label=f"{MODEL_LABELS[model_key]} (overall)",
        )

        if len(seen) > 0:
            ax.scatter(
                n - 0.15,
                seen.mean(),
                s=80,
                color=MODEL_COLORS[model_key],
                marker="^",
                zorder=3,
                alpha=0.7,
            )
        if len(heldout) > 0:
            ax.scatter(
                n + 0.15,
                heldout.mean(),
                s=80,
                color=MODEL_COLORS[model_key],
                marker="v",
                zorder=3,
                alpha=0.7,
            )

    # Custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Overall",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Seen structures",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Held-out structures",
        ),
    ]
    for mk, label in MODEL_LABELS.items():
        if mk in frames:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=MODEL_COLORS[mk],
                    markersize=10,
                    label=label,
                )
            )

    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlabel("Number of Training Structures", fontsize=12)
    ax.set_ylabel("Mean Dice Score", fontsize=12)
    ax.set_title("Segmentation Quality vs Training Diversity", fontsize=14)
    ax.set_xlim(0, 16)
    ax.set_ylim(0.3, 1.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "dice_vs_num_structures.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_bottleneck_health(health_frames):
    """Bar chart: dead channel percentage by layer, for each model."""
    models = [k for k in MODEL_LABELS if k in health_frames]

    fig, ax = plt.subplots(figsize=(14, 5))
    layers = [
        "input_block",
        "downsamples.0",
        "downsamples.1",
        "downsamples.2",
        "bottleneck",
        "upsamples.0",
        "upsamples.1",
        "upsamples.2",
        "upsamples.3",
        "output_block",
    ]
    x = np.arange(len(layers))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        df = health_frames[model]
        vals = []
        for layer in layers:
            row = df[df["layer"] == layer]
            if len(row) > 0:
                vals.append(row["pct_dead"].values[0])
            else:
                vals.append(0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Dead Channels (%)", fontsize=12)
    ax.set_title("Bottleneck Collapse: Dead Channel Percentage by Layer", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    out = FIGURES_DIR / "bottleneck_health.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_heldout_heatmap(frames):
    """Heatmap: held-out Dice for structures not seen during training."""
    # Show models that have held-out structures
    models_with_heldout = [
        k
        for k in [
            "2_structs",
            "4_structs",
            "7_structs",
            "14_structs",
            "megaseg_pretrained",
        ]
        if k in frames
    ]
    if not models_with_heldout:
        return

    # Get all structures
    all_structs = sorted(
        set().union(*[set(frames[m]["structure"]) for m in models_with_heldout])
    )

    # Build matrix: rows = structures, cols = models (only held-out values)
    data = []
    for s in all_structs:
        row = {"structure": s}
        for m in models_with_heldout:
            df = frames[m]
            srow = df[(df["structure"] == s) & (df["seen"] == False)]
            if len(srow) > 0:
                row[MODEL_LABELS[m]] = srow["dice_mean"].values[0]
            else:
                row[MODEL_LABELS[m]] = np.nan  # seen, not held-out
        data.append(row)

    heatmap_df = pd.DataFrame(data).set_index("structure")

    # Drop rows where both are NaN (structures seen by both)
    heatmap_df = heatmap_df.dropna(how="all")

    if heatmap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Dice Score"},
    )
    ax.set_title("Held-out Structure Performance", fontsize=14)
    ax.set_ylabel("")

    fig.tight_layout()
    out = FIGURES_DIR / "heldout_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def print_summary_table(frames):
    """Print a summary table to stdout."""
    print("\n" + "=" * 80)
    print("ABLATION DIVERSITY SUMMARY")
    print("=" * 80)

    for model_key in [
        "2_structs",
        "4_structs",
        "7_structs",
        "14_structs",
        "megaseg_pretrained",
    ]:
        if model_key not in frames:
            continue
        df = frames[model_key]
        seen = df[df["seen"] == True]
        heldout = df[df["seen"] == False]

        print(f"\n{MODEL_LABELS[model_key]}:")
        print(
            f"  Overall Dice:   {df['dice_mean'].mean():.4f} +/- {df['dice_mean'].std():.4f}"
        )
        if len(seen) > 0:
            print(
                f"  Seen Dice:      {seen['dice_mean'].mean():.4f} +/- {seen['dice_mean'].std():.4f} (n={len(seen)})"
            )
        if len(heldout) > 0:
            print(
                f"  Held-out Dice:  {heldout['dice_mean'].mean():.4f} +/- {heldout['dice_mean'].std():.4f} (n={len(heldout)})"
            )

    print("\n" + "=" * 80)


def main():
    print("Loading results...")
    frames = load_all_results()
    health_frames = load_all_health()

    if not frames:
        print("No results found!")
        sys.exit(1)

    print(f"Found results for: {list(frames.keys())}")
    print_summary_table(frames)

    print("\nGenerating figures...")
    plot_dice_by_structure(frames)
    plot_seen_vs_heldout(frames)
    plot_dice_vs_num_structures(frames)
    plot_bottleneck_health(health_frames)
    plot_heldout_heatmap(frames)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
