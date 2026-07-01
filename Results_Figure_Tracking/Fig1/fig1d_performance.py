"""
Fig 1d — MegaSeg vs specialist-model performance (13 training structures).

Self-contained regeneration script. The matched per-structure Dice scores
(specialist model and MegaSeg/DynUNet, mean +/- std) are embedded below, so it
reproduces `fig1d_performance.pdf` in this folder with no external files.

Source of the numbers: matched diagonal (model_struct == test_struct) of
../Fig2-HeatMap_SpModel/heatmap_summary_stats.csv  (eval_type "individual" =
specialist model; eval_type "megaseg", backbone "dynunet" = MegaSeg).
Structure ("organelle") names follow Viana et al., Nature 2023, Fig. 1a.

Usage:
    python fig1d_performance.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Nature-style rcParams (Arial, editable PDF text) ──────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_PDF = Path(__file__).parent / "fig1d_performance.pdf"

# (Viana structure name, specialist [mean, std], MegaSeg-DynUNet [mean, std])
DATA = [
    ("Actin bundles",       [0.6746, 0.1118], [0.8071, 0.0335]),
    ("ER (SERCA)",          [0.7937, 0.0329], [0.8570, 0.0154]),
    ("Adherens junctions",  [0.8834, 0.0082], [0.9654, 0.0025]),
    ("Desmosomes",          [0.6894, 0.0497], [0.9262, 0.0411]),
    ("Gap junctions",       [0.7255, 0.0313], [0.9719, 0.0031]),
    ("Actomyosin bundles",  [0.7234, 0.0747], [0.8734, 0.0142]),
    ("Nuclear pores",       [0.7854, 0.0415], [0.9544, 0.0075]),
    ("Endosomes",           [0.6671, 0.0744], [0.9364, 0.0257]),
    ("ER (SEC61β)",    [0.8324, 0.0276], [0.8814, 0.0175]),
    ("Nuclear speckles",    [0.7809, 0.0204], [0.8824, 0.0061]),
    ("Golgi",               [0.7980, 0.0409], [0.8887, 0.0154]),
    ("Tight junctions",     [0.7843, 0.0191], [0.9404, 0.0078]),
    ("Mitochondria",        [0.9147, 0.0190], [0.9348, 0.0169]),
]


def main():
    names = [d[0] for d in DATA]
    ind_means = np.array([d[1][0] for d in DATA])
    ind_stds = np.array([d[1][1] for d in DATA])
    meg_means = np.array([d[2][0] for d in DATA])
    meg_stds = np.array([d[2][1] for d in DATA])

    # Sort by improvement (MegaSeg - specialist), largest first.
    deltas = meg_means - ind_means
    order = np.argsort(deltas)[::-1]
    names = [names[i] for i in order]
    ind_means, ind_stds = ind_means[order], ind_stds[order]
    meg_means, meg_stds = meg_means[order], meg_stds[order]
    deltas = deltas[order]

    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor="white")
    x = np.arange(len(names))
    bar_w = 0.35

    ax.bar(x - bar_w / 2, ind_means, bar_w, yerr=ind_stds,
           color="#bdc3c7", edgecolor="white", linewidth=0.3,
           error_kw=dict(lw=0.5, capsize=1.5, capthick=0.5),
           label="Specialist model", zorder=3)
    ax.bar(x + bar_w / 2, meg_means, bar_w, yerr=meg_stds,
           color="#2980b9", edgecolor="white", linewidth=0.3,
           error_kw=dict(lw=0.5, capsize=1.5, capthick=0.5),
           label="MegaSeg (DynUNet)", zorder=3)

    # Delta annotations above the MegaSeg bars
    for i, d in enumerate(deltas):
        y_top = meg_means[i] + meg_stds[i] + 0.015
        ax.text(x[i] + bar_w / 2, y_top, f"+{d:.2f}", ha="center", va="bottom",
                fontsize=4.5, color="#27ae60", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, fontweight="bold", rotation=45, ha="right")
    ax.set_ylabel("Dice Score", fontsize=7, fontweight="bold")
    ax.set_ylim(0.5, 1.08)
    ax.set_xlim(-0.6, len(names) - 0.4)

    # Specialist mean baseline
    ind_overall = float(np.mean(ind_means))
    ax.axhline(ind_overall, color="#bdc3c7", linewidth=0.6, linestyle="--", zorder=1)
    ax.text(len(names) - 0.5, ind_overall + 0.008,
            f"Specialist mean: {ind_overall:.3f}",
            fontsize=5, color="#7f8c8d", ha="right")

    ax.legend(fontsize=6, frameon=True, edgecolor="#d5d8dc", loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_title("MegaSeg outperforms specialist models on all 13 structures",
                 fontsize=8, fontweight="bold", pad=8)

    fig.savefig(str(OUT_PDF), dpi=300, facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
