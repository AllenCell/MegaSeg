"""
Visualize drug perturbation v2 segmentation results (local TIFFs).

Generates comparison figures showing raw input + predictions from each model type
(classic, megaseg, individual) for drug-perturbed cell FOVs.

Figure layout:
- One figure per structure, rows = FOVs grouped by condition, cols = [Raw, Classic, Individual, MegaSeg]
- Each cell shows a max-intensity projection (MIP) of the Z-stack
- Predictions are binarized (threshold 128) and shown as colored overlays on the raw image
- An overview grid figure shows one representative FOV per structure

Data source: local single-channel ZYX TIFFs in
  /allen/aics/rep_learn/MegaSeg/data/drug_perturbation/{structure}/{condition}/

Usage:
    .venv/bin/python scripts/visualize_drug_perturbation_v2.py
    .venv/bin/python scripts/visualize_drug_perturbation_v2.py --structures FBL TOMM20
    .venv/bin/python scripts/visualize_drug_perturbation_v2.py --mode grid
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRED_THRESHOLD = 128

BENCHMARK_DIR = Path(__file__).resolve().parent.parent

MEGASEG_PRED_DIR = (
    BENCHMARK_DIR / "outputs/eval/uint8/megaseg/dynunet/drug_perturbation_v2"
)
INDIVIDUAL_PRED_DIR = (
    BENCHMARK_DIR / "outputs/eval/uint8/individual/drug_perturbation_v2"
)
CLASSIC_PRED_DIR = BENCHMARK_DIR / "outputs/eval/uint8/classic/drug_perturbation_v2"
EVAL_CSV_DIR = BENCHMARK_DIR / "data/drug_perturbation_v2/eval_csvs"

RESULTS_DIR = BENCHMARK_DIR / "results/drug_perturbation_v2"
FIGURES_DIR = RESULTS_DIR / "figures"

ALL_STRUCTURES = ["FBL", "NUP153", "ST6GAL1", "TOMM20", "TUBA1B"]

# Display-friendly condition labels
CONDITION_LABELS = {
    "control": "Control",
    "actinomycin": "Actinomycin D",
    "thapsigargin": "Thapsigargin",
    "rotenone_high": "Rotenone (high)",
    "rotenone_low": "Rotenone (low)",
    "nocodazole": "Nocodazole",
}

# Overlay colors per model type
COLOR_CLASSIC = (1, 0.3, 0)  # orange
COLOR_MEGASEG = (0, 1, 0)  # green
COLOR_INDIVIDUAL = (0, 0.6, 1)  # blue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_fov_table() -> pd.DataFrame:
    """Build a table of all FOVs with structure, condition, file paths, and stems.

    Reads eval CSVs and extracts condition from the directory structure.
    """
    rows = []
    for structure in ALL_STRUCTURES:
        csv_path = EVAL_CSV_DIR / f"{structure}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping {structure}")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            fp = Path(row["file_path"])
            condition = fp.parent.name
            fname = fp.name
            # Stem used for matching predictions (strip .tiff extension)
            stem = fname.replace(".tiff", "").replace(".tif", "")
            rows.append(
                {
                    "structure": structure,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS.get(condition, condition),
                    "raw_path": str(fp),
                    "filename": fname,
                    "stem": stem,
                }
            )
    return pd.DataFrame(rows)


def load_raw(raw_path: str) -> np.ndarray:
    """Load a local single-channel ZYX TIFF as float32."""
    img = tifffile.imread(raw_path)
    return img.astype(np.float32)


def load_prediction(pred_dir: Path, stem: str) -> np.ndarray | None:
    """Load a prediction TIF, checking split/ subdir and direct dir."""
    # DL models output to split/ subdir
    split_dir = pred_dir / "split"
    if not split_dir.exists():
        split_dir = pred_dir

    if not split_dir.exists():
        return None

    # Try exact match with common extensions
    for ext in [".tif", ".ome.tif", ".tiff", ".ome.tiff"]:
        path = split_dir / f"{stem}{ext}"
        if path.exists():
            return tifffile.imread(str(path))

    # Fuzzy match: stem might differ slightly (e.g. trailing dot)
    for f in split_dir.iterdir():
        if f.suffix in (".tif", ".tiff"):
            # Strip all extensions for comparison
            f_stem = (
                f.name.replace(".ome.tif", "")
                .replace(".ome.tiff", "")
                .replace(".tif", "")
                .replace(".tiff", "")
            )
            if f_stem == stem or stem in f_stem or f_stem in stem:
                return tifffile.imread(str(f))

    return None


def mip(volume: np.ndarray) -> np.ndarray:
    """Max intensity projection along Z axis."""
    return volume.max(axis=0)


def crop_to_shape(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop 2D image to target shape if larger."""
    if img.shape == target_shape:
        return img
    if img.ndim == 2:
        h, w = img.shape
        th, tw = target_shape
        if h > th:
            start = (h - th) // 2
            img = img[start : start + th, :]
        if w > tw:
            start = (w - tw) // 2
            img = img[:, start : start + tw]
    return img


def make_overlay(
    raw_mip: np.ndarray,
    pred_mip: np.ndarray,
    color: tuple = (0, 1, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Create an RGB overlay of binarized prediction on raw MIP.

    Raw is shown as grayscale background.
    Prediction is binarized at PRED_THRESHOLD and overlaid in `color`.
    """
    raw_norm = raw_mip.astype(np.float32)
    nz = raw_norm[raw_norm > 0]
    if nz.size > 0:
        p1, p99 = np.percentile(nz, [1, 99])
    else:
        p1, p99 = 0, 1
    if p99 > p1:
        raw_norm = np.clip((raw_norm - p1) / (p99 - p1), 0, 1)
    else:
        raw_norm = np.clip(raw_norm / (raw_norm.max() + 1e-8), 0, 1)

    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

    mask = pred_mip > PRED_THRESHOLD
    for c in range(3):
        rgb[:, :, c] = np.where(
            mask,
            rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
            rgb[:, :, c],
        )

    return np.clip(rgb, 0, 1)


def visualize_structure(fov_table: pd.DataFrame, structure: str) -> None:
    """Generate comparison figure for a single structure.

    Rows = all FOVs sorted by condition (drug first, control last).
    Cols = [Raw, Classic, Individual, MegaSeg].
    """
    sub = fov_table[fov_table["structure"] == structure].copy()
    sub["is_control"] = sub["condition"] == "control"
    sub = sub.sort_values(["is_control", "condition", "filename"]).reset_index(
        drop=True
    )

    n_fovs = len(sub)
    col_labels = ["Raw (MIP)", "Classic", "Individual", "MegaSeg"]
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_fovs,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_fovs),
        squeeze=False,
    )

    for row_idx, (_, fov) in enumerate(sub.iterrows()):
        stem = fov["stem"]
        cond_label = fov["condition_label"]

        # Load raw
        try:
            raw = load_raw(fov["raw_path"])
            raw_mip_img = mip(raw)
            target_shape = raw_mip_img.shape
        except Exception as e:
            print(f"  ERROR loading raw for {stem}: {e}")
            continue

        col_idx = 0

        # Raw
        axes[row_idx, col_idx].imshow(raw_mip_img, cmap="gray")
        axes[row_idx, col_idx].set_ylabel(cond_label, fontsize=9, fontweight="bold")
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        col_idx += 1

        # Classic
        classic_pred = load_prediction(CLASSIC_PRED_DIR / structure, stem)
        if classic_pred is not None:
            c_mip = crop_to_shape(mip(classic_pred), target_shape)
            axes[row_idx, col_idx].imshow(
                make_overlay(raw_mip_img, c_mip, color=COLOR_CLASSIC)
            )
        else:
            axes[row_idx, col_idx].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=14,
                transform=axes[row_idx, col_idx].transAxes,
                color="gray",
            )
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        col_idx += 1

        # Individual
        indiv_pred = load_prediction(INDIVIDUAL_PRED_DIR / structure, stem)
        if indiv_pred is not None:
            i_mip = crop_to_shape(mip(indiv_pred), target_shape)
            axes[row_idx, col_idx].imshow(
                make_overlay(raw_mip_img, i_mip, color=COLOR_INDIVIDUAL)
            )
        else:
            axes[row_idx, col_idx].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=14,
                transform=axes[row_idx, col_idx].transAxes,
                color="gray",
            )
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        col_idx += 1

        # MegaSeg
        mega_pred = load_prediction(MEGASEG_PRED_DIR / structure, stem)
        if mega_pred is not None:
            m_mip = crop_to_shape(mip(mega_pred), target_shape)
            axes[row_idx, col_idx].imshow(
                make_overlay(raw_mip_img, m_mip, color=COLOR_MEGASEG)
            )
        else:
            axes[row_idx, col_idx].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=14,
                transform=axes[row_idx, col_idx].transAxes,
                color="gray",
            )
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])

    # Column titles
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle(
        f"{structure} - Drug Perturbation v2 Segmentation Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"{structure}_comparison.{ext}"
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {structure}_comparison.[pdf|png]")


def visualize_overview_grid(fov_table: pd.DataFrame) -> None:
    """Generate a compact overview grid: one row per structure, showing
    one drug-treated FOV with Raw + all model predictions."""

    n_structs = len(ALL_STRUCTURES)
    col_labels = ["Raw (MIP)", "Classic", "Individual", "MegaSeg"]
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_structs,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_structs),
        squeeze=False,
    )

    for row_idx, structure in enumerate(ALL_STRUCTURES):
        sub = fov_table[fov_table["structure"] == structure]
        # Pick first non-control FOV
        non_ctrl = sub[sub["condition"] != "control"]
        if len(non_ctrl) > 0:
            fov = non_ctrl.iloc[0]
        else:
            fov = sub.iloc[0]

        stem = fov["stem"]
        cond_label = fov["condition_label"]

        # Load raw
        try:
            raw = load_raw(fov["raw_path"])
            raw_mip_img = mip(raw)
            target_shape = raw_mip_img.shape
        except Exception as e:
            print(f"  ERROR loading raw for {stem}: {e}")
            for c in range(n_cols):
                axes[row_idx, c].set_visible(False)
            continue

        # Row label
        axes[row_idx, 0].set_ylabel(
            f"{structure}\n({cond_label})", fontsize=10, fontweight="bold"
        )

        # Col 0: Raw
        axes[row_idx, 0].imshow(raw_mip_img, cmap="gray")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Col 1: Classic
        pred = load_prediction(CLASSIC_PRED_DIR / structure, stem)
        if pred is not None:
            p_mip = crop_to_shape(mip(pred), target_shape)
            axes[row_idx, 1].imshow(
                make_overlay(raw_mip_img, p_mip, color=COLOR_CLASSIC)
            )
        else:
            axes[row_idx, 1].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=12,
                transform=axes[row_idx, 1].transAxes,
                color="gray",
            )
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Col 2: Individual
        pred = load_prediction(INDIVIDUAL_PRED_DIR / structure, stem)
        if pred is not None:
            p_mip = crop_to_shape(mip(pred), target_shape)
            axes[row_idx, 2].imshow(
                make_overlay(raw_mip_img, p_mip, color=COLOR_INDIVIDUAL)
            )
        else:
            axes[row_idx, 2].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=12,
                transform=axes[row_idx, 2].transAxes,
                color="gray",
            )
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

        # Col 3: MegaSeg
        pred = load_prediction(MEGASEG_PRED_DIR / structure, stem)
        if pred is not None:
            p_mip = crop_to_shape(mip(pred), target_shape)
            axes[row_idx, 3].imshow(
                make_overlay(raw_mip_img, p_mip, color=COLOR_MEGASEG)
            )
        else:
            axes[row_idx, 3].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                fontsize=12,
                transform=axes[row_idx, 3].transAxes,
                color="gray",
            )
        axes[row_idx, 3].set_xticks([])
        axes[row_idx, 3].set_yticks([])

    # Column titles
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle(
        "Drug Perturbation v2 - Model Comparison Overview",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"overview_grid.{ext}"
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview_grid.[pdf|png]")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize drug perturbation v2 segmentation comparison"
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=ALL_STRUCTURES,
        choices=ALL_STRUCTURES,
        help="Structures to visualize",
    )
    parser.add_argument(
        "--mode",
        choices=["per_structure", "grid", "both"],
        default="both",
        help="Figure mode: per_structure, grid, or both",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building FOV table from eval CSVs...")
    fov_table = build_fov_table()
    print(
        f"  Found {len(fov_table)} FOVs across {fov_table['structure'].nunique()} structures"
    )

    if args.mode in ("per_structure", "both"):
        for structure in args.structures:
            print(f"\n--- {structure} ---")
            visualize_structure(fov_table, structure)

    if args.mode in ("grid", "both"):
        print("\n--- Overview Grid ---")
        visualize_overview_grid(fov_table)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
