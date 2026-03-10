"""
Visualize drug perturbation segmentation results.

Generates comparison figures showing raw input + predictions from each model type
(classic, megaseg, individual) for drug-perturbed cell FOVs.

Figure layout:
- One figure per structure, with rows = selected FOVs, cols = [Raw, Classic, MegaSeg, Individual]
- Each cell shows a max-intensity projection (MIP) of the Z-stack
- Predictions are binarized (threshold 128) and shown as colored overlays on the raw image
- FOVs are grouped by drug condition and labeled accordingly

Usage:
    uv run python scripts/visualize_drug_perturbation.py [--structures ACTB NUP153 ...]
    uv run python scripts/visualize_drug_perturbation.py --mode grid  # overview grid
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import s3fs
import tifffile
from matplotlib.colors import Normalize
from skimage.segmentation import find_boundaries

# Binarization threshold for soft predictions
PRED_THRESHOLD = 128

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent.parent

MEGASEG_PRED_DIR = (
    BENCHMARK_DIR / "outputs/eval/uint8/megaseg/dynunet/drug_perturbation"
)
INDIVIDUAL_PRED_DIR = BENCHMARK_DIR / "outputs/eval/uint8/individual/drug_perturbation"
CLASSIC_PRED_DIR = BENCHMARK_DIR / "outputs/eval/uint8/classic/drug_perturbation"
EVAL_CSV_DIR = BENCHMARK_DIR / "data/drug_perturbations/eval_csvs"
MERGED_CSV = BENCHMARK_DIR / "data/drug_perturbations/merged_drug_perturbations.csv"

RESULTS_DIR = BENCHMARK_DIR / "results/drug_perturbation"
FIGURES_DIR = RESULTS_DIR / "figures"

ALL_STRUCTURES = ["ACTB", "NUP153", "ST6GAL1", "TJP1", "TOMM20", "TUBA1B"]

# Structures with individual model checkpoints
INDIVIDUAL_STRUCTURES = {"NUP153", "ST6GAL1", "TJP1", "TOMM20"}
# Structures with classic workflows
CLASSIC_STRUCTURES = {"ACTB", "NUP153", "ST6GAL1", "TJP1", "TOMM20", "TUBA1B"}


def simplify_drug(solution: str) -> str:
    """Map verbose Solution string to short drug name."""
    sol = solution.lower()
    if "jasplakinolide" in sol:
        return "Jasplakinolide"
    elif "latrunculin" in sol:
        return "Latrunculin A"
    elif "thapsigargin" in sol:
        return "Thapsigargin"
    elif "brefeldin" in sol:
        return "Brefeldin A"
    elif "monensin" in sol:
        return "Monensin"
    elif "staurosporine" in sol:
        return "Staurosporine"
    elif "rotenone" in sol:
        return "Rotenone"
    elif "nocodazole" in sol:
        return "Nocodazole"
    elif "paclitaxel" in sol:
        return "Paclitaxel"
    elif "dmso" in sol or "control" in sol:
        return "DMSO Control"
    return solution


def build_file_drug_mapping() -> pd.DataFrame:
    """Build a mapping from file_name to (Gene, Drug)."""
    df = pd.read_csv(MERGED_CSV)
    df["Drug"] = df["Solution"].apply(simplify_drug)
    # Extract file stem from file_path (HTTPS URLs)
    df["stem"] = (
        df["file_name"]
        .str.replace(".ome.tiff", "", regex=False)
        .str.replace(".ome.tif", "", regex=False)
    )
    return df[["file_name", "stem", "Gene", "Drug", "file_path"]].drop_duplicates()


def load_raw_from_s3(
    s3_url: str, fs: s3fs.S3FileSystem, channel: int = 1
) -> np.ndarray:
    """Load structure channel from S3 as ZYX float32."""
    s3_path = s3_url.replace("s3://", "").replace(
        "https://s3.us-west-2.amazonaws.com/", ""
    )
    with fs.open(s3_path, "rb") as f:
        img = tifffile.imread(f)
    if img.ndim == 4:
        return img[:, channel, :, :].astype(np.float32)
    return img.astype(np.float32)


def load_prediction(pred_dir: Path, stem: str) -> np.ndarray | None:
    """Load a prediction TIF from a split/ directory, matching by stem."""
    split_dir = pred_dir / "split"
    if not split_dir.exists():
        # Classic doesn't have split/ subdirectory
        split_dir = pred_dir

    # Try exact match first
    for ext in [".ome.tif", ".ome.tiff"]:
        path = split_dir / f"{stem}{ext}"
        if path.exists():
            return tifffile.imread(str(path))

    # Fuzzy match: stem might be slightly different
    for f in split_dir.iterdir():
        if f.suffix in (".tif", ".tiff") and stem in f.stem:
            return tifffile.imread(str(f))

    return None


def mip(volume: np.ndarray) -> np.ndarray:
    """Max intensity projection along Z axis."""
    return volume.max(axis=0)


def crop_to_shape(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center-crop to target shape if larger."""
    if img.shape == target_shape:
        return img
    # For 2D (after MIP)
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

    - Raw is shown as grayscale background
    - Prediction is binarized at PRED_THRESHOLD and overlaid in `color`
    """
    # Normalize raw to [0, 1]
    raw_norm = raw_mip.astype(np.float32)
    p1, p99 = (
        np.percentile(raw_norm[raw_norm > 0], [1, 99])
        if (raw_norm > 0).any()
        else (0, 1)
    )
    if p99 > p1:
        raw_norm = np.clip((raw_norm - p1) / (p99 - p1), 0, 1)
    else:
        raw_norm = np.clip(raw_norm / (raw_norm.max() + 1e-8), 0, 1)

    # Make RGB from grayscale
    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

    # Binarize prediction
    mask = pred_mip > PRED_THRESHOLD

    # Overlay: blend color where mask is True
    for c in range(3):
        rgb[:, :, c] = np.where(
            mask,
            rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
            rgb[:, :, c],
        )

    return np.clip(rgb, 0, 1)


def select_representative_fovs(
    drug_mapping: pd.DataFrame, structure: str, n_per_drug: int = 1
) -> pd.DataFrame:
    """Select representative FOVs: one per drug condition."""
    sub = drug_mapping[drug_mapping["Gene"] == structure].copy()
    # Pick the first FOV per drug condition (deterministic)
    selected = sub.groupby("Drug").first().reset_index()
    # Sort: drug conditions first, then DMSO control last
    selected["is_control"] = selected["Drug"] == "DMSO Control"
    selected = selected.sort_values(["is_control", "Drug"]).reset_index(drop=True)
    return selected


def visualize_structure(
    structure: str,
    drug_mapping: pd.DataFrame,
    fs: s3fs.S3FileSystem,
    n_per_drug: int = 1,
) -> None:
    """Generate comparison figure for a single structure."""
    selected = select_representative_fovs(drug_mapping, structure, n_per_drug)
    n_fovs = len(selected)

    has_individual = structure in INDIVIDUAL_STRUCTURES
    has_classic = structure in CLASSIC_STRUCTURES

    # Determine columns
    col_labels = ["Raw (MIP)"]
    if has_classic:
        col_labels.append("Classic")
    col_labels.append("MegaSeg")
    if has_individual:
        col_labels.append("Individual")
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_fovs,
        n_cols,
        figsize=(4 * n_cols, 4 * n_fovs),
        squeeze=False,
    )

    # Build s3 URL mapping from eval CSV
    eval_csv = pd.read_csv(EVAL_CSV_DIR / f"{structure}.csv")
    eval_csv["stem"] = (
        eval_csv["file_path"]
        .str.split("/")
        .str[-1]
        .str.replace(".ome.tiff", "", regex=False)
        .str.replace(".ome.tif", "", regex=False)
    )

    for row_idx, (_, fov) in enumerate(selected.iterrows()):
        stem = fov["stem"]
        drug = fov["Drug"]

        # Find S3 URL for this FOV from eval CSV
        match = eval_csv[eval_csv["stem"] == stem]
        if len(match) == 0:
            # Try partial match
            match = eval_csv[eval_csv["stem"].str.contains(stem[:40], regex=False)]
        if len(match) == 0:
            print(f"  WARNING: no S3 URL found for {stem}")
            continue
        s3_url = match["file_path"].iloc[0]

        # Load raw
        try:
            raw = load_raw_from_s3(s3_url, fs)
            raw_mip = mip(raw)
            target_shape = raw_mip.shape
        except Exception as e:
            print(f"  ERROR loading raw for {stem}: {e}")
            continue

        col_idx = 0

        # Raw
        axes[row_idx, col_idx].imshow(raw_mip, cmap="gray")
        axes[row_idx, col_idx].set_ylabel(drug, fontsize=11, fontweight="bold")
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=12)
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        col_idx += 1

        # Define overlay colors per model type
        color_classic = (1, 0.3, 0)  # orange
        color_megaseg = (0, 1, 0)  # green
        color_indiv = (0, 0.6, 1)  # blue

        # Classic
        if has_classic:
            classic_pred = load_prediction(CLASSIC_PRED_DIR / structure, stem)
            if classic_pred is not None:
                classic_mip = mip(classic_pred)
                classic_mip = crop_to_shape(classic_mip, target_shape)
                overlay = make_overlay(raw_mip, classic_mip, color=color_classic)
                axes[row_idx, col_idx].imshow(overlay)
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
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=12)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            col_idx += 1

        # MegaSeg
        mega_pred = load_prediction(MEGASEG_PRED_DIR / structure, stem)
        if mega_pred is not None:
            mega_mip = mip(mega_pred)
            mega_mip = crop_to_shape(mega_mip, target_shape)
            overlay = make_overlay(raw_mip, mega_mip, color=color_megaseg)
            axes[row_idx, col_idx].imshow(overlay)
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
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=12)
        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        col_idx += 1

        # Individual
        if has_individual:
            indiv_pred = load_prediction(INDIVIDUAL_PRED_DIR / structure, stem)
            if indiv_pred is not None:
                indiv_mip = mip(indiv_pred)
                indiv_mip = crop_to_shape(indiv_mip, target_shape)
                overlay = make_overlay(raw_mip, indiv_mip, color=color_indiv)
                axes[row_idx, col_idx].imshow(overlay)
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
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_labels[col_idx], fontsize=12)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            col_idx += 1

    fig.suptitle(
        f"{structure} — Drug Perturbation Segmentation Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"{structure}_comparison.{ext}"
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {structure}_comparison.[pdf|png]")


def visualize_overview_grid(
    drug_mapping: pd.DataFrame,
    fs: s3fs.S3FileSystem,
) -> None:
    """Generate a compact overview grid: one row per structure, showing
    one drug-treated FOV with Raw + all available model predictions."""

    n_structs = len(ALL_STRUCTURES)
    # Columns: Raw, Classic, MegaSeg, Individual
    col_labels = ["Raw (MIP)", "Classic", "MegaSeg", "Individual"]
    n_cols = len(col_labels)

    fig, axes = plt.subplots(
        n_structs,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_structs),
        squeeze=False,
    )

    for row_idx, structure in enumerate(ALL_STRUCTURES):
        selected = select_representative_fovs(drug_mapping, structure)
        # Pick first non-control FOV
        non_ctrl = selected[selected["Drug"] != "DMSO Control"]
        if len(non_ctrl) > 0:
            fov = non_ctrl.iloc[0]
        else:
            fov = selected.iloc[0]

        stem = fov["stem"]
        drug = fov["Drug"]

        # S3 URL
        eval_csv = pd.read_csv(EVAL_CSV_DIR / f"{structure}.csv")
        eval_csv["stem"] = (
            eval_csv["file_path"]
            .str.split("/")
            .str[-1]
            .str.replace(".ome.tiff", "", regex=False)
            .str.replace(".ome.tif", "", regex=False)
        )
        match = eval_csv[eval_csv["stem"] == stem]
        if len(match) == 0:
            match = eval_csv[eval_csv["stem"].str.contains(stem[:40], regex=False)]
        if len(match) == 0:
            print(f"  WARNING: no S3 URL for {stem}")
            for c in range(n_cols):
                axes[row_idx, c].set_visible(False)
            continue
        s3_url = match["file_path"].iloc[0]

        # Load raw
        try:
            raw = load_raw_from_s3(s3_url, fs)
            raw_mip = mip(raw)
            target_shape = raw_mip.shape
        except Exception as e:
            print(f"  ERROR loading raw for {stem}: {e}")
            continue

        # Row label
        axes[row_idx, 0].set_ylabel(
            f"{structure}\n({drug})", fontsize=10, fontweight="bold"
        )

        # Overlay colors (same as per-structure figures)
        color_classic = (1, 0.3, 0)  # orange
        color_megaseg = (0, 1, 0)  # green
        color_indiv = (0, 0.6, 1)  # blue

        # Col 0: Raw
        axes[row_idx, 0].imshow(raw_mip, cmap="gray")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Col 1: Classic
        if structure in CLASSIC_STRUCTURES:
            pred = load_prediction(CLASSIC_PRED_DIR / structure, stem)
            if pred is not None:
                pred_mip = crop_to_shape(mip(pred), target_shape)
                overlay = make_overlay(raw_mip, pred_mip, color=color_classic)
                axes[row_idx, 1].imshow(overlay)
            else:
                axes[row_idx, 1].text(
                    0.5,
                    0.5,
                    "Pending",
                    ha="center",
                    va="center",
                    fontsize=12,
                    transform=axes[row_idx, 1].transAxes,
                    color="gray",
                )
        else:
            axes[row_idx, 1].text(
                0.5,
                0.5,
                "No workflow",
                ha="center",
                va="center",
                fontsize=12,
                transform=axes[row_idx, 1].transAxes,
                color="gray",
            )
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Col 2: MegaSeg
        pred = load_prediction(MEGASEG_PRED_DIR / structure, stem)
        if pred is not None:
            pred_mip = crop_to_shape(mip(pred), target_shape)
            overlay = make_overlay(raw_mip, pred_mip, color=color_megaseg)
            axes[row_idx, 2].imshow(overlay)
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

        # Col 3: Individual
        if structure in INDIVIDUAL_STRUCTURES:
            pred = load_prediction(INDIVIDUAL_PRED_DIR / structure, stem)
            if pred is not None:
                pred_mip = crop_to_shape(mip(pred), target_shape)
                overlay = make_overlay(raw_mip, pred_mip, color=color_indiv)
                axes[row_idx, 3].imshow(overlay)
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
        else:
            axes[row_idx, 3].text(
                0.5,
                0.5,
                "No model",
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
        "Drug Perturbation Segmentation — Model Comparison Overview",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"overview_grid.{ext}"
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overview_grid.[pdf|png]")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize drug perturbation segmentation comparison"
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
    parser.add_argument(
        "--n-per-drug",
        type=int,
        default=1,
        help="Number of FOVs per drug condition (per_structure mode)",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building file-drug mapping...")
    drug_mapping = build_file_drug_mapping()

    print("Connecting to S3...")
    fs = s3fs.S3FileSystem(anon=True)

    if args.mode in ("per_structure", "both"):
        for structure in args.structures:
            print(f"\n--- {structure} ---")
            visualize_structure(structure, drug_mapping, fs, args.n_per_drug)

    if args.mode in ("grid", "both"):
        print("\n--- Overview Grid ---")
        visualize_overview_grid(drug_mapping, fs)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
