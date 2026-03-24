#!/usr/bin/env python
"""
Quantitative evaluation of MegaSeg benchmark segmentation models.

Three-phase pipeline:
  compute — reads prediction/GT TIFF pairs, computes metrics, writes per-sample CSV
  report  — reads per-sample CSV, generates comparison tables
  plot    — reads per-sample CSV, generates publication-quality visualization plots
  filter  — removes bottom-N low-signal samples per structure, regenerates reports & plots

Usage:
    python scripts/evaluate_benchmark.py compute --dtype uint8 --workers 16
    python scripts/evaluate_benchmark.py report --dtype uint8 --metric dice
    python scripts/evaluate_benchmark.py plot --dtype uint8
    python scripts/evaluate_benchmark.py plot --dtype uint8 --metric dice
    python scripts/evaluate_benchmark.py filter --dtype uint8 --bottom-n 10
"""

import argparse
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRUCTURES = [
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
]
# NOTE: NPM1 excluded - not used in training and no test data available.

BACKBONES = ["dynunet", "attention_unet", "swin_unetr", "unet"]

METRIC_COLS = ["dice", "iou", "precision", "recall", "accuracy"]

# Structures to include in plots (exclude NPM1 — nucleolar marker with
# fundamentally different signal characteristics that distorts visual scales)
PLOT_STRUCTURES = [s for s in STRUCTURES if s != "NPM1"]

EVAL_TYPES = ["individual", "individual_test_new", "megaseg", "cross_eval"]

# Plot configuration
MODEL_COLORS = {
    "individual": "#7f7f7f",
    "megaseg_dynunet": "#1f77b4",
    "megaseg_attention_unet": "#ff7f0e",
    "megaseg_swin_unetr": "#2ca02c",
    "megaseg_unet": "#d62728",
}

BACKBONE_COLORS = {
    "dynunet": "#1f77b4",
    "attention_unet": "#ff7f0e",
    "swin_unetr": "#2ca02c",
    "unet": "#d62728",
}

BACKBONE_LABELS = {
    "dynunet": "DynUNet",
    "attention_unet": "AttentionUNet",
    "swin_unetr": "SwinUNETR",
    "unet": "UNet",
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute segmentation metrics from two binary boolean masks.

    Both empty -> perfect agreement (1.0). One empty -> zero overlap.
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    dice_denom = 2 * tp + fp + fn
    dice = (2 * tp / dice_denom) if dice_denom > 0 else 1.0

    iou_denom = tp + fp + fn
    iou = (tp / iou_denom) if iou_denom > 0 else 1.0

    prec_denom = tp + fp
    precision = (tp / prec_denom) if prec_denom > 0 else 1.0

    rec_denom = tp + fn
    recall = (tp / rec_denom) if rec_denom > 0 else 1.0

    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total) if total > 0 else 1.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def binarize_prediction(pred: np.ndarray, dtype: str) -> np.ndarray:
    """Binarize prediction: uint8 >= 128, float >= 0.5."""
    if pred.dtype == np.uint8:
        return pred >= 128
    elif np.issubdtype(pred.dtype, np.floating):
        return pred >= 0.5
    else:
        if dtype == "uint8":
            return pred.astype(np.float64) >= 128
        else:
            return pred.astype(np.float64) >= 0.5


def binarize_gt(gt: np.ndarray) -> np.ndarray:
    """Binarize ground truth mask (uint8 0/255) to boolean."""
    return gt > 0


def extract_structure_from_filename(filename: str) -> str:
    """Extract structure name from a filename like 'ST6GAL1_4567_IM.tiff'.

    Pattern: {STRUCT}_{DIGITS}[_suffix].ext where suffix is IM/GT/seg.
    """
    stem = Path(filename).stem  # e.g. "ST6GAL1_4567_IM"
    # Remove known suffixes
    for suffix in ("_IM", "_GT", "_seg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Now stem = "ST6GAL1_4567" — split on last underscore before pure digits
    match = re.match(r"^(.+)_(\d+)$", stem)
    if match:
        return match.group(1)
    return stem  # fallback


# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalJob:
    """Defines one evaluation task: a set of predictions to compare against GT."""

    eval_type: str  # individual, individual_test_new, megaseg, cross_eval
    dtype: str  # uint8, float32
    backbone: str | None  # None for individual/cross_eval
    model_struct: str  # structure the model was trained on
    test_struct: str  # structure being tested on
    test_set: str  # "standard" or "test_new"


def generate_jobs(
    dtype: str,
    eval_type: str | None = None,
    backbone: str | None = None,
    structure: str | None = None,
) -> list[EvalJob]:
    """Enumerate all evaluation jobs, optionally filtered."""
    jobs: list[EvalJob] = []
    structs = [structure] if structure else STRUCTURES

    # --- individual ---
    if eval_type in (None, "individual"):
        for s in structs:
            jobs.append(EvalJob("individual", dtype, None, s, s, "standard"))

    # --- individual_test_new ---
    if eval_type in (None, "individual_test_new"):
        for s in structs:
            jobs.append(EvalJob("individual_test_new", dtype, None, s, s, "test_new"))

    # --- megaseg ---
    if eval_type in (None, "megaseg"):
        bbs = [backbone] if backbone else BACKBONES
        for bb in bbs:
            for s in structs:
                jobs.append(EvalJob("megaseg", dtype, bb, s, s, "standard"))
            # test_new: one job per backbone
            if not structure:  # test_new is not per-structure
                jobs.append(
                    EvalJob("megaseg", dtype, bb, "test_new", "test_new", "test_new")
                )

    # --- cross_eval ---
    if eval_type in (None, "cross_eval"):
        for model_s in structs:
            test_structs = [structure] if structure else STRUCTURES
            for test_s in test_structs:
                jobs.append(
                    EvalJob("cross_eval", dtype, None, model_s, test_s, "standard")
                )

    return jobs


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_pred_dir(job: EvalJob, project_root: Path) -> Path:
    """Resolve the prediction seg/ directory for a given job."""
    base = project_root / "outputs" / "eval" / job.dtype

    if job.eval_type == "individual":
        return base / "individual" / f"ms13-{job.model_struct}" / "seg"

    elif job.eval_type == "individual_test_new":
        return base / "individual_test_new" / f"ms13-{job.model_struct}" / "seg"

    elif job.eval_type == "megaseg":
        if job.test_set == "test_new":
            return base / "megaseg" / job.backbone / "test_new" / "seg"
        else:
            return base / "megaseg" / job.backbone / job.model_struct / "seg"

    elif job.eval_type == "cross_eval":
        return (
            base
            / "cross_eval"
            / f"ms13-{job.model_struct}"
            / f"tested_on_{job.test_struct}"
            / "seg"
        )

    else:
        raise ValueError(f"Unknown eval_type: {job.eval_type}")


def resolve_test_csv(job: EvalJob, project_root: Path) -> Path:
    """Resolve the test CSV path for a given job."""
    data_root = project_root / ".." / "data"

    if job.test_set == "test_new":
        return data_root / "standardized_test" / "test_new.csv"
    else:
        return data_root / "variance-100x" / job.test_struct / "test.csv"


def derive_prediction_path(raw_filename: str, pred_seg_dir: Path) -> Path:
    """Map a raw image filename to its prediction file in pred_seg_dir.

    Handles .tif/.tiff extension swapping.
    """
    name = Path(raw_filename).name
    exact = pred_seg_dir / name
    if exact.is_file():
        return exact

    stem = exact.stem
    ext = exact.suffix.lower()
    if ext == ".tif":
        alt = pred_seg_dir / (stem + ".tiff")
    elif ext == ".tiff":
        alt = pred_seg_dir / (stem + ".tif")
    else:
        return exact

    if alt.is_file():
        return alt

    return exact  # Return original path; caller handles missing file


# ---------------------------------------------------------------------------
# Sample evaluation (multiprocessing worker)
# ---------------------------------------------------------------------------


def _evaluate_sample(args: tuple) -> dict:
    """Evaluate one prediction/GT pair. Designed for multiprocessing.Pool.

    Args is a tuple: (pred_path_str, gt_path_str, dtype, metadata_dict)
    Returns a dict with metadata + metric columns (or NaN on failure).
    """
    pred_path_str, gt_path_str, dtype, metadata = args
    pred_path = Path(pred_path_str)
    gt_path = Path(gt_path_str)

    nan_result = {**metadata}
    for m in METRIC_COLS:
        nan_result[m] = np.nan

    # Load GT
    try:
        gt_img = tifffile.imread(str(gt_path))
    except Exception as e:
        log.warning("Failed to load GT %s: %s", gt_path, e)
        return nan_result

    gt_mask = binarize_gt(gt_img)

    # Load prediction
    if not pred_path.is_file():
        log.warning("Prediction not found: %s", pred_path)
        return nan_result

    try:
        pred_img = tifffile.imread(str(pred_path))
    except Exception as e:
        log.warning("Failed to load prediction %s: %s", pred_path, e)
        return nan_result

    pred_mask = binarize_prediction(pred_img, dtype)

    # Shape check
    if gt_mask.shape != pred_mask.shape:
        log.warning(
            "Shape mismatch for %s: GT %s vs Pred %s",
            metadata.get("sample_id", "?"),
            gt_mask.shape,
            pred_mask.shape,
        )
        return nan_result

    # Compute metrics
    metrics = compute_metrics(gt_mask, pred_mask)
    return {**metadata, **metrics}


# ---------------------------------------------------------------------------
# Compute orchestrator
# ---------------------------------------------------------------------------


def _build_work_items(job: EvalJob, project_root: Path) -> list[tuple]:
    """Build (pred_path, gt_path, dtype, metadata) tuples for one job."""
    pred_dir = resolve_pred_dir(job, project_root)
    csv_path = resolve_test_csv(job, project_root)

    if not pred_dir.is_dir():
        log.warning("Pred dir missing, skipping job: %s", pred_dir)
        return []

    if not csv_path.is_file():
        log.warning("Test CSV missing, skipping job: %s", csv_path)
        return []

    df = pd.read_csv(csv_path)
    if "raw" not in df.columns or "seg" not in df.columns:
        log.warning("CSV missing raw/seg columns: %s", csv_path)
        return []

    items = []
    for _, row in df.iterrows():
        raw_path = str(row["raw"])
        gt_path = str(row["seg"])

        # For individual_test_new, the pred dir is per-structure (ms13-{S}/seg/)
        # but the test CSV (test_new.csv) is shared across all structures.
        # We need to filter the CSV to rows whose filename matches the structure.
        if job.eval_type == "individual_test_new":
            file_struct = extract_structure_from_filename(Path(raw_path).name)
            if file_struct != job.model_struct:
                continue

        pred_file = derive_prediction_path(raw_path, pred_dir)

        sample_id = Path(raw_path).stem  # e.g. "ACTN1_10917" or "DSP_3970_IM"

        # Determine test_struct from filename for test_new jobs
        if job.test_set == "test_new":
            test_struct = extract_structure_from_filename(Path(raw_path).name)
        else:
            test_struct = job.test_struct

        metadata = {
            "eval_type": job.eval_type,
            "dtype": job.dtype,
            "backbone": job.backbone or "",
            "model_struct": job.model_struct,
            "test_struct": test_struct,
            "test_set": job.test_set,
            "sample_id": sample_id,
        }

        items.append((str(pred_file), gt_path, job.dtype, metadata))

    return items


def compute_all(
    project_root: Path,
    dtype: str,
    workers: int,
    eval_type: str | None = None,
    backbone: str | None = None,
    structure: str | None = None,
) -> None:
    """Run the full compute phase."""
    results_dir = project_root / "results" / dtype
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "per_sample_metrics.csv"

    # Load existing results for incremental support
    existing_keys: set[tuple] = set()
    existing_df = None
    if csv_path.is_file():
        existing_df = pd.read_csv(csv_path)
        for _, row in existing_df.iterrows():
            key = (
                row["eval_type"],
                row["dtype"],
                str(row.get("backbone", "")),
                row["model_struct"],
                row["test_struct"],
                row["sample_id"],
            )
            existing_keys.add(key)
        log.info("Loaded %d existing results from %s", len(existing_keys), csv_path)

    # Generate jobs
    jobs = generate_jobs(
        dtype, eval_type=eval_type, backbone=backbone, structure=structure
    )
    log.info("Generated %d jobs", len(jobs))

    # Build work items, skipping already-computed samples
    all_items: list[tuple] = []
    for job in jobs:
        items = _build_work_items(job, project_root)
        for item in items:
            _, _, _, meta = item
            key = (
                meta["eval_type"],
                meta["dtype"],
                meta["backbone"],
                meta["model_struct"],
                meta["test_struct"],
                meta["sample_id"],
            )
            if key not in existing_keys:
                all_items.append(item)

    if not all_items:
        log.info("All samples already computed. Nothing to do.")
        return

    log.info(
        "Computing metrics for %d new samples using %d workers...",
        len(all_items),
        workers,
    )

    # Run with multiprocessing
    results: list[dict] = []
    with Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_evaluate_sample, all_items, chunksize=32),
            total=len(all_items),
            desc="Computing metrics",
        ):
            results.append(result)

    # Combine with existing results
    new_df = pd.DataFrame(results)
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Sort for deterministic output
    combined_df = combined_df.sort_values(
        ["eval_type", "backbone", "model_struct", "test_struct", "sample_id"]
    ).reset_index(drop=True)

    combined_df.to_csv(csv_path, index=False)
    log.info("Wrote %d total results to %s", len(combined_df), csv_path)

    # Print summary
    n_nan = combined_df[METRIC_COLS[0]].isna().sum()
    if n_nan > 0:
        log.warning("%d samples had NaN metrics (missing/failed predictions)", n_nan)


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------


def _load_results(project_root: Path, dtype: str) -> pd.DataFrame:
    """Load the per-sample metrics CSV."""
    csv_path = project_root / "results" / dtype / "per_sample_metrics.csv"
    if not csv_path.is_file():
        log.error("Results CSV not found: %s — run 'compute' first", csv_path)
        sys.exit(1)
    return pd.read_csv(csv_path)


def report_summary_by_structure(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Mean +/- std per (eval_type, backbone, test_struct)."""
    group_cols = ["eval_type", "backbone", "test_struct", "test_set"]
    grouped = df.groupby(group_cols, dropna=False)[METRIC_COLS]

    summary = grouped.mean()
    summary_std = grouped.std()
    summary_std.columns = [f"{c}_std" for c in summary_std.columns]
    summary = pd.concat([summary, summary_std], axis=1)
    summary["n_samples"] = grouped.size().values
    summary = summary.reset_index()

    out_path = output_dir / "summary_by_structure.csv"
    summary.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    return summary


def report_individual_vs_megaseg(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "dice",
) -> pd.DataFrame:
    """Side-by-side comparison of individual vs each megaseg backbone per structure.

    Only uses 'standard' test set (per-structure test data).
    """
    mask = df["eval_type"].isin(["individual", "megaseg"]) & (
        df["test_set"] == "standard"
    )
    subset = df[mask].copy()

    subset["model_label"] = subset.apply(
        lambda r: r["eval_type"]
        if r["eval_type"] == "individual"
        else f"megaseg_{r['backbone']}",
        axis=1,
    )

    means = subset.groupby(["test_struct", "model_label"])[metric].mean().reset_index()
    pivoted = means.pivot(index="test_struct", columns="model_label", values=metric)

    # Add deltas (megaseg - individual)
    if "individual" in pivoted.columns:
        for col in pivoted.columns:
            if col.startswith("megaseg_"):
                pivoted[f"delta_{col}"] = pivoted[col] - pivoted["individual"]

    # Add OVERALL row
    overall = pivoted.mean()
    overall.name = "OVERALL"
    pivoted = pd.concat([pivoted, overall.to_frame().T])

    pivoted = pivoted.reset_index().rename(columns={"test_struct": "structure"})

    out_path = output_dir / "comparison_individual_vs_megaseg.csv"
    pivoted.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    print(f"\n{'=' * 80}")
    print(f"Individual vs MegaSeg ({metric}, standard test set)")
    print(f"{'=' * 80}")
    print(pivoted.round(4).to_string(index=False))
    print()

    return pivoted


def report_backbone_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "dice",
) -> pd.DataFrame:
    """Compare all 4 megaseg backbones per structure (standard test set)."""
    mask = (df["eval_type"] == "megaseg") & (df["test_set"] == "standard")
    subset = df[mask]

    means = subset.groupby(["test_struct", "backbone"])[metric].mean().reset_index()
    pivoted = means.pivot(index="test_struct", columns="backbone", values=metric)

    # Add OVERALL row
    overall = pivoted.mean()
    overall.name = "OVERALL"
    pivoted = pd.concat([pivoted, overall.to_frame().T])
    pivoted = pivoted.reset_index().rename(columns={"test_struct": "structure"})

    out_path = output_dir / "comparison_backbones.csv"
    pivoted.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    print(f"\n{'=' * 80}")
    print(f"Backbone Comparison ({metric}, megaseg, standard test set)")
    print(f"{'=' * 80}")
    print(pivoted.round(4).to_string(index=False))
    print()

    return pivoted


def report_test_new_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "dice",
) -> pd.DataFrame:
    """Compare individual vs megaseg backbones on the held-out test_new set.

    Groups by test_struct (structure extracted from filename) to show
    per-structure performance on out-of-distribution data.
    """
    mask = df["test_set"] == "test_new"
    subset = df[mask].copy()

    subset["model_label"] = subset.apply(
        lambda r: "individual"
        if r["eval_type"] == "individual_test_new"
        else f"megaseg_{r['backbone']}",
        axis=1,
    )

    means = subset.groupby(["test_struct", "model_label"])[metric].mean().reset_index()
    pivoted = means.pivot(index="test_struct", columns="model_label", values=metric)

    # Add OVERALL row
    overall = pivoted.mean()
    overall.name = "OVERALL"
    pivoted = pd.concat([pivoted, overall.to_frame().T])
    pivoted = pivoted.reset_index().rename(columns={"test_struct": "structure"})

    out_path = output_dir / "test_new_comparison.csv"
    pivoted.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    print(f"\n{'=' * 80}")
    print(f"Test-New Comparison ({metric}, held-out test_new set)")
    print(f"{'=' * 80}")
    print(pivoted.round(4).to_string(index=False))
    print()

    return pivoted


def report_cross_eval_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "dice",
) -> pd.DataFrame:
    """14x14 heatmap: rows = model trained on, cols = tested on, values = mean metric.

    Diagonal entries = same-structure performance (should match individual eval).
    Off-diagonal = transfer performance.
    """
    mask = df["eval_type"] == "cross_eval"
    subset = df[mask]

    means = subset.groupby(["model_struct", "test_struct"])[metric].mean().reset_index()
    heatmap = means.pivot(index="model_struct", columns="test_struct", values=metric)

    # Ensure consistent ordering
    heatmap = heatmap.reindex(index=STRUCTURES, columns=STRUCTURES)

    out_path = output_dir / "cross_eval_heatmap.csv"
    heatmap.to_csv(out_path)
    log.info("Wrote %s", out_path)

    print(f"\n{'=' * 80}")
    print(f"Cross-Eval Heatmap ({metric}, rows=trained_on, cols=tested_on)")
    print(f"{'=' * 80}")
    print(heatmap.round(4).to_string())
    print()

    # Print summary stats
    diag = pd.Series(
        [
            heatmap.loc[s, s]
            for s in STRUCTURES
            if s in heatmap.index and s in heatmap.columns
        ],
        index=[s for s in STRUCTURES if s in heatmap.index and s in heatmap.columns],
    )
    off_diag = []
    for m in STRUCTURES:
        for t in STRUCTURES:
            if m != t and m in heatmap.index and t in heatmap.columns:
                val = heatmap.loc[m, t]
                if not np.isnan(val):
                    off_diag.append(val)

    if not diag.empty:
        print(f"Diagonal mean (same-structure): {diag.mean():.4f}")
    if off_diag:
        print(f"Off-diagonal mean (transfer):   {np.mean(off_diag):.4f}")
        print(f"Diagonal - off-diagonal gap:    {diag.mean() - np.mean(off_diag):.4f}")
    print()

    return heatmap


# ---------------------------------------------------------------------------
# Report orchestrator
# ---------------------------------------------------------------------------


def report_all(
    project_root: Path,
    dtype: str,
    metric: str = "dice",
    df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run all report generators.

    If *df* is provided it is used directly; otherwise the per-sample CSV is
    loaded from ``results/{dtype}/per_sample_metrics.csv``.
    """
    if df is None:
        df = _load_results(project_root, dtype)
    if output_dir is None:
        output_dir = project_root / "results" / dtype
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loaded %d samples from results CSV", len(df))
    log.info("Eval types present: %s", sorted(df["eval_type"].unique()))
    log.info("Generating reports with metric=%s ...", metric)

    # 1. Summary by structure (always)
    report_summary_by_structure(df, output_dir)

    # 2. Individual vs MegaSeg (only if both present)
    has_individual = "individual" in df["eval_type"].values
    has_megaseg = "megaseg" in df["eval_type"].values
    if has_individual and has_megaseg:
        report_individual_vs_megaseg(df, output_dir, metric)
    else:
        log.info("Skipping individual-vs-megaseg (need both eval types)")

    # 3. Backbone comparison (only if megaseg present)
    if has_megaseg:
        report_backbone_comparison(df, output_dir, metric)
    else:
        log.info("Skipping backbone comparison (no megaseg results)")

    # 4. Cross-eval heatmap (only if cross_eval present)
    if "cross_eval" in df["eval_type"].values:
        report_cross_eval_heatmap(df, output_dir, metric)
    else:
        log.info("Skipping cross-eval heatmap (no cross_eval results)")

    # 5. Test-new comparison (only if test_new data present)
    has_test_new = "test_new" in df["test_set"].values
    if has_test_new:
        report_test_new_comparison(df, output_dir, metric)
    else:
        log.info("Skipping test_new comparison (no test_new results)")

    log.info("All reports written to %s", output_dir)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def setup_plot_style() -> None:
    """Configure matplotlib/seaborn for publication-quality figures."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)
    log.info("Saved %s.{pdf,png} to %s", name, outdir)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_cross_eval_heatmap(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 1: Cross-evaluation heatmap (excluding NPM1)."""
    mask = df["eval_type"] == "cross_eval"
    subset = df[mask]
    if subset.empty:
        log.warning("No cross_eval data; skipping heatmap")
        return

    means = subset.groupby(["model_struct", "test_struct"])[metric].mean().reset_index()
    heatmap_data = means.pivot(
        index="model_struct", columns="test_struct", values=metric
    )
    heatmap_data = heatmap_data.reindex(index=PLOT_STRUCTURES, columns=PLOT_STRUCTURES)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric.capitalize(), "shrink": 0.8},
        ax=ax,
    )

    # Highlight diagonal
    for i in range(len(PLOT_STRUCTURES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", lw=2))

    ax.set_title(
        f"{metric.upper()} — Cross-Evaluation: Individual Models", fontsize=14, pad=12
    )
    ax.set_xlabel("Tested On", fontsize=12)
    ax.set_ylabel("Trained On", fontsize=12)
    ax.tick_params(axis="both", labelsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save_figure(fig, outdir, "cross_eval_heatmap")


def plot_cross_eval_clustermap(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Hierarchically clustered cross-evaluation heatmap (Ward linkage).

    Same data as plot_cross_eval_heatmap but with rows/columns reordered by
    Ward hierarchical clustering to reveal structure groupings.
    """
    from scipy.cluster.hierarchy import linkage

    mask = df["eval_type"] == "cross_eval"
    subset = df[mask]
    if subset.empty:
        log.warning("No cross_eval data; skipping clustermap")
        return

    means = subset.groupby(["model_struct", "test_struct"])[metric].mean().reset_index()
    heatmap_data = means.pivot(
        index="model_struct", columns="test_struct", values=metric
    )
    heatmap_data = heatmap_data.reindex(index=PLOT_STRUCTURES, columns=PLOT_STRUCTURES)
    heatmap_data = heatmap_data.fillna(0)

    # Compute Ward linkage for both axes
    row_linkage = linkage(heatmap_data.values, method="ward", metric="euclidean")
    col_linkage = linkage(heatmap_data.values.T, method="ward", metric="euclidean")

    g = sns.clustermap(
        heatmap_data,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        figsize=(11, 10),
        cbar_kws={"label": metric.capitalize()},
        dendrogram_ratio=(0.12, 0.12),
    )

    g.ax_heatmap.set_xlabel("Tested On", fontsize=12)
    g.ax_heatmap.set_ylabel("Trained On", fontsize=12)
    g.ax_heatmap.tick_params(axis="both", labelsize=9)
    g.figure.suptitle(
        f"{metric.upper()} — Cross-Evaluation Clustermap (Ward linkage)",
        fontsize=14,
        y=1.02,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(outdir / "cross_eval_clustermap.pdf", bbox_inches="tight")
    g.figure.savefig(outdir / "cross_eval_clustermap.png", bbox_inches="tight")
    plt.close(g.figure)
    log.info("Saved cross_eval_clustermap.{pdf,png} to %s", outdir)


def plot_individual_vs_megaseg(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 2: Grouped bar chart — individual vs 4 MegaSeg backbones per structure."""
    mask = df["eval_type"].isin(["individual", "megaseg"]) & (
        df["test_set"] == "standard"
    )
    subset = df[mask].copy()
    if subset.empty:
        log.warning("No individual/megaseg data; skipping bar chart")
        return

    # Filter to PLOT_STRUCTURES only
    subset = subset[subset["test_struct"].isin(PLOT_STRUCTURES)]

    subset["model_label"] = subset.apply(
        lambda r: "individual"
        if r["eval_type"] == "individual"
        else f"megaseg_{r['backbone']}",
        axis=1,
    )

    stats = (
        subset.groupby(["test_struct", "model_label"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    model_order = [
        "individual",
        "megaseg_dynunet",
        "megaseg_attention_unet",
        "megaseg_swin_unetr",
        "megaseg_unet",
    ]
    model_labels = {
        "individual": "Individual",
        **{f"megaseg_{k}": v for k, v in BACKBONE_LABELS.items()},
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    n_structs = len(PLOT_STRUCTURES)
    n_models = len(model_order)
    bar_width = 0.15
    x = np.arange(n_structs)

    for i, model in enumerate(model_order):
        model_data = stats[stats["model_label"] == model].set_index("test_struct")
        means = [
            model_data.loc[s, "mean"] if s in model_data.index else 0
            for s in PLOT_STRUCTURES
        ]
        stds = [
            model_data.loc[s, "std"] if s in model_data.index else 0
            for s in PLOT_STRUCTURES
        ]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            label=model_labels.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(
        f"{metric.upper()} — Individual vs MegaSeg by Structure", fontsize=14, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PLOT_STRUCTURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "individual_vs_megaseg")


def plot_backbone_comparison(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 3: Grouped bar chart — 4 MegaSeg backbones per structure."""
    mask = (df["eval_type"] == "megaseg") & (df["test_set"] == "standard")
    subset = df[mask]
    if subset.empty:
        log.warning("No megaseg data; skipping backbone comparison")
        return

    # Filter to PLOT_STRUCTURES only
    subset = subset[subset["test_struct"].isin(PLOT_STRUCTURES)]

    stats = (
        subset.groupby(["test_struct", "backbone"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    n_structs = len(PLOT_STRUCTURES)
    n_bb = len(BACKBONES)
    bar_width = 0.18
    x = np.arange(n_structs)

    for i, bb in enumerate(BACKBONES):
        bb_data = stats[stats["backbone"] == bb].set_index("test_struct")
        means = [
            bb_data.loc[s, "mean"] if s in bb_data.index else 0 for s in PLOT_STRUCTURES
        ]
        stds = [
            bb_data.loc[s, "std"] if s in bb_data.index else 0 for s in PLOT_STRUCTURES
        ]
        offset = (i - n_bb / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            label=BACKBONE_LABELS.get(bb, bb),
            color=BACKBONE_COLORS.get(bb, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.upper()} — MegaSeg Backbone Comparison", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(PLOT_STRUCTURES, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "backbone_comparison")


def plot_delta_improvement(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 4: Horizontal bar chart — MegaSeg improvement over individual, 2x2 grid."""
    mask_ind = (df["eval_type"] == "individual") & (df["test_set"] == "standard")
    mask_mega = (df["eval_type"] == "megaseg") & (df["test_set"] == "standard")
    ind_means = df[mask_ind].groupby("test_struct")[metric].mean()
    mega_means = df[mask_mega].groupby(["backbone", "test_struct"])[metric].mean()

    if ind_means.empty or mega_means.empty:
        log.warning("Insufficient data for delta improvement chart")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, bb in enumerate(BACKBONES):
        ax = axes[idx]
        if bb in mega_means.index:
            bb_means = mega_means.loc[bb]
        else:
            ax.set_visible(False)
            continue

        # Compute deltas and sort (filter to PLOT_STRUCTURES)
        common = sorted(
            set(bb_means.index) & set(ind_means.index) & set(PLOT_STRUCTURES)
        )
        deltas = pd.Series(
            {s: bb_means[s] - ind_means[s] for s in common}
        ).sort_values()

        colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas.values]
        y_pos = np.arange(len(deltas))

        ax.barh(y_pos, deltas.values, color=colors, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(deltas.index, fontsize=9)
        ax.set_xlabel(f"Δ {metric.capitalize()}", fontsize=10)
        ax.set_title(BACKBONE_LABELS.get(bb, bb), fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

        # Annotate values
        for j, (val, name) in enumerate(zip(deltas.values, deltas.index)):
            ha = "left" if val >= 0 else "right"
            offset = 0.005 if val >= 0 else -0.005
            ax.text(val + offset, j, f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    fig.suptitle(
        f"{metric.upper()} — MegaSeg Improvement Over Individual",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, outdir, "delta_improvement")


def plot_test_new_generalization(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 5: Grouped bar chart — individual vs MegaSeg on held-out test_new structures."""
    mask = df["test_set"] == "test_new"
    subset = df[mask].copy()
    if subset.empty:
        log.warning("No test_new data; skipping generalization chart")
        return

    subset["model_label"] = subset.apply(
        lambda r: "individual"
        if r["eval_type"] == "individual_test_new"
        else f"megaseg_{r['backbone']}",
        axis=1,
    )

    stats = (
        subset.groupby(["test_struct", "model_label"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    structures = sorted(stats["test_struct"].unique())

    model_order = [
        "individual",
        "megaseg_dynunet",
        "megaseg_attention_unet",
        "megaseg_swin_unetr",
        "megaseg_unet",
    ]
    model_labels = {
        "individual": "Individual",
        **{f"megaseg_{k}": v for k, v in BACKBONE_LABELS.items()},
    }
    present_models = [m for m in model_order if m in stats["model_label"].values]

    fig, ax = plt.subplots(figsize=(max(10, len(structures) * 0.8), 6))
    n_structs = len(structures)
    n_models = len(present_models)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(n_structs)

    for i, model in enumerate(present_models):
        model_data = stats[stats["model_label"] == model].set_index("test_struct")
        means = [
            model_data.loc[s, "mean"] if s in model_data.index else 0
            for s in structures
        ]
        stds = [
            model_data.loc[s, "std"] if s in model_data.index else 0 for s in structures
        ]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            label=model_labels.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            alpha=0.85,
        )

    ax.set_xlabel("Structure (held-out)", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(
        f"{metric.upper()} — Generalization to Novel Structures (test_new)",
        fontsize=14,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "test_new_generalization")


def plot_summary_overview(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    """Plot 6: Overall mean metric by eval type — high-level comparison."""
    # Filter to PLOT_STRUCTURES for structure-based eval types
    plot_df = df[
        df["test_struct"].isin(PLOT_STRUCTURES) | (df["test_set"] == "test_new")
    ]
    categories = []

    # Individual
    ind = plot_df[
        (plot_df["eval_type"] == "individual") & (plot_df["test_set"] == "standard")
    ]
    if not ind.empty:
        categories.append(
            (
                "Individual",
                ind[metric].mean(),
                ind[metric].std(),
                MODEL_COLORS["individual"],
            )
        )

    # MegaSeg backbones
    for bb in BACKBONES:
        mega = plot_df[
            (plot_df["eval_type"] == "megaseg")
            & (plot_df["backbone"] == bb)
            & (plot_df["test_set"] == "standard")
        ]
        if not mega.empty:
            label = f"MegaSeg\n{BACKBONE_LABELS.get(bb, bb)}"
            categories.append(
                (label, mega[metric].mean(), mega[metric].std(), BACKBONE_COLORS[bb])
            )

    # Cross-eval diagonal vs off-diagonal
    cross = plot_df[plot_df["eval_type"] == "cross_eval"]
    if not cross.empty:
        diag = cross[cross["model_struct"] == cross["test_struct"]]
        off_diag = cross[cross["model_struct"] != cross["test_struct"]]
        if not diag.empty:
            categories.append(
                (
                    "Cross-Eval\n(same struct)",
                    diag[metric].mean(),
                    diag[metric].std(),
                    "#9467bd",
                )
            )
        if not off_diag.empty:
            categories.append(
                (
                    "Cross-Eval\n(transfer)",
                    off_diag[metric].mean(),
                    off_diag[metric].std(),
                    "#e377c2",
                )
            )

    if not categories:
        log.warning("No data for summary overview")
        return

    labels, means, stds, colors = zip(*categories)

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.2), 5))
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.6)

    # Reference line at individual mean
    ind_data = plot_df[
        (plot_df["eval_type"] == "individual") & (plot_df["test_set"] == "standard")
    ]
    if not ind_data.empty:
        ind_mean = ind_data[metric].mean()
        ax.axhline(
            y=ind_mean,
            color=MODEL_COLORS["individual"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Individual baseline ({ind_mean:.3f})",
        )

    # Annotate bar values
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Model / Evaluation Type", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.upper()} — Summary Overview", fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, outdir, "summary_overview")


# ---------------------------------------------------------------------------
# Plot orchestrator
# ---------------------------------------------------------------------------


def plot_all(
    project_root: Path,
    dtype: str,
    metric: str | None = None,
    outdir: Path | None = None,
    df: pd.DataFrame | None = None,
) -> None:
    """Generate all visualization plots.

    If *df* is provided it is used directly; otherwise the per-sample CSV is
    loaded from ``results/{dtype}/per_sample_metrics.csv``.
    """
    if df is None:
        df = _load_results(project_root, dtype)
    if outdir is None:
        outdir = project_root / "results" / dtype / "figures"

    setup_plot_style()

    metrics = [metric] if metric else METRIC_COLS
    for m in metrics:
        metric_dir = outdir / m
        log.info("Generating plots for metric=%s in %s", m, metric_dir)

        plot_cross_eval_heatmap(df, m, metric_dir)
        plot_cross_eval_clustermap(df, m, metric_dir)
        plot_individual_vs_megaseg(df, m, metric_dir)
        plot_backbone_comparison(df, m, metric_dir)
        plot_delta_improvement(df, m, metric_dir)
        plot_test_new_generalization(df, m, metric_dir)
        plot_summary_overview(df, m, metric_dir)

    log.info("All plots saved to %s", outdir)


# ---------------------------------------------------------------------------
# Low-signal sample filtering
# ---------------------------------------------------------------------------


def identify_low_signal_samples(
    df: pd.DataFrame,
    bottom_n: int = 10,
    metric: str = "dice",
    reference_backbone: str = "dynunet",
) -> pd.DataFrame:
    """Identify the bottom-N samples per structure by metric.

    Uses MegaSeg predictions with *reference_backbone* on the standard test
    set as the reference distribution.  Returns a DataFrame with columns
    ``[test_struct, sample_id, <metric>]`` for the excluded samples, sorted
    by structure then metric ascending.
    """
    ref = df[
        (df["eval_type"] == "megaseg")
        & (df["backbone"] == reference_backbone)
        & (df["test_set"] == "standard")
    ].copy()

    if ref.empty:
        raise ValueError(
            f"No megaseg/{reference_backbone}/standard rows found in data "
            f"— cannot identify low-signal samples"
        )

    excluded_rows = []
    for struct in sorted(ref["test_struct"].unique()):
        struct_df = ref[ref["test_struct"] == struct].sort_values(metric)
        bottom = struct_df.head(bottom_n)
        excluded_rows.append(bottom[["test_struct", "sample_id", metric]])

    excluded = pd.concat(excluded_rows, ignore_index=True)
    log.info(
        "Identified %d low-signal samples (%d structures x bottom %d by %s)",
        len(excluded),
        excluded["test_struct"].nunique(),
        bottom_n,
        metric,
    )
    return excluded


def filter_all(
    project_root: Path,
    dtype: str,
    bottom_n: int = 10,
    metric: str = "dice",
    reference_backbone: str = "dynunet",
    plot_metric: str | None = None,
) -> None:
    """Filter low-signal samples, save artefacts, regenerate reports & plots.

    Steps:
      1. Load the full per-sample CSV.
      2. Identify the bottom-N samples per structure (from megaseg/dynunet).
      3. Remove those samples from ALL eval types.
      4. Save filtered CSV + excluded-samples manifest.
      5. Re-run reports and plots into ``results/{dtype}/filtered/``.
    """
    df = _load_results(project_root, dtype)
    out_dir = project_root / "results" / dtype / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Identify bottom-N ---
    excluded = identify_low_signal_samples(
        df,
        bottom_n=bottom_n,
        metric=metric,
        reference_backbone=reference_backbone,
    )

    # Save the excluded manifest
    manifest_path = out_dir / "excluded_samples.csv"
    excluded.to_csv(manifest_path, index=False)
    log.info("Saved excluded-sample manifest to %s", manifest_path)

    # --- Step 2: Build exclusion set and filter ---
    exclude_keys = set(zip(excluded["test_struct"], excluded["sample_id"]))

    mask_keep = ~df.apply(
        lambda r: (r["test_struct"], r["sample_id"]) in exclude_keys, axis=1
    )
    df_filtered = df[mask_keep].copy()

    n_removed = len(df) - len(df_filtered)
    log.info(
        "Filtered: %d -> %d rows (%d removed across all eval types)",
        len(df),
        len(df_filtered),
        n_removed,
    )

    # Save filtered per-sample CSV
    filtered_csv = out_dir / "per_sample_metrics.csv"
    df_filtered.to_csv(filtered_csv, index=False)
    log.info("Saved filtered per-sample CSV to %s", filtered_csv)

    # Print removal breakdown by eval type
    for et in sorted(df["eval_type"].unique()):
        orig = len(df[df["eval_type"] == et])
        filt = len(df_filtered[df_filtered["eval_type"] == et])
        log.info("  %-25s %5d -> %5d  (-%d)", et, orig, filt, orig - filt)

    # --- Step 3: Reports ---
    report_all(
        project_root=project_root,
        dtype=dtype,
        metric=metric,
        df=df_filtered,
        output_dir=out_dir,
    )

    # --- Step 4: Plots ---
    plot_all(
        project_root=project_root,
        dtype=dtype,
        metric=plot_metric,
        outdir=out_dir / "figures",
        df=df_filtered,
    )

    log.info("All filtered artefacts written to %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MegaSeg benchmark evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("PROJECT_ROOT", ".")),
        help="Project root directory (default: $PROJECT_ROOT or cwd)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- compute --
    p_compute = sub.add_parser("compute", help="Compute metrics from TIFF pairs")
    p_compute.add_argument("--dtype", required=True, choices=["uint8", "float32"])
    p_compute.add_argument("--workers", type=int, default=min(cpu_count(), 16))
    p_compute.add_argument("--eval-type", choices=EVAL_TYPES, default=None)
    p_compute.add_argument("--backbone", choices=BACKBONES, default=None)
    p_compute.add_argument("--structure", choices=STRUCTURES, default=None)

    # -- report --
    p_report = sub.add_parser("report", help="Generate comparison tables from CSV")
    p_report.add_argument("--dtype", required=True, choices=["uint8", "float32"])
    p_report.add_argument("--metric", choices=METRIC_COLS, default="dice")

    # -- plot --
    p_plot = sub.add_parser("plot", help="Generate visualization plots from CSV")
    p_plot.add_argument("--dtype", required=True, choices=["uint8", "float32"])
    p_plot.add_argument(
        "--metric",
        choices=METRIC_COLS,
        default=None,
        help="Generate plots for this metric only (default: all)",
    )
    p_plot.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: results/{dtype}/figures)",
    )

    # -- filter --
    p_filter = sub.add_parser(
        "filter",
        help="Remove bottom-N low-signal samples per structure, regenerate reports & plots",
    )
    p_filter.add_argument("--dtype", required=True, choices=["uint8", "float32"])
    p_filter.add_argument(
        "--bottom-n",
        type=int,
        default=10,
        help="Number of lowest-scoring samples to remove per structure (default: 10)",
    )
    p_filter.add_argument(
        "--metric",
        choices=METRIC_COLS,
        default="dice",
        help="Metric used to rank samples for exclusion (default: dice)",
    )
    p_filter.add_argument(
        "--reference-backbone",
        choices=BACKBONES,
        default="dynunet",
        help="MegaSeg backbone used to identify low-signal samples (default: dynunet)",
    )
    p_filter.add_argument(
        "--plot-metric",
        choices=METRIC_COLS,
        default=None,
        help="Generate plots for this metric only (default: all)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "compute":
        compute_all(
            project_root=args.project_root,
            dtype=args.dtype,
            workers=args.workers,
            eval_type=args.eval_type,
            backbone=args.backbone,
            structure=args.structure,
        )
    elif args.command == "report":
        report_all(
            project_root=args.project_root,
            dtype=args.dtype,
            metric=args.metric,
        )
    elif args.command == "plot":
        plot_all(
            project_root=args.project_root,
            dtype=args.dtype,
            metric=args.metric,
            outdir=args.outdir,
        )
    elif args.command == "filter":
        filter_all(
            project_root=args.project_root,
            dtype=args.dtype,
            bottom_n=args.bottom_n,
            metric=args.metric,
            reference_backbone=args.reference_backbone,
            plot_metric=args.plot_metric,
        )


if __name__ == "__main__":
    main()
