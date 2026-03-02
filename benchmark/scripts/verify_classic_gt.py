#!/usr/bin/env python
"""Verify that classic segmentation workflows reproduce the ground truth (GT).

This script compares classic workflow outputs against GT segmentation masks to
detect discrepancies caused by library version changes (skimage 0.20->0.26, etc.).

Phase 1: Compare existing classic workflow outputs (ms_classic/{STRUCT}/) against
          GT from test_new.csv. No re-running needed.
Phase 2: Re-run classic workflows on samples from per-structure CSVs (train/val/test)
          and compare fresh output against GT seg files.

Usage (run with msseg conda env):
    /allen/aics/users/ashwin.samudre/miniconda3/envs/msseg/bin/python \\
        benchmark/scripts/verify_classic_gt.py [--phase1] [--phase2] [--samples N]
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from classic_workflows import (
    MS_CLASSIC_DIR,
    STRUCTURES,
    STRUCTURES_WITH_CSVS,
    STRUCTURES_WITH_OUTPUTS,
    dice_score,
    extract_structure_from_filename,
    load_image_3d,
    load_workflow_function,
)

# ---------------------------------------------------------------------------
# Paths (project-specific, not in shared module)
# ---------------------------------------------------------------------------
TEST_NEW_CSV = Path("/allen/aics/rep_learn/MegaSeg/data/standardized_test/test_new.csv")
VARIANCE_DIR = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x")


# ---------------------------------------------------------------------------
# Phase 1: Compare existing classic outputs vs test_new.csv GT
# ---------------------------------------------------------------------------
def run_phase1() -> pd.DataFrame:
    """Compare existing ms_classic outputs against test_new.csv GT."""
    print("=" * 70)
    print("PHASE 1: Comparing existing classic outputs vs test_new.csv GT")
    print("=" * 70)

    df_test = pd.read_csv(TEST_NEW_CSV)
    results = []

    for structure in sorted(STRUCTURES_WITH_OUTPUTS):
        print(f"\n--- {structure} ---")
        classic_dir = MS_CLASSIC_DIR / structure

        # Find test_new.csv images that belong to this structure
        for _, row in df_test.iterrows():
            raw_path = row["raw"]
            gt_path = row["seg"]
            filename = os.path.basename(raw_path)
            img_struct = extract_structure_from_filename(filename)

            if img_struct != structure:
                continue

            # Construct expected classic output path
            # Classic output naming: {STRUCT}_{ID}_IM_classic_seg.tiff
            classic_filename = filename.replace(".tiff", "_classic_seg.tiff")
            classic_path = classic_dir / classic_filename

            if not classic_path.exists():
                print(f"  MISSING: {classic_path.name}")
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test_new",
                        "source": "existing",
                        "dice": None,
                        "error": "classic output file missing",
                    }
                )
                continue

            try:
                pred = load_image_3d(str(classic_path))
                gt = load_image_3d(gt_path)
                d = dice_score(pred, gt)
                print(f"  {filename}: Dice = {d:.6f}")
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test_new",
                        "source": "existing",
                        "dice": d,
                        "error": None,
                    }
                )
            except Exception as e:
                print(f"  ERROR on {filename}: {e}")
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test_new",
                        "source": "existing",
                        "dice": None,
                        "error": str(e),
                    }
                )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Phase 2: Fresh workflow runs on per-structure CSV samples
# ---------------------------------------------------------------------------
def run_phase2(samples_per_split: int = 3) -> pd.DataFrame:
    """Run classic workflows on sampled images and compare vs GT."""
    print("\n" + "=" * 70)
    print(
        f"PHASE 2: Fresh workflow runs on per-structure CSVs "
        f"({samples_per_split} samples/split)"
    )
    print("=" * 70)

    results = []

    for structure in sorted(STRUCTURES_WITH_CSVS):
        print(f"\n--- {structure} ---")

        # Load the workflow function (with patches if needed)
        workflow_fn = load_workflow_function(structure)
        if workflow_fn is None:
            print(f"  SKIP: Could not load workflow for {structure}")
            continue

        # Load per-structure CSVs
        struct_csv_dir = VARIANCE_DIR / structure
        splits_loaded = {}
        for split_name in ("train", "val", "test"):
            csv_path = struct_csv_dir / f"{split_name}.csv"
            if csv_path.exists():
                splits_loaded[split_name] = pd.read_csv(csv_path)
            else:
                print(f"  WARNING: {csv_path.name} not found")

        if not splits_loaded:
            print(f"  SKIP: No CSVs found for {structure}")
            continue

        # Sample and process
        for split_name, df_split in splits_loaded.items():
            n = min(samples_per_split, len(df_split))
            sampled = df_split.sample(n=n, random_state=42)

            for _, row in sampled.iterrows():
                raw_path = row["raw"]
                gt_path = row["seg"]
                filename = os.path.basename(raw_path)

                if not os.path.exists(raw_path):
                    print(f"  MISSING raw: {filename}")
                    results.append(
                        {
                            "structure": structure,
                            "image_id": filename,
                            "split": split_name,
                            "source": "fresh_run",
                            "dice": None,
                            "error": "raw file missing",
                        }
                    )
                    continue

                if not os.path.exists(gt_path):
                    print(f"  MISSING seg: {os.path.basename(gt_path)}")
                    results.append(
                        {
                            "structure": structure,
                            "image_id": filename,
                            "split": split_name,
                            "source": "fresh_run",
                            "dice": None,
                            "error": "seg file missing",
                        }
                    )
                    continue

                try:
                    # Load image
                    struct_img = load_image_3d(raw_path)

                    # Run workflow (output_type="array" returns the seg directly)
                    pred = workflow_fn(struct_img, output_type="array")

                    # Load GT
                    gt = load_image_3d(gt_path)

                    # Compute Dice
                    d = dice_score(pred, gt)
                    print(f"  [{split_name}] {filename}: Dice = {d:.6f}")
                    results.append(
                        {
                            "structure": structure,
                            "image_id": filename,
                            "split": split_name,
                            "source": "fresh_run",
                            "dice": d,
                            "error": None,
                        }
                    )

                except Exception as e:
                    print(f"  [{split_name}] ERROR {filename}: {e}")
                    traceback.print_exc()
                    results.append(
                        {
                            "structure": structure,
                            "image_id": filename,
                            "split": split_name,
                            "source": "fresh_run",
                            "dice": None,
                            "error": str(e),
                        }
                    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Phase 3: Re-run workflows, save outputs, compare fresh vs existing
# ---------------------------------------------------------------------------
def run_phase3(
    n_samples: int = 20,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Re-run classic workflows on test_new.csv images, save fresh outputs,
    and compare them against existing classic outputs in ms_classic/{STRUCT}/.

    Args:
        n_samples: Total number of images to process (~20).
        output_dir: Directory to save fresh segmentation outputs.
    """
    print("\n" + "=" * 70)
    print(
        f"PHASE 3: Re-run workflows, save outputs, compare fresh vs existing "
        f"(sampling {n_samples} images)"
    )
    print("=" * 70)

    if output_dir is None:
        output_dir = str(
            Path(
                "/allen/aics/rep_learn/MegaSeg/benchmark/results/classic_fresh_outputs"
            )
        )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving fresh outputs to: {out_path}")

    # Load test_new.csv
    df_test = pd.read_csv(TEST_NEW_CSV)

    # Build candidate list: images whose structure has existing outputs AND
    # a loadable workflow
    candidates = []
    for _, row in df_test.iterrows():
        raw_path = row["raw"]
        gt_path = row["seg"]
        filename = os.path.basename(raw_path)
        structure = extract_structure_from_filename(filename)

        if structure not in STRUCTURES_WITH_OUTPUTS:
            continue
        if structure not in STRUCTURES:
            continue

        # Check existing classic output exists
        classic_filename = filename.replace(".tiff", "_classic_seg.tiff")
        classic_path = MS_CLASSIC_DIR / structure / classic_filename
        if not classic_path.exists():
            continue

        candidates.append(
            {
                "structure": structure,
                "filename": filename,
                "raw_path": raw_path,
                "gt_path": gt_path,
                "classic_path": str(classic_path),
            }
        )

    print(f"Found {len(candidates)} candidate images with existing classic outputs")

    # Sample
    df_cand = pd.DataFrame(candidates)
    if len(df_cand) > n_samples:
        # Stratified-ish: sample proportionally per structure, at least 1 each
        df_cand = (
            df_cand.groupby("structure", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(
                        len(x), max(1, n_samples // len(df_cand["structure"].unique()))
                    ),
                    random_state=42,
                )
            )
            .reset_index(drop=True)
        )
        # If still over, trim
        if len(df_cand) > n_samples:
            df_cand = df_cand.sample(n=n_samples, random_state=42).reset_index(
                drop=True
            )

    print(
        f"Processing {len(df_cand)} images across {df_cand['structure'].nunique()} structures"
    )

    # Load workflows (cache per structure)
    workflow_cache: dict[str, Optional[Callable]] = {}
    results = []

    for _, row in df_cand.iterrows():
        structure = row["structure"]
        filename = row["filename"]
        raw_path = row["raw_path"]
        gt_path = row["gt_path"]
        classic_path = row["classic_path"]

        # Load workflow (cached)
        if structure not in workflow_cache:
            print(f"\n--- Loading workflow: {structure} ---")
            workflow_cache[structure] = load_workflow_function(structure)

        workflow_fn = workflow_cache[structure]
        if workflow_fn is None:
            print(f"  SKIP {filename}: workflow failed to load")
            results.append(
                {
                    "structure": structure,
                    "image_id": filename,
                    "split": "test_new",
                    "source": "phase3_rerun",
                    "dice_fresh_vs_existing": None,
                    "dice_fresh_vs_gt": None,
                    "dice_existing_vs_gt": None,
                    "fresh_output_path": None,
                    "error": "workflow load failed",
                }
            )
            continue

        try:
            # Load raw image
            struct_img = load_image_3d(raw_path)

            # Run workflow
            pred_fresh = workflow_fn(struct_img, output_type="array")

            # Save fresh output
            fresh_filename = filename.replace(".tiff", f"_fresh_seg.tiff")
            fresh_out_path = out_path / structure / fresh_filename
            fresh_out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as uint8 {0, 255} to match classic output format
            pred_save = ((pred_fresh > 0).astype(np.uint8)) * 255
            from tifffile import imwrite

            imwrite(str(fresh_out_path), pred_save)

            # Load existing classic output
            pred_existing = load_image_3d(classic_path)

            # Load GT
            gt = load_image_3d(gt_path)

            # Compute all three Dice comparisons
            d_fresh_existing = dice_score(pred_fresh, pred_existing)
            d_fresh_gt = dice_score(pred_fresh, gt)
            d_existing_gt = dice_score(pred_existing, gt)

            print(
                f"  {filename}: "
                f"fresh_vs_existing={d_fresh_existing:.6f}  "
                f"fresh_vs_gt={d_fresh_gt:.6f}  "
                f"existing_vs_gt={d_existing_gt:.6f}"
            )

            results.append(
                {
                    "structure": structure,
                    "image_id": filename,
                    "split": "test_new",
                    "source": "phase3_rerun",
                    "dice_fresh_vs_existing": d_fresh_existing,
                    "dice_fresh_vs_gt": d_fresh_gt,
                    "dice_existing_vs_gt": d_existing_gt,
                    "fresh_output_path": str(fresh_out_path),
                    "error": None,
                }
            )

        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            traceback.print_exc()
            results.append(
                {
                    "structure": structure,
                    "image_id": filename,
                    "split": "test_new",
                    "source": "phase3_rerun",
                    "dice_fresh_vs_existing": None,
                    "dice_fresh_vs_gt": None,
                    "dice_existing_vs_gt": None,
                    "fresh_output_path": None,
                    "error": str(e),
                }
            )

    return pd.DataFrame(results)


def run_phase3_per_struct(
    samples_per_struct: int = 3,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Re-run classic workflows on per-structure test.csv images, save fresh
    outputs, and compare against GT seg files.

    Uses only the test split from each structure's CSV in variance-100x/.

    Args:
        samples_per_struct: Number of test images to process per structure.
        output_dir: Directory to save fresh segmentation outputs.
    """
    print("\n" + "=" * 70)
    print(
        f"PHASE 3b: Re-run workflows on per-structure test.csv "
        f"({samples_per_struct} samples/structure), saving outputs"
    )
    print("=" * 70)

    if output_dir is None:
        output_dir = str(
            Path(
                "/allen/aics/rep_learn/MegaSeg/benchmark/results"
                "/classic_fresh_outputs_per_struct"
            )
        )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving fresh outputs to: {out_path}")

    # Load workflows (cache per structure)
    workflow_cache: dict[str, Optional[Callable]] = {}
    results = []

    for structure in sorted(STRUCTURES_WITH_CSVS):
        print(f"\n--- {structure} ---")

        # Load test.csv for this structure
        test_csv = VARIANCE_DIR / structure / "test.csv"
        if not test_csv.exists():
            print(f"  SKIP: {test_csv} not found")
            continue

        df_test = pd.read_csv(test_csv)
        n = min(samples_per_struct, len(df_test))
        sampled = df_test.sample(n=n, random_state=42)

        # Load workflow (cached)
        if structure not in workflow_cache:
            workflow_cache[structure] = load_workflow_function(structure)

        workflow_fn = workflow_cache[structure]
        if workflow_fn is None:
            print(f"  SKIP: Could not load workflow for {structure}")
            for _, row in sampled.iterrows():
                filename = os.path.basename(row["raw"])
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test",
                        "source": "phase3b_per_struct",
                        "dice_fresh_vs_gt": None,
                        "fresh_output_path": None,
                        "error": "workflow load failed",
                    }
                )
            continue

        for _, row in sampled.iterrows():
            raw_path = row["raw"]
            gt_path = row["seg"]
            filename = os.path.basename(raw_path)

            if not os.path.exists(raw_path):
                print(f"  MISSING raw: {filename}")
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test",
                        "source": "phase3b_per_struct",
                        "dice_fresh_vs_gt": None,
                        "fresh_output_path": None,
                        "error": "raw file missing",
                    }
                )
                continue

            if not os.path.exists(gt_path):
                print(f"  MISSING seg: {os.path.basename(gt_path)}")
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test",
                        "source": "phase3b_per_struct",
                        "dice_fresh_vs_gt": None,
                        "fresh_output_path": None,
                        "error": "seg file missing",
                    }
                )
                continue

            try:
                # Load raw image
                struct_img = load_image_3d(raw_path)

                # Run workflow
                pred_fresh = workflow_fn(struct_img, output_type="array")

                # Save fresh output
                fresh_filename = filename.replace(".tiff", "_fresh_seg.tiff")
                fresh_out_path = out_path / structure / fresh_filename
                fresh_out_path.parent.mkdir(parents=True, exist_ok=True)

                pred_save = ((pred_fresh > 0).astype(np.uint8)) * 255
                from tifffile import imwrite

                imwrite(str(fresh_out_path), pred_save)

                # Load GT
                gt = load_image_3d(gt_path)

                # Compute Dice
                d = dice_score(pred_fresh, gt)
                print(f"  [test] {filename}: Dice = {d:.6f}  -> {fresh_out_path.name}")

                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test",
                        "source": "phase3b_per_struct",
                        "dice_fresh_vs_gt": d,
                        "fresh_output_path": str(fresh_out_path),
                        "error": None,
                    }
                )

            except Exception as e:
                print(f"  [test] ERROR {filename}: {e}")
                traceback.print_exc()
                results.append(
                    {
                        "structure": structure,
                        "image_id": filename,
                        "split": "test",
                        "source": "phase3b_per_struct",
                        "dice_fresh_vs_gt": None,
                        "fresh_output_path": None,
                        "error": str(e),
                    }
                )

    return pd.DataFrame(results)


def print_phase3b_summary(df: pd.DataFrame) -> None:
    """Print Phase 3b summary."""
    if df.empty:
        print("\nPHASE 3b: No results.")
        return

    print(f"\n{'=' * 70}")
    print("PHASE 3b SUMMARY")
    print(f"{'=' * 70}")

    valid = df[df["dice_fresh_vs_gt"].notna()].copy()
    errors = df[df["dice_fresh_vs_gt"].isna()]

    if not valid.empty:
        summary = (
            valid.groupby("structure")["dice_fresh_vs_gt"]
            .agg(["count", "mean", "min", "max", "std"])
            .round(6)
        )
        summary.columns = ["N", "Mean Dice", "Min Dice", "Max Dice", "Std"]
        print("\nFresh output vs GT (per-structure test.csv):")
        print(summary.to_string())

        print(
            f"\nOverall: N={len(valid)}, "
            f"Mean={valid['dice_fresh_vs_gt'].mean():.6f}, "
            f"Min={valid['dice_fresh_vs_gt'].min():.6f}, "
            f"Max={valid['dice_fresh_vs_gt'].max():.6f}"
        )

        # Show all rows sorted
        print("\nAll comparisons (sorted by Dice):")
        for _, row in valid.sort_values("dice_fresh_vs_gt").iterrows():
            print(
                f"  {row['structure']:>12s} | {row['image_id']:>30s} | "
                f"Dice={row['dice_fresh_vs_gt']:.6f}"
            )

    if not errors.empty:
        print(f"\nErrors ({len(errors)}):")
        for _, row in errors.iterrows():
            print(f"  {row['structure']:>12s} | {row['image_id']}: {row['error']}")


def print_phase3_summary(df: pd.DataFrame) -> None:
    """Print Phase 3 summary with all three Dice comparisons."""
    if df.empty:
        print("\nPHASE 3: No results.")
        return

    print(f"\n{'=' * 70}")
    print("PHASE 3 SUMMARY")
    print(f"{'=' * 70}")

    valid = df[df["dice_fresh_vs_existing"].notna()].copy()
    errors = df[df["dice_fresh_vs_existing"].isna()]

    if not valid.empty:
        # Per-structure summary for fresh_vs_existing
        summary = (
            valid.groupby("structure")["dice_fresh_vs_existing"]
            .agg(["count", "mean", "min", "max"])
            .round(6)
        )
        summary.columns = ["N", "Mean", "Min", "Max"]
        print("\nFresh vs Existing classic output (Dice):")
        print(summary.to_string())

        print(
            f"\nOverall fresh_vs_existing: N={len(valid)}, "
            f"Mean={valid['dice_fresh_vs_existing'].mean():.6f}, "
            f"Min={valid['dice_fresh_vs_existing'].min():.6f}, "
            f"Max={valid['dice_fresh_vs_existing'].max():.6f}"
        )

        # Show all rows sorted by fresh_vs_existing Dice
        print("\nAll comparisons (sorted by fresh_vs_existing Dice):")
        print(
            f"  {'Structure':>12s} | {'Image':>30s} | {'fresh_v_exist':>13s} | {'fresh_v_gt':>10s} | {'exist_v_gt':>10s}"
        )
        print(f"  {'-' * 12} | {'-' * 30} | {'-' * 13} | {'-' * 10} | {'-' * 10}")
        for _, row in valid.sort_values("dice_fresh_vs_existing").iterrows():
            print(
                f"  {row['structure']:>12s} | {row['image_id']:>30s} | "
                f"{row['dice_fresh_vs_existing']:>13.6f} | "
                f"{row['dice_fresh_vs_gt']:>10.6f} | "
                f"{row['dice_existing_vs_gt']:>10.6f}"
            )

    if not errors.empty:
        print(f"\nErrors ({len(errors)}):")
        for _, row in errors.iterrows():
            print(f"  {row['structure']:>12s} | {row['image_id']}: {row['error']}")


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, phase_label: str) -> None:
    """Print a summary table of results."""
    if df.empty:
        print(f"\n{phase_label}: No results.")
        return

    print(f"\n{'=' * 70}")
    print(f"{phase_label} SUMMARY")
    print(f"{'=' * 70}")

    valid = df[df["dice"].notna()].copy()
    errors = df[df["dice"].isna()]

    if not valid.empty:
        # Per-structure summary
        summary = (
            valid.groupby("structure")["dice"]
            .agg(["count", "mean", "min", "max", "std"])
            .round(6)
        )
        summary.columns = ["N", "Mean Dice", "Min Dice", "Max Dice", "Std"]
        print(f"\nPer-structure Dice scores:")
        print(summary.to_string())

        # Overall
        print(
            f"\nOverall: N={len(valid)}, Mean Dice={valid['dice'].mean():.6f}, "
            f"Min={valid['dice'].min():.6f}, Max={valid['dice'].max():.6f}"
        )

        # Flag anything with Dice < 1.0 (potential discrepancy)
        imperfect = valid[valid["dice"] < 1.0]
        if not imperfect.empty:
            print(f"\nImages with Dice < 1.0 ({len(imperfect)} of {len(valid)}):")
            for _, row in imperfect.sort_values("dice").iterrows():
                print(
                    f"  {row['structure']:>12s} | {row['split']:>8s} | "
                    f"{row['image_id']}: {row['dice']:.6f}"
                )
        else:
            print("\nAll images have Dice = 1.0 (perfect match).")

    if not errors.empty:
        print(f"\nErrors ({len(errors)}):")
        for _, row in errors.iterrows():
            print(f"  {row['structure']:>12s} | {row['image_id']}: {row['error']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify classic segmentation workflows reproduce GT."
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        help="Run Phase 1: compare existing outputs vs test_new.csv GT",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Run Phase 2: fresh workflow runs on per-structure CSVs",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples per split for Phase 2 (default: 3)",
    )
    parser.add_argument(
        "--phase3",
        action="store_true",
        help="Run Phase 3: re-run workflows, save outputs, compare fresh vs existing",
    )
    parser.add_argument(
        "--phase3-per-struct",
        action="store_true",
        help="Run Phase 3b: re-run workflows on per-structure test.csv, save outputs",
    )
    parser.add_argument(
        "--save-outputs",
        type=str,
        default=None,
        help="Directory to save fresh segmentation outputs (Phase 3/3b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV (optional)",
    )
    args = parser.parse_args()

    # Default: run phases 1+2 if none specified
    if (
        not args.phase1
        and not args.phase2
        and not args.phase3
        and not args.phase3_per_struct
    ):
        args.phase1 = True
        args.phase2 = True

    all_results = []

    if args.phase1:
        df1 = run_phase1()
        print_summary(df1, "PHASE 1")
        all_results.append(df1)

    if args.phase2:
        df2 = run_phase2(samples_per_split=args.samples)
        print_summary(df2, "PHASE 2")
        all_results.append(df2)

    if args.phase3:
        df3 = run_phase3(n_samples=args.samples, output_dir=args.save_outputs)
        print_phase3_summary(df3)
        # Save Phase 3 results separately (different columns than Phase 1/2)
        p3_out = Path(
            "/allen/aics/rep_learn/MegaSeg/benchmark/results"
            "/classic_gt_verification_phase3.csv"
        )
        p3_out.parent.mkdir(parents=True, exist_ok=True)
        df3.to_csv(p3_out, index=False)
        print(f"\nPhase 3 results saved to: {p3_out}")

    if args.phase3_per_struct:
        df3b = run_phase3_per_struct(
            samples_per_struct=args.samples, output_dir=args.save_outputs
        )
        print_phase3b_summary(df3b)
        p3b_out = Path(
            "/allen/aics/rep_learn/MegaSeg/benchmark/results"
            "/classic_gt_verification_phase3b.csv"
        )
        p3b_out.parent.mkdir(parents=True, exist_ok=True)
        df3b.to_csv(p3b_out, index=False)
        print(f"\nPhase 3b results saved to: {p3b_out}")

    # Save combined Phase 1+2 results (Phase 3/3b have different columns)
    phase12 = [r for r in all_results]
    if phase12:
        combined = pd.concat(phase12, ignore_index=True)
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path(
                "/allen/aics/rep_learn/MegaSeg/benchmark/results"
                "/classic_gt_verification.csv"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
