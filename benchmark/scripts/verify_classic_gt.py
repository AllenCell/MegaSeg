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
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from aicsimageio import AICSImage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MS_CLASSIC_DIR = Path("/allen/aics/lumenoid/inner_outer_seg/megaseg_exp/ms_classic")
TEST_NEW_CSV = Path("/allen/aics/rep_learn/MegaSeg/data/standardized_test/test_new.csv")
VARIANCE_DIR = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x")

# ---------------------------------------------------------------------------
# Structure registry: script filename, workflow function name, status
# ---------------------------------------------------------------------------
STRUCTURES = {
    # structure: (script_basename, workflow_func_name, broken_reason_or_None)
    "ACTN1": ("ms_seg_ACTN1.py", "Workflow_actn1", None),
    "CETN2": ("classic_segmentation_CETN2.py", "Workflow_cetn2", "peak_local_max"),
    "CTNNB1": ("classic_segmentation_CTNNB1.py", "Workflow_ctnnb1", None),
    "FBL": ("classic_segmentation_FBL.py", "Workflow_fbl", None),
    "LAMP1": ("classic_segmentation_LAMP1.py", "Workflow_lamp1", None),
    "LMNB1": (
        "classic_segmentation_LMNB1.py",
        "Workflow_lmnb1_interphase",
        "watershed_import",
    ),
    "MYH10": ("classic_segmentation_MYH10.py", "Workflow_myh10", None),
    "NPM1": ("classic_segmentation_NPM1.py", "Workflow_npm1", None),
    "NUP153": ("classic_segmentation_NUP153.py", "Workflow_nup153", None),
    "RAB5A": ("classic_segmentation_RAB5A.py", "Workflow_rab5a", None),
    "SEC61B": ("classic_segmentation_SEC61B.py", "Workflow_sec61b", None),
    "SEC61B_DUAL": (
        "classic_segmentation_SEC61B_DUAL.py",
        "Workflow_sec61b_dual",
        None,
    ),
    "SLC25A17": (
        "classic_segmentation_SLC25A17.py",
        "Workflow_slc25a17",
        "peak_local_max",
    ),
    "SON": ("classic_segmentation_SON.py", "Workflow_son", None),
    "ST6GAL1": ("classic_segmentation_ST6GAL1.py", "Workflow_st6gal1", None),
    "TOMM20": ("classic_segmentation_TOMM20.py", "Workflow_tomm20", "main_only"),
    "TUBA1B": ("classic_segmentation_TUBA1B.py", "Workflow_tuba1b", None),
}

# Structures that have per-structure CSVs in variance-100x/
STRUCTURES_WITH_CSVS = {
    "ACTN1",
    "CTNNB1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TOMM20",
}

# Structures with existing classic outputs (13 populated directories)
STRUCTURES_WITH_OUTPUTS = {
    "ACTN1",
    "CTNNB1",
    "FBL",
    "LAMP1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SEC61B_DUAL",
    "SON",
    "ST6GAL1",
    "TUBA1B",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def load_image_3d(path: str) -> np.ndarray:
    """Load a TIFF and squeeze to 3D (Z, Y, X)."""
    img = AICSImage(path)
    return np.squeeze(img.data)


def extract_structure_from_filename(filename: str) -> str:
    """Extract structure name from filenames like 'NPM1_13054_IM.tiff'."""
    # Handle both _IM.tiff and plain .tiff naming
    base = filename.replace("_IM.tiff", "").replace("_GT.tiff", "")
    base = base.replace("_seg.tiff", "").replace(".tiff", "")
    # Structure name is everything before the last underscore + numeric ID
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def patch_script_source(script_path: Path, broken_reason: str) -> str:
    """Read script source and apply patches for known issues."""
    source = script_path.read_text()

    if broken_reason == "watershed_import":
        # Fix: move watershed from skimage.morphology to skimage.segmentation
        source = source.replace(
            "from skimage.morphology import remove_small_objects, watershed, dilation, ball",
            "from skimage.morphology import remove_small_objects, dilation, ball\n"
            "from skimage.segmentation import watershed",
        )

    elif broken_reason == "peak_local_max":
        # Fix: convert peak_local_max coordinate output to boolean mask
        # Insert a helper function after imports, before the Workflow def
        helper = '''
def _peak_local_max_mask(image, **kwargs):
    """Wrapper: convert peak_local_max coords to boolean mask (skimage >=0.24 compat)."""
    from skimage.feature import peak_local_max as _plm
    coords = _plm(image, **kwargs)
    mask = np.zeros(image.shape, dtype=bool)
    if len(coords) > 0:
        mask[tuple(coords.T)] = True
    return mask
'''
        # Insert helper before the first "def Workflow_" line
        idx = source.find("\ndef Workflow_")
        if idx == -1:
            idx = source.find("def Workflow_")
        source = source[:idx] + helper + source[idx:]

        # Replace peak_local_max calls with the wrapper
        source = source.replace(
            "local_maxi = peak_local_max(",
            "local_maxi = _peak_local_max_mask(",
        )

    # For "main_only" (TOMM20), we only need the Workflow function,
    # not the broken main(). No source patching needed since we import
    # just the workflow function.

    return source


def load_workflow_function(
    structure: str,
) -> Optional[Callable]:
    """Dynamically import the workflow function for a structure."""
    script_basename, func_name, broken_reason = STRUCTURES[structure]
    script_path = MS_CLASSIC_DIR / script_basename

    if not script_path.exists():
        print(f"  WARNING: Script not found: {script_path}")
        return None

    try:
        if broken_reason and broken_reason != "main_only":
            # Apply source patches and compile
            patched_source = patch_script_source(script_path, broken_reason)
            module_name = f"classic_seg_{structure}_patched"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(compile(patched_source, str(script_path), "exec"), module.__dict__)
        else:
            # Normal import
            module_name = f"classic_seg_{structure}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        return getattr(module, func_name)

    except Exception as e:
        print(f"  ERROR importing {structure}: {e}")
        traceback.print_exc()
        return None


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
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV (optional)",
    )
    args = parser.parse_args()

    # Default: run both phases if neither specified
    if not args.phase1 and not args.phase2:
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

    # Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
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
