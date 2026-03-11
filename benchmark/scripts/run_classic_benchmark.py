#!/usr/bin/env python
"""Run classic aicssegmentation workflows on all test samples and compute metrics.

Runs the actual aicssegmentation classic workflows (the same workflows used to
create the production GT masks) on the 67-sample test set for each structure.
Only structures with available classic workflow scripts are evaluated (10 of 14).

Outputs:
    results/uint8/classic_workflow_metrics.csv - per-sample metrics

Usage (requires msseg conda env):
    PYTHON=/allen/aics/users/ashwin.samudre/miniconda3/envs/msseg/bin/python
    $PYTHON benchmark/scripts/run_classic_benchmark.py
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread

# --- Add scripts dir to path for classic_workflows import ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from classic_workflows import (
    STRUCTURES_WITH_CSVS,
    load_workflow_function,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Worktree is sparse-checked-out (no data/ dir), so point DATA_DIR at the main clone.
MAIN_CLONE = Path("/allen/aics/rep_learn/MegaSeg")
WORKTREE_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MAIN_CLONE / "data" / "variance-100x"
OUTPUT_CSV = (
    WORKTREE_ROOT / "benchmark" / "results" / "uint8" / "classic_workflow_metrics.csv"
)

# The 10 structures that have both test CSVs and classic workflow scripts
STRUCTURES = sorted(STRUCTURES_WITH_CSVS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute Dice, IoU, precision, recall, accuracy between binary masks."""
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()

    total = tp + fp + fn + tn

    dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    accuracy = float((tp + tn) / total) if total > 0 else 1.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_existing() -> pd.DataFrame:
    """Load previously saved results for resume support."""
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"  Loaded {len(df)} existing rows from {OUTPUT_CSV}")
        return df
    return pd.DataFrame()


def _save(all_rows: list[dict]) -> None:
    """Save all rows to CSV (overwrites)."""
    result_df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)


def main():
    print(f"Classic workflow benchmark")
    print(f"Structures: {STRUCTURES}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    # Resume support: load any previously saved rows
    existing_df = _load_existing()
    completed_structs = set()
    all_rows: list[dict] = []
    if len(existing_df) > 0:
        all_rows = existing_df.to_dict("records")
        # A structure is complete if it has 67 rows (full test set)
        for struct in STRUCTURES:
            n_existing = len(existing_df[existing_df["test_struct"] == struct])
            if n_existing >= 67:
                completed_structs.add(struct)
                print(f"  RESUME: {struct} already done ({n_existing} rows)")

    total_errors = 0

    for structure in STRUCTURES:
        if structure in completed_structs:
            continue

        # Load test CSV
        test_csv = DATA_DIR / structure / "test.csv"
        if not test_csv.exists():
            print(f"  SKIP {structure}: no test.csv at {test_csv}")
            continue

        df = pd.read_csv(test_csv)
        n_samples = len(df)
        print(f"\n--- {structure} ({n_samples} samples) ---", flush=True)

        # Load workflow function
        workflow_fn = load_workflow_function(structure)
        if workflow_fn is None:
            print(f"  SKIP: Could not load workflow for {structure}")
            continue

        # Remove any partial rows for this structure before re-running
        all_rows = [r for r in all_rows if r["test_struct"] != structure]

        struct_errors = 0
        struct_t0 = time.time()

        for idx, row in df.iterrows():
            raw_path = row["raw"]
            seg_path = row["seg"]

            # Extract sample ID from filename
            raw_name = Path(raw_path).stem
            sample_id = raw_name  # e.g., ACTN1_11347

            if not Path(raw_path).exists():
                print(f"  MISSING raw: {raw_path}")
                struct_errors += 1
                continue
            if not Path(seg_path).exists():
                print(f"  MISSING seg: {seg_path}")
                struct_errors += 1
                continue

            try:
                # Load raw image
                raw_img = np.squeeze(imread(raw_path))

                # Run classic workflow
                pred = workflow_fn(raw_img, output_type="array")
                pred_bin = (pred > 0).astype(bool)

                # Load GT
                gt_img = np.squeeze(imread(seg_path))
                gt_bin = (gt_img > 0).astype(bool)

                # Compute metrics
                metrics = compute_metrics(pred_bin, gt_bin)

                all_rows.append(
                    {
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "accuracy": metrics["accuracy"],
                        "eval_type": "classic_workflow",
                        "test_struct": structure,
                        "sample_id": sample_id,
                    }
                )

            except Exception as e:
                print(f"  ERROR {sample_id}: {e}")
                traceback.print_exc()
                struct_errors += 1

        struct_elapsed = time.time() - struct_t0
        struct_rows = [r for r in all_rows if r["test_struct"] == structure]
        if struct_rows:
            struct_dice = np.mean([r["dice"] for r in struct_rows])
            print(
                f"  {structure}: {len(struct_rows)}/{n_samples} done, "
                f"mean Dice={struct_dice:.4f}, "
                f"errors={struct_errors}, {struct_elapsed:.1f}s",
                flush=True,
            )
        total_errors += struct_errors

        # Incremental save after each structure
        _save(all_rows)
        print(f"  [saved {len(all_rows)} total rows to {OUTPUT_CSV}]", flush=True)

    # Final summary
    if all_rows:
        result_df = pd.DataFrame(all_rows)
        print(f"\n{'=' * 60}")
        print(f"Final: {len(result_df)} rows in {OUTPUT_CSV}")
        print(f"Total errors: {total_errors}")
        print()

        print("Per-structure summary:")
        for struct in STRUCTURES:
            sub = result_df[result_df["test_struct"] == struct]
            if len(sub) > 0:
                print(
                    f"  {struct:10s}: n={len(sub):3d}, "
                    f"Dice={sub['dice'].mean():.4f} +/- {sub['dice'].std():.4f}"
                )

        print(
            f"\n  OVERALL: n={len(result_df)}, "
            f"Dice={result_df['dice'].mean():.4f} +/- {result_df['dice'].std():.4f}"
        )
    else:
        print("\nNo results generated!")


if __name__ == "__main__":
    main()
