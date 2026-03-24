#!/usr/bin/env python
"""Classic aicssegmentation cross-eval: run 14 workflows on 14 test structures.

Runs each of the 14 classic workflows on every structure's test set
(67 images each), producing:
  - Prediction TIFFs in {output_dir}/{WORKFLOW}/tested_on_{TEST_STRUCT}/
  - Per-sample metrics CSV (same schema as per_sample_metrics.csv)
  - 14x14 heatmap CSV (rows=workflow, cols=test_struct, values=mean Dice)

Supports --skip-existing to resume interrupted runs, --metrics-only to
recompute metrics from saved TIFFs without re-running workflows, and
--workflows to run a subset in parallel.

Usage:
    python scripts/run_classic_cross_eval.py
    python scripts/run_classic_cross_eval.py --skip-existing
    python scripts/run_classic_cross_eval.py --metrics-only
    python scripts/run_classic_cross_eval.py --dry-run
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# ---------------------------------------------------------------------------
# Ensure benchmark/scripts is on sys.path for sibling imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from classic_workflows import load_image_3d, load_workflow_function


# ---------------------------------------------------------------------------
# Metric helpers (inlined from evaluate_benchmark.py to avoid heavy imports)
# ---------------------------------------------------------------------------


def binarize_gt(gt: np.ndarray) -> np.ndarray:
    """Binarize ground truth mask (uint8 0/255) to boolean."""
    return gt > 0


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 14 classic workflows - the 10 with local scripts in ms_classic/ plus
# 4 imported from the installed aicssegmentation package (ATP2A2, DSP, GJA1, TJP1).
WORKFLOWS = [
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

# All 14 test structures.
TEST_STRUCTS = [
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # benchmark/
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "data" / "variance-100x"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "uint8" / "classic_cross_eval"
DEFAULT_METRICS_CSV = (
    PROJECT_ROOT / "results" / "uint8" / "classic_cross_eval_metrics.csv"
)
DEFAULT_HEATMAP_CSV = (
    PROJECT_ROOT / "results" / "uint8" / "classic_cross_eval_heatmap.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_sample_id(raw_path: str) -> str:
    """Extract sample ID from a raw path like .../ACTN1_10917.tiff -> ACTN1_10917."""
    name = Path(raw_path).stem
    # Strip _IM suffix if present
    if name.endswith("_IM"):
        name = name[:-3]
    return name


def load_test_csv(data_dir: Path, test_struct: str) -> pd.DataFrame:
    """Load test.csv for a structure, returning DataFrame with raw/seg columns."""
    csv_path = data_dir / test_struct / "test.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping {test_struct}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "raw" not in df.columns or "seg" not in df.columns:
        print(f"  WARNING: {csv_path} missing raw/seg columns")
        return pd.DataFrame()
    return df


def format_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Core: run predictions + compute metrics
# ---------------------------------------------------------------------------


def run_cross_eval(
    data_dir: Path,
    output_dir: Path,
    workflows: list[str] | None = None,
    skip_existing: bool = False,
    metrics_only: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Run all workflow x test_struct combinations.

    Returns a list of metric dicts (one per image).
    """
    wf_list = workflows if workflows is not None else WORKFLOWS
    total_cells = len(wf_list) * len(TEST_STRUCTS)
    total_images = total_cells * 67  # approximate
    results = []
    global_count = 0
    global_errors = 0
    global_skipped = 0
    t_start = time.time()

    for wi, workflow in enumerate(wf_list, 1):
        print(f"\n{'=' * 70}")
        print(f"Workflow {wi}/{len(wf_list)}: {workflow}")
        print(f"{'=' * 70}")

        # Load workflow function once per workflow (expensive import)
        workflow_fn = None
        if not dry_run and not metrics_only:
            workflow_fn = load_workflow_function(workflow)
            if workflow_fn is None:
                print(f"  FATAL: Could not load workflow for {workflow}, skipping all")
                global_skipped += len(TEST_STRUCTS) * 67
                continue

        for ti, test_struct in enumerate(TEST_STRUCTS, 1):
            cell_label = f"  [{workflow} -> {test_struct}] ({ti}/{len(TEST_STRUCTS)})"
            print(f"\n{cell_label}")

            # Load test CSV
            test_df = load_test_csv(data_dir, test_struct)
            if test_df.empty:
                continue

            # Output directory for this cell
            cell_out = output_dir / workflow / f"tested_on_{test_struct}"
            if not dry_run:
                cell_out.mkdir(parents=True, exist_ok=True)

            cell_ok = 0
            cell_err = 0

            for _, row in test_df.iterrows():
                raw_path = row["raw"]
                gt_path = row["seg"]
                sample_id = extract_sample_id(raw_path)
                out_name = f"{sample_id}_seg.tiff"
                out_path = cell_out / out_name

                global_count += 1

                if dry_run:
                    print(f"    [dry-run] {sample_id} -> {out_path}")
                    cell_ok += 1
                    continue

                try:
                    # --- Prediction ---
                    pred_arr = None

                    if metrics_only or (skip_existing and out_path.exists()):
                        # Load existing prediction
                        if out_path.exists():
                            pred_arr = tifffile.imread(str(out_path))
                        else:
                            # metrics_only but no TIFF - record NaN
                            results.append(
                                _nan_record(workflow, test_struct, sample_id)
                            )
                            cell_err += 1
                            continue
                    else:
                        # Run workflow
                        t0 = time.time()
                        struct_img = load_image_3d(raw_path)
                        pred_arr = workflow_fn(struct_img, output_type="array")

                        # Save as uint8 {0, 255}
                        pred_save = ((pred_arr > 0).astype(np.uint8)) * 255
                        tifffile.imwrite(str(out_path), pred_save)
                        pred_arr = pred_save  # use saved version for metrics
                        elapsed = time.time() - t0

                        if global_count <= 5 or global_count % 100 == 0:
                            print(f"    {sample_id} ({elapsed:.1f}s)")

                    # --- Metrics ---
                    gt_arr = tifffile.imread(str(gt_path))
                    gt_bin = binarize_gt(gt_arr)
                    pred_bin = binarize_prediction(pred_arr, "uint8")
                    metrics = compute_metrics(gt_bin, pred_bin)

                    results.append(
                        {
                            "eval_type": "classic_cross_eval",
                            "dtype": "uint8",
                            "backbone": "",
                            "model_struct": workflow,
                            "test_struct": test_struct,
                            "test_set": "standard",
                            "sample_id": sample_id,
                            **metrics,
                        }
                    )
                    cell_ok += 1

                except Exception as e:
                    print(f"    ERROR {sample_id}: {e}")
                    results.append(_nan_record(workflow, test_struct, sample_id))
                    cell_err += 1
                    global_errors += 1

            elapsed_so_far = time.time() - t_start
            rate = global_count / max(elapsed_so_far, 1)
            remaining = (total_images - global_count) / max(rate, 0.01)
            print(
                f"  {cell_label}: {cell_ok} ok, {cell_err} err "
                f"| total: {global_count}/{total_images} "
                f"| elapsed: {format_elapsed(elapsed_so_far)} "
                f"| ETA: {format_elapsed(remaining)}"
            )

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE: {global_count} images processed in {format_elapsed(elapsed_total)}")
    print(f"  OK: {global_count - global_errors - global_skipped}")
    print(f"  Errors: {global_errors}")
    print(f"  Skipped (workflow load fail): {global_skipped}")
    print(f"{'=' * 70}")

    return results


def _nan_record(workflow: str, test_struct: str, sample_id: str) -> dict:
    """Create a NaN metric record for failed images."""
    return {
        "eval_type": "classic_cross_eval",
        "dtype": "uint8",
        "backbone": "",
        "model_struct": workflow,
        "test_struct": test_struct,
        "test_set": "standard",
        "sample_id": sample_id,
        "dice": float("nan"),
        "iou": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "accuracy": float("nan"),
    }


# ---------------------------------------------------------------------------
# Output: metrics CSV + heatmap CSV
# ---------------------------------------------------------------------------


def write_metrics_csv(results: list[dict], csv_path: Path) -> None:
    """Write per-sample metrics CSV."""
    df = pd.DataFrame(results)
    cols = [
        "eval_type",
        "dtype",
        "backbone",
        "model_struct",
        "test_struct",
        "test_set",
        "sample_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "accuracy",
    ]
    df = df[cols]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics CSV: {csv_path} ({len(df)} rows)")


def write_heatmap_csv(results: list[dict], csv_path: Path) -> None:
    """Write NxM heatmap CSV (mean Dice per workflow x test_struct)."""
    df = pd.DataFrame(results)
    if df.empty:
        print("  WARNING: No results to write heatmap")
        return

    pivot = df.pivot_table(
        index="model_struct",
        columns="test_struct",
        values="dice",
        aggfunc="mean",
    )

    # Reorder rows and columns
    pivot = pivot.reindex(index=WORKFLOWS, columns=TEST_STRUCTS)
    pivot.index.name = "model_struct"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path)
    print(f"Heatmap CSV: {csv_path} ({pivot.shape[0]} x {pivot.shape[1]})")

    # Print summary
    print(f"\nHeatmap preview (mean Dice):")
    print(pivot.round(3).to_string())

    # Diagonal (same-structure) stats
    diag_vals = []
    for s in WORKFLOWS:
        if s in pivot.columns:
            val = pivot.loc[s, s]
            if not (isinstance(val, float) and math.isnan(val)):
                diag_vals.append(val)
    if diag_vals:
        print(f"\nDiagonal mean Dice (same-structure): {np.mean(diag_vals):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Classic aicssegmentation cross-eval (14 workflows x 14 structures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"Directory with per-structure test CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for prediction TIFFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=str(DEFAULT_METRICS_CSV),
        help=f"Output path for per-sample metrics CSV (default: {DEFAULT_METRICS_CSV})",
    )
    parser.add_argument(
        "--heatmap-csv",
        type=str,
        default=str(DEFAULT_HEATMAP_CSV),
        help=f"Output path for heatmap CSV (default: {DEFAULT_HEATMAP_CSV})",
    )
    parser.add_argument(
        "--workflows",
        type=str,
        default=None,
        help="Comma-separated list of workflows to run (default: all 14). "
        "E.g. --workflows ACTN1,DSP,TJP1",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have prediction TIFFs (resume support)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics from existing TIFFs (no workflow runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running anything",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    metrics_csv = Path(args.metrics_csv)
    heatmap_csv = Path(args.heatmap_csv)

    # Parse --workflows filter
    wf_list = None
    if args.workflows:
        wf_list = [w.strip() for w in args.workflows.split(",")]
        unknown = [w for w in wf_list if w not in WORKFLOWS]
        if unknown:
            print(f"ERROR: Unknown workflow(s): {unknown}", file=sys.stderr)
            print(f"  Valid workflows: {WORKFLOWS}", file=sys.stderr)
            sys.exit(1)

    active_wf = wf_list if wf_list is not None else WORKFLOWS

    print("Classic aicssegmentation cross-eval")
    print(f"  Workflows:    {len(active_wf)} {active_wf}")
    print(f"  Test structs: {len(TEST_STRUCTS)} {TEST_STRUCTS}")
    print(f"  Data dir:     {data_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Metrics CSV:  {metrics_csv}")
    print(f"  Heatmap CSV:  {heatmap_csv}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Metrics only:  {args.metrics_only}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    # Validate data dir
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Run
    results = run_cross_eval(
        data_dir=data_dir,
        output_dir=output_dir,
        workflows=wf_list,
        skip_existing=args.skip_existing,
        metrics_only=args.metrics_only,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n[dry-run] No outputs written.")
        return

    # Write outputs (only full aggregation when running all workflows)
    if wf_list is None:
        write_metrics_csv(results, metrics_csv)
        write_heatmap_csv(results, heatmap_csv)
    else:
        print(
            f"\nSubset run ({len(wf_list)} workflows). "
            "Run with --metrics-only (no --workflows) to aggregate all results."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
