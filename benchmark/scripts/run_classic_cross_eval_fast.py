#!/usr/bin/env python
"""Fast classic aicssegmentation cross-eval with concurrent.futures.

Same logic as run_classic_cross_eval.py but uses ProcessPoolExecutor to
parallelize image processing across CPU cores. Each worker process lazily
loads and caches workflow functions, so tasks are sorted by workflow to
minimize redundant imports.

Self-contained: inlines everything from classic_workflows.py so this single
file can be copied to another machine (requires the msseg conda env).

Usage:
    # Full 14x14 run with 8 workers
    python run_classic_cross_eval_fast.py --workers 8

    # Resume interrupted run (skip existing TIFFs)
    python run_classic_cross_eval_fast.py --workers 8 --skip-existing

    # Only specific workflows
    python run_classic_cross_eval_fast.py --workers 4 --workflows ACTN1,DSP,TJP1

    # Recompute metrics from existing TIFFs (no segmentation, fast)
    python run_classic_cross_eval_fast.py --metrics-only

    # Dry run to see task list
    python run_classic_cross_eval_fast.py --dry-run

Environment:
    conda activate msseg
    # or: /allen/aics/users/ashwin.samudre/miniconda3/envs/msseg/bin/python
"""

import argparse
import ast
import importlib
import importlib.util
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import tifffile
from aicsimageio import AICSImage


# ===================================================================
# Inlined from classic_workflows.py - workflow registry & loading
# ===================================================================

MS_CLASSIC_DIR = Path("/allen/aics/lumenoid/inner_outer_seg/megaseg_exp/ms_classic")

# structure: (script_basename, workflow_func_name, broken_reason_or_None)
# script_basename=None -> import from installed aicssegmentation package.
STRUCTURES = {
    "ACTN1": ("ms_seg_ACTN1.py", "Workflow_actn1", None),
    "ATP2A2": (None, "Workflow_atp2a2", None),
    "CETN2": ("classic_segmentation_CETN2.py", "Workflow_cetn2", "peak_local_max"),
    "CTNNB1": ("classic_segmentation_CTNNB1.py", "Workflow_ctnnb1", None),
    "DSP": (None, "Workflow_dsp", None),
    "FBL": ("classic_segmentation_FBL.py", "Workflow_fbl", None),
    "GJA1": (None, "Workflow_gja1", None),
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
    "TJP1": (None, "Workflow_tjp1", None),
    "TOMM20": ("classic_segmentation_TOMM20.py", "Workflow_tomm20", "main_only"),
    "TUBA1B": ("classic_segmentation_TUBA1B.py", "Workflow_tuba1b", None),
}


def load_image_3d(path: str) -> np.ndarray:
    """Load a TIFF and squeeze to 3D (Z, Y, X)."""
    img = AICSImage(path)
    return np.squeeze(img.data)


def patch_script_source(script_path: Path, broken_reason: str) -> str:
    """Read script source and apply patches for known issues."""
    source = script_path.read_text()

    if broken_reason == "watershed_import":
        source = source.replace(
            "from skimage.morphology import remove_small_objects, watershed, dilation, ball",
            "from skimage.morphology import remove_small_objects, dilation, ball\n"
            "from skimage.segmentation import watershed",
        )
    elif broken_reason == "peak_local_max":
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
        idx = source.find("\ndef Workflow_")
        if idx == -1:
            idx = source.find("def Workflow_")
        source = source[:idx] + helper + source[idx:]
        source = source.replace(
            "local_maxi = peak_local_max(",
            "local_maxi = _peak_local_max_mask(",
        )
    elif broken_reason == "main_only":
        tree = ast.parse(source)
        last_keep_line = 0
        for node in ast.iter_child_nodes(tree):
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                end = getattr(node, "end_lineno", None)
                if end is not None:
                    last_keep_line = max(last_keep_line, end)
        if last_keep_line > 0:
            lines = source.split("\n")
            source = "\n".join(lines[:last_keep_line])

    return source


def load_workflow_function(structure: str) -> Optional[Callable]:
    """Dynamically import the workflow function for a structure.

    Structures with script_basename=None are imported from the installed
    aicssegmentation.structure_wrapper package. Others loaded from local
    scripts in MS_CLASSIC_DIR with patches applied for known breakage.
    """
    if structure not in STRUCTURES:
        print(f"  ERROR: Unknown structure '{structure}'", file=sys.stderr)
        return None

    script_basename, func_name, broken_reason = STRUCTURES[structure]

    try:
        if script_basename is None:
            module_path = f"aicssegmentation.structure_wrapper.seg_{structure.lower()}"
            module = importlib.import_module(module_path)
            return getattr(module, func_name)

        script_path = MS_CLASSIC_DIR / script_basename
        if not script_path.exists():
            print(f"  WARNING: Script not found: {script_path}", file=sys.stderr)
            return None

        if broken_reason:
            patched_source = patch_script_source(script_path, broken_reason)
            module_name = f"classic_seg_{structure}_patched"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(compile(patched_source, str(script_path), "exec"), module.__dict__)
        else:
            module_name = f"classic_seg_{structure}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        return getattr(module, func_name)

    except Exception as e:
        print(f"  ERROR importing {structure}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


# ===================================================================
# Metric helpers (inlined from evaluate_benchmark.py)
# ===================================================================


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
    """Compute segmentation metrics from two binary boolean masks."""
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


# ===================================================================
# Constants
# ===================================================================

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


# ===================================================================
# Helpers
# ===================================================================


def extract_sample_id(raw_path: str) -> str:
    """Extract sample ID from raw path like .../ACTN1_10917.tiff -> ACTN1_10917."""
    name = Path(raw_path).stem
    if name.endswith("_IM"):
        name = name[:-3]
    return name


def format_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ===================================================================
# Per-process workflow cache (populated lazily in worker processes)
# ===================================================================

_WF_CACHE: dict[str, Optional[Callable]] = {}


def _process_image(task: tuple) -> dict:
    """Process a single image in a worker process.

    Args:
        task: (workflow_name, test_struct, raw_path, gt_path, out_path,
               skip_existing, metrics_only)

    Returns:
        Result dict with metrics (or NaN on failure). Contains "_error"
        key if an exception occurred.
    """
    (
        wf_name,
        test_struct,
        raw_path,
        gt_path,
        out_path_str,
        skip_existing,
        metrics_only,
    ) = task
    out_path = Path(out_path_str)
    sample_id = extract_sample_id(raw_path)

    try:
        pred_arr = None

        if metrics_only or (skip_existing and out_path.exists()):
            # Load existing prediction
            if out_path.exists():
                pred_arr = tifffile.imread(str(out_path))
            else:
                return _nan_record(wf_name, test_struct, sample_id)
        else:
            # Lazy-load workflow function (cached per process)
            if wf_name not in _WF_CACHE:
                _WF_CACHE[wf_name] = load_workflow_function(wf_name)
            wf_fn = _WF_CACHE[wf_name]
            if wf_fn is None:
                return {
                    **_nan_record(wf_name, test_struct, sample_id),
                    "_error": f"Could not load workflow {wf_name}",
                }

            # Run workflow
            struct_img = load_image_3d(raw_path)
            pred_arr = wf_fn(struct_img, output_type="array")

            # Save as uint8 {0, 255}
            pred_save = ((pred_arr > 0).astype(np.uint8)) * 255
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(out_path), pred_save)
            pred_arr = pred_save

        # Compute metrics against GT
        gt_arr = tifffile.imread(str(gt_path))
        gt_bin = binarize_gt(gt_arr)
        pred_bin = binarize_prediction(pred_arr, "uint8")
        metrics = compute_metrics(gt_bin, pred_bin)

        return {
            "eval_type": "classic_cross_eval",
            "dtype": "uint8",
            "backbone": "",
            "model_struct": wf_name,
            "test_struct": test_struct,
            "test_set": "standard",
            "sample_id": sample_id,
            **metrics,
        }

    except Exception as e:
        return {
            **_nan_record(wf_name, test_struct, sample_id),
            "_error": str(e),
        }


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


# ===================================================================
# Task building
# ===================================================================


def build_task_list(
    data_dir: Path,
    output_dir: Path,
    wf_list: list[str],
    skip_existing: bool,
    metrics_only: bool,
) -> list[tuple]:
    """Build the full list of (wf, struct, raw, gt, out, skip, metrics) tasks.

    Tasks are sorted by workflow so that worker processes can cache each
    workflow function and reuse it for consecutive tasks.
    """
    tasks = []
    skipped_existing = 0

    for wf in wf_list:
        for test_struct in TEST_STRUCTS:
            csv_path = data_dir / test_struct / "test.csv"
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found, skipping")
                continue

            df = pd.read_csv(csv_path)
            if "raw" not in df.columns or "seg" not in df.columns:
                print(f"  WARNING: {csv_path} missing raw/seg columns")
                continue

            cell_out = output_dir / wf / f"tested_on_{test_struct}"

            for _, row in df.iterrows():
                raw_path = str(row["raw"])
                gt_path = str(row["seg"])
                sample_id = extract_sample_id(raw_path)
                out_path = str(cell_out / f"{sample_id}_seg.tiff")

                # Pre-filter: if skip_existing and TIFF exists, we still
                # need to compute metrics, so include the task. The worker
                # will skip the segmentation step.
                tasks.append(
                    (
                        wf,
                        test_struct,
                        raw_path,
                        gt_path,
                        out_path,
                        skip_existing,
                        metrics_only,
                    )
                )

    return tasks


# ===================================================================
# Output writers
# ===================================================================


def write_metrics_csv(results: list[dict], csv_path: Path) -> None:
    """Write per-sample metrics CSV."""
    # Strip internal _error key before writing
    clean = [{k: v for k, v in r.items() if k != "_error"} for r in results]
    df = pd.DataFrame(clean)
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
    clean = [{k: v for k, v in r.items() if k != "_error"} for r in results]
    df = pd.DataFrame(clean)
    if df.empty:
        print("  WARNING: No results to write heatmap")
        return

    pivot = df.pivot_table(
        index="model_struct",
        columns="test_struct",
        values="dice",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=WORKFLOWS, columns=TEST_STRUCTS)
    pivot.index.name = "model_struct"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(csv_path)
    print(f"Heatmap CSV: {csv_path} ({pivot.shape[0]} x {pivot.shape[1]})")

    print("\nHeatmap preview (mean Dice):")
    print(pivot.round(3).to_string())

    # Diagonal (same-structure) stats
    diag_vals = []
    for s in WORKFLOWS:
        if s in pivot.columns and s in pivot.index:
            val = pivot.loc[s, s]
            if not (isinstance(val, float) and math.isnan(val)):
                diag_vals.append(val)
    if diag_vals:
        print(f"\nDiagonal mean Dice (same-structure): {np.mean(diag_vals):.4f}")


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast classic aicssegmentation cross-eval "
            "(14 workflows x 14 structures, concurrent.futures)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
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
        help="Comma-separated workflow subset (default: all 14). "
        "E.g. --workflows ACTN1,DSP,TJP1",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip segmentation for images that already have prediction TIFFs",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics from existing TIFFs (no segmentation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task count and exit without processing",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    metrics_csv = Path(args.metrics_csv)
    heatmap_csv = Path(args.heatmap_csv)

    # Parse --workflows filter
    wf_list = WORKFLOWS
    if args.workflows:
        wf_list = [w.strip() for w in args.workflows.split(",")]
        unknown = [w for w in wf_list if w not in WORKFLOWS]
        if unknown:
            print(f"ERROR: Unknown workflow(s): {unknown}", file=sys.stderr)
            print(f"  Valid: {WORKFLOWS}", file=sys.stderr)
            sys.exit(1)

    print("Classic aicssegmentation cross-eval (FAST)")
    print(f"  Workers:      {args.workers}")
    print(f"  Workflows:    {len(wf_list)} {wf_list}")
    print(f"  Test structs: {len(TEST_STRUCTS)} {TEST_STRUCTS}")
    print(f"  Data dir:     {data_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Metrics CSV:  {metrics_csv}")
    print(f"  Heatmap CSV:  {heatmap_csv}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Metrics only:  {args.metrics_only}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Build task list
    print("Building task list...")
    tasks = build_task_list(
        data_dir,
        output_dir,
        wf_list,
        args.skip_existing,
        args.metrics_only,
    )
    print(
        f"  {len(tasks)} tasks ({len(wf_list)} workflows x {len(TEST_STRUCTS)} structs x ~67 images)"
    )

    if args.dry_run:
        # Show a few sample tasks
        for t in tasks[:5]:
            sid = extract_sample_id(t[2])
            print(f"  [dry-run] {t[0]} -> {t[1]}: {sid}")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more")
        print("\n[dry-run] No outputs written.")
        return

    if not tasks:
        print("No tasks to process.")
        return

    # Pre-create all output directories to avoid mkdir races
    print("Creating output directories...")
    seen_dirs = set()
    for t in tasks:
        d = str(Path(t[4]).parent)
        if d not in seen_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
            seen_dirs.add(d)
    print(f"  {len(seen_dirs)} directories ready")

    # Process with worker pool
    results = []
    errors = 0
    t_start = time.time()

    print(f"\nProcessing {len(tasks)} images with {args.workers} workers...\n")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_image, t): t for t in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if "_error" in result:
                errors += 1
                task = futures[future]
                sid = extract_sample_id(task[2])
                print(f"  ERROR [{task[0]} -> {task[1]}] {sid}: {result['_error']}")

            results.append(result)

            # Progress every 100 images or at the end
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 0.1)
                eta = (len(tasks) - i) / max(rate, 0.01)
                print(
                    f"  [{i:>6}/{len(tasks)}] "
                    f"{rate:>5.1f} img/s | "
                    f"elapsed {format_elapsed(elapsed)} | "
                    f"ETA {format_elapsed(eta)} | "
                    f"errors {errors}"
                )

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE: {len(results)} images in {format_elapsed(elapsed_total)}")
    print(f"  OK:     {len(results) - errors}")
    print(f"  Errors: {errors}")
    print(f"  Rate:   {len(results) / max(elapsed_total, 0.1):.1f} img/s")
    print(f"{'=' * 70}")

    # Write outputs
    is_subset = args.workflows is not None
    if is_subset:
        print(
            f"\nSubset run ({len(wf_list)} workflows). "
            "Run with --metrics-only (no --workflows) to aggregate all results."
        )
    else:
        write_metrics_csv(results, metrics_csv)
        write_heatmap_csv(results, heatmap_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
