"""Run classic segmentation workflows on Caroline's timelapse microscopy data.

Structure-to-workflow mapping (from issue #10, comment 5):
  - NPM1 data     -> NPM1 classic workflow
  - Mitochondria   -> TOMM20 classic workflow
  - Tubulin_MultiCh -> TUBA1B classic workflow
  - PCNA           -> SKIP (no classic workflow)

Data is pre-extracted as single-channel TIF files in data/caroline_data/raw/.
See extract_npm1_scene3.py and extract_timelapse_tifs.py for extraction.

Usage:
    python scripts/run_classic_timelapse.py [--structures NPM1 TOMM20 TUBA1B]

Requires: aicssegmentation, tifffile, numpy, scipy, scikit-image
Run with: msseg conda env
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, dilation, remove_small_objects, erosion

from aicssegmentation.core.pre_processing_utils import (
    edge_preserving_smoothing_3d,
    image_smoothing_gaussian_3d,
    intensity_normalization,
)
from aicssegmentation.core.seg_dot import dot_3d, dot_slice_by_slice
from aicssegmentation.core.vessel import vesselness3D, vesselnessSliceBySlice


# ---------------------------------------------------------------------------
# Workflow functions
# ---------------------------------------------------------------------------


def workflow_npm1(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for NPM1 (nucleophosmin, nucleolar granular component).

    Ported from aicssegmentation.structure_wrapper.seg_npm1.Workflow_npm1.
    Algorithm: intensity norm -> gaussian smooth -> triangle threshold + median
    threshold -> dilation -> per-object local otsu -> dot_slice_by_slice
    (bright + dark) -> combine -> remove small objects
    """
    intensity_norm_param = [0.5, 15]
    gaussian_smoothing_sigma = 1
    gaussian_smoothing_truncate_range = 3.0
    dot_2d_sigma = 2
    dot_2d_sigma_extra = 1
    dot_2d_cutoff = 0.025
    minArea = 5
    low_level_min_size = 700

    # Step 1: intensity normalization
    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)

    # Step 2: gaussian smoothing
    structure_img_smooth = image_smoothing_gaussian_3d(
        struct_img,
        sigma=gaussian_smoothing_sigma,
        truncate_range=gaussian_smoothing_truncate_range,
    )

    # Step 3: low-level thresholding (triangle + median)
    global_tri = threshold_triangle(structure_img_smooth)
    global_median = np.percentile(structure_img_smooth, 50)
    th_low_level = (global_tri + global_median) / 2
    bw_low_level = structure_img_smooth > th_low_level
    bw_low_level = remove_small_objects(
        bw_low_level, min_size=low_level_min_size, connectivity=1
    )

    # Step 4: per-object local Otsu
    bw_low_level_dilated = dilation(bw_low_level, footprint=ball(2))
    bw_high_level = np.zeros_like(bw_low_level)
    lab_low, num_obj = label(bw_low_level_dilated, return_num=True)
    for idx in range(1, num_obj + 1):
        single_obj = lab_low == idx
        local_otsu = threshold_otsu(structure_img_smooth[single_obj > 0])
        bw_high_level[
            np.logical_and(structure_img_smooth > local_otsu * 1.2, single_obj)
        ] = True

    # Step 5: 2D dot detection (bright dots)
    response_bright = dot_slice_by_slice(structure_img_smooth, log_sigma=dot_2d_sigma)
    bw_extra = response_bright > dot_2d_cutoff

    # Step 6: 2D dot detection (dark dots for hole-filling)
    response_dark = dot_slice_by_slice(
        structure_img_smooth, log_sigma=dot_2d_sigma_extra
    )
    bw_extra_dark = response_dark > dot_2d_cutoff

    # Combine
    bw = np.logical_or(bw_high_level, bw_extra)
    bw = np.logical_or(bw, bw_extra_dark)

    # Clean up
    seg = remove_small_objects(bw, min_size=minArea, connectivity=1)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


def workflow_tomm20(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for TOMM20 (mitochondria).

    Same as in run_classic_drug_perturbation.py.
    """
    intensity_norm_param = [3.5, 15]
    gaussian_smoothing_sigma = 1
    gaussian_smoothing_truncate_range = 3.0
    vesselness_sigma = [1.5]
    vesselness_cutoff = 0.16
    minArea = 10

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = image_smoothing_gaussian_3d(
        struct_img,
        sigma=gaussian_smoothing_sigma,
        truncate_range=gaussian_smoothing_truncate_range,
    )

    response = vesselnessSliceBySlice(
        structure_img_smooth, sigmas=vesselness_sigma, tau=1, whiteonblack=True
    )
    bw = response > vesselness_cutoff

    seg = remove_small_objects(bw, min_size=minArea, connectivity=1)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


def workflow_tuba1b(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for TUBA1B (alpha-tubulin / microtubules).

    Same as in run_classic_drug_perturbation.py.
    """
    intensity_norm_param = [1.5, 8.0]
    vesselness_sigma = [1]
    vesselness_cutoff = 0.01
    minArea = 20

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    response = vesselness3D(
        structure_img_smooth, sigmas=vesselness_sigma, tau=1, whiteonblack=True
    )
    bw = response > vesselness_cutoff

    seg = remove_small_objects(bw, min_size=minArea, connectivity=1)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


WORKFLOWS = {
    "NPM1": workflow_npm1,
    "TOMM20": workflow_tomm20,
    "TUBA1B": workflow_tuba1b,
}

# Map dataset names to classic workflow names and TIF directories
DATASET_CONFIG = {
    "NPM1": {
        "workflow": "NPM1",
        "tif_dir": ".",  # TIFs are directly in caroline_data/raw/
        "glob_pattern": "20171107_*_scene*_T*_C01.tif",  # Airyscan NPM1
        "extra_patterns": [
            "20180227_N01_001(*_scene00_T*_C00.tif",  # ZSD 20x NPM1
            "20180315_N01_001(*_scene00_T*_C00.tif",  # ZSD 100x NPM1
        ],
    },
    "Mitochondria": {
        "workflow": "TOMM20",
        "tif_dir": "Mitochondria",
        "glob_pattern": "*.tif",
    },
    "Tubulin_MultiCh": {
        "workflow": "TUBA1B",
        "tif_dir": "Tubulin_MultiCh",
        "glob_pattern": "*.tif",
    },
}

ALL_DATASETS = list(DATASET_CONFIG.keys())


def find_tif_files(raw_dir: Path, dataset: str) -> list[Path]:
    """Find all TIF files for a dataset."""
    config = DATASET_CONFIG[dataset]
    tif_dir = raw_dir / config["tif_dir"] if config["tif_dir"] != "." else raw_dir

    files = sorted(tif_dir.glob(config["glob_pattern"]))

    # NPM1 has multiple file patterns (Airyscan + ZSD)
    if "extra_patterns" in config:
        for pattern in config["extra_patterns"]:
            # Handle parentheses in glob patterns by using raw dir listing
            pass  # The main pattern should catch Airyscan; ZSD files are also in raw_dir

    if dataset == "NPM1":
        # NPM1 files are all directly in raw_dir with various naming patterns
        all_tifs = sorted(raw_dir.glob("*.tif"))
        # Filter to only NPM1-related files (exclude Thumbs.db etc.)
        files = [f for f in all_tifs if f.suffix == ".tif"]

    return files


def run_classic(
    dataset: str,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    """Run classic segmentation for a single dataset."""
    config = DATASET_CONFIG[dataset]
    workflow_fn = WORKFLOWS[config["workflow"]]

    tif_files = find_tif_files(raw_dir, dataset)
    if not tif_files:
        print(f"WARNING: No TIF files found for {dataset}")
        return

    out_dir = output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(tif_files)
    print(f"\n[{dataset}] Processing {n_total} FOVs with {config['workflow']} workflow")
    print(f"  Output: {out_dir}")

    for i, tif_path in enumerate(tif_files):
        stem = tif_path.stem
        out_name = f"{stem}.tif"
        out_path = out_dir / out_name

        if out_path.exists():
            print(f"  [{i + 1}/{n_total}] SKIP (exists): {stem}")
            continue

        try:
            # Load single-channel 3D image
            struct_img = tifffile.imread(str(tif_path))
            struct_img = struct_img.astype(np.float32)

            print(f"  [{i + 1}/{n_total}] {stem} shape={struct_img.shape} ...", end="")

            # Run segmentation
            seg = workflow_fn(struct_img)

            # Save
            tifffile.imwrite(str(out_path), seg)
            print(f" OK -> {seg.shape}")

        except Exception as e:
            print(f"\n  [{i + 1}/{n_total}] ERROR: {stem}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run classic segmentation on timelapse data (issue #10)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Base directory for raw TIF files (default: data/caroline_data/raw)",
    )
    parser.add_argument(
        "--output-base",
        default="outputs/eval/uint8/classic/timelapse",
        help="Base output directory (relative to benchmark/)",
    )
    args = parser.parse_args()

    benchmark_dir = Path(__file__).resolve().parent.parent
    raw_dir = (
        Path(args.raw_dir)
        if args.raw_dir
        else benchmark_dir / "data" / "caroline_data" / "raw"
    )
    output_base = benchmark_dir / args.output_base

    print("Classic segmentation - timelapse evaluation (issue #10)")
    print(f"Datasets: {args.datasets}")
    print(f"Raw dir: {raw_dir}")
    print(f"Output base: {output_base}")

    for dataset in args.datasets:
        run_classic(dataset, raw_dir, output_base)

    print("\nDone!")


if __name__ == "__main__":
    main()
