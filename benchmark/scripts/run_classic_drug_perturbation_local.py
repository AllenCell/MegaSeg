"""
Run classic segmentation workflows on local drug perturbation v2 data.

Unlike run_classic_drug_perturbation.py (which loads multi-channel images
from S3), this script loads single-channel ZYX TIFFs from local NFS paths
organized by structure/condition directories.

The data lives at:
    /allen/aics/rep_learn/MegaSeg/data/drug_perturbation/{STRUCTURE}/{condition}/

Usage:
    python scripts/run_classic_drug_perturbation_local.py [--structures FBL NUP153 ...]

Structures supported: FBL, NUP153, ST6GAL1, TOMM20, TUBA1B
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, dilation, remove_small_objects

from aicssegmentation.core.pre_processing_utils import (
    edge_preserving_smoothing_3d,
    image_smoothing_gaussian_3d,
    intensity_normalization,
)
from aicssegmentation.core.seg_dot import dot_3d, dot_3d_wrapper, dot_slice_by_slice
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.vessel import vesselness3D, vesselnessSliceBySlice


# ---------------------------------------------------------------------------
# Workflow functions
# ---------------------------------------------------------------------------


def workflow_fbl(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for FBL (Fibrillarin - nucleolus marker).

    Ported from aicssegmentation.structure_wrapper.seg_fbl.Workflow_fbl.
    Uses triangle+median threshold for coarse detection, per-object Otsu for
    refinement, and dot_slice_by_slice for fine structure capture.
    """
    intensity_norm_param = [0.5, 18]
    gaussian_smoothing_sigma = 1
    gaussian_smoothing_truncate_range = 3.0
    dot_2d_sigma = 1
    dot_2d_cutoff = 0.01
    minArea = 5
    low_level_min_size = 700

    # Pre-processing
    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = image_smoothing_gaussian_3d(
        struct_img,
        sigma=gaussian_smoothing_sigma,
        truncate_range=gaussian_smoothing_truncate_range,
    )

    # Step 1: low-level thresholding (triangle + median average)
    global_tri = threshold_triangle(structure_img_smooth)
    global_median = np.percentile(structure_img_smooth, 50)
    th_low_level = (global_tri + global_median) / 2
    bw_low_level = structure_img_smooth > th_low_level
    bw_low_level = remove_small_objects(
        bw_low_level, min_size=low_level_min_size, connectivity=1
    )

    # Step 2: per-object local Otsu thresholding
    bw_high_level = np.zeros_like(bw_low_level)
    lab_low, num_obj = label(bw_low_level, return_num=True, connectivity=1)
    for idx in range(num_obj):
        single_obj = lab_low == (idx + 1)
        local_otsu = threshold_otsu(structure_img_smooth[single_obj])
        bw_high_level[np.logical_and(structure_img_smooth > local_otsu, single_obj)] = 1

    # Step 3: finer segmentation via 2D dot detection
    response2d = dot_slice_by_slice(structure_img_smooth, log_sigma=dot_2d_sigma)
    bw_finer = remove_small_objects(
        response2d > dot_2d_cutoff, min_size=minArea, connectivity=1
    )

    # Merge finer detection into coarse segmentation
    bw_high_level[bw_finer > 0] = 1

    # Post-processing
    seg = remove_small_objects(bw_high_level > 0, min_size=minArea, connectivity=1)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


def workflow_nup153(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for NUP153 (nuclear pores)."""
    intensity_norm_param = [1, 13.5]
    s3_param = [[1, 0.07]]
    minArea = 7

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    bw = dot_3d_wrapper(structure_img_smooth, s3_param)

    seg = remove_small_objects(bw, min_size=minArea, connectivity=1)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


def workflow_st6gal1(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for ST6GAL1 (Golgi)."""
    intensity_norm_param = [9, 19]
    gaussian_smoothing_sigma = 1
    gaussian_smoothing_truncate_range = 3.0
    cell_wise_min_area = 1200
    dot_3d_sigma = 1.6
    dot_3d_cutoff = 0.02
    minArea = 10
    thin_dist = 1
    thin_dist_preserve = 1.6

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = image_smoothing_gaussian_3d(
        struct_img,
        sigma=gaussian_smoothing_sigma,
        truncate_range=gaussian_smoothing_truncate_range,
    )

    # Global thresholding + cell-wise filtering
    global_thresh = threshold_triangle(structure_img_smooth)
    bw_low_level = structure_img_smooth > global_thresh
    bw_low_level = remove_small_objects(
        bw_low_level, min_size=cell_wise_min_area, connectivity=1
    )
    bw_low_level = dilation(bw_low_level, footprint=ball(2))

    # Dot detection
    bw_high_level = dot_3d(structure_img_smooth, log_sigma=dot_3d_sigma)
    bw_high_level = bw_high_level > dot_3d_cutoff

    # Combine
    bw = np.logical_and(bw_low_level, bw_high_level)
    bw = remove_small_objects(bw, min_size=minArea, connectivity=1)

    # Topology preserving thinning
    bw = topology_preserving_thinning(bw, thin_dist, thin_dist_preserve)

    seg = bw.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


def workflow_tomm20(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for TOMM20 (mitochondria)."""
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
    """Classic segmentation for TUBA1B (alpha-tubulin / microtubules)."""
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
    "FBL": workflow_fbl,
    "NUP153": workflow_nup153,
    "ST6GAL1": workflow_st6gal1,
    "TOMM20": workflow_tomm20,
    "TUBA1B": workflow_tuba1b,
}

ALL_STRUCTURES = list(WORKFLOWS.keys())


def load_local_tiff(filepath: str) -> np.ndarray:
    """Load a single-channel ZYX TIFF from local filesystem.

    The drug perturbation v2 data is pre-extracted single-channel uint16,
    shape (Z, Y, X) = (100, 624, 924).
    """
    img = tifffile.imread(filepath)
    if img.ndim == 3:
        return img.astype(np.float32)
    elif img.ndim == 4:
        # Unexpected multi-channel - take channel 0
        print(f"  WARNING: multi-channel image {filepath}, using channel 0")
        return img[:, 0, :, :].astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape {img.shape} for {filepath}")


def run_structure(
    structure: str,
    csv_path: str,
    output_dir: str,
) -> None:
    """Run classic segmentation for a single structure from local TIFFs."""
    workflow_fn = WORKFLOWS[structure]
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["file_path"])

    os.makedirs(output_dir, exist_ok=True)

    n_total = len(df)
    print(f"\n[{structure}] Processing {n_total} unique FOVs")
    print(f"  Output: {output_dir}")

    for idx, (_, row) in enumerate(df.iterrows()):
        filepath = row["file_path"]
        filename = os.path.basename(filepath)
        stem = filename.replace(".tiff", "").replace(".tif", "")
        out_name = f"{stem}.ome.tif"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            print(f"  [{idx + 1}/{n_total}] SKIP (exists): {stem}")
            continue

        try:
            struct_img = load_local_tiff(filepath)

            seg = workflow_fn(struct_img)

            tifffile.imwrite(out_path, seg)
            print(f"  [{idx + 1}/{n_total}] OK: {stem} -> {seg.shape}")
        except Exception as e:
            print(f"  [{idx + 1}/{n_total}] ERROR: {stem}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run classic segmentation on local drug perturbation v2 data"
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=ALL_STRUCTURES,
        choices=ALL_STRUCTURES,
        help="Structures to process (default: all)",
    )
    parser.add_argument(
        "--csv-dir",
        default="data/drug_perturbation_v2/eval_csvs",
        help="Directory containing per-structure eval CSVs",
    )
    parser.add_argument(
        "--output-base",
        default="outputs/eval/uint8/classic/drug_perturbation_v2",
        help="Base output directory",
    )
    args = parser.parse_args()

    # Resolve paths relative to benchmark dir
    benchmark_dir = Path(__file__).resolve().parent.parent
    csv_dir = benchmark_dir / args.csv_dir
    output_base = benchmark_dir / args.output_base

    print("Classic segmentation - drug perturbation v2 (local data)")
    print(f"Structures: {args.structures}")
    print(f"CSV dir: {csv_dir}")
    print(f"Output base: {output_base}")

    for structure in args.structures:
        csv_path = csv_dir / f"{structure}.csv"
        if not csv_path.exists():
            print(f"WARNING: CSV not found for {structure}: {csv_path}")
            continue
        output_dir = output_base / structure
        run_structure(structure, str(csv_path), str(output_dir))

    print("\nDone!")


if __name__ == "__main__":
    main()
