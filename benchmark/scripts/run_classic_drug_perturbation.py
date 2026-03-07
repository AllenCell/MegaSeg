"""
Run classic segmentation workflows on drug perturbation FOVs.

Adapted from the per-structure Jupyter notebooks in segmenter-classic/.
Workflows are from aicssegmentation with structure-specific parameters.

Usage:
    uv run python scripts/run_classic_drug_perturbation.py [--structures ACTB NUP153 ...]

Covers ACTB, NUP153, ST6GAL1, TJP1, TOMM20, TUBA1B.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import s3fs
import tifffile
from scipy.ndimage import zoom
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, dilation, remove_small_objects

from aicssegmentation.core.pre_processing_utils import (
    edge_preserving_smoothing_3d,
    image_smoothing_gaussian_3d,
    intensity_normalization,
)
from aicssegmentation.core.seg_dot import dot_3d, dot_3d_wrapper
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.vessel import vesselness3D, vesselnessSliceBySlice


# ---------------------------------------------------------------------------
# Workflow functions — ported directly from the classic segmentation notebooks
# ---------------------------------------------------------------------------


def workflow_actb(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for ACTB (Beta-Actin filaments)."""
    intensity_norm_param = [3, 15]
    vesselness_sigma_1 = [2]
    vesselness_cutoff_1 = 0.1
    vesselness_sigma_2 = [1]
    vesselness_cutoff_2 = 0.04
    minArea = 15

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = edge_preserving_smoothing_3d(struct_img)

    response_1 = vesselness3D(
        structure_img_smooth, sigmas=vesselness_sigma_1, tau=1, whiteonblack=True
    )
    response_2 = vesselness3D(
        structure_img_smooth, sigmas=vesselness_sigma_2, tau=1, whiteonblack=True
    )
    bw = np.logical_or(
        response_1 > vesselness_cutoff_1, response_2 > vesselness_cutoff_2
    )

    seg = remove_small_objects(bw, min_size=minArea, connectivity=1)
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


def workflow_tjp1(struct_img: np.ndarray) -> np.ndarray:
    """Classic segmentation for TJP1 (tight junctions).

    Ported from aicssegmentation.structure_wrapper.seg_tjp1.Workflow_tjp1.
    """
    intensity_norm_param = [3, 17]
    gaussian_smoothing_sigma = 1
    gaussian_smoothing_truncate_range = 3.0
    vesselness_sigma = [1.5]
    vesselness_cutoff = 0.2
    minArea = 15

    struct_img = intensity_normalization(struct_img, scaling_param=intensity_norm_param)
    structure_img_smooth = image_smoothing_gaussian_3d(
        struct_img,
        sigma=gaussian_smoothing_sigma,
        truncate_range=gaussian_smoothing_truncate_range,
    )

    response = vesselness3D(
        structure_img_smooth, sigmas=vesselness_sigma, tau=1, whiteonblack=True
    )
    bw = response > vesselness_cutoff

    seg = remove_small_objects(bw > 0, min_size=minArea, connectivity=1)
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg


WORKFLOWS = {
    "ACTB": workflow_actb,
    "NUP153": workflow_nup153,
    "ST6GAL1": workflow_st6gal1,
    "TJP1": workflow_tjp1,
    "TOMM20": workflow_tomm20,
    "TUBA1B": workflow_tuba1b,
}

ALL_STRUCTURES = list(WORKFLOWS.keys())


def load_image_from_s3(
    s3_url: str, fs: s3fs.S3FileSystem, channel: int = 1
) -> np.ndarray:
    """Load a single channel from an S3 image.

    Images are ZCYX format with shape (Z, C, Y, X).
    Channel 0 = brightfield, Channel 1 = structure (GFP 488nm).
    """
    s3_path = s3_url.replace("s3://", "")
    with fs.open(s3_path, "rb") as f:
        img = tifffile.imread(f)

    # img shape is (Z, C, Y, X) = (100, 2, 600, 900)
    if img.ndim == 4:
        return img[:, channel, :, :]  # ZYX
    elif img.ndim == 3:
        return img  # already ZYX
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


def run_structure(
    structure: str,
    csv_path: str,
    output_dir: str,
    fs: s3fs.S3FileSystem,
) -> None:
    """Run classic segmentation for a single structure."""
    workflow_fn = WORKFLOWS[structure]
    df = pd.read_csv(csv_path)
    # Deduplicate by file_path
    df = df.drop_duplicates(subset=["file_path"])

    os.makedirs(output_dir, exist_ok=True)

    n_total = len(df)
    print(f"\n[{structure}] Processing {n_total} unique FOVs")
    print(f"  Output: {output_dir}")

    for i, row in df.iterrows():
        s3_url = row["file_path"]
        filename = s3_url.split("/")[-1]
        # Match output naming convention: stem.ome.tif (not .tiff)
        stem = filename.replace(".ome.tiff", "").replace(".ome.tif", "")
        out_name = f"{stem}.ome.tif"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            print(f"  [{i + 1}/{n_total}] SKIP (exists): {stem}")
            continue

        try:
            # Load structure channel from S3
            struct_img = load_image_from_s3(s3_url, fs, channel=1)
            struct_img = struct_img.astype(np.float32)

            # Run segmentation
            seg = workflow_fn(struct_img)

            # Save
            tifffile.imwrite(out_path, seg)
            print(f"  [{i + 1}/{n_total}] OK: {stem} -> {seg.shape}")
        except Exception as e:
            print(f"  [{i + 1}/{n_total}] ERROR: {stem}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Run classic segmentation on drug perturbation data"
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
        default="data/drug_perturbations/eval_csvs",
        help="Directory containing per-structure eval CSVs",
    )
    parser.add_argument(
        "--output-base",
        default="outputs/eval/uint8/classic/drug_perturbation",
        help="Base output directory",
    )
    args = parser.parse_args()

    # Resolve paths relative to benchmark dir
    benchmark_dir = Path(__file__).resolve().parent.parent
    csv_dir = benchmark_dir / args.csv_dir
    output_base = benchmark_dir / args.output_base

    fs = s3fs.S3FileSystem(anon=True)

    print(f"Classic segmentation — drug perturbation evaluation")
    print(f"Structures: {args.structures}")
    print(f"CSV dir: {csv_dir}")
    print(f"Output base: {output_base}")

    for structure in args.structures:
        csv_path = csv_dir / f"{structure}.csv"
        if not csv_path.exists():
            print(f"WARNING: CSV not found for {structure}: {csv_path}")
            continue
        output_dir = output_base / structure
        run_structure(structure, str(csv_path), str(output_dir), fs)

    print("\nDone!")


if __name__ == "__main__":
    main()
