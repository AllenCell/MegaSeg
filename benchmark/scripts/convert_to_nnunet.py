#!/usr/bin/env python
"""Convert MegaSeg TIFF data to nnU-Net v2 format.

Reads train/val/test CSVs for all 14 structures and converts to:
  nnUNet_raw/Dataset001_MegaSeg/
    imagesTr/{case}_0000.nii.gz
    labelsTr/{case}.nii.gz
    imagesTs/{case}_0000.nii.gz
    labelsTs/{case}.nii.gz   (kept separate for evaluation)
    dataset.json

Binary segmentation: background=0, foreground=1.
Train+val are merged into imagesTr/labelsTr (nnU-Net does its own splits).
Test goes to imagesTs/labelsTs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile
from concurrent.futures import ProcessPoolExecutor, as_completed

STRUCTURES = [
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

DATA_CSV_DIR = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x")


def tiff_to_nifti(tiff_path: str, out_path: str, is_label: bool = False) -> None:
    """Convert a TIFF file to NIfTI (.nii.gz).

    For labels, normalizes to {0, 1} regardless of input encoding (uint8 0/255
    or uint16 0/65535).
    """
    img = tifffile.imread(tiff_path)

    if is_label:
        # Binarize: any nonzero -> 1
        img = (img > 0).astype(np.uint8)

    # nnU-Net expects (X, Y, Z) orientation in NIfTI, but we just need consistent
    # axes. Use identity affine with 1mm isotropic spacing.
    # TIFF is (Z, Y, X) - transpose to (X, Y, Z) for NIfTI convention.
    img = img.transpose(2, 1, 0)

    affine = np.eye(4)
    nii = nib.Nifti1Image(img, affine)
    nib.save(nii, out_path)


def convert_single(args_tuple):
    """Convert a single raw+seg pair. Used by ProcessPoolExecutor."""
    raw_path, seg_path, case_id, img_dir, lbl_dir = args_tuple
    img_out = os.path.join(img_dir, f"{case_id}_0000.nii.gz")
    lbl_out = os.path.join(lbl_dir, f"{case_id}.nii.gz")

    if os.path.exists(img_out) and os.path.exists(lbl_out):
        return case_id, True  # already done

    tiff_to_nifti(raw_path, img_out, is_label=False)
    tiff_to_nifti(seg_path, lbl_out, is_label=True)
    return case_id, False


def main():
    parser = argparse.ArgumentParser(
        description="Convert MegaSeg data to nnU-Net format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/allen/aics/rep_learn/MegaSeg/benchmark/nnUNet_raw/Dataset001_MegaSeg",
        help="Output directory for nnU-Net dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_ts = out_dir / "imagesTs"
    labels_ts = out_dir / "labelsTs"

    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all conversion jobs
    train_jobs = []
    test_jobs = []
    train_cases = []
    test_cases = []

    for struct in STRUCTURES:
        csv_dir = DATA_CSV_DIR / struct

        # Train + val -> imagesTr (nnU-Net does its own cross-validation)
        for split in ["train", "val"]:
            csv_path = csv_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"WARNING: {csv_path} not found, skipping")
                continue

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                raw_path = row["raw"]
                seg_path = row["seg"]
                # Extract case ID: e.g. ACTN1_11269 from the path
                basename = Path(raw_path).stem  # e.g. ACTN1_11269
                case_id = basename

                train_jobs.append(
                    (raw_path, seg_path, case_id, str(images_tr), str(labels_tr))
                )
                train_cases.append(case_id)

        # Test -> imagesTs
        csv_path = csv_dir / "test.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            raw_path = row["raw"]
            seg_path = row["seg"]
            basename = Path(raw_path).stem
            case_id = basename

            test_jobs.append(
                (raw_path, seg_path, case_id, str(images_ts), str(labels_ts))
            )
            test_cases.append(case_id)

    print(f"Total train+val: {len(train_jobs)}, test: {len(test_jobs)}")

    if args.dry_run:
        print("Dry run - not converting. Sample jobs:")
        for j in train_jobs[:3]:
            print(f"  Train: {j[0]} -> {j[2]}")
        for j in test_jobs[:3]:
            print(f"  Test:  {j[0]} -> {j[2]}")
        return

    # Convert in parallel
    all_jobs = train_jobs + test_jobs
    done = 0
    skipped = 0
    total = len(all_jobs)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_single, job): job for job in all_jobs}
        for future in as_completed(futures):
            case_id, was_cached = future.result()
            done += 1
            if was_cached:
                skipped += 1
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] converted ({skipped} cached)")

    print(f"Done: {done} files converted, {skipped} were already cached.")

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "fluorescence"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    json_path = out_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Wrote {json_path}")
    print(f"  Training cases: {len(train_cases)}")
    print(f"  Test cases: {len(test_cases)}")


if __name__ == "__main__":
    main()
