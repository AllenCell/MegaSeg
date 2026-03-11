"""Extract single-channel TIFs from CZI files for timelapse datasets.

Reads the original manifests, extracts the specified channel at each
scene/timepoint, and saves as individual TIF files. This pre-processing
step decouples CZI reading (requires bioio) from segmentation (requires
aicssegmentation).

Usage:
    python scripts/extract_timelapse_tifs.py [--datasets NPM1 Mitochondria Tubulin_MultiCh]

Requires: bioio, bioio-czi, tifffile
Run with: test-bioio conda env
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from bioio import BioImage

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BENCHMARK_DIR / "data" / "caroline_data" / "raw"

# Original manifests (with CZI paths)
MANIFESTS = {
    "Mitochondria": "/allen/aics/rep_learn/amil/MegaSeg/benchmark/cyto-dl/ip_manifest_mitochondria.csv",
    "Tubulin_MultiCh": "/allen/aics/rep_learn/amil/MegaSeg/benchmark/cyto-dl/ip_manifest_tubulin_multich.csv",
}

# NPM1 scene 3 is handled by extract_npm1_scene3.py separately;
# all other NPM1 TIFs already exist on disk.


def extract_dataset(dataset_name: str, csv_path: str, output_dir: Path) -> None:
    """Extract TIFs from CZI files for a single dataset."""
    df = pd.read_csv(csv_path)
    out_dir = output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{dataset_name}] {len(df)} rows from {csv_path}")
    print(f"  Output: {out_dir}")

    # Group by path to avoid re-opening the same CZI
    for czi_path, group in df.groupby("path"):
        print(f"\n  CZI: {Path(czi_path).name}")
        if not os.path.exists(czi_path):
            print(f"    ERROR: file not found: {czi_path}")
            continue

        img = BioImage(czi_path)
        print(f"    Scenes: {img.scenes}")

        for _, row in group.iterrows():
            scene = row["scene"]
            tp = row["timepoint"]
            ch = row["source_channel"]

            # Build output filename: CZIname_sceneNN_TXXXX_CYY.tif
            czi_stem = Path(czi_path).stem
            out_name = (
                f"{czi_stem}_scene{int(scene):02d}_T{int(tp):04d}_C{int(ch):02d}.tif"
            )
            out_path = out_dir / out_name

            if out_path.exists():
                print(f"    SKIP (exists): {out_name}")
                continue

            # Map manifest scene (1-based float) to BioImage scene name (0-based str)
            scene_idx = int(scene) - 1
            scene_name = img.scenes[scene_idx]
            img.set_scene(scene_name)

            print(f"    Extracting scene={int(scene)}, T={int(tp)}, C={int(ch)} ...")
            data = img.get_image_data("ZYX", T=int(tp), C=int(ch))
            print(f"      Shape: {data.shape}, dtype: {data.dtype}")

            tifffile.imwrite(str(out_path), data)
            print(f"      Saved: {out_name}")

    print(f"\n[{dataset_name}] Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract TIFs from CZI files for timelapse datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(MANIFESTS.keys()),
        choices=list(MANIFESTS.keys()),
        help="Datasets to extract (default: all)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        csv_path = MANIFESTS[dataset]
        extract_dataset(dataset, csv_path, RAW_DIR)

    print("\nAll done!")


if __name__ == "__main__":
    main()
