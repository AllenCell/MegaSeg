"""Extract NPM1 Airyscan scene 3 TIFs from the source CZI.

The caroline_data/raw/ directory has pre-extracted TIFs for scenes 1 and 2
but is missing scene 3 (5 timepoints). This script extracts them.

Usage:
    python scripts/extract_npm1_scene3.py
"""

from pathlib import Path

import numpy as np
import tifffile
from bioio import BioImage

CZI_PATH = (
    "/allen/programs/allencell/data/proj0/fbf/bff/809/3a2/280/c69/443/dd1/"
    "844/85d/38/20171107_C03_001_Airyscan Processing.czi"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "caroline_data" / "raw"

# Scene 3 in 1-based manifest = scene '2' in 0-based BioImage
# (scenes are '0', '1', '2')
SCENE_NAME = "2"
CHANNEL = 1
# Timepoints from the manifest (rows 67-71)
TIMEPOINTS = [0, 11, 23, 35, 47]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading CZI: {CZI_PATH}")
    print("This is a ~33GB file, initial load may take a few minutes...")

    img = BioImage(CZI_PATH)
    print(f"Scenes available: {img.scenes}")

    img.set_scene(SCENE_NAME)
    print(f"Set scene to '{SCENE_NAME}'")
    print(
        f"Scene dims: T={img.dims.T}, C={img.dims.C}, "
        f"Z={img.dims.Z}, Y={img.dims.Y}, X={img.dims.X}"
    )

    for tp in TIMEPOINTS:
        # Match existing naming convention from scenes 1/2:
        # 20171107_C03_001_Airyscan Processing_scene0N_TXXXX_CYY.tif
        out_name = (
            f"20171107_C03_001_Airyscan Processing_scene03_T{tp:04d}_C{CHANNEL:02d}.tif"
        )
        out_path = OUTPUT_DIR / out_name

        if out_path.exists():
            print(f"  SKIP (exists): {out_name}")
            continue

        print(f"  Extracting T={tp}, C={CHANNEL} ...")
        data = img.get_image_data("ZYX", T=tp, C=CHANNEL)
        print(f"    Shape: {data.shape}, dtype: {data.dtype}")

        tifffile.imwrite(str(out_path), data)
        print(f"    Saved: {out_name}")

    print("Done!")


if __name__ == "__main__":
    main()
