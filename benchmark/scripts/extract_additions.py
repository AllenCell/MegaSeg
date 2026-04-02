"""Extract TIFs from Caroline's additional data (new Tubulin CZI + Cardio SLD files).

Reads ip_manifest_megaseg_caroline_additions.csv and extracts each row as a
single-channel 3D TIF. The Tubulin CZI uses bioio-czi; the Cardio SLD files
use bioio-bioformats (requires Java).

Output directories:
    Tubulin_MultiCh_2/  - new Tubulin CZI (separate from existing Tubulin_MultiCh)
    Cardio_MultiCh/     - new Cardio data (5 channels per scene)

Usage:
    python scripts/extract_additions.py [--datasets Tubulin_MultiCh Cardio_MultiCh]

Requires: bioio, bioio-czi, bioio-bioformats, tifffile, pandas
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

ADDITIONS_CSV = (
    "/allen/aics/rep_learn/MegaSeg/data/caroline_data/"
    "ip_manifest_megaseg_caroline_additions.csv"
)

# Map structure name to output subdirectory
OUTPUT_DIRS = {
    "Tubulin_MultiCh": "Tubulin_MultiCh_2",
    "Cardio_MultiCh": "Cardio_MultiCh",
}


def extract_rows(df: pd.DataFrame, structure: str, output_dir: Path) -> None:
    """Extract TIFs for a single structure type."""
    sub_df = df[df["structure"] == structure]
    out_dir = output_dir / OUTPUT_DIRS[structure]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{structure}] {len(sub_df)} rows -> {out_dir}")

    # Group by source file to avoid re-opening
    for src_path, group in sub_df.groupby("path"):
        fname = Path(src_path).name
        ext = Path(src_path).suffix.lower()
        print(f"\n  Source: {fname} ({ext})")

        if not os.path.exists(src_path):
            print(f"    ERROR: file not found: {src_path}")
            continue

        # Open with appropriate reader
        # bioio auto-detects: .czi uses bioio-czi, .sld uses bioio-bioformats
        img = BioImage(src_path)
        print(f"    Scenes available: {img.scenes}")

        for _, row in group.iterrows():
            scene = int(row["scene"])
            tp = int(row["timepoint"])
            ch = int(row["source_channel"])

            stem = Path(src_path).stem
            out_name = f"{stem}_scene{scene:02d}_T{tp:04d}_C{ch:02d}.tif"
            out_path = out_dir / out_name

            if out_path.exists():
                print(f"    SKIP (exists): {out_name}")
                continue

            # Map scene number to BioImage scene
            # CZI scenes are typically 1-indexed in the manifest
            # SLD scenes may also be 1-indexed
            # Use positional index: scene N maps to img.scenes[N-1] for 1-based,
            # or img.scenes[N] for 0-based.
            # The manifest for existing Tubulin used 1-based scenes.
            # For Cardio (C08), scenes go 0-9; for Cardio (C09), scenes go 1-9.
            # We'll try the scene value as-is first, then fall back.
            try:
                # Try treating scene as a positional index into scenes list
                # For 1-based: scene 1 -> index 0
                scene_idx = scene - 1 if scene >= 1 else scene
                if scene_idx < 0 or scene_idx >= len(img.scenes):
                    # For 0-based manifests (like C08 with scenes 0-9),
                    # scene 0 -> index 0
                    scene_idx = scene
                scene_name = img.scenes[scene_idx]
            except (IndexError, KeyError):
                print(f"    ERROR: scene {scene} not found in {img.scenes}")
                continue

            img.set_scene(scene_name)

            print(f"    Extracting scene={scene}, T={tp}, C={ch} ...")
            try:
                data = img.get_image_data("ZYX", T=tp, C=ch)
            except Exception as e:
                print(f"      ERROR extracting: {e}")
                continue

            print(f"      Shape: {data.shape}, dtype: {data.dtype}")
            tifffile.imwrite(str(out_path), data)
            print(f"      Saved: {out_name}")

    print(f"\n[{structure}] Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract TIFs from Caroline's additional data"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(OUTPUT_DIRS.keys()),
        choices=list(OUTPUT_DIRS.keys()),
        help="Structure types to extract (default: all)",
    )
    parser.add_argument(
        "--csv",
        default=ADDITIONS_CSV,
        help="Path to additions CSV manifest",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    for dataset in args.datasets:
        extract_rows(df, dataset, RAW_DIR)

    print("\nAll done!")


if __name__ == "__main__":
    main()
