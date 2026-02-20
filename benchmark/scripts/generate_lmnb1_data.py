#!/usr/bin/env python3
"""Generate standardized LMNB1 CSV data manifests from downloaded TIFF files.

Creates train.csv, val.csv, test.csv in benchmark/data/lmnb1/ with the same
format used by the existing benchmark structures:

    ,raw,seg
    0,/allen/.../raw/0000_raw.tiff,/allen/.../seg/0000_seg.tiff

Source data (downloaded by Suraj) lives at:
    /allen/aics/lumenoid/inner_outer_seg/megaseg_exp/fine-tune/data_downloads/
"""

import re
import sys
from pathlib import Path

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────

DATA_ROOT = Path(
    "/allen/aics/lumenoid/inner_outer_seg/megaseg_exp/fine-tune/data_downloads"
)

SPLITS = {
    "train": "train_lmnb1_s3_cytodl",
    "val": "val_lmnb1_s3_cytodl",
    "test": "test_lmnb1_s3_cytodl",
}

# From the original LMNB1 dataset download (267 FOVs total)
EXPECTED_COUNTS = {"train": 160, "val": 40, "test": 67}

# Pattern to extract the numeric ID from filenames like "0000_raw.tiff"
_STEM_RE = re.compile(r"^(\d+)_(?:raw|seg)$")


def _stem_key(filename: str) -> str:
    """Extract the numeric identifier from a raw/seg TIFF filename."""
    stem = Path(filename).stem
    m = _STEM_RE.match(stem)
    return m.group(1) if m else stem


def generate_csv(split: str, subdir: str, output_path: Path) -> pd.DataFrame:
    """Generate a standardized CSV for one split."""
    raw_dir = DATA_ROOT / subdir / "raw"
    seg_dir = DATA_ROOT / subdir / "seg"

    if not raw_dir.is_dir():
        print(f"ERROR: Raw directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    if not seg_dir.is_dir():
        print(f"ERROR: Seg directory not found: {seg_dir}", file=sys.stderr)
        sys.exit(1)

    raw_by_key = {
        _stem_key(f.name): f.name
        for f in sorted(raw_dir.iterdir())
        if f.suffix == ".tiff"
    }
    seg_by_key = {
        _stem_key(f.name): f.name
        for f in sorted(seg_dir.iterdir())
        if f.suffix == ".tiff"
    }

    if len(raw_by_key) != len(seg_by_key):
        print(
            f"ERROR: Mismatch in {subdir}: {len(raw_by_key)} raw vs "
            f"{len(seg_by_key)} seg files",
            file=sys.stderr,
        )
        sys.exit(1)

    missing_in_seg = set(raw_by_key) - set(seg_by_key)
    missing_in_raw = set(seg_by_key) - set(raw_by_key)
    if missing_in_seg or missing_in_raw:
        print(
            f"ERROR: Key mismatch in {subdir}: "
            f"{len(missing_in_seg)} raw-only, {len(missing_in_raw)} seg-only",
            file=sys.stderr,
        )
        sys.exit(1)

    expected = EXPECTED_COUNTS[split]
    if len(raw_by_key) != expected:
        print(
            f"  WARNING: Expected {expected} {split} files, found {len(raw_by_key)}",
            file=sys.stderr,
        )

    rows = []
    for key in sorted(raw_by_key):
        rows.append(
            {
                "raw": str(raw_dir / raw_by_key[key]),
                "seg": str(seg_dir / seg_by_key[key]),
            }
        )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "lmnb1"

    print(f"Source data: {DATA_ROOT}")
    print(f"Output dir:  {output_dir}\n")

    for split, subdir in SPLITS.items():
        output_path = output_dir / f"{split}.csv"
        df = generate_csv(split, subdir, output_path)
        print(f"  {split}.csv: {len(df)} rows -> {output_path}")

    print("\nDone. Generated 3 CSVs.")


if __name__ == "__main__":
    main()
