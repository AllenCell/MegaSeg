#!/usr/bin/env python3
"""Copy LMNB1 data CSVs from user's finetune setup into benchmark.

The user's CSVs are pre-validated and use the correct format (raw,seg columns,
no index). We copy the test CSV into a dedicated test/ subdirectory because
eval.py predict configs point data.path to this directory, and cyto-dl's
DataframeDatamodule auto-discovers CSVs by filename prefix.

Source: /allen/aics/users/ashwin.samudre/EMT_image_analysis/
  Colony_mask_training_inference/data/megaseg_finetune_lamin/

Layout created:
  benchmark/data/lmnb1/
    test_lmnb1.csv          <- full test CSV (67 rows, for evaluation script)
    test/
      test_lmnb1.csv        <- same test CSV in isolated dir (for eval.py predict)
"""

import shutil
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

SOURCE_ROOT = Path(
    "/allen/aics/users/ashwin.samudre/EMT_image_analysis/"
    "Colony_mask_training_inference/data/megaseg_finetune_lamin"
)

# Test CSV lives in its own subdirectory in the source too
SOURCE_TEST_CSV = SOURCE_ROOT / "test" / "test_lmnb1_s3_cytodl_local.csv"

EXPECTED_TEST_ROWS = 67


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "lmnb1"
    test_subdir = output_dir / "test"

    print(f"Source: {SOURCE_ROOT}")
    print(f"Output: {output_dir}\n")

    # Validate source exists
    if not SOURCE_TEST_CSV.is_file():
        print(f"ERROR: Source test CSV not found: {SOURCE_TEST_CSV}", file=sys.stderr)
        sys.exit(1)

    # Validate row count
    with open(SOURCE_TEST_CSV) as f:
        lines = f.readlines()
    data_rows = len(lines) - 1  # subtract header
    if data_rows != EXPECTED_TEST_ROWS:
        print(
            f"WARNING: Expected {EXPECTED_TEST_ROWS} test rows, found {data_rows}",
            file=sys.stderr,
        )

    # Create directories
    test_subdir.mkdir(parents=True, exist_ok=True)

    # Copy test CSV to root (for evaluation script)
    dest_root = output_dir / "test_lmnb1.csv"
    shutil.copy2(SOURCE_TEST_CSV, dest_root)
    print(f"  Copied test CSV -> {dest_root} ({data_rows} rows)")

    # Copy test CSV to test/ subdirectory (for eval.py predict configs)
    dest_test = test_subdir / "test_lmnb1.csv"
    shutil.copy2(SOURCE_TEST_CSV, dest_test)
    print(f"  Copied test CSV -> {dest_test} ({data_rows} rows)")

    # Validate format (no index column)
    header = lines[0].strip()
    if header != "raw,seg":
        print(f"WARNING: Unexpected header: '{header}' (expected 'raw,seg')")

    print("\nDone.")


if __name__ == "__main__":
    main()
