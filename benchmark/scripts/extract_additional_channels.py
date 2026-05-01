#!/usr/bin/env python
"""Extract individual channels from multi-channel OME-TIFFs for additional structures.

Reads data from the shared source directory (variance_additional_structures),
extracts each channel as a separate ZYX uint16 TIFF, and writes them to
data/additional_structures_extracted/{STRUCT}/{filename}_ch{N}.tiff.

Also generates predict-only CSVs for cyto_dl eval:
  data/additional_structures_eval_csvs/all.csv          (all extracted volumes)
  data/additional_structures_eval_csvs/{STRUCT}.csv     (per-structure)

CSV format: file_path,split  (split=predict for all rows).

Usage:
    uv run python scripts/extract_additional_channels.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import tifffile


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STRUCTURES = ["CTCF", "FUS", "LC3B", "PCNA"]

# Absolute path to shared source data
SOURCE_DIR = Path("/allen/aics/rep_learn/MegaSeg/data/variance_additional_structures")

# Output paths relative to benchmark/ (PROJECT_ROOT)
OUTPUT_REL = Path("data/additional_structures_extracted")
CSV_REL = Path("data/additional_structures_eval_csvs")


def extract_channels(project_root: Path, dry_run: bool = False) -> list[dict]:
    """Extract all channels from all structures.

    Returns list of dicts: {structure, source_file, channel, output_path}.
    """
    out_base = project_root / OUTPUT_REL
    records = []

    for struct in STRUCTURES:
        struct_src = SOURCE_DIR / struct
        struct_out = out_base / struct

        if not struct_src.exists():
            print(f"WARNING: Source dir not found: {struct_src}", file=sys.stderr)
            continue

        tiff_files = sorted(struct_src.glob("*.tiff")) + sorted(
            struct_src.glob("*.tif")
        )
        if not tiff_files:
            print(f"WARNING: No TIFFs in {struct_src}", file=sys.stderr)
            continue

        if not dry_run:
            struct_out.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {struct} ({len(tiff_files)} files) ---")

        for tiff_path in tiff_files:
            img = tifffile.imread(str(tiff_path))
            stem = tiff_path.stem

            # Handle ZYX (3D) vs CZYX (4D)
            if img.ndim == 3:
                n_channels = 1
                channels = [img]
            elif img.ndim == 4:
                n_channels = img.shape[0]
                channels = [img[c] for c in range(n_channels)]
            else:
                print(
                    f"  WARNING: Unexpected shape {img.shape} for {tiff_path.name}, "
                    f"skipping",
                    file=sys.stderr,
                )
                continue

            print(
                f"  {tiff_path.name}: shape={img.shape}, "
                f"dtype={img.dtype}, {n_channels} ch"
            )

            for ch_idx, ch_data in enumerate(channels):
                out_name = f"{stem}_ch{ch_idx}.tiff"
                out_path = struct_out / out_name

                record = {
                    "structure": struct,
                    "source_file": tiff_path.name,
                    "channel": ch_idx,
                    "output_path": str(out_path.resolve()),
                }
                records.append(record)

                if dry_run:
                    print(f"    ch{ch_idx} -> {out_name}  shape={ch_data.shape}")
                    continue

                if out_path.exists():
                    print(f"    ch{ch_idx} -> {out_name}  (exists, skipping)")
                    continue

                tifffile.imwrite(str(out_path), ch_data)
                print(f"    ch{ch_idx} -> {out_name}  shape={ch_data.shape}")

    return records


def write_csvs(records: list[dict], project_root: Path, dry_run: bool = False):
    """Write predict-only CSVs from extraction records."""
    csv_dir = project_root / CSV_REL
    if not dry_run:
        csv_dir.mkdir(parents=True, exist_ok=True)

    all_lines = ["file_path,split"]
    by_struct: dict[str, list[str]] = {}

    for rec in records:
        line = f"{rec['output_path']},predict"
        all_lines.append(line)
        by_struct.setdefault(rec["structure"], []).append(line)

    # Write all.csv
    all_csv = csv_dir / "all.csv"
    if dry_run:
        print(f"\n[dry-run] Would write {all_csv} ({len(records)} rows)")
    else:
        all_csv.write_text("\n".join(all_lines) + "\n")
        print(f"\nWrote {all_csv} ({len(records)} rows)")

    # Write per-structure CSVs
    for struct in sorted(by_struct):
        lines = ["file_path,split"] + by_struct[struct]
        csv_path = csv_dir / f"{struct}.csv"
        if dry_run:
            print(f"[dry-run] Would write {csv_path} ({len(by_struct[struct])} rows)")
        else:
            csv_path.write_text("\n".join(lines) + "\n")
            print(f"Wrote {csv_path} ({len(by_struct[struct])} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Benchmark project root (default: parent of scripts/)",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    print(f"Project root: {project_root}")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {project_root / OUTPUT_REL}")
    print(f"CSVs: {project_root / CSV_REL}")

    if not SOURCE_DIR.exists():
        print(f"ERROR: Source data not found at {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    records = extract_channels(project_root, dry_run=args.dry_run)

    if not records:
        print("\nNo channels extracted!", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal extracted: {len(records)} channel-volumes")
    write_csvs(records, project_root, dry_run=args.dry_run)
    print("\nDone!")


if __name__ == "__main__":
    main()
