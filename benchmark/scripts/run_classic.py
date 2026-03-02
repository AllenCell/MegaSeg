#!/usr/bin/env python
"""Run classic aicssegmentation workflows on arbitrary input data.

Applies the same classic segmentation workflows used as baselines in the
MegaSeg benchmark. Supports running one or more structures on a directory
of TIFFs or a CSV manifest with a ``raw`` column.

Usage (run with msseg conda env):
    PYTHON=/allen/aics/users/ashwin.samudre/miniconda3/envs/msseg/bin/python

    # Run all structures on a directory of TIFFs
    $PYTHON benchmark/scripts/run_classic.py \\
        --input /path/to/tiffs/ --output-dir /path/to/output/

    # Run specific structures
    $PYTHON benchmark/scripts/run_classic.py \\
        --structures ACTN1 NPM1 --input data.csv --output-dir ./out/

    # List available structures
    $PYTHON benchmark/scripts/run_classic.py --list

    # Dry run (show what would be processed)
    $PYTHON benchmark/scripts/run_classic.py \\
        --input /path/to/tiffs/ --output-dir ./out/ --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tifffile import imwrite

from classic_workflows import (
    STRUCTURES,
    extract_structure_from_filename,
    load_image_3d,
    load_workflow_function,
)

# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def discover_inputs(input_path: str, structures: list[str]) -> list[dict]:
    """Discover input images from a CSV or directory.

    Returns a list of dicts with keys: structure, raw_path, image_id.
    """
    p = Path(input_path)
    items = []

    if p.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(p)
        if "raw" not in df.columns:
            print(
                f"ERROR: CSV must have a 'raw' column. Found: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

        for _, row in df.iterrows():
            raw_path = row["raw"]
            filename = Path(raw_path).name
            structure = extract_structure_from_filename(filename)
            if structure in structures:
                items.append(
                    {
                        "structure": structure,
                        "raw_path": raw_path,
                        "image_id": filename,
                    }
                )

    elif p.is_dir():
        tiff_files = sorted(p.glob("*.tiff")) + sorted(p.glob("*.tif"))
        for tiff_path in tiff_files:
            filename = tiff_path.name
            structure = extract_structure_from_filename(filename)
            if structure in structures:
                items.append(
                    {
                        "structure": structure,
                        "raw_path": str(tiff_path),
                        "image_id": filename,
                    }
                )

    else:
        print(
            f"ERROR: Input must be a .csv file or directory: {input_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return items


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------


def make_output_filename(image_id: str, structure: str) -> str:
    """Generate output filename: {STRUCT}_{ID}_seg.tiff"""
    # Strip known suffixes to get the base
    base = image_id
    for suffix in ("_IM.tiff", "_GT.tiff", ".tiff", ".tif"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}_seg.tiff"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    items: list[dict],
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Run classic workflows on discovered items.

    Returns summary dict with counts.
    """
    # Group by structure for efficient workflow loading
    by_structure: dict[str, list[dict]] = {}
    for item in items:
        by_structure.setdefault(item["structure"], []).append(item)

    total = len(items)
    processed = 0
    errors = 0
    skipped = 0

    for structure in sorted(by_structure):
        struct_items = by_structure[structure]
        print(f"\n--- {structure} ({len(struct_items)} images) ---")

        if dry_run:
            for item in struct_items:
                out_name = make_output_filename(item["image_id"], structure)
                out_path = output_dir / structure / out_name
                print(f"  [dry-run] {item['image_id']} -> {out_path}")
            processed += len(struct_items)
            continue

        # Load workflow
        workflow_fn = load_workflow_function(structure)
        if workflow_fn is None:
            print(f"  SKIP: Could not load workflow for {structure}")
            skipped += len(struct_items)
            continue

        # Create output subdirectory
        struct_out = output_dir / structure
        struct_out.mkdir(parents=True, exist_ok=True)

        for item in struct_items:
            raw_path = item["raw_path"]
            image_id = item["image_id"]
            out_name = make_output_filename(image_id, structure)
            out_path = struct_out / out_name

            if not Path(raw_path).exists():
                print(f"  MISSING: {raw_path}")
                errors += 1
                continue

            try:
                t0 = time.time()
                struct_img = load_image_3d(raw_path)
                pred = workflow_fn(struct_img, output_type="array")

                # Save as uint8 {0, 255}
                pred_save = ((pred > 0).astype(np.uint8)) * 255
                imwrite(str(out_path), pred_save)

                elapsed = time.time() - t0
                print(f"  {image_id} -> {out_name}  ({elapsed:.1f}s)")
                processed += 1

            except Exception as e:
                print(f"  ERROR {image_id}: {e}")
                errors += 1

    return {
        "total": total,
        "processed": processed,
        "errors": errors,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_structures():
    """Print available structures and exit."""
    print("Available structures (17):")
    print()
    for name, (script, func, broken) in sorted(STRUCTURES.items()):
        status = f"  (patched: {broken})" if broken else ""
        print(f"  {name:<15s}  {func}{status}")


def main():
    parser = argparse.ArgumentParser(
        description="Run classic aicssegmentation workflows on input data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --list
  %(prog)s --input /path/to/tiffs/ --output-dir ./out/
  %(prog)s --structures ACTN1 NPM1 --input data.csv --output-dir ./out/
  %(prog)s --input /path/to/tiffs/ --output-dir ./out/ --dry-run
""",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available structures and exit",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=None,
        metavar="STRUCT",
        help="Structures to run (default: all). Use --list to see options.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV (must have 'raw' column) or directory of TIFFs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save segmentation outputs (organized by structure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running workflows",
    )
    args = parser.parse_args()

    # --list mode
    if args.list:
        list_structures()
        return

    # Validate required args
    if args.input is None:
        parser.error("--input is required (unless using --list)")
    if args.output_dir is None:
        parser.error("--output-dir is required (unless using --list)")

    # Resolve structures
    if args.structures is None:
        structures = list(STRUCTURES.keys())
        print(f"Running all {len(structures)} structures")
    else:
        # Validate structure names
        unknown = [s for s in args.structures if s not in STRUCTURES]
        if unknown:
            print(f"ERROR: Unknown structures: {unknown}", file=sys.stderr)
            print(f"Use --list to see available structures.", file=sys.stderr)
            sys.exit(1)
        structures = args.structures
        print(f"Running {len(structures)} structures: {structures}")

    # Discover inputs
    items = discover_inputs(args.input, structures)
    if not items:
        print("No matching input images found.")
        return

    struct_counts = {}
    for item in items:
        struct_counts[item["structure"]] = struct_counts.get(item["structure"], 0) + 1
    print(f"\nFound {len(items)} images across {len(struct_counts)} structures:")
    for s, n in sorted(struct_counts.items()):
        print(f"  {s}: {n}")

    # Run pipeline
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_pipeline(items, output_dir, dry_run=args.dry_run)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"{'DRY RUN ' if args.dry_run else ''}SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total images:   {summary['total']}")
    print(f"  Processed:      {summary['processed']}")
    print(f"  Errors:         {summary['errors']}")
    print(f"  Skipped:        {summary['skipped']}")
    print(f"  Output dir:     {args.output_dir}")


if __name__ == "__main__":
    main()
