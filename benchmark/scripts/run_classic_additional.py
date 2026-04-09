#!/usr/bin/env python
"""Run classic segmentation workflows on additional structures channel-volumes.

Runs all 14 classic workflows on every extracted single-channel TIFF from
the additional structures data. This creates ~574 predictions (41 volumes
x 14 workflows).

Output: outputs/eval/classic/additional_structures/{WORKFLOW}/{volume}_seg.tiff

Usage (with msseg conda env):
    PYTHON=/allen/aics/users/ashwin.samudre/miniconda3/envs/msseg/bin/python
    $PYTHON scripts/run_classic_additional.py [--dry-run] [--workflows ACTN1 NPM1]

Prerequisites: run extract_additional_channels.py first.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tifffile import imwrite

# Add scripts dir to path for classic_workflows import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classic_workflows import load_image_3d, load_workflow_function


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# All 14 classic workflows that match the individual DL models
ALL_WORKFLOWS = [
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

STRUCTURES = ["CTCF", "FUS", "LC3B", "PCNA"]

# Relative to project root
EXTRACTED_DIR = Path("data/additional_structures_extracted")
OUTPUT_DIR = Path("outputs/eval/classic/additional_structures")


def discover_inputs(project_root: Path) -> list[dict]:
    """Find all extracted single-channel TIFFs.

    Returns list of dicts: {structure, path, stem}.
    """
    base = project_root / EXTRACTED_DIR
    items = []

    for struct in STRUCTURES:
        struct_dir = base / struct
        if not struct_dir.exists():
            print(f"WARNING: {struct_dir} not found", file=sys.stderr)
            continue

        tiffs = sorted(struct_dir.glob("*.tiff")) + sorted(struct_dir.glob("*.tif"))
        for t in tiffs:
            items.append(
                {
                    "structure": struct,
                    "path": str(t),
                    "stem": t.stem,
                }
            )

    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=ALL_WORKFLOWS,
        help="Which workflows to run (default: all 14)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    workflows = args.workflows

    print(f"Project root: {project_root}")
    print(f"Workflows: {workflows}")
    print(f"Input dir: {project_root / EXTRACTED_DIR}")
    print(f"Output dir: {project_root / OUTPUT_DIR}")
    print()

    # Discover extracted volumes
    items = discover_inputs(project_root)
    if not items:
        print(
            "ERROR: No extracted volumes found. Run extract_additional_channels.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(items)} extracted volumes across {len(STRUCTURES)} structures")
    total_jobs = len(items) * len(workflows)
    print(
        f"Total predictions: {total_jobs} ({len(items)} volumes x {len(workflows)} workflows)"
    )
    print()

    if args.dry_run:
        for wf in workflows:
            out_dir = project_root / OUTPUT_DIR / wf
            print(f"[dry-run] {wf}: {len(items)} predictions -> {out_dir}")
        return

    # Run all workflows
    processed = 0
    errors = 0
    skipped = 0
    t_start = time.time()

    for wf_idx, wf_name in enumerate(workflows):
        print(f"\n{'=' * 60}")
        print(f"[{wf_idx + 1}/{len(workflows)}] Workflow: {wf_name}")
        print(f"{'=' * 60}")

        # Load workflow function once per workflow
        wf_fn = load_workflow_function(wf_name)
        if wf_fn is None:
            print(f"  SKIP: Could not load workflow for {wf_name}")
            skipped += len(items)
            continue

        out_dir = project_root / OUTPUT_DIR / wf_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            stem = item["stem"]
            struct = item["structure"]
            out_name = f"{stem}_seg.tiff"
            out_path = out_dir / out_name

            # Skip if already done
            if out_path.exists():
                print(f"  {struct}/{stem} -> (exists, skipping)")
                processed += 1
                continue

            try:
                t0 = time.time()
                img = load_image_3d(item["path"])
                pred = wf_fn(img, output_type="array")

                # Save as uint8 {0, 255}
                pred_save = ((pred > 0).astype(np.uint8)) * 255
                imwrite(str(out_path), pred_save)

                elapsed = time.time() - t0
                processed += 1
                print(f"  {struct}/{stem} -> {out_name}  ({elapsed:.1f}s)")

            except Exception as e:
                errors += 1
                print(f"  ERROR {struct}/{stem}: {e}", file=sys.stderr)

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed_total / 60:.1f} min")
    print(f"  Processed: {processed}/{total_jobs}")
    print(f"  Errors: {errors}")
    print(f"  Skipped (workflow load fail): {skipped}")


if __name__ == "__main__":
    main()
