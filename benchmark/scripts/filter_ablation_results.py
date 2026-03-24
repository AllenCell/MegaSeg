#!/usr/bin/env python
"""Filter ablation results by removing low-signal samples.

Matches PR #16's approach: uses the same excluded samples identified by the
main benchmark's MegaSeg DynUNet Dice scores. Keeps original results intact
and saves filtered versions to results/ablation/filtered/.

For AAVS1 (not in PR #16's benchmark): the bottom 10 samples by Dice score
from the 14-struct model are excluded (the 14-struct model trained on AAVS1,
making it the appropriate reference, analogous to PR #16's approach).

Final sample counts: 57 samples per structure x 14 structures = 798 per model.

Usage:
    python scripts/filter_ablation_results.py
    python scripts/filter_ablation_results.py --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation"
FILTERED_DIR = RESULTS_DIR / "filtered"

# Excluded samples from PR #16 (bottom 10 by MegaSeg DynUNet Dice per structure).
# NPM1 is included in the original list but not used in ablation analysis.
# AAVS1: not in PR #16 benchmark; bottom 10 by 14-struct model Dice excluded instead.
EXCLUDED_SAMPLES = {
    "AAVS1": [
        "AAVS1_10627",
        "AAVS1_17146",
        "AAVS1_10272",
        "AAVS1_11161",
        "AAVS1_10754",
        "AAVS1_15209",
        "AAVS1_10854",
        "AAVS1_10708",
        "AAVS1_10312",
        "AAVS1_10753",
    ],
    "ACTN1": [
        "ACTN1_11101",
        "ACTN1_11115",
        "ACTN1_10934",
        "ACTN1_11053",
        "ACTN1_11399",
        "ACTN1_11459",
        "ACTN1_11031",
        "ACTN1_11366",
        "ACTN1_11547",
        "ACTN1_10917",
    ],
    "ATP2A2": [
        "ATP2A2_146596",
        "ATP2A2_146639",
        "ATP2A2_147125",
        "ATP2A2_145986",
        "ATP2A2_142049",
        "ATP2A2_146627",
        "ATP2A2_142001",
        "ATP2A2_147204",
        "ATP2A2_146973",
        "ATP2A2_142034",
    ],
    "CTNNB1": [
        "CTNNB1_13729",
        "CTNNB1_13892",
        "CTNNB1_13782",
        "CTNNB1_13748",
        "CTNNB1_13632",
        "CTNNB1_13733",
        "CTNNB1_13821",
        "CTNNB1_13822",
        "CTNNB1_13898",
        "CTNNB1_13592",
    ],
    "DSP": [
        "DSP_4931",
        "DSP_4951",
        "DSP_4956",
        "DSP_4914",
        "DSP_4997",
        "DSP_4960",
        "DSP_4596",
        "DSP_4090",
        "DSP_4560",
        "DSP_15725",
    ],
    "GJA1": [
        "GJA1_10723",
        "GJA1_10901",
        "GJA1_10238",
        "GJA1_11887",
        "GJA1_10563",
        "GJA1_10501",
        "GJA1_10254",
        "GJA1_11846",
        "GJA1_10531",
        "GJA1_10798",
    ],
    "MYH10": [
        "MYH10_7137",
        "MYH10_7078",
        "MYH10_7603",
        "MYH10_7462",
        "MYH10_7527",
        "MYH10_7549",
        "MYH10_6626",
        "MYH10_7143",
        "MYH10_7547",
        "MYH10_7600",
    ],
    "NUP153": [
        "NUP153_145497",
        "NUP153_146810",
        "NUP153_147186",
        "NUP153_145466",
        "NUP153_145501",
        "NUP153_146844",
        "NUP153_145441",
        "NUP153_145406",
        "NUP153_145619",
        "NUP153_145576",
    ],
    "RAB5A": [
        "RAB5A_149458",
        "RAB5A_149256",
        "RAB5A_149262",
        "RAB5A_144295",
        "RAB5A_143779",
        "RAB5A_148001",
        "RAB5A_149246",
        "RAB5A_147854",
        "RAB5A_147995",
        "RAB5A_149475",
    ],
    "SEC61B": [
        "SEC61B_5302",
        "SEC61B_4804",
        "SEC61B_4669",
        "SEC61B_4810",
        "SEC61B_5080",
        "SEC61B_4640",
        "SEC61B_5313",
        "SEC61B_17222",
        "SEC61B_4628",
        "SEC61B_5439",
    ],
    "SON": [
        "SON_136086",
        "SON_136198",
        "SON_136259",
        "SON_136068",
        "SON_136032",
        "SON_136031",
        "SON_137152",
        "SON_136221",
        "SON_136202",
        "SON_137085",
    ],
    "ST6GAL1": [
        "ST6GAL1_5517",
        "ST6GAL1_5956",
        "ST6GAL1_5873",
        "ST6GAL1_5903",
        "ST6GAL1_5511",
        "ST6GAL1_5871",
        "ST6GAL1_5665",
        "ST6GAL1_5963",
        "ST6GAL1_5556",
        "ST6GAL1_5518",
    ],
    "TJP1": [
        "TJP1_7206",
        "TJP1_7213",
        "TJP1_7184",
        "TJP1_7487",
        "TJP1_6969",
        "TJP1_7787",
        "TJP1_6457",
        "TJP1_7311",
        "TJP1_7415",
        "TJP1_7374",
    ],
    "TOMM20": [
        "TOMM20_3837",
        "TOMM20_3490",
        "TOMM20_3817",
        "TOMM20_3133",
        "TOMM20_3046",
        "TOMM20_3707",
        "TOMM20_3337",
        "TOMM20_2874",
        "TOMM20_1390",
        "TOMM20_1329",
    ],
}

# Flatten to a set of sample IDs for fast lookup
ALL_EXCLUDED_IDS = set()
for ids in EXCLUDED_SAMPLES.values():
    ALL_EXCLUDED_IDS.update(ids)

MODEL_KEYS = [
    "2_structs",
    "4_structs",
    "7_structs",
    "14_structs",
    "megaseg_pretrained",
]

# Training structure membership (for recomputing seen/held-out)
TRAINING_STRUCTS = {
    "2_structs": {"TOMM20", "TJP1"},
    "4_structs": {"ACTN1", "ATP2A2", "CTNNB1", "MYH10"},
    "7_structs": {
        "ACTN1",
        "ATP2A2",
        "CTNNB1",
        "MYH10",
        "RAB5A",
        "ST6GAL1",
        "TOMM20",
    },
    "14_structs": {
        "AAVS1",
        "ACTN1",
        "ATP2A2",
        "CTNNB1",
        "DSP",
        "GJA1",
        "MYH10",
        "NUP153",
        "RAB5A",
        "SEC61B",
        "SON",
        "ST6GAL1",
        "TJP1",
        "TOMM20",
    },
    "megaseg_pretrained": {
        "ACTN1",
        "ATP2A2",
        "CTNNB1",
        "DSP",
        "GJA1",
        "MYH10",
        "NUP153",
        "RAB5A",
        "SEC61B",
        "SON",
        "ST6GAL1",
        "TJP1",
        "TOMM20",
    },
}


def filter_per_sample(model_key: str, dry_run: bool = False) -> dict:
    """Filter a model's per-sample CSV, removing excluded samples.

    Returns summary dict with counts.
    """
    src = RESULTS_DIR / f"metrics_{model_key}_per_sample.csv"
    if not src.exists():
        log.warning("Per-sample CSV not found: %s", src)
        return {"model": model_key, "status": "missing"}

    df = pd.read_csv(src)
    n_before = len(df)

    # Filter out excluded samples
    mask = ~df["sample_id"].isin(ALL_EXCLUDED_IDS)
    df_filtered = df[mask].copy()
    n_after = len(df_filtered)
    n_removed = n_before - n_after

    # Per-structure breakdown
    removed_by_struct = {}
    for struct in sorted(df["structure"].unique()):
        struct_before = len(df[df["structure"] == struct])
        struct_after = len(df_filtered[df_filtered["structure"] == struct])
        removed_by_struct[struct] = struct_before - struct_after

    log.info(
        "  %s: %d -> %d rows (-%d removed)",
        model_key,
        n_before,
        n_after,
        n_removed,
    )

    if not dry_run:
        # Save filtered per-sample CSV
        dst = FILTERED_DIR / f"metrics_{model_key}_per_sample.csv"
        df_filtered.to_csv(dst, index=False)

        # Recompute per-structure summary from filtered data
        training = TRAINING_STRUCTS.get(model_key, set())
        summary_rows = []
        for struct in sorted(df_filtered["structure"].unique()):
            sdf = df_filtered[df_filtered["structure"] == struct]
            seen = struct in training
            summary_rows.append(
                {
                    "structure": struct,
                    "seen": seen,
                    "n_samples": len(sdf),
                    "dice_mean": float(sdf["dice"].mean()),
                    "dice_std": float(sdf["dice"].std()),
                    "hd95_mean": float(sdf["hausdorff_95"].mean()),
                    "hd95_std": float(sdf["hausdorff_95"].std()),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_dst = FILTERED_DIR / f"metrics_{model_key}_per_structure.csv"
        summary_df.to_csv(summary_dst, index=False)

    return {
        "model": model_key,
        "status": "filtered",
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_removed,
        "removed_by_structure": removed_by_struct,
    }


def copy_health_files():
    """Copy health CSVs unchanged (they're about model architecture, not samples)."""
    for model_key in MODEL_KEYS:
        src = RESULTS_DIR / f"health_{model_key}.csv"
        if src.exists():
            dst = FILTERED_DIR / f"health_{model_key}.csv"
            dst.write_bytes(src.read_bytes())
            log.info("  Copied health: %s", model_key)


def save_excluded_manifest():
    """Save the excluded samples manifest (matches PR #16 format)."""
    rows = []
    for struct, ids in sorted(EXCLUDED_SAMPLES.items()):
        for sample_id in ids:
            rows.append({"test_struct": struct, "sample_id": sample_id})
    df = pd.DataFrame(rows)
    dst = FILTERED_DIR / "excluded_samples.csv"
    df.to_csv(dst, index=False)
    log.info("Saved excluded manifest: %d samples -> %s", len(df), dst)


def main():
    parser = argparse.ArgumentParser(
        description="Filter ablation results by removing low-signal samples"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be filtered, don't write files",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also regenerate comparison plots from filtered data",
    )
    args = parser.parse_args()

    log.info("Filtering ablation results...")
    log.info(
        "Excluded: %d samples across %d structures",
        len(ALL_EXCLUDED_IDS),
        len(EXCLUDED_SAMPLES),
    )

    if not args.dry_run:
        FILTERED_DIR.mkdir(parents=True, exist_ok=True)
        (FILTERED_DIR / "figures").mkdir(exist_ok=True)

    # Filter each model's results
    all_summaries = []
    for model_key in MODEL_KEYS:
        summary = filter_per_sample(model_key, dry_run=args.dry_run)
        all_summaries.append(summary)

    if not args.dry_run:
        # Copy health files unchanged
        copy_health_files()

        # Save excluded manifest
        save_excluded_manifest()

    # Print summary table
    print("\n" + "=" * 70)
    print("FILTERING SUMMARY")
    print("=" * 70)
    for s in all_summaries:
        if s["status"] == "missing":
            print(f"  {s['model']}: MISSING (skipped)")
            continue
        print(
            f"  {s['model']}: {s['n_before']} -> {s['n_after']} "
            f"(-{s['n_removed']} removed)"
        )
        for struct, n in sorted(s["removed_by_structure"].items()):
            if n > 0:
                print(f"    {struct}: -{n}")
    print("=" * 70)

    if not args.dry_run:
        print(f"\nFiltered results saved to: {FILTERED_DIR}")

    # Optionally regenerate plots
    if args.plot and not args.dry_run:
        log.info("Regenerating comparison plots from filtered data...")
        from plot_ablation_comparison import (
            load_all_health,
            load_all_results,
            plot_bottleneck_health,
            plot_dice_by_structure,
            plot_dice_vs_num_structures,
            plot_heldout_heatmap,
            plot_seen_vs_heldout,
            print_summary_table,
        )

        # Monkey-patch the results and figures directories
        import plot_ablation_comparison as pac

        pac.RESULTS_DIR = FILTERED_DIR
        pac.FIGURES_DIR = FILTERED_DIR / "figures"
        pac.FIGURES_DIR.mkdir(exist_ok=True)

        frames = load_all_results()
        health_frames = load_all_health()
        if frames:
            print_summary_table(frames)
            plot_dice_by_structure(frames)
            plot_seen_vs_heldout(frames)
            plot_dice_vs_num_structures(frames)
            plot_bottleneck_health(health_frames)
            plot_heldout_heatmap(frames)
            print(f"\nFiltered figures saved to: {FILTERED_DIR / 'figures'}")


if __name__ == "__main__":
    main()
