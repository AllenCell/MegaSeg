"""
MegaSeg Figure — 12 unseen structures, MegaSeg-only, good vs bad side-by-side.

For each structure, shows a good case and a bad case in a single row (10 columns).
Good case = best-Dice z-slice + best local-Dice crop.
Bad case  = worst-Dice z-slice (with enough foreground) + worst local-Dice crop.

Layout per row (1 row per structure):
  Good: Raw | GT | Prob | Mask | Error  |  Bad: Raw | GT | Prob | Mask | Error

Usage:
    python megaseg_fig2b_12structs_megaseg_only_v2.py [--crop 320] [--dpi 400]
"""

import os
import csv
import json
import argparse
import numpy as np
import tifffile as tif
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════
MEGASEG_PRED_DIR = (
    "/allen/aics/lumenoid/megaseg-benchmark/outputs/eval/uint8/"
    "megaseg/dynunet/test_new/seg"
)
TEST_CSV = (
    "/allen/aics/lumenoid/megaseg-benchmark/data/megaseg_13structs/"
    "test_new/test_new.csv"
)

FIGURE_DIR = Path(__file__).parent

STRUCTURES = [
    "AAVS1", "ACTB", "CETN2", "FBL", "HIST1H2BJ", "LAMP1",
    "LMNB1", "NPM1", "PXN", "SLC25A17", "SMC1A", "TUBA1B",
]

PART1_STRUCTS = STRUCTURES[:6]
PART2_STRUCTS = STRUCTURES[6:]

PIXEL_SIZE_UM = 0.10833
SCALEBAR_UM = 10
PROB_CMAP = "inferno"

# Minimum GT foreground requirements
MIN_SLICE_FG = 200          # min foreground pixels for a z-slice to be considered
MIN_CROP_FG = 1500          # min foreground pixels within a crop patch
MIN_CROP_FG_FRAC = 0.01     # min fraction of crop area that must be GT foreground (1%)

# Force specific samples for structures where the default pick has sparse GT
SAMPLE_OVERRIDES = {
    "LAMP1": "LAMP1_150879_IM",       # denser GT (max slice fg ~59k vs ~1.3k)
    "CETN2": "CETN2_9471_IM",         # slightly better max-fg slices
    "SLC25A17": "SLC25A17_147878_IM",  # denser slices (max fg ~2.2k vs ~0.7k)
}

# Use a different sample for the bad case (good/bad from different volumes)
BAD_SAMPLE_OVERRIDES = {
    "FBL": "FBL_9076_IM",             # bad from the other sample
}


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════
def fix_unc_path(path: str) -> str:
    p = path.strip()
    return "/" + p if p.startswith("/allen") else p


def to_prob(vol):
    if vol.dtype == np.uint8:
        return vol.astype(np.float32) / 255.0
    elif vol.dtype == np.uint16:
        return vol.astype(np.float32) / 65535.0
    elif np.issubdtype(vol.dtype, np.floating):
        return vol.astype(np.float32)
    else:
        mx = vol.max()
        return vol.astype(np.float32) / mx if mx > 0 else vol.astype(np.float32)


def compute_dice(gt_bool, pred_bool):
    tp = np.sum(gt_bool & pred_bool)
    denom = 2.0 * tp + np.sum(~gt_bool & pred_bool) + np.sum(gt_bool & ~pred_bool)
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def read_test_csv():
    samples_by_struct = {}
    with open(TEST_CSV, newline="") as f:
        for r in csv.DictReader(f):
            raw_path = fix_unc_path(r["raw"])
            seg_path = fix_unc_path(r["seg"])
            stem = os.path.basename(raw_path).replace(".tiff", "")
            struct = stem.replace("_IM", "").rsplit("_", 1)[0]
            samples_by_struct.setdefault(struct, []).append({
                "raw": raw_path, "seg": seg_path, "stem": stem,
            })
    return samples_by_struct


def per_slice_dice(gt_vol, pred_bool_vol):
    """Return list of (z, dice, fg_count) for slices with enough foreground."""
    results = []
    for z in range(gt_vol.shape[0]):
        g = gt_vol[z] > 0
        fg = np.sum(g)
        if fg < MIN_SLICE_FG:
            continue
        p = pred_bool_vol[z]
        tp = np.sum(g & p)
        denom = 2.0 * tp + np.sum(~g & p) + np.sum(g & ~p)
        d = float(2.0 * tp / denom) if denom > 0 else 0.0
        results.append((z, d, int(fg)))
    return results


def smart_crop(shape_2d, crop_size, ref_mask=None):
    h, w = shape_2d
    crop_size = min(crop_size, h, w)
    if ref_mask is not None and ref_mask.sum() > 0:
        ys, xs = np.where(ref_mask > 0)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
    else:
        cy, cx = h // 2, w // 2
    y0 = max(0, min(cy - crop_size // 2, h - crop_size))
    x0 = max(0, min(cx - crop_size // 2, w - crop_size))
    return y0, x0, crop_size


def _has_enough_gt(gt_crop, crop_size):
    """Check if a crop has enough GT foreground (absolute count AND fraction)."""
    fg = np.sum(gt_crop)
    frac = fg / (crop_size * crop_size)
    return fg >= MIN_CROP_FG and frac >= MIN_CROP_FG_FRAC


def find_crop(gt_slice, pred_bool_slice, crop_size, mode="best", step=20):
    """Find crop that maximizes (mode='best') or minimizes (mode='worst') local Dice.
    Only considers crops with enough GT foreground."""
    h, w = gt_slice.shape
    cs = min(crop_size, h, w)
    radius = cs // 2
    gt_bool = gt_slice > 0

    ys_fg, xs_fg = np.where(gt_bool)
    if len(ys_fg) == 0:
        return smart_crop(gt_slice.shape, crop_size)

    if mode == "best":
        target_d = -1.0
        compare = lambda d, t: d > t
    else:  # worst
        target_d = 2.0
        compare = lambda d, t: d < t

    target_pos = None

    for cy in range(radius, h - radius + 1, step):
        for cx in range(radius, w - radius + 1, step):
            y0 = max(0, min(cy - cs // 2, h - cs))
            x0 = max(0, min(cx - cs // 2, w - cs))
            g = gt_bool[y0:y0 + cs, x0:x0 + cs]
            if not _has_enough_gt(g, cs):
                continue
            p = pred_bool_slice[y0:y0 + cs, x0:x0 + cs]
            tp = np.sum(g & p)
            denom = 2.0 * tp + np.sum(~g & p) + np.sum(g & ~p)
            d = float(2.0 * tp / denom) if denom > 0 else 0.0
            if compare(d, target_d):
                target_d = d
                target_pos = (y0, x0)

    # Also try GT centroid
    cy_g, cx_g = int(np.mean(ys_fg)), int(np.mean(xs_fg))
    y0 = max(0, min(cy_g - cs // 2, h - cs))
    x0 = max(0, min(cx_g - cs // 2, w - cs))
    g = gt_bool[y0:y0 + cs, x0:x0 + cs]
    if _has_enough_gt(g, cs):
        p = pred_bool_slice[y0:y0 + cs, x0:x0 + cs]
        tp = np.sum(g & p)
        denom = 2.0 * tp + np.sum(~g & p) + np.sum(g & ~p)
        d = float(2.0 * tp / denom) if denom > 0 else 0.0
        if compare(d, target_d):
            target_pos = (y0, x0)

    # Fallback to GT centroid if nothing passed the threshold
    if target_pos is None:
        target_pos = (max(0, min(cy_g - cs // 2, h - cs)),
                      max(0, min(cx_g - cs // 2, w - cs)))

    return target_pos[0], target_pos[1], cs


def make_error_map(gt_mask, pred_mask):
    gt = gt_mask > 0.5
    pred = pred_mask > 0.5
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[gt & pred] = [0.85, 0.75, 0.25]
    rgb[~gt & pred] = [0.82, 0.35, 0.45]
    rgb[gt & ~pred] = [0.30, 0.55, 0.75]
    rgb[~gt & ~pred] = [0.12, 0.12, 0.12]
    return rgb


def add_scalebar(ax, crop_pixels):
    bar_px = SCALEBAR_UM / PIXEL_SIZE_UM
    margin = 8
    bar_y = crop_pixels - margin
    bar_x_end = crop_pixels - margin
    bar_x_start = bar_x_end - bar_px
    ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
            color="white", linewidth=2, solid_capstyle="butt")
    ax.text((bar_x_start + bar_x_end) / 2, bar_y - 5,
            f"{SCALEBAR_UM} \u00b5m", color="white", fontsize=6,
            ha="center", va="bottom", fontweight="bold")


# ══════════════════════════════════════════════════════════════
# Data loading — good + bad case per structure
# ══════════════════════════════════════════════════════════════
def extract_panels(raw_vol, gt_vol, mega_prob_vol, mega_bool_vol, z, crop_size, mode):
    """Extract 5 panels (raw, gt, prob, mask, error) for a given z-slice and crop mode."""
    raw_z = raw_vol[z]
    gt_z = gt_vol[z]
    mega_prob_z = mega_prob_vol[z]
    mega_bool_z = mega_bool_vol[z]

    y0, x0, cs = find_crop(gt_z, mega_bool_z, crop_size, mode=mode)
    sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))

    raw_crop = raw_z[sl].astype(np.float32)
    rmin, rmax = raw_crop.min(), raw_crop.max()
    raw_crop = (raw_crop - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(raw_crop)

    gt_crop = (gt_z[sl] > 0).astype(np.float32)
    mega_prob = mega_prob_z[sl[0], sl[1]]
    mega_mask = (mega_prob >= 0.5).astype(np.float32)
    mega_err = make_error_map(gt_crop, mega_mask)

    return {
        "raw": raw_crop, "gt": gt_crop,
        "mega_prob": mega_prob, "mega_mask": mega_mask, "mega_err": mega_err,
        "z_slice": z, "y_start": y0, "x_start": x0, "crop_size": cs,
    }


def _select_good_bad_slices(gt, mega_prob_vol):
    """Select good and bad z-slices.

    Good: best Dice among top-50% foreground slices.
    Bad:  most foreground among slices with below-median Dice (in fg-rich set).
          Prioritizes GT visibility over worst-case Dice.

    Returns (good_slice, bad_slice) as (z, dice, fg) tuples, or (None, None).
    """
    mega_bool_vol = mega_prob_vol >= 0.5
    slices = per_slice_dice(gt, mega_bool_vol)
    if not slices:
        return None, None

    # Top 50% by foreground
    slices.sort(key=lambda x: x[2], reverse=True)
    n_keep = max(2, len(slices) // 2)
    fg_rich = slices[:n_keep]

    # Good: best Dice
    fg_rich_by_dice = sorted(fg_rich, key=lambda x: x[1])
    good_slice = fg_rich_by_dice[-1]

    # Bad: among bottom half by Dice, pick the one with MOST foreground
    n_bottom = max(1, len(fg_rich_by_dice) // 2)
    bottom_half = fg_rich_by_dice[:n_bottom]
    bad_slice = max(bottom_half, key=lambda x: x[2])  # most fg

    return good_slice, bad_slice


def load_single_case(sample, crop_size, mode="best"):
    """Load a single good or bad case from a sample volume."""
    raw = tif.imread(sample["raw"])
    gt = tif.imread(sample["seg"])
    mega_path = os.path.join(MEGASEG_PRED_DIR, f"{sample['stem']}.tif")
    mega = tif.imread(mega_path)
    mega_prob_vol = to_prob(mega)
    mega_bool_vol = mega_prob_vol >= 0.5

    good_slice, bad_slice = _select_good_bad_slices(gt, mega_prob_vol)
    if good_slice is None:
        return None

    z, dice, _ = good_slice if mode == "best" else bad_slice
    panels = extract_panels(raw, gt, mega_prob_vol, mega_bool_vol, z, crop_size, mode=mode)
    panels["slice_dice"] = dice
    return panels


def load_good_bad(sample, crop_size):
    """Load both good and bad cases from the same sample volume."""
    raw = tif.imread(sample["raw"])
    gt = tif.imread(sample["seg"])
    mega_path = os.path.join(MEGASEG_PRED_DIR, f"{sample['stem']}.tif")
    mega = tif.imread(mega_path)
    mega_prob_vol = to_prob(mega)
    mega_bool_vol = mega_prob_vol >= 0.5

    good_slice, bad_slice = _select_good_bad_slices(gt, mega_prob_vol)
    if good_slice is None:
        return None, None

    z_best, dice_best, _ = good_slice
    z_worst, dice_worst, _ = bad_slice

    good = extract_panels(raw, gt, mega_prob_vol, mega_bool_vol, z_best, crop_size, mode="best")
    good["slice_dice"] = dice_best

    bad = extract_panels(raw, gt, mega_prob_vol, mega_bool_vol, z_worst, crop_size, mode="worst")
    bad["slice_dice"] = dice_worst

    return good, bad


def load_all_data(samples_by_struct, crop_size):
    """For each structure, try all available samples (best Dice first)
    and pick the first one that produces valid good/bad slices."""
    all_data = {}
    meta = {"crop_size": crop_size, "samples": {}}

    for struct in STRUCTURES:
        if struct not in samples_by_struct:
            print(f"  {struct}: SKIP")
            continue

        # Score all samples by whole-volume Dice
        scored = []
        for s in samples_by_struct[struct]:
            mega_path = os.path.join(MEGASEG_PRED_DIR, f"{s['stem']}.tif")
            if not os.path.exists(mega_path):
                continue
            gt = tif.imread(s["seg"])
            pred = tif.imread(mega_path)
            d = compute_dice(gt > 0, to_prob(pred) >= 0.5)
            scored.append((d, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            print(f"  {struct}: SKIP (no valid samples)")
            continue

        # Apply sample override if specified
        if struct in SAMPLE_OVERRIDES:
            override_stem = SAMPLE_OVERRIDES[struct]
            override = [(d, s) for d, s in scored if s["stem"] == override_stem]
            if override:
                scored = override + [(d, s) for d, s in scored if s["stem"] != override_stem]

        # Check if bad case should come from a different sample
        bad_override_stem = BAD_SAMPLE_OVERRIDES.get(struct)
        bad_override_sample = None
        if bad_override_stem:
            for _, s in scored:
                if s["stem"] == bad_override_stem:
                    bad_override_sample = s
                    break

        loaded = False
        for vol_dice, sample in scored:
            if bad_override_sample and sample["stem"] == bad_override_stem:
                # Skip the bad-override sample as primary — it's used only for bad case
                continue

            if bad_override_sample:
                # Good from primary sample, bad from override sample
                good = load_single_case(sample, crop_size, mode="best")
                bad = load_single_case(bad_override_sample, crop_size, mode="worst")
                bad_stem = bad_override_stem
            else:
                # Both from the same sample
                good, bad = load_good_bad(sample, crop_size)
                bad_stem = sample["stem"]

            if good is None or bad is None:
                print(f"    {sample['stem']}: skipped (no valid slices with enough GT)")
                continue

            good_gt_fg = np.sum(good["gt"] > 0.5)
            bad_gt_fg = np.sum(bad["gt"] > 0.5)
            good_frac = good_gt_fg / (good["crop_size"] ** 2)
            bad_frac = bad_gt_fg / (bad["crop_size"] ** 2)

            all_data[struct] = {"good": good, "bad": bad}
            meta[struct] = {
                "good_stem": sample["stem"], "bad_stem": bad_stem,
                "vol_dice": vol_dice,
                "good_z": good["z_slice"], "good_slice_dice": good["slice_dice"],
                "good_crop": (good["y_start"], good["x_start"]),
                "good_gt_fg": int(good_gt_fg), "good_gt_frac": round(good_frac, 4),
                "bad_z": bad["z_slice"], "bad_slice_dice": bad["slice_dice"],
                "bad_crop": (bad["y_start"], bad["x_start"]),
                "bad_gt_fg": int(bad_gt_fg), "bad_gt_frac": round(bad_frac, 4),
            }
            good_src = sample["stem"]
            bad_src = bad_stem if bad_stem != good_src else ""
            bad_label = f" [{bad_src}]" if bad_src else ""
            print(f"  {struct}: good={good_src}  "
                  f"(z={good['z_slice']}, dice={good['slice_dice']:.3f}, "
                  f"gt_fg={good_gt_fg}, {good_frac:.1%})  "
                  f"bad{bad_label}  "
                  f"(z={bad['z_slice']}, dice={bad['slice_dice']:.3f}, "
                  f"gt_fg={bad_gt_fg}, {bad_frac:.1%})")
            loaded = True
            break

        if not loaded:
            print(f"  {struct}: SKIP (no sample had valid slices)")

    return all_data, meta


# ══════════════════════════════════════════════════════════════
# Figure — 10 columns per row (good 5 + bad 5)
# ══════════════════════════════════════════════════════════════
def create_figure(all_data, save_path, dpi=400, struct_list=None):
    structs = struct_list or list(all_data.keys())
    structs = [s for s in structs if s in all_data]
    n_rows = len(structs)
    n_cols = 10  # 5 good + 5 bad

    col_titles = [
        "Raw", "Pseudo-\nLabels", "MegaSeg\nProb.", "MegaSeg\nMask", "MegaSeg\nError",
        "Raw", "Pseudo-\nLabels", "MegaSeg\nProb.", "MegaSeg\nMask", "MegaSeg\nError",
    ]

    fig = plt.figure(
        figsize=(n_cols * 2.2, n_rows * 2.4 + 1.6), facecolor="white")

    # Use nested gridspec: outer splits good|bad, inner is 5 columns each
    outer_gs = gridspec.GridSpec(
        n_rows, 2, wspace=0.08, hspace=0.10,
        left=0.05, right=0.98, top=0.94, bottom=0.01)

    for row, struct in enumerate(structs):
        d = all_data[struct]

        for half, (key, label_text) in enumerate([("good", "Good"), ("bad", "Bad")]):
            panels_data = d[key]
            crop_px = panels_data["crop_size"]

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 5, subplot_spec=outer_gs[row, half], wspace=0.03)

            panels = [
                (panels_data["raw"],       "gray",     False),
                (panels_data["gt"],        "gray",     False),
                (panels_data["mega_prob"], PROB_CMAP,  False),
                (panels_data["mega_mask"], "gray",     False),
                (panels_data["mega_err"],  None,       True),
            ]

            for col, (img, cmap, is_rgb) in enumerate(panels):
                ax = fig.add_subplot(inner_gs[0, col])

                if is_rgb:
                    ax.imshow(img, interpolation="nearest", aspect="equal")
                else:
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1,
                              interpolation="nearest", aspect="equal")

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
                    spine.set_color("#555555")

                # Scale bar on raw panels
                if col == 0:
                    add_scalebar(ax, crop_px)

                # Structure label on leftmost column of good panel
                if half == 0 and col == 0:
                    ax.set_ylabel(struct, fontsize=16, fontweight="bold",
                                  rotation=90, labelpad=10, va="center")

                # Column titles on first row
                if row == 0:
                    ax.set_title(col_titles[half * 5 + col], fontsize=12,
                                 fontweight="bold", pad=6)

    # Panel group labels at top
    fig.text(0.27, 0.97, "Good Case", fontsize=14, fontweight="bold",
             ha="center", va="bottom", color="#2171B5")
    fig.text(0.76, 0.97, "Bad Case", fontsize=14, fontweight="bold",
             ha="center", va="bottom", color="#CB181D")

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    print(f"  Figure saved: {save_path}")
    plt.close(fig)


def save_colorbar(out_path):
    fig, ax = plt.subplots(figsize=(4, 0.4), facecolor="white")
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=PROB_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Prediction Probability", fontsize=10, fontweight="bold")
    cb.ax.tick_params(labelsize=9)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Colorbar saved: {out_path}")


def save_error_legend(out_path):
    fig, ax = plt.subplots(figsize=(6, 0.5), facecolor="white")
    ax.axis("off")
    legend_handles = [
        Patch(facecolor=(0.85, 0.75, 0.25), edgecolor="gray", label="True Positive"),
        Patch(facecolor=(0.82, 0.35, 0.45), edgecolor="gray", label="False Positive"),
        Patch(facecolor=(0.30, 0.55, 0.75), edgecolor="gray", label="False Negative"),
        Patch(facecolor=(0.12, 0.12, 0.12), edgecolor="gray", label="True Negative"),
    ]
    ax.legend(handles=legend_handles, loc="center", ncol=4,
              fontsize=10, frameon=True, edgecolor="black")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Error legend saved: {out_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="MegaSeg Fig -- 12 unseen structures, good vs bad side-by-side")
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()

    prefix = args.prefix or "fig2b_12structs_megaseg_only_v2"

    # Step 1: Read CSV
    print("Reading test_new.csv...")
    samples_by_struct = read_test_csv()

    # Step 2: Load good + bad cases
    print(f"\nLoading good/bad cases (crop={args.crop})...")
    all_data, metadata = load_all_data(samples_by_struct, args.crop)

    if not all_data:
        print("No data loaded.")
        return

    # Step 3: Figures
    path1 = FIGURE_DIR / f"{prefix}_part1.pdf"
    print(f"\nGenerating Part 1 ({len(PART1_STRUCTS)} structures)...")
    create_figure(all_data, str(path1), args.dpi, struct_list=PART1_STRUCTS)

    path2 = FIGURE_DIR / f"{prefix}_part2.pdf"
    print(f"Generating Part 2 ({len(PART2_STRUCTS)} structures)...")
    create_figure(all_data, str(path2), args.dpi, struct_list=PART2_STRUCTS)

    cbar_path = FIGURE_DIR / f"{prefix}_colorbar.pdf"
    save_colorbar(cbar_path)

    legend_path = FIGURE_DIR / f"{prefix}_error_legend.pdf"
    save_error_legend(legend_path)

    meta_out = FIGURE_DIR / f"{prefix}_metadata.json"
    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata: {meta_out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
