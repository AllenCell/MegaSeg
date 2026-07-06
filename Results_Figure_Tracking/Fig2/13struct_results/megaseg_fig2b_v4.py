"""
MegaSeg Figure 2B v4: Drug-perturbation-v2d styling.

Changes from v3:
  - Inline colorbars removed → separate colorbar PDF
  - Probability colormap: "inferno" (from drug perturbation v2d)
  - Muted error map palette (gold/rose/steel-blue/dark-gray)
  - Scale bars on raw panels
  - Drug-perturbation-v2d visual conventions (spines, spacing)

Layout per row (one per structure):
  Raw | Pseudo-Labels | MegaSeg Prob | MegaSeg Mask | MegaSeg Error |
  Individual Prob | Individual Mask | Individual Error

Usage:
    python megaseg_fig2b_v4.py [--seed 42] [--crop 320] [--dpi 400]
    python megaseg_fig2b_v4.py --metadata fig2b_seed42_crop320_metadata.json
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

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
MEGASEG_PRED_ROOT = "/allen/aics/lumenoid/megaseg-benchmark/outputs/eval/uint8/megaseg/dynunet"
INDIVIDUAL_PRED_ROOT = "/allen/aics/lumenoid/megaseg-benchmark/outputs/eval/uint8/individual"
DATA_ROOT = "/allen/aics/lumenoid/megaseg-benchmark/data/megaseg_13structs"

STRUCTURES = [
    "ACTN1", "ATP2A2", "CTNNB1", "DSP", "GJA1", "MYH10",
    "NUP153", "RAB5A", "SEC61B", "SON", "ST6GAL1", "TJP1", "TOMM20",
]

LEFT_STRUCTS = STRUCTURES[:7]   # ACTN1 .. NUP153
RIGHT_STRUCTS = STRUCTURES[7:]  # RAB5A .. TOMM20

FIGURE_DIR = Path(__file__).parent

PIXEL_SIZE_UM = 0.10833  # μm per pixel
SCALEBAR_UM = 10  # 10 μm scale bar
PROB_CMAP = "inferno"

# Structures to force-resample
RESAMPLE_STRUCTS = {"ATP2A2"}


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────
def fix_unc_path(path: str) -> str:
    p = path.strip()
    if p.startswith("/allen"):
        return "/" + p
    return p


def read_test_csv(struct: str) -> list[dict]:
    csv_path = os.path.join(DATA_ROOT, struct, "test", "test.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            stem = os.path.basename(r["raw"]).replace(".tiff", "")
            rows.append({
                "raw": fix_unc_path(r["raw"]),
                "seg": fix_unc_path(r["seg"]),
                "stem": stem,
            })
    return rows


def get_pred_path(struct, stem, model_type):
    if model_type == "megaseg":
        return os.path.join(MEGASEG_PRED_ROOT, struct, "seg", f"{stem}.tif")
    else:
        return os.path.join(INDIVIDUAL_PRED_ROOT, f"ms13-{struct}", "seg", f"{stem}.tif")


def find_valid_samples(struct):
    rows = read_test_csv(struct)
    valid = []
    for r in rows:
        mega_path = get_pred_path(struct, r["stem"], "megaseg")
        ind_path = get_pred_path(struct, r["stem"], "individual")
        if os.path.exists(mega_path) and os.path.exists(ind_path):
            r["megaseg_pred"] = mega_path
            r["individual_pred"] = ind_path
            valid.append(r)
    return valid


def pick_best_z_slice(volume_3d):
    fg_per_slice = np.sum(volume_3d > 0, axis=(1, 2))
    return int(np.argmax(fg_per_slice))


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


def make_error_map(gt_mask, pred_mask):
    """Muted, publication-friendly error map.

    Muted gold   = True Positive
    Soft rose    = False Positive
    Steel blue   = False Negative
    Dark gray    = True Negative
    """
    gt = gt_mask > 0.5
    pred = pred_mask > 0.5
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    tn = ~gt & ~pred
    rgb[tp] = [0.85, 0.75, 0.25]   # Muted gold
    rgb[fp] = [0.82, 0.35, 0.45]   # Soft rose
    rgb[fn] = [0.30, 0.55, 0.75]   # Steel blue
    rgb[tn] = [0.12, 0.12, 0.12]   # Dark gray
    return rgb


def add_scalebar(ax, crop_pixels):
    """White 10 um scale bar in lower-right corner."""
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


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def load_sample(sample, crop_size):
    raw = tif.imread(sample["raw"])
    gt = tif.imread(sample["seg"])
    mega = tif.imread(sample["megaseg_pred"])
    ind = tif.imread(sample["individual_pred"])

    z = pick_best_z_slice(gt)
    raw_z, gt_z, mega_z, ind_z = raw[z], gt[z], mega[z], ind[z]

    y0, x0, cs = smart_crop(gt_z.shape, crop_size, ref_mask=gt_z)
    sl = (slice(y0, y0+cs), slice(x0, x0+cs))

    raw_crop = raw_z[sl].astype(np.float32)
    rmin, rmax = raw_crop.min(), raw_crop.max()
    if rmax > rmin:
        raw_crop = (raw_crop - rmin) / (rmax - rmin)
    else:
        raw_crop = np.zeros_like(raw_crop)

    gt_crop = gt_z[sl].astype(np.float32) / 255.0
    mega_prob = mega_z[sl].astype(np.float32) / 255.0
    ind_prob = ind_z[sl].astype(np.float32) / 255.0

    mega_mask = (mega_prob >= 0.5).astype(np.float32)
    ind_mask = (ind_prob >= 0.5).astype(np.float32)

    mega_err = make_error_map(gt_crop, mega_mask)
    ind_err = make_error_map(gt_crop, ind_mask)

    return {
        "raw": raw_crop,
        "gt": gt_crop,
        "mega_mask": mega_mask,
        "mega_prob": mega_prob,
        "mega_err": mega_err,
        "ind_mask": ind_mask,
        "ind_prob": ind_prob,
        "ind_err": ind_err,
        "z_slice": z, "y_start": y0, "x_start": x0, "crop_size": cs,
    }


def load_all_data(seed, crop_size, metadata_path=None, resample_structs=None, resample_seed=None):
    resample_structs = resample_structs or set()
    resample_seed = resample_seed if resample_seed is not None else seed
    prev_meta = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path) as f:
            prev_meta = json.load(f)
        print(f"Reusing samples from {metadata_path}")
    else:
        np.random.seed(seed)

    all_data = {}
    new_meta = {"seed": seed, "crop_size": crop_size, "samples": {}}

    for struct in STRUCTURES:
        print(f"  {struct}...", end=" ")
        valid = find_valid_samples(struct)
        if not valid:
            print("SKIP (no valid samples)")
            continue

        if struct in resample_structs:
            np.random.seed(resample_seed + hash(struct) % 10000)
            sample = valid[np.random.randint(len(valid))]
            print(f"(resampled: {sample['stem']})", end=" ")
        elif prev_meta and struct in prev_meta["samples"]:
            prev = prev_meta["samples"][struct]
            sample = next((v for v in valid if v["stem"] == prev["stem"]), None)
            if sample is None:
                sample = valid[np.random.randint(len(valid))]
                print(f"(prev sample gone, new: {sample['stem']})", end=" ")
        else:
            sample = valid[np.random.randint(len(valid))]

        try:
            data = load_sample(sample, crop_size)
            all_data[struct] = data
            new_meta["samples"][struct] = {
                "stem": sample["stem"],
                "raw_path": sample["raw"],
                "seg_path": sample["seg"],
                "megaseg_pred_path": sample["megaseg_pred"],
                "individual_pred_path": sample["individual_pred"],
                "z_slice": data["z_slice"],
                "y_start": data["y_start"],
                "x_start": data["x_start"],
                "crop_size": data["crop_size"],
            }
            print(f"{sample['stem']} z={data['z_slice']}")
        except Exception as e:
            print(f"ERROR: {e}")

    return all_data, new_meta


# ──────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────
def create_figure(all_data, save_path, dpi=400, struct_list=None):
    """
    8-column layout (no inline colorbars):
      Raw | Pseudo-Labels | MegaSeg Prob | MegaSeg Mask | MegaSeg Error |
      Individual Prob | Individual Mask | Individual Error
    """
    structs = struct_list or list(all_data.keys())
    structs = [s for s in structs if s in all_data]
    n_rows = len(structs)
    n_cols = 8

    col_titles = [
        "Raw", "Pseudo-\nLabels",
        "MegaSeg\nProb.", "MegaSeg\nMask", "MegaSeg\nError",
        "Specialized\nProb.", "Specialized\nMask", "Specialized\nError",
    ]

    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 2.8 + 1.0), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.01)

    for row, struct in enumerate(structs):
        d = all_data[struct]
        crop_px = d["crop_size"]

        # (image, cmap, is_rgb)
        panels = [
            (d["raw"],       "gray",     False),
            (d["gt"],        "gray",     False),
            (d["mega_prob"], PROB_CMAP,  False),
            (d["mega_mask"], "gray",     False),
            (d["mega_err"],  None,       True),
            (d["ind_prob"],  PROB_CMAP,  False),
            (d["ind_mask"],  "gray",     False),
            (d["ind_err"],   None,       True),
        ]

        for col, (img, cmap, is_rgb) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])

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

            # Scale bar on raw image only
            if col == 0:
                add_scalebar(ax, crop_px)

            # Row label
            if col == 0:
                ax.set_ylabel(struct, fontsize=16, fontweight="bold",
                              rotation=90, labelpad=10, va="center")

            # Column titles on first row
            if row == 0:
                ax.set_title(col_titles[col], fontsize=14, fontweight="bold", pad=8)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    print(f"\nFigure saved: {save_path}")
    plt.close(fig)


def save_colorbar(out_path):
    """Save a separate horizontal colorbar for probability maps."""
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
    print(f"Colorbar saved: {out_path}")


def save_error_legend(out_path):
    """Save a separate horizontal legend for error map colors."""
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
    print(f"Error legend saved: {out_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MegaSeg Fig 2B v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to metadata JSON to reuse same samples")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Output filename prefix (default: auto)")
    parser.add_argument("--resample-seed", type=int, default=None,
                        help="Seed for resampling")
    args = parser.parse_args()

    prefix = args.prefix or f"fig2b_v4_seed{args.seed}"
    meta_out = FIGURE_DIR / f"{prefix}_metadata.json"

    # Default: try to reuse v1 metadata for consistency
    meta_in = args.metadata
    if meta_in is None:
        default_meta = FIGURE_DIR / f"fig2b_seed{args.seed}_crop{args.crop}_metadata.json"
        if default_meta.exists():
            meta_in = str(default_meta)

    print(f"Loading data (seed={args.seed}, crop={args.crop})...")
    all_data, metadata = load_all_data(args.seed, args.crop, meta_in,
                                       resample_structs=RESAMPLE_STRUCTS,
                                       resample_seed=args.resample_seed)

    if not all_data:
        print("No data loaded.")
        return

    # Part 1: first 7 structures
    path1 = FIGURE_DIR / f"{prefix}_part1.pdf"
    print(f"\nGenerating Part 1 ({len(LEFT_STRUCTS)} structures)...")
    create_figure(all_data, str(path1), args.dpi, struct_list=LEFT_STRUCTS)

    # Part 2: remaining 6 structures
    path2 = FIGURE_DIR / f"{prefix}_part2.pdf"
    print(f"Generating Part 2 ({len(RIGHT_STRUCTS)} structures)...")
    create_figure(all_data, str(path2), args.dpi, struct_list=RIGHT_STRUCTS)

    # Separate colorbar and error legend
    cbar_path = FIGURE_DIR / f"{prefix}_colorbar.pdf"
    save_colorbar(cbar_path)

    legend_path = FIGURE_DIR / f"{prefix}_error_legend.pdf"
    save_error_legend(legend_path)

    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_out}")


if __name__ == "__main__":
    main()
