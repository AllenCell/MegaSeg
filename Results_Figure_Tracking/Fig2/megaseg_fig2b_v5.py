"""
MegaSeg Figure 2B v5: 7 main structures + 6 supplementary structures.

Split from v4's 7+6 (alphabetical) layout:
  Main (7): MYH10, RAB5A, SEC61B, ST6GAL1, TJP1, TOMM20, NUP153
  Supp (6): ACTN1, ATP2A2, CTNNB1, DSP, GJA1, SON

Other changes from v4:
  - Larger per-panel size (3.5 in instead of 2.8 in)
  - Higher default DPI (600 instead of 400) — matches the prior "bigsb" output
  - Output filenames use _main / _supp instead of _part1 / _part2
  - Reuses fig2b_v4_seed42_resampled_metadata.json by default so the FOV
    samples, z-slice picks, and crop coordinates are identical to the
    current bigsb PDFs

Layout per row (one per structure) is unchanged:
  Raw | Pseudo-Labels | Specialized Prob | Specialized Mask | Specialized Error |
  MegaSeg Prob | MegaSeg Mask | MegaSeg Error

Usage:
    python megaseg_fig2b_v5.py
    python megaseg_fig2b_v5.py --dpi 800 --panel 4.0
    python megaseg_fig2b_v5.py --metadata path/to/other_metadata.json
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
MEGASEG_PRED_ROOT = "//allen/aics/lumenoid/megaseg-benchmark/outputs/eval/uint8/megaseg/dynunet"
INDIVIDUAL_PRED_ROOT = "//allen/aics/lumenoid/megaseg-benchmark/outputs/eval/uint8/individual"
DATA_ROOT = "//allen/aics/lumenoid/megaseg-benchmark/data/megaseg_13structs"

# Order chosen by user for the main figure
MAIN_STRUCTS = [
    "MYH10", "RAB5A", "SEC61B", "ST6GAL1", "TJP1", "TOMM20", "NUP153",
]
SUPP_STRUCTS = ["ACTN1", "ATP2A2", "CTNNB1", "DSP", "GJA1", "SON"]
STRUCTURES = MAIN_STRUCTS + SUPP_STRUCTS

# Gene -> cellular structure name (used for row labels)
STRUCTURE_LABELS = {
    "MYH10":   "actomyosin bundles",
    "RAB5A":   "endosomes",
    "SEC61B":  "ER (Sec61 beta)",
    "ST6GAL1": "Golgi",
    "TJP1":    "tight junctions",
    "TOMM20":  "mitochondria",
    "NUP153":  "nuclear pores",
    "ACTN1":   "actin bundles",
    "ATP2A2":  "ER (SERCA2)",
    "CTNNB1":  "adherens junctions",
    "DSP":     "desmosomes",
    "GJA1":    "gap junctions",
    "SON":     "nuclear speckles",
}

FIGURE_DIR = Path(__file__).parent
DEFAULT_METADATA = FIGURE_DIR / "fig2b_v4_seed42_resampled_metadata.json"

PIXEL_SIZE_UM = 0.10833  # μm per pixel
SCALEBAR_UM = 10  # 10 μm scale bar
PROB_CMAP = "inferno"


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
    gt = gt_mask > 0.5
    pred = pred_mask > 0.5
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred
    tn = ~gt & ~pred
    # Wong (2011) colorblind-safe palette on white background
    rgb[tp] = [0.000, 0.620, 0.451]   # #009E73  bluish green
    rgb[fp] = [0.835, 0.369, 0.000]   # #D55E00  vermillion
    rgb[fn] = [0.000, 0.447, 0.698]   # #0072B2  blue
    rgb[tn] = [1.000, 1.000, 1.000]   # white
    return rgb


def add_scalebar(ax, crop_pixels):
    """White 10 µm scale bar in lower-right corner."""
    bar_px = SCALEBAR_UM / PIXEL_SIZE_UM
    margin = 8
    bar_y = crop_pixels - margin
    bar_x_end = crop_pixels - margin
    bar_x_start = bar_x_end - bar_px
    ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
            color="white", linewidth=2, solid_capstyle="butt")
    ax.text((bar_x_start + bar_x_end) / 2, bar_y - 5,
            f"{SCALEBAR_UM} µm", color="white", fontsize=6,
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
    sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))

    raw_crop = raw_z[sl].astype(np.float32)
    rmin, rmax = raw_crop.min(), raw_crop.max()
    raw_crop = (raw_crop - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(raw_crop)

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
        "mega_mask": mega_mask, "mega_prob": mega_prob, "mega_err": mega_err,
        "ind_mask": ind_mask, "ind_prob": ind_prob, "ind_err": ind_err,
        "z_slice": z, "y_start": y0, "x_start": x0, "crop_size": cs,
    }


def load_all_data(seed, crop_size, metadata_path):
    """Always require an existing metadata JSON so samples stay reproducible."""
    with open(metadata_path) as f:
        prev_meta = json.load(f)
    print(f"Reusing samples from {metadata_path}")

    all_data = {}
    new_meta = {"seed": seed, "crop_size": crop_size, "samples": {}}

    for struct in STRUCTURES:
        print(f"  {struct}...", end=" ", flush=True)
        if struct not in prev_meta["samples"]:
            print("SKIP (not in metadata)")
            continue

        valid = find_valid_samples(struct)
        if not valid:
            print("SKIP (no valid samples)")
            continue

        prev = prev_meta["samples"][struct]
        sample = next((v for v in valid if v["stem"] == prev["stem"]), None)
        if sample is None:
            print(f"SKIP (prev sample {prev['stem']} no longer valid)")
            continue

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
def create_figure(all_data, save_path, dpi, struct_list, panel_in):
    structs = [s for s in struct_list if s in all_data]
    n_rows = len(structs)
    n_cols = 8

    col_titles = [
        "Raw", "Pseudo-\nLabels",
        "Specialized Model\nProb.", "Specialized Model\nMask", "Specialized Model\nError",
        "MegaSeg\nProb.", "MegaSeg\nMask", "MegaSeg\nError",
    ]

    fig = plt.figure(
        figsize=(n_cols * panel_in, n_rows * panel_in + 1.0),
        facecolor="white",
    )
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.01)

    for row, struct in enumerate(structs):
        d = all_data[struct]
        crop_px = d["crop_size"]
        panels = [
            (d["raw"],       "gray",     False),
            (d["gt"],        "gray",     False),
            (d["ind_prob"],  PROB_CMAP,  False),
            (d["ind_mask"],  "gray",     False),
            (d["ind_err"],   None,       True),
            (d["mega_prob"], PROB_CMAP,  False),
            (d["mega_mask"], "gray",     False),
            (d["mega_err"],  None,       True),
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
            if col == 0:
                add_scalebar(ax, crop_px)
                ax.set_ylabel(STRUCTURE_LABELS.get(struct, struct),
                              fontsize=20, fontweight="bold",
                              rotation=90, labelpad=12, va="center")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=18, fontweight="bold", pad=10)

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
    handles = [
        Patch(facecolor=(0.000, 0.620, 0.451), edgecolor="gray", label="True Positive"),
        Patch(facecolor=(0.835, 0.369, 0.000), edgecolor="gray", label="False Positive"),
        Patch(facecolor=(0.000, 0.447, 0.698), edgecolor="gray", label="False Negative"),
        Patch(facecolor=(1.000, 1.000, 1.000), edgecolor="gray", label="True Negative"),
    ]
    ax.legend(handles=handles, loc="center", ncol=4,
              fontsize=10, frameon=True, edgecolor="black")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  Error legend saved: {out_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MegaSeg Fig 2B v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--dpi", type=int, default=600,
                        help="Output DPI (default 600; v4 bigsb-equivalent)")
    parser.add_argument("--panel", type=float, default=3.5,
                        help="Per-panel size in inches (default 3.5; v4 was 2.8)")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA),
                        help="Metadata JSON to reuse same FOV samples")
    parser.add_argument("--prefix", type=str, default="fig2b_v5_seed42")
    args = parser.parse_args()

    print(f"Loading data (seed={args.seed}, crop={args.crop}, "
          f"panel={args.panel}in, dpi={args.dpi})...")
    all_data, metadata = load_all_data(args.seed, args.crop, args.metadata)
    if not all_data:
        print("No data loaded.")
        return

    main_path = FIGURE_DIR / f"{args.prefix}_main.pdf"
    print(f"\nGenerating MAIN ({len(MAIN_STRUCTS)} structures: {MAIN_STRUCTS})...")
    create_figure(all_data, str(main_path), args.dpi,
                  struct_list=MAIN_STRUCTS, panel_in=args.panel)

    supp_path = FIGURE_DIR / f"{args.prefix}_supp.pdf"
    print(f"\nGenerating SUPP ({len(SUPP_STRUCTS)} structures: {SUPP_STRUCTS})...")
    create_figure(all_data, str(supp_path), args.dpi,
                  struct_list=SUPP_STRUCTS, panel_in=args.panel)

    cbar_path = FIGURE_DIR / f"{args.prefix}_colorbar.pdf"
    save_colorbar(cbar_path)
    legend_path = FIGURE_DIR / f"{args.prefix}_error_legend.pdf"
    save_error_legend(legend_path)

    meta_out = FIGURE_DIR / f"{args.prefix}_metadata.json"
    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_out}")


if __name__ == "__main__":
    main()
