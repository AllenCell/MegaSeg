"""
SOTA qualitative comparison — 13 structures × 8 methods.

Rows: one per structure (median-Dice sample from MegaSeg DynUNet).
Columns: Raw | Pseudo-Labels | Classic | MegaSeg | nnU-Net | Individual | Otsu | CellPose | StarDist | SAM

Error maps: TP (gold), FP (rose), FN (steel blue), TN (dark gray).
One scale bar per row (bottom-right of Raw column).

Usage:
    python fig_sota_qualitative.py
"""

import gzip
from pathlib import Path

import numpy as np
import tifffile as tif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK = Path("/allen/aics/rep_learn/megaseg/benchmark")
LUMENOID = Path("/allen/aics/lumenoid/megaseg-benchmark")
OUTPUTS = BENCHMARK / "outputs" / "eval" / "uint8"
RAW_ROOT = Path("/allen/aics/assay-dev/users/Suraj/megaSeg/14structs_rand/test")

GENE_TO_STRUCTURE = {
    "ACTN1": "Actin bundles",
    "ATP2A2": "ER (SERCA)",
    "CTNNB1": "Adherens junctions",
    "DSP": "Desmosomes",
    "GJA1": "Gap junctions",
    "MYH10": "Actomyosin bundles",
    "NUP153": "Nuclear pores",
    "RAB5A": "Endosomes",
    "SEC61B": "ER (SEC61β)",
    "SON": "Nuclear speckles",
    "ST6GAL1": "Golgi",
    "TJP1": "Tight junctions",
    "TOMM20": "Mitochondria",
}

SAMPLES = {
    "ACTN1": "ACTN1_11506",
    "ATP2A2": "ATP2A2_146647",
    "CTNNB1": "CTNNB1_13280",
    "DSP": "DSP_17059",
    "GJA1": "GJA1_17005",
    "MYH10": "MYH10_6339",
    "NUP153": "NUP153_140757",
    "RAB5A": "RAB5A_148014",
    "SEC61B": "SEC61B_16982",
    "SON": "SON_137205",
    "ST6GAL1": "ST6GAL1_15403",
    "TJP1": "TJP1_6465",
    "TOMM20": "TOMM20_2975",
}

METHODS = ["classic", "megaseg", "nnunet", "individual", "otsu", "cellpose", "stardist", "sam"]
METHOD_LABELS = {
    "classic": "Classic\n(aicsseg)",
    "megaseg": "MegaSeg\n(DynUNet)",
    "nnunet": "nnU-Net v2",
    "individual": "Individual\nDynUNet",
    "otsu": "Otsu",
    "cellpose": "CellPose v4",
    "stardist": "StarDist",
    "sam": "SAM\n(vit_b)",
}

CROP_SIZE = 256
PIXEL_SIZE_UM = 0.108
SCALEBAR_UM = 10


def _pred_path(method, struct, sid):
    if method == "megaseg":
        return LUMENOID / "outputs/eval/uint8/megaseg/dynunet" / struct / "seg" / f"{sid}.tif"
    if method == "individual":
        return LUMENOID / f"outputs/eval/uint8/individual/ms13-{struct}/seg/{sid}.tif"
    if method == "nnunet":
        return OUTPUTS / "nnunet_3d_fullres" / f"{sid}.nii.gz"
    if method == "classic":
        return BENCHMARK / "results/uint8/classic_cross_eval" / struct / f"tested_on_{struct}" / f"{sid}_seg.tiff"
    if method == "cellpose":
        return OUTPUTS / "cellpose_zeroshot" / struct / f"{sid}_pred.tiff"
    if method == "stardist":
        return OUTPUTS / "stardist_zeroshot" / struct / f"{sid}_pred.tiff"
    if method == "otsu":
        return OUTPUTS / "otsu_baseline" / struct / f"{sid}_pred.tiff"
    if method == "sam":
        return OUTPUTS / "sam_zeroshot" / struct / f"{sid}_pred.tiff"
    raise ValueError(method)


def load_nifti(path):
    with gzip.open(str(path), "rb") as f:
        data = f.read()
    hdr_size = int.from_bytes(data[:4], "little")
    if hdr_size not in (348, 540):
        hdr_size = int.from_bytes(data[:4], "big")
    if hdr_size == 348:
        dims = np.frombuffer(data[40:56], dtype=np.int16)
        dt_code = int.from_bytes(data[70:72], "little")
        offset = int(np.frombuffer(data[108:112], dtype=np.float32)[0])
    else:
        dims = np.frombuffer(data[48:80], dtype=np.int64)
        dt_code = int.from_bytes(data[12:14], "little")
        offset = int(np.frombuffer(data[168:176], dtype=np.float64)[0])
    ndim = dims[0]
    shape = tuple(dims[1:ndim + 1])
    dt_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8, 512: np.uint16}
    dtype = dt_map.get(dt_code, np.float32)
    vol = np.frombuffer(data[offset:], dtype=dtype)[:np.prod(shape)].reshape(shape)
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol


def load_pred(method, struct, sid):
    path = _pred_path(method, struct, sid)
    if not path.exists():
        return None
    if method == "nnunet":
        vol = load_nifti(path)
    else:
        vol = tif.imread(str(path))
    if vol.dtype == np.uint8:
        return (vol >= 128).astype(np.bool_)
    return (vol > 0).astype(np.bool_)


def mip(vol):
    return np.max(vol, axis=0)


def normalize_percentile(img, p_lo=1, p_hi=95):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return np.clip(img, 0, 1)


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


def make_error_map(gt, pred):
    h, w = gt.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[gt & pred] = [0.85, 0.75, 0.25]
    rgb[~gt & pred] = [0.82, 0.35, 0.45]
    rgb[gt & ~pred] = [0.30, 0.55, 0.75]
    rgb[~gt & ~pred] = [0.12, 0.12, 0.12]
    return rgb


def add_scalebar(ax, crop_px):
    bar_px = SCALEBAR_UM / PIXEL_SIZE_UM
    margin = 8
    bar_y = crop_px - margin
    bar_x_end = crop_px - margin
    bar_x_start = bar_x_end - bar_px
    ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y],
            color="white", linewidth=2, solid_capstyle="butt")
    ax.text((bar_x_start + bar_x_end) / 2, bar_y - 5,
            f"{SCALEBAR_UM} µm", color="white", fontsize=8,
            ha="center", va="bottom", fontweight="bold")


def main():
    structures = list(GENE_TO_STRUCTURE.keys())
    n_rows = len(structures)
    n_cols = 2 + len(METHODS)  # Raw + GT + 8 methods
    col_labels = ["Raw", "Pseudo-\nLabels"] + [METHOD_LABELS[m] for m in METHODS]

    with plt.style.context("seaborn-v0_8-white"):
        fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2 + 1.5), facecolor="white")
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            wspace=0.02, hspace=0.05,
            top=0.96, bottom=0.04, left=0.08, right=0.99,
        )

        for row_idx, struct in enumerate(structures):
            sid = SAMPLES[struct]
            struct_name = GENE_TO_STRUCTURE[struct]
            print(f"  [{row_idx + 1}/{n_rows}] {struct_name} ({sid})...")

            raw_path = RAW_ROOT / f"{sid}.tiff"
            gt_path = RAW_ROOT / f"{sid}_seg.tiff"

            raw_vol = tif.imread(str(raw_path)).astype(np.float32)
            gt_vol = (tif.imread(str(gt_path)) > 0).astype(np.bool_)

            raw_mip = normalize_percentile(mip(raw_vol))
            gt_mip = mip(gt_vol.astype(np.uint8)) > 0

            y0, x0, cs = smart_crop(gt_mip.shape, CROP_SIZE, ref_mask=gt_mip)

            def crop(img):
                return img[y0:y0 + cs, x0:x0 + cs]

            raw_crop = crop(raw_mip)
            gt_crop = crop(gt_mip)

            for col_idx in range(n_cols):
                ax = fig.add_subplot(gs[row_idx, col_idx])

                if col_idx == 0:
                    ax.imshow(raw_crop, cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest", aspect="equal")
                    add_scalebar(ax, cs)
                elif col_idx == 1:
                    ax.imshow(gt_crop.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest", aspect="equal")
                else:
                    method = METHODS[col_idx - 2]
                    pred_vol = load_pred(method, struct, sid)
                    if pred_vol is not None:
                        pred_mip = mip(pred_vol.astype(np.uint8)) > 0
                        ax.imshow(make_error_map(gt_crop, crop(pred_mip)),
                                  interpolation="nearest", aspect="equal")
                    else:
                        ax.imshow(np.zeros((cs, cs, 3)), aspect="equal")
                        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                                ha="center", va="center", fontsize=10, color="gray")

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.5)
                    spine.set_color("#888888")

                if row_idx == 0:
                    ax.set_title(col_labels[col_idx], fontsize=14,
                                 fontweight="bold", pad=6)

                if col_idx == 0:
                    ax.set_ylabel(struct_name, fontsize=13, fontweight="bold",
                                  labelpad=8)

        # Error map legend
        legend_handles = [
            Patch(facecolor=(0.85, 0.75, 0.25), edgecolor="#666666",
                  linewidth=1.2, label="True Positive"),
            Patch(facecolor=(0.82, 0.35, 0.45), edgecolor="#666666",
                  linewidth=1.2, label="False Positive"),
            Patch(facecolor=(0.30, 0.55, 0.75), edgecolor="#666666",
                  linewidth=1.2, label="False Negative"),
            Patch(facecolor=(0.12, 0.12, 0.12), edgecolor="#666666",
                  linewidth=1.2, label="True Negative"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=16, frameon=True, edgecolor="#999999",
                   fancybox=True, shadow=False, framealpha=0.95,
                   bbox_to_anchor=(0.52, -0.005),
                   handlelength=2.5, handleheight=1.5, handletextpad=0.8,
                   columnspacing=4.0, borderpad=0.6)

    fig.savefig(str(SCRIPT_DIR / "fig_sota_qualitative.pdf"),
                dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"  Saved: {SCRIPT_DIR / 'fig_sota_qualitative.pdf'}")


if __name__ == "__main__":
    main()
