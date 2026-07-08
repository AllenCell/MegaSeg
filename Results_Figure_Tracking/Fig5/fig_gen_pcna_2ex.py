"""
MegaSeg Generalization — PCNA, compact 2-example panel.

Two examples in a single row, each shown as Raw + MegaSeg Prob + MegaSeg Mask
at a single z-slice. Fully driven by fig_gen_pcna_2ex_config.json for
reproducibility.

Run with the bioio env (needs BioImage for the CZI raw channel):
  C:\\Users\\suraj.mishra\\AppData\\Local\\anaconda3\\envs\\bioio\\python.exe fig_gen_pcna_2ex.py
"""

import json
from pathlib import Path

import numpy as np
import tifffile as tif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})

DIR = Path(__file__).parent
CFG = json.loads((DIR / "fig_gen_pcna_2ex_config.json").read_text())


# ── helpers ──────────────────────────────────────────────────────
def normalize_img(img, p_lo=1, p_hi=99):
    lo, hi = np.percentile(img, p_lo), np.percentile(img, p_hi)
    return np.clip((img - lo) / (hi - lo) if hi > lo else np.zeros_like(img), 0, 1)


def best_z(prob, thr):
    fg = (prob >= thr).sum(axis=(1, 2))
    return int(np.argmax(fg))


def smart_crop(shape_2d, crop_size, ref_mask=None, center_xy=None):
    h, w = shape_2d
    cs = min(crop_size, h, w)
    if center_xy is not None:
        cx, cy = int(center_xy[0]), int(center_xy[1])
    elif ref_mask is not None and ref_mask.sum() > 0:
        ys, xs = np.where(ref_mask > 0)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
    else:
        cy, cx = h // 2, w // 2
    y0 = max(0, min(cy - cs // 2, h - cs))
    x0 = max(0, min(cx - cs // 2, w - cs))
    return y0, x0, cs


def add_scalebar(ax, pixel_size_um, scalebar_um, fontsize=8):
    bar_px = scalebar_um / pixel_size_um
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = abs(x1 - x0), abs(y1 - y0)
    m = min(w, h) * 0.04
    right_x = max(x0, x1) - m
    bottom_y = max(y0, y1) - m
    ax.plot([right_x - bar_px, right_x], [bottom_y, bottom_y],
            color="white", linewidth=2.0, solid_capstyle="butt")
    ax.text(right_x - bar_px / 2, bottom_y - min(w, h) * 0.03,
            f"{scalebar_um} µm", color="white", fontsize=fontsize,
            ha="center", va="bottom", fontweight="bold")


def style_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color("#555")


# ── load one example ─────────────────────────────────────────────
def load_example(ex, bio):
    scene, t = ex["scene"], ex["T"]
    pred_path = (Path(CFG["pred_root"]) /
                 CFG["pred_template"].format(t=t, scene=scene))
    prob = tif.imread(str(pred_path)).astype(np.float32) / 255.0
    thr = CFG["threshold"] / 255.0

    z = ex["z"] if ex.get("z") is not None else best_z(prob, thr)
    prob_z = prob[z]
    mask_z = (prob_z >= thr).astype(np.float32)

    bio.set_scene(scene)
    raw_vol = bio.get_image_data("ZYX", T=t, C=CFG["raw_channel"])
    raw_z = normalize_img(raw_vol[z].astype(np.float32),
                          CFG.get("p_lo", 1), CFG.get("p_hi", 99))

    y0, x0, cs = smart_crop(prob_z.shape, CFG["crop_size_px"],
                            ref_mask=mask_z > 0,
                            center_xy=ex.get("center_xy"))
    sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))
    print(f"    {scene} T={t} z={z} crop={cs} @({x0},{y0})")
    return {"raw": raw_z[sl], "prob": prob_z[sl], "mask": mask_z[sl],
            "label": ex.get("label", scene)}


# ── build figure ─────────────────────────────────────────────────
def main():
    from bioio import BioImage
    print(f"Loading CZI: {CFG['czi']}")
    bio = BioImage(CFG["czi"])

    examples = [load_example(ex, bio) for ex in CFG["examples"]]

    n_ex = len(examples)
    cols = CFG["columns"]           # 3: Raw, Prob, Mask
    n_col = len(cols)
    keys = ["raw", "prob", "mask"]
    cmaps = ["gray", CFG["prob_cmap"], "gray"]

    # Single row of n_ex groups, each group = n_col image tiles, with a
    # small gap between the two example groups.
    cell = 1.9
    fig = plt.figure(figsize=(n_ex * n_col * cell + (n_ex - 1) * 0.25,
                              cell + 0.75), facecolor="white")
    outer = gridspec.GridSpec(1, n_ex, wspace=0.14,
                              left=0.01, right=0.99, top=0.80, bottom=0.02)

    for g, ex in enumerate(examples):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_col, subplot_spec=outer[g], wspace=0.04)
        for c in range(n_col):
            ax = fig.add_subplot(inner[c])
            img = ex[keys[c]]
            if keys[c] == "mask":
                ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            elif keys[c] == "prob":
                ax.imshow(img, cmap=cmaps[c], vmin=0, vmax=1,
                          interpolation="bilinear", aspect="equal")
            else:  # raw
                ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                          interpolation="bilinear", aspect="equal")
                add_scalebar(ax, CFG["pixel_size_um"], CFG["scalebar_um"])
            style_ax(ax)
            ax.set_title(cols[c], fontsize=7.5, fontweight="bold", pad=3)
            if c == 0:
                ax.set_ylabel(ex["label"], fontsize=8, fontweight="bold",
                              rotation=90, labelpad=5, va="center")

    out = DIR / CFG["output"]
    fig.savefig(str(out), facecolor="white", pad_inches=0.04)
    fig.savefig(str(out.with_suffix(".png")), facecolor="white",
                pad_inches=0.04, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
