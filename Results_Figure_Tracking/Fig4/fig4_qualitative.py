"""
Fig 4 — drug-perturbation qualitative comparison (2-column), config-driven.

One script + one JSON (fig4_qualitative_config.json). Reads local TIFF sources
(raw + classic / specialist / megaseg predictions), one folder per structure,
and renders the 2-column qualitative figure:

    per row, 6 columns:
      Raw | Classic Mask | Specialist Prob | Specialist Mask | MegaSeg Prob | MegaSeg Mask

Samples are auto-discovered from the raw source (one folder per structure):
control rows are detected by `control_token` in the filename; the drug label
comes from the structure's `drug` plus the timepoint parsed from the filename.
Files/dirs in `ignore_names` (e.g. "later/") are skipped.

Usage:
    python fig4_qualitative.py
    python fig4_qualitative.py --config fig4_qualitative_config.json
"""

import argparse
import json
import os
import re
import zlib
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
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5,
})

DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────
# Sample discovery
# ──────────────────────────────────────────────────────────────
def parse_timepoint(name):
    m = re.search(r"(\d+)\s*h(?:r|rs|ours)?(?:[_\-.]|$)", name, re.I)
    return f"{m.group(1)}hr" if m else ""


def discover_rows(cfg):
    """Build the ordered row list from the raw source (control first)."""
    raw_dir = cfg["sources"]["raw"]["dir"]
    ctrl_tok = cfg.get("control_token", "control").lower()
    ignore = set(cfg.get("ignore_names", []))
    rows = []
    for sc in cfg["structures"]:
        s, drug, bundle = sc["name"], sc["drug"], sc["bundle"]
        files = [f for f in sorted(os.listdir(f"{cfg['_data_root']}/{raw_dir}/{s}"))
                 if f.endswith(".tiff") and f not in ignore]
        srows = []
        for f in files:
            stem = f[:-5]                       # strip .tiff
            is_ctrl = ctrl_tok in stem.lower()
            base = stem.replace("_control", "")  # model outputs drop the suffix
            tp = parse_timepoint(stem)
            label = "Control" if is_ctrl else (f"{drug} {tp}".strip())
            crop_cfg = sc.get("crop", {})
            crop_strategy = crop_cfg.get("control" if is_ctrl else "drug", "centroid")
            crop_center = None
            z_override = None
            contrast = None
            ov = sc.get("crop_overrides", {}).get(base) or sc.get("crop_overrides", {}).get(stem)
            if ov:
                if "center_xy" in ov:
                    crop_strategy, crop_center = "manual", ov["center_xy"]
                if "z" in ov:
                    z_override = ov["z"]
                contrast = ov.get("contrast")
            srows.append(dict(struct=s, drug=drug, bundle=bundle, stem=stem,
                              base=base, is_ctrl=is_ctrl, tp=tp, label=label,
                              source_override=sc.get("source_override", {}),
                              crop_strategy=crop_strategy, crop_center=crop_center,
                              z_override=z_override, contrast=contrast))
        # optionally keep only control + one drug timepoint
        keep_tp = sc.get("keep_drug_tp")
        if keep_tp is not None:
            srows = [r for r in srows if r["is_ctrl"] or r["tp"] == keep_tp]
        # control first, then by timepoint (numeric), then name
        srows.sort(key=lambda r: (not r["is_ctrl"],
                                  int(re.sub(r"\D", "", r["tp"]) or 0), r["label"]))
        rows.extend(srows)
    return rows


# ──────────────────────────────────────────────────────────────
# Loading / slice selection
# ──────────────────────────────────────────────────────────────
def _path(cfg, src, row):
    s = cfg["sources"][src]
    ov = row.get("source_override", {}).get(src, {})
    d = ov.get("dir", s["dir"])
    folder = ov.get("struct_folder", row["struct"])
    template = ov.get("template", s["template"])
    return f"{cfg['_data_root']}/{d}/{folder}/{template.format(stem=row['stem'], base=row['base'])}"


def pick_center_signal_z(spec_m, meg_m, seed_str, center_frac=0.5, sig_rel=0.15):
    """Pick a z-slice near the center of the stack that carries signal.

    Signal is gauged from the ML masks (specialist + megaseg); the classic
    mask is ignored because it can dump large noise blobs on background slices.
    Among the central `center_frac` band, slices with foreground above
    `sig_rel` * max are candidates; one is chosen with a per-sample seeded RNG
    (stable across re-runs). Falls back to the whole central band if no slice
    shows signal.
    """
    n_z = spec_m.shape[0]
    signal = spec_m.sum((1, 2)) + meg_m.sum((1, 2))
    lo = int(round(n_z * (0.5 - center_frac / 2)))
    hi = int(round(n_z * (0.5 + center_frac / 2)))
    band = np.arange(max(lo, 0), max(min(hi, n_z), lo + 1))
    thr = sig_rel * signal.max() if signal.max() > 0 else -1
    cand = band[signal[band] > thr]
    if cand.size == 0:
        cand = band
    rng = np.random.default_rng(zlib.crc32(seed_str.encode()) & 0xFFFFFFFF)
    return int(rng.choice(cand))


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


def load_row_data(cfg, row):
    thr = cfg["threshold"]
    cs_cfg = cfg["crop_size_px"]
    raw = tif.imread(_path(cfg, "raw", row)).astype(np.float32)
    classic = tif.imread(_path(cfg, "classic", row))
    spec = tif.imread(_path(cfg, "specialist", row))
    meg = tif.imread(_path(cfg, "megaseg", row))

    classic_m = (classic >= thr).astype(np.uint8)
    spec_m = (spec >= thr).astype(np.uint8)
    meg_m = (meg >= thr).astype(np.uint8)
    if row.get("z_override") is not None:
        z = int(row["z_override"])
    else:
        z = pick_center_signal_z(spec_m, meg_m, row["base"],
                                 center_frac=cfg.get("z_center_fraction", 0.5),
                                 sig_rel=cfg.get("z_signal_rel_threshold", 0.15))

    raw_z = raw[z]
    c = row.get("contrast") or {}
    p_lo, p_hi, gamma = c.get("p_lo", 1), c.get("p_hi", 99), c.get("gamma", 1.0)
    lo, hi = np.percentile(raw_z, p_lo), np.percentile(raw_z, p_hi)
    raw_z = np.clip((raw_z - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(raw_z)
    if gamma != 1.0:
        raw_z = np.power(raw_z, gamma)

    H, W = raw_z.shape
    cs = min(cs_cfg, H, W)
    strat = row.get("crop_strategy", "centroid")
    if strat == "manual" and row.get("crop_center"):   # pinned [x, y] center
        cx, cy = int(row["crop_center"][0]), int(row["crop_center"][1])
    elif strat == "brightest":             # center on the densest/brightest region
        from scipy.ndimage import uniform_filter
        heat = uniform_filter(raw_z.astype(np.float32), size=max(8, cs // 2))
        cy, cx = map(int, np.unravel_index(int(np.argmax(heat)), heat.shape))
    else:                                   # centroid of the megaseg mask (default)
        ref = meg_m[z]
        if ref.sum() > 0:
            ys, xs = np.where(ref > 0)
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy, cx = H // 2, W // 2
    y0 = max(0, min(cy - cs // 2, H - cs))
    x0 = max(0, min(cx - cs // 2, W - cs))
    sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))
    return {
        "raw": raw_z[sl],
        "classic_mask": classic_m[z][sl].astype(np.float32),
        "spec_prob": spec[z][sl].astype(np.float32) / 255.0,
        "spec_mask": spec_m[z][sl].astype(np.float32),
        "meg_prob": meg[z][sl].astype(np.float32) / 255.0,
        "meg_mask": meg_m[z][sl].astype(np.float32),
        "crop": cs,
    }


# ──────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────
def add_scalebar(ax, crop_px, scalebar_um, pixel_um):
    bar_px = scalebar_um / pixel_um
    margin = 8
    y = crop_px - margin
    x_end = crop_px - margin
    x_start = x_end - bar_px
    ax.plot([x_start, x_end], [y, y], color="white", linewidth=2, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2, y - 5, f"{scalebar_um} µm", color="white",
            fontsize=6, ha="center", va="bottom", fontweight="bold")


def panels_for(cfg, data):
    cmap = cfg["prob_cmap"]
    return [
        (data["raw"], "gray", False),
        (data["classic_mask"], "gray", False),
        (data["spec_prob"], cmap, True),
        (data["spec_mask"], "gray", False),
        (data["meg_prob"], cmap, True),
        (data["meg_mask"], "gray", False),
    ]


def build_2col(cfg, rows, all_data, out_path):
    col_titles = cfg["columns"]
    n_pc = len(col_titles)
    idx = {id(r): d for r, d in zip(rows, all_data)}
    left = [(r, idx[id(r)]) for r in rows if r["bundle"] == "left"]
    right = [(r, idx[id(r)]) for r in rows if r["bundle"] == "right"]
    n_rows = max(len(left), len(right))
    print(f"  2-col: left={len(left)} rows, right={len(right)} rows")

    col_gap = 0.06
    width_ratios = [1.0] * n_pc + [col_gap] + [1.0] * n_pc
    fig = plt.figure(figsize=(sum(width_ratios) * 2.4, n_rows * 2.8 + 1.0), facecolor="white")
    gs = gridspec.GridSpec(n_rows, len(width_ratios), width_ratios=width_ratios,
                           wspace=0.03, hspace=0.08,
                           left=0.06, right=0.99, top=0.965, bottom=0.01)

    def draw(bundle, col_off, is_left):
        for ri in range(n_rows):
            if ri >= len(bundle):
                continue
            row, data = bundle[ri]
            for pc, (img, cmap, is_prob) in enumerate(panels_for(cfg, data)):
                ax = fig.add_subplot(gs[ri, col_off + pc])
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.3); sp.set_color("#555555")
                if is_left and pc == 0:
                    add_scalebar(ax, data["crop"], cfg["scalebar_um"], cfg["pixel_size_um"])
                if ri == 0:
                    ax.set_title(col_titles[pc], fontsize=10, fontweight="bold", pad=6)
                if pc == 0:
                    ax.set_ylabel(f"{row['struct']} — {row['label']}", fontsize=9,
                                  fontweight="bold", rotation=90, labelpad=10, va="center")
        # separators between structures within the bundle
        prev = None
        for i, (row, _) in enumerate(bundle):
            if prev is not None and row["struct"] != prev:
                p0 = gs[i, col_off].get_position(fig)
                pN = gs[i, col_off + n_pc - 1].get_position(fig)
                y = p0.y1 + 0.004
                fig.add_artist(plt.Line2D([p0.x0, pN.x1], [y, y], transform=fig.transFigure,
                                          color="#888888", linewidth=1.2))
            prev = row["struct"]

    draw(left, 0, True)
    draw(right, n_pc + 1, False)
    fig.savefig(str(out_path), dpi=cfg["render_dpi"], bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    fig.savefig(str(Path(out_path).with_suffix(".png")), dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def build_single(cfg, rows, all_data, out_path):
    """All rows stacked in one column (6 panels wide)."""
    col_titles = cfg["columns"]
    n_pc = len(col_titles)
    n_rows = len(rows)
    print(f"  single column: {n_rows} rows")
    fig = plt.figure(figsize=(n_pc * 2.4, n_rows * 2.8 + 1.0), facecolor="white")
    gs = gridspec.GridSpec(n_rows, n_pc, wspace=0.03, hspace=0.08,
                           left=0.09, right=0.99, top=0.97, bottom=0.01)
    for ri, (row, data) in enumerate(zip(rows, all_data)):
        for pc, (img, cmap, is_prob) in enumerate(panels_for(cfg, data)):
            ax = fig.add_subplot(gs[ri, pc])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3); sp.set_color("#555555")
            if pc == 0:
                add_scalebar(ax, data["crop"], cfg["scalebar_um"], cfg["pixel_size_um"])
                ax.set_ylabel(f"{row['struct']} — {row['label']}", fontsize=9,
                              fontweight="bold", rotation=90, labelpad=10, va="center")
            if ri == 0:
                ax.set_title(col_titles[pc], fontsize=10, fontweight="bold", pad=6)
    prev = None
    for i, row in enumerate(rows):
        if prev is not None and row["struct"] != prev:
            p0 = gs[i, 0].get_position(fig)
            pN = gs[i, n_pc - 1].get_position(fig)
            y = p0.y1 + 0.004
            fig.add_artist(plt.Line2D([p0.x0, pN.x1], [y, y], transform=fig.transFigure,
                                      color="#888888", linewidth=1.2))
        prev = row["struct"]
    fig.savefig(str(out_path), dpi=cfg["render_dpi"], bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    fig.savefig(str(Path(out_path).with_suffix(".png")), dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate the Fig4 drug-perturbation figure")
    ap.add_argument("--config", default="fig4_qualitative_config.json")
    args = ap.parse_args()
    cfg = json.load(open(DIR / args.config, encoding="utf-8"))
    # Resolve the local data location (absolute, or relative to this script)
    dr = cfg.get("data_root") or "."
    cfg["_data_root"] = dr if os.path.isabs(dr) else str((DIR / dr).resolve())

    rows = discover_rows(cfg)
    print(f"Discovered {len(rows)} rows:")
    for r in rows:
        print(f"  [{r['bundle']:5s}] {r['struct']:8s} {r['label']}")

    all_data = []
    for i, r in enumerate(rows):
        print(f"  loading {i+1}/{len(rows)}: {r['struct']} — {r['label']}")
        all_data.append(load_row_data(cfg, r))

    if cfg.get("layout") == "single":
        build_single(cfg, rows, all_data, DIR / cfg["output"])
    else:
        build_2col(cfg, rows, all_data, DIR / cfg["output"])


if __name__ == "__main__":
    main()
