#!/usr/bin/env python
"""Visualize timelapse segmentation results.

Generates comparison figures for classic, individual-model, and MegaSeg
segmentations on Caroline's timelapse microscopy data.

Figure types
  comparison  Side-by-side panels per FOV and timepoint (overlay + grayscale)
  overlay     All methods' contours on a single raw MIP per FOV and timepoint
  temporal    Rows = methods, columns = timepoints per FOV group (overlay + gray)

Each comparison / temporal figure is produced in two display modes:
  overlay  Segmentation shown as colored alpha-blend on raw
  gray     Segmentation shown as separate grayscale panels

Usage
  python visualize_timelapse.py                    # curated subset, all figures
  python visualize_timelapse.py --datasets NPM1    # NPM1 only
  python visualize_timelapse.py --all-fovs          # every FOV, not curated
  python visualize_timelapse.py --figures temporal   # temporal montages only
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_REPO = Path("/allen/aics/rep_learn/MegaSeg")

RAW_DIRS: dict[str, Path] = {
    "NPM1": ORIGINAL_REPO / "data" / "caroline_data" / "raw",
    "Mitochondria": (BENCHMARK_DIR / "data" / "caroline_data" / "raw" / "Mitochondria"),
    "Tubulin_MultiCh": (
        BENCHMARK_DIR / "data" / "caroline_data" / "raw" / "Tubulin_MultiCh"
    ),
}

CLASSIC_DIRS: dict[str, Path] = {
    "NPM1": (
        BENCHMARK_DIR / "outputs" / "eval" / "uint8" / "classic" / "timelapse" / "NPM1"
    ),
    "Mitochondria": (
        BENCHMARK_DIR
        / "outputs"
        / "eval"
        / "uint8"
        / "classic"
        / "timelapse"
        / "Mitochondria"
    ),
    "Tubulin_MultiCh": (
        BENCHMARK_DIR
        / "outputs"
        / "eval"
        / "uint8"
        / "classic"
        / "timelapse"
        / "Tubulin_MultiCh"
    ),
}

INDIVIDUAL_DIRS: dict[str, Path] = {
    "NPM1": (
        BENCHMARK_DIR
        / "outputs"
        / "eval"
        / "uint8"
        / "individual"
        / "timelapse"
        / "NPM1"
        / "seg"
    ),
    "Mitochondria": (
        BENCHMARK_DIR
        / "outputs"
        / "eval"
        / "uint8"
        / "individual"
        / "timelapse"
        / "TOMM20"
        / "seg"
    ),
}

MEGASEG_DIRS: dict[str, Path] = {
    "NPM1": (
        BENCHMARK_DIR
        / "outputs"
        / "eval"
        / "uint8"
        / "megaseg"
        / "dynunet"
        / "timelapse"
        / "NPM1"
        / "seg"
    ),
    "Mitochondria": (
        ORIGINAL_REPO
        / "benchmark"
        / "outputs"
        / "eval"
        / "uint8"
        / "megaseg"
        / "dynunet"
        / "Mitochondria"
        / "seg"
    ),
    "Tubulin_MultiCh": (
        ORIGINAL_REPO
        / "benchmark"
        / "outputs"
        / "eval"
        / "uint8"
        / "megaseg"
        / "dynunet"
        / "Tubulin_MultiCh"
        / "seg"
    ),
}

VISUALS_DIR = BENCHMARK_DIR / "outputs" / "visuals"

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------

METHOD_COLORS: dict[str, tuple[float, float, float]] = {
    "Classic": (1.0, 0.3, 0.0),  # orange
    "MegaSeg": (0.0, 1.0, 0.0),  # green
    "Individual": (0.0, 0.6, 1.0),  # blue
}

OVERLAY_ALPHA = 0.4
PRED_THRESHOLD = 128
SAVE_DPI = 200

DATASET_METHODS: dict[str, list[str]] = {
    "NPM1": ["Classic", "Individual", "MegaSeg"],
    "Mitochondria": ["Classic", "Individual", "MegaSeg"],
    "Tubulin_MultiCh": ["Classic", "MegaSeg"],
}

# ---------------------------------------------------------------------------
# Curated FOV groups (one representative per microscope type / CZI)
# ---------------------------------------------------------------------------

CURATED_FOVS: dict[str, list[str]] = {
    "NPM1": [
        "20171107_C03_001_Airyscan Processing_scene01",  # Airyscan 40x
        "20180315_N01_001(1)_scene00",  # ZSD 20x
        "20180227_N01_001(1)_scene00",  # ZSD 100x
    ],
    "Mitochondria": [
        "20170424_C01_001_Airyscan Processing_scene01",
    ],
    "Tubulin_MultiCh": [
        "20170718_C01_001_Airyscan Processing_scene01",  # CZI 1
        "20171031_C01_001_Airyscan_Processing_scene01",  # CZI 2
    ],
}

FOV_LABELS: dict[str, str] = {
    "20171107_C03_001_Airyscan Processing_scene01": "NPM1 - Airyscan 40x",
    "20180315_N01_001(1)_scene00": "NPM1 - ZSD 20x",
    "20180227_N01_001(1)_scene00": "NPM1 - ZSD 100x",
    "20170424_C01_001_Airyscan Processing_scene01": "Mitochondria - Airyscan",
    "20170718_C01_001_Airyscan Processing_scene01": "Tubulin - CZI 1 Scene 1",
    "20171031_C01_001_Airyscan_Processing_scene01": "Tubulin - CZI 2 Scene 1",
}

# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------

# Raw / classic stems:  {fov_group}_T{NNNN}_C{NN}
_RAW_STEM_RE = re.compile(r"^(.+)_T(\d+)_C(\d+)$")

# Seg suffix:  _dimension_order_out_CZYX_C_{c}_T_{t}_scene_{s}
_SEG_SUFFIX_RE = re.compile(r"_dimension_order_out_CZYX_C_(\d+)_T_(\d+)_scene_(.+)$")

# FOV group contains scene:  {prefix}_scene{NN}
_SCENE_RE = re.compile(r"^(.+)_scene(\d+)$")


def parse_raw_stem(stem: str) -> dict | None:
    """Extract (fov_group, timepoint) from a raw/classic filename stem."""
    m = _RAW_STEM_RE.match(stem)
    if m is None:
        return None
    return {"fov_group": m.group(1), "timepoint": int(m.group(2))}


def strip_seg_suffix(stem: str) -> dict | None:
    """Strip the _dimension_order_out_CZYX_* suffix from a seg stem.

    Returns {base, channel, timepoint, scene} or None.
    """
    m = _SEG_SUFFIX_RE.search(stem)
    if m is None:
        return None
    return {
        "base": stem[: m.start()],
        "channel": int(m.group(1)),
        "timepoint": int(m.group(2)),
        "scene": m.group(3),
    }


def split_fov_group(fov_group: str) -> tuple[str, str] | None:
    """Split a FOV group string into (prefix, scene_number_str)."""
    m = _SCENE_RE.match(fov_group)
    if m is None:
        return None
    return m.group(1), m.group(2)


def sanitize(name: str) -> str:
    """Make a filename-safe version of a FOV group name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")


# ---------------------------------------------------------------------------
# File discovery and matching
# ---------------------------------------------------------------------------


def _build_direct_seg_lookup(seg_dir: Path | None) -> dict[str, Path]:
    """For NPM1: strip suffix from seg filename to recover raw stem."""
    lookup: dict[str, Path] = {}
    if seg_dir is None or not seg_dir.exists():
        return lookup
    for f in seg_dir.glob("*.tif"):
        parsed = strip_seg_suffix(f.stem)
        if parsed is not None:
            lookup[parsed["base"]] = f
    return lookup


def _build_czi_seg_lookup(
    seg_dir: Path | None,
    raw_scenes_by_prefix: dict[str, dict[str, set[int]]],
) -> dict[tuple[str, int], Path]:
    """For CZI-based datasets: match seg files to (fov_group, timepoint).

    Uses positional scene matching -- sort unique raw scenes and seg scenes
    within each CZI prefix and pair them by rank order.
    """
    lookup: dict[tuple[str, int], Path] = {}
    if seg_dir is None or not seg_dir.exists():
        return lookup

    # Group seg files by (prefix, seg_scene)
    seg_by_prefix: dict[str, dict[str, list[tuple[int, Path]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in sorted(seg_dir.glob("*.tif")):
        parsed = strip_seg_suffix(f.stem)
        if parsed is None:
            continue
        seg_by_prefix[parsed["base"]][parsed["scene"]].append((parsed["timepoint"], f))

    # Build scene map per prefix via positional matching
    for prefix, seg_scenes in seg_by_prefix.items():
        if prefix not in raw_scenes_by_prefix:
            continue
        raw_scenes = raw_scenes_by_prefix[prefix]
        sorted_raw = sorted(raw_scenes.keys(), key=int)
        sorted_seg = sorted(
            seg_scenes.keys(),
            key=lambda s: int(s) if s.isdigit() else 0,
        )

        scene_map: dict[str, str] = {}
        for i, seg_s in enumerate(sorted_seg):
            if i < len(sorted_raw):
                scene_map[seg_s] = sorted_raw[i]

        for seg_s, raw_s in scene_map.items():
            fov_group = f"{prefix}_scene{raw_s}"
            for tp, path in seg_scenes[seg_s]:
                lookup[(fov_group, tp)] = path

    return lookup


def discover_files(dataset: str) -> dict[str, list[dict]]:
    """Discover and match raw, classic, individual, and megaseg files.

    Returns {fov_group: [{timepoint, raw, classic, individual, megaseg}, ...]}
    with entries sorted by timepoint within each group.
    """
    raw_dir = RAW_DIRS[dataset]
    classic_dir = CLASSIC_DIRS.get(dataset)
    individual_dir = INDIVIDUAL_DIRS.get(dataset)
    megaseg_dir = MEGASEG_DIRS.get(dataset)

    # Parse all raw files
    raw_entries = []
    for f in sorted(raw_dir.glob("*.tif")):
        parsed = parse_raw_stem(f.stem)
        if parsed is not None:
            raw_entries.append({**parsed, "path": f, "stem": f.stem})

    # Classic shares filenames with raw
    classic_lookup: dict[str, Path] = {}
    if classic_dir and classic_dir.exists():
        classic_lookup = {f.stem: f for f in classic_dir.glob("*.tif")}

    # --- NPM1: direct stem matching ---
    if dataset == "NPM1":
        individual_lookup = _build_direct_seg_lookup(individual_dir)
        megaseg_lookup = _build_direct_seg_lookup(megaseg_dir)

        results: dict[str, list[dict]] = defaultdict(list)
        for e in raw_entries:
            results[e["fov_group"]].append(
                {
                    "timepoint": e["timepoint"],
                    "raw": e["path"],
                    "classic": classic_lookup.get(e["stem"]),
                    "individual": individual_lookup.get(e["stem"]),
                    "megaseg": megaseg_lookup.get(e["stem"]),
                }
            )

    # --- CZI-based: positional scene + timepoint matching ---
    else:
        raw_scenes_by_prefix: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for e in raw_entries:
            parts = split_fov_group(e["fov_group"])
            if parts is not None:
                prefix, scene = parts
                raw_scenes_by_prefix[prefix][scene].add(e["timepoint"])

        individual_lookup_czi = _build_czi_seg_lookup(
            individual_dir, raw_scenes_by_prefix
        )
        megaseg_lookup_czi = _build_czi_seg_lookup(megaseg_dir, raw_scenes_by_prefix)

        results = defaultdict(list)
        for e in raw_entries:
            fov = e["fov_group"]
            tp = e["timepoint"]
            results[fov].append(
                {
                    "timepoint": tp,
                    "raw": e["path"],
                    "classic": classic_lookup.get(e["stem"]),
                    "individual": individual_lookup_czi.get((fov, tp)),
                    "megaseg": megaseg_lookup_czi.get((fov, tp)),
                }
            )

    for fov_group in results:
        results[fov_group].sort(key=lambda x: x["timepoint"])

    return dict(results)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def load_mip(path: Path) -> np.ndarray:
    """Load a 3-D TIFF and return the max-intensity projection (float32)."""
    vol = tifffile.imread(str(path))
    if vol.ndim == 2:
        return vol.astype(np.float32)
    return vol.max(axis=0).astype(np.float32)


def normalize(img: np.ndarray) -> np.ndarray:
    """Percentile-based contrast stretch to [0, 1]."""
    nonzero = img[img > 0]
    if nonzero.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo = np.percentile(nonzero, 1)
    hi = np.percentile(nonzero, 99)
    if hi <= lo:
        hi = lo + 1
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def crop_to_shape(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop img to target (H, W), preserving extra channels."""
    h, w = img.shape[:2]
    th, tw = target_shape
    if h == th and w == tw:
        return img
    y0 = max(0, (h - th) // 2)
    x0 = max(0, (w - tw) // 2)
    return img[y0 : y0 + th, x0 : x0 + tw]


def make_overlay(
    raw_norm: np.ndarray,
    seg_mip: np.ndarray,
    color: tuple[float, float, float],
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Alpha-blend a method's color onto raw grayscale where seg > threshold."""
    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1).copy()
    mask = seg_mip >= PRED_THRESHOLD
    for c in range(3):
        rgb[:, :, c][mask] = rgb[:, :, c][mask] * (1 - alpha) + color[c] * alpha
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure as both png and pdf, then close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(str(out), dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path.with_suffix('.png').name}")


def _na_axes(ax: plt.Axes, text: str = "N/A") -> None:
    """Fill an axes with a centered placeholder label."""
    ax.set_facecolor("#f0f0f0")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=14,
        color="#999999",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Comparison grid (one per FOV + timepoint, in overlay or gray mode)
# ---------------------------------------------------------------------------


def make_comparison_grid(
    fov_group: str,
    tp: int,
    loaded: dict[tuple, np.ndarray],
    methods: list[str],
    mode: str,
    output_dir: Path,
    dataset: str,
) -> None:
    """Generate a single-row comparison grid.

    mode: 'overlay' - colored overlay on raw; 'gray' - grayscale panels.
    """
    raw_mip = loaded.get((tp, "raw"))
    if raw_mip is None:
        return

    raw_norm = normalize(raw_mip)
    raw_shape = raw_mip.shape[:2]

    cols = ["Raw"] + methods
    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)

    for c, label in enumerate(cols):
        ax = axes[0, c]
        if label == "Raw":
            ax.imshow(raw_norm, cmap="gray", vmin=0, vmax=1)
        else:
            seg_mip = loaded.get((tp, label))
            if seg_mip is None:
                _na_axes(ax, "No model" if label == "Individual" else "N/A")
            elif mode == "overlay":
                seg_cropped = crop_to_shape(seg_mip, raw_shape)
                overlay = make_overlay(raw_norm, seg_cropped, METHOD_COLORS[label])
                ax.imshow(overlay)
            else:
                seg_cropped = crop_to_shape(seg_mip, raw_shape)
                ax.imshow(seg_cropped, cmap="gray")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    title = FOV_LABELS.get(fov_group, fov_group)
    fig.suptitle(f"{title} - T{tp:04d}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    name = f"{sanitize(fov_group)}_T{tp:04d}_{mode}"
    _save(fig, output_dir / "comparison" / dataset / name)


# ---------------------------------------------------------------------------
# Temporal montage (one per FOV group, in overlay or gray mode)
# ---------------------------------------------------------------------------


def make_temporal_montage(
    fov_group: str,
    entries: list[dict],
    loaded: dict[tuple, np.ndarray],
    methods: list[str],
    mode: str,
    output_dir: Path,
    dataset: str,
) -> None:
    """Rows = [Raw] + methods, columns = timepoints."""
    rows = ["Raw"] + methods
    n_rows = len(rows)
    n_cols = len(entries)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for col_idx, entry in enumerate(entries):
        tp = entry["timepoint"]
        raw_mip = loaded.get((tp, "raw"))
        raw_norm = normalize(raw_mip) if raw_mip is not None else None
        raw_shape = raw_mip.shape[:2] if raw_mip is not None else None

        for row_idx, label in enumerate(rows):
            ax = axes[row_idx, col_idx]
            if label == "Raw":
                if raw_norm is not None:
                    ax.imshow(raw_norm, cmap="gray", vmin=0, vmax=1)
                else:
                    _na_axes(ax)
            else:
                seg_mip = loaded.get((tp, label))
                if seg_mip is None or raw_norm is None:
                    _na_axes(ax, "No model" if label == "Individual" else "N/A")
                elif mode == "overlay":
                    seg_cropped = crop_to_shape(seg_mip, raw_shape)
                    overlay = make_overlay(raw_norm, seg_cropped, METHOD_COLORS[label])
                    ax.imshow(overlay)
                else:
                    seg_cropped = crop_to_shape(seg_mip, raw_shape)
                    ax.imshow(seg_cropped, cmap="gray")

            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"T{tp:04d}", fontsize=11, fontweight="bold")

    title = FOV_LABELS.get(fov_group, fov_group)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    name = f"{sanitize(fov_group)}_{mode}"
    _save(fig, output_dir / "temporal" / dataset / name)


# ---------------------------------------------------------------------------
# Contour overlay (one per FOV + timepoint)
# ---------------------------------------------------------------------------


def make_contour_figure(
    fov_group: str,
    tp: int,
    loaded: dict[tuple, np.ndarray],
    methods: list[str],
    output_dir: Path,
    dataset: str,
) -> None:
    """Overlay contour boundaries from every method on a single raw MIP."""
    raw_mip = loaded.get((tp, "raw"))
    if raw_mip is None:
        return

    raw_norm = normalize(raw_mip)
    raw_shape = raw_mip.shape[:2]
    rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1).copy()

    legend_entries: list[tuple[str, tuple]] = []
    for method in methods:
        seg_mip = loaded.get((tp, method))
        if seg_mip is None:
            continue
        seg_cropped = crop_to_shape(seg_mip, raw_shape)
        mask = seg_cropped >= PRED_THRESHOLD
        if not mask.any():
            continue
        boundaries = find_boundaries(mask, mode="outer")
        color = METHOD_COLORS[method]
        for c in range(3):
            rgb[:, :, c][boundaries] = color[c]
        legend_entries.append((method, color))

    h, w = raw_shape
    aspect = h / w
    fig_w = 6
    fig_h = max(4, fig_w * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])

    if legend_entries:
        patches = [Patch(facecolor=c, label=m) for m, c in legend_entries]
        ax.legend(
            handles=patches,
            loc="upper right",
            fontsize=9,
            framealpha=0.8,
            edgecolor="white",
        )

    title = FOV_LABELS.get(fov_group, fov_group)
    ax.set_title(f"{title} - T{tp:04d}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    name = f"{sanitize(fov_group)}_T{tp:04d}"
    _save(fig, output_dir / "overlay" / dataset / name)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_fov_group(
    fov_group: str,
    entries: list[dict],
    methods: list[str],
    output_dir: Path,
    dataset: str,
    figure_types: set[str],
) -> None:
    """Load all images for a FOV group and generate requested figures."""
    label = FOV_LABELS.get(fov_group, fov_group)
    print(f"\n  {label}  ({len(entries)} timepoints)")

    # Load all MIP projections once
    loaded: dict[tuple, np.ndarray] = {}
    for entry in entries:
        tp = entry["timepoint"]
        if entry["raw"] is not None:
            loaded[(tp, "raw")] = load_mip(entry["raw"])
        for method in methods:
            seg_path = entry.get(method.lower())
            if seg_path is not None:
                loaded[(tp, method)] = load_mip(seg_path)

    print(f"    Loaded {len(loaded)} MIP projections")

    # Per-timepoint figures
    for entry in entries:
        tp = entry["timepoint"]
        if "comparison" in figure_types:
            for mode in ("overlay", "gray"):
                make_comparison_grid(
                    fov_group, tp, loaded, methods, mode, output_dir, dataset
                )
        if "overlay" in figure_types:
            make_contour_figure(fov_group, tp, loaded, methods, output_dir, dataset)

    # Per-FOV-group figures
    if "temporal" in figure_types:
        for mode in ("overlay", "gray"):
            make_temporal_montage(
                fov_group, entries, loaded, methods, mode, output_dir, dataset
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize timelapse segmentation results.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_METHODS.keys()),
        choices=list(DATASET_METHODS.keys()),
        help="Datasets to visualize (default: all).",
    )
    parser.add_argument(
        "--all-fovs",
        action="store_true",
        help="Visualize every FOV group, not just the curated subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VISUALS_DIR,
        help=f"Root output directory for figures (default: {VISUALS_DIR}).",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["comparison", "overlay", "temporal"],
        choices=["comparison", "overlay", "temporal"],
        help="Figure types to generate (default: all).",
    )
    args = parser.parse_args()
    figure_types = set(args.figures)

    for dataset in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset}")
        print(f"{'=' * 60}")

        all_fovs = discover_files(dataset)
        methods = DATASET_METHODS[dataset]
        print(f"  Discovered {len(all_fovs)} FOV groups, methods: {methods}")

        if args.all_fovs:
            selected = all_fovs
        else:
            curated = set(CURATED_FOVS.get(dataset, []))
            selected = {k: v for k, v in all_fovs.items() if k in curated}
            if not selected:
                print("  WARNING: no curated FOVs matched, falling back to all FOVs")
                selected = all_fovs
            missing = curated - set(selected.keys())
            if missing:
                for m in sorted(missing):
                    print(f"  WARNING: curated FOV not found: {m}")

        print(f"  Selected {len(selected)} FOV groups for visualization")
        print(f"  Figure types: {sorted(figure_types)}")

        for fov_group, entries in sorted(selected.items()):
            process_fov_group(
                fov_group,
                entries,
                methods,
                args.output_dir,
                dataset,
                figure_types,
            )

    print(f"\nDone. Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
