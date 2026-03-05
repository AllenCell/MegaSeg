"""Shared infrastructure for running classic aicssegmentation workflows.

Provides the structure registry, dynamic import with source patching for
broken scripts, and image loading utilities. Used by both the verification
script (verify_classic_gt.py) and the run pipeline (run_classic.py).
"""

import ast
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from aicsimageio import AICSImage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MS_CLASSIC_DIR = Path("/allen/aics/lumenoid/inner_outer_seg/megaseg_exp/ms_classic")

# ---------------------------------------------------------------------------
# Structure registry: script filename, workflow function name, patch reason
# ---------------------------------------------------------------------------
STRUCTURES = {
    # structure: (script_basename, workflow_func_name, broken_reason_or_None)
    "ACTN1": ("ms_seg_ACTN1.py", "Workflow_actn1", None),
    "CETN2": ("classic_segmentation_CETN2.py", "Workflow_cetn2", "peak_local_max"),
    "CTNNB1": ("classic_segmentation_CTNNB1.py", "Workflow_ctnnb1", None),
    "FBL": ("classic_segmentation_FBL.py", "Workflow_fbl", None),
    "LAMP1": ("classic_segmentation_LAMP1.py", "Workflow_lamp1", None),
    "LMNB1": (
        "classic_segmentation_LMNB1.py",
        "Workflow_lmnb1_interphase",
        "watershed_import",
    ),
    "MYH10": ("classic_segmentation_MYH10.py", "Workflow_myh10", None),
    "NPM1": ("classic_segmentation_NPM1.py", "Workflow_npm1", None),
    "NUP153": ("classic_segmentation_NUP153.py", "Workflow_nup153", None),
    "RAB5A": ("classic_segmentation_RAB5A.py", "Workflow_rab5a", None),
    "SEC61B": ("classic_segmentation_SEC61B.py", "Workflow_sec61b", None),
    "SEC61B_DUAL": (
        "classic_segmentation_SEC61B_DUAL.py",
        "Workflow_sec61b_dual",
        None,
    ),
    "SLC25A17": (
        "classic_segmentation_SLC25A17.py",
        "Workflow_slc25a17",
        "peak_local_max",
    ),
    "SON": ("classic_segmentation_SON.py", "Workflow_son", None),
    "ST6GAL1": ("classic_segmentation_ST6GAL1.py", "Workflow_st6gal1", None),
    "TOMM20": ("classic_segmentation_TOMM20.py", "Workflow_tomm20", "main_only"),
    "TUBA1B": ("classic_segmentation_TUBA1B.py", "Workflow_tuba1b", None),
}

# Structures that have per-structure CSVs in variance-100x/
STRUCTURES_WITH_CSVS = {
    "ACTN1",
    "CTNNB1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TOMM20",
}

# Structures with existing classic outputs (13 populated directories)
STRUCTURES_WITH_OUTPUTS = {
    "ACTN1",
    "CTNNB1",
    "FBL",
    "LAMP1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SEC61B_DUAL",
    "SON",
    "ST6GAL1",
    "TUBA1B",
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_3d(path: str) -> np.ndarray:
    """Load a TIFF and squeeze to 3D (Z, Y, X)."""
    img = AICSImage(path)
    return np.squeeze(img.data)


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------
def extract_structure_from_filename(filename: str) -> str:
    """Extract structure name from filenames like 'NPM1_13054_IM.tiff'."""
    base = filename.replace("_IM.tiff", "").replace("_GT.tiff", "")
    base = base.replace("_seg.tiff", "").replace(".tiff", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


# ---------------------------------------------------------------------------
# Dice score
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


# ---------------------------------------------------------------------------
# Source patching for broken workflow scripts
# ---------------------------------------------------------------------------
def patch_script_source(script_path: Path, broken_reason: str) -> str:
    """Read script source and apply patches for known issues.

    Patches:
        watershed_import: skimage.morphology.watershed moved to
            skimage.segmentation in skimage 0.26.
        peak_local_max: returns Nx3 coordinates instead of boolean mask
            in skimage >=0.24. Inject wrapper that converts back.
        main_only: module-level notebook code triggers numpy 2.x breakage
            at import time. AST-based stripping of executable statements.
    """
    source = script_path.read_text()

    if broken_reason == "watershed_import":
        source = source.replace(
            "from skimage.morphology import remove_small_objects, watershed, dilation, ball",
            "from skimage.morphology import remove_small_objects, dilation, ball\n"
            "from skimage.segmentation import watershed",
        )

    elif broken_reason == "peak_local_max":
        helper = '''
def _peak_local_max_mask(image, **kwargs):
    """Wrapper: convert peak_local_max coords to boolean mask (skimage >=0.24 compat)."""
    from skimage.feature import peak_local_max as _plm
    coords = _plm(image, **kwargs)
    mask = np.zeros(image.shape, dtype=bool)
    if len(coords) > 0:
        mask[tuple(coords.T)] = True
    return mask
'''
        idx = source.find("\ndef Workflow_")
        if idx == -1:
            idx = source.find("def Workflow_")
        source = source[:idx] + helper + source[idx:]

        source = source.replace(
            "local_maxi = peak_local_max(",
            "local_maxi = _peak_local_max_mask(",
        )

    elif broken_reason == "main_only":
        tree = ast.parse(source)
        last_keep_line = 0
        for node in ast.iter_child_nodes(tree):
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                end = getattr(node, "end_lineno", None)
                if end is not None:
                    last_keep_line = max(last_keep_line, end)

        if last_keep_line > 0:
            lines = source.split("\n")
            source = "\n".join(lines[:last_keep_line])

    return source


def load_workflow_function(
    structure: str,
) -> Optional[Callable]:
    """Dynamically import the workflow function for a structure.

    Applies source patches for scripts with known incompatibilities.
    Returns the callable workflow function, or None on failure.
    """
    if structure not in STRUCTURES:
        print(f"  ERROR: Unknown structure '{structure}'", file=sys.stderr)
        return None

    script_basename, func_name, broken_reason = STRUCTURES[structure]
    script_path = MS_CLASSIC_DIR / script_basename

    if not script_path.exists():
        print(f"  WARNING: Script not found: {script_path}", file=sys.stderr)
        return None

    try:
        if broken_reason:
            # Apply source patches and compile
            patched_source = patch_script_source(script_path, broken_reason)
            module_name = f"classic_seg_{structure}_patched"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(compile(patched_source, str(script_path), "exec"), module.__dict__)
        else:
            # Normal import
            module_name = f"classic_seg_{structure}"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        return getattr(module, func_name)

    except Exception as e:
        print(f"  ERROR importing {structure}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None
