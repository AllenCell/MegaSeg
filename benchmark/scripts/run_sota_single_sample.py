#!/usr/bin/env python
"""Run all 8 SOTA comparison methods on a single input volume.

Produces prediction TIFFs and an optional max-projection comparison figure
for one sample, making it easy to iterate on sample selection for the paper
qualitative figure.

Methods: MegaSeg (DynUNet), Individual DynUNet, nnU-Net v2, Classic
(aicsseg), Otsu, CellPose v4, StarDist, SAM (vit_b).

Each method runs in its own subprocess with the appropriate virtualenv,
so this script itself only needs numpy/tifffile/matplotlib (the benchmark
.venv).

Usage:
    # By sample ID and structure (looks up paths from test.csv):
    python scripts/run_sota_single_sample.py --structure ACTN1 --sample-id ACTN1_11347

    # By explicit raw TIFF path:
    python scripts/run_sota_single_sample.py --structure ACTN1 --raw /path/to/raw.tiff

    # Skip specific methods:
    python scripts/run_sota_single_sample.py --structure ACTN1 --sample-id ACTN1_11347 --skip sam stardist

    # Only run specific methods:
    python scripts/run_sota_single_sample.py --structure ACTN1 --sample-id ACTN1_11347 --only megaseg classic otsu
"""

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # benchmark/ (may be a worktree)
MAIN_BENCHMARK = Path("/allen/aics/rep_learn/megaseg/benchmark")
DATA_DIR = Path("/allen/aics/rep_learn/megaseg/data/variance-100x")
CHECKPOINT_DIR = MAIN_BENCHMARK / "checkpoints"

ENVS = {
    "benchmark": MAIN_BENCHMARK / ".venv" / "bin" / "python",
    "cellpose": MAIN_BENCHMARK / "envs" / "cellpose" / "bin" / "python",
    "stardist": MAIN_BENCHMARK / "envs" / "stardist" / "bin" / "python",
    "microsam": MAIN_BENCHMARK / "envs" / "microsam" / "bin" / "python",
    "nnunet": MAIN_BENCHMARK / "envs" / "nnunet" / "bin" / "python",
}

ALL_METHODS = [
    "megaseg", "individual", "nnunet", "classic", "otsu",
    "cellpose", "stardist", "sam",
]


def resolve_sample(structure: str, sample_id: str):
    """Look up raw and seg paths from test.csv."""
    import pandas as pd
    csv = DATA_DIR / structure / "test.csv"
    if not csv.exists():
        sys.exit(f"ERROR: {csv} not found")
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        sid = Path(row["raw"]).stem
        if sid == sample_id:
            return row["raw"], row["seg"]
    sys.exit(f"ERROR: sample_id '{sample_id}' not found in {csv}")


def run_subprocess(python_bin, script_code, label, gpu=0):
    """Run a Python snippet in a subprocess with the given interpreter."""
    t0 = time.time()
    print(f"  [{label}] running...", end="", flush=True)
    result = subprocess.run(
        [str(python_bin), "-c", script_code],
        capture_output=True, text=True,
        env={
            **dict(__import__("os").environ),
            "CUDA_VISIBLE_DEVICES": str(gpu),
        },
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f" FAILED ({elapsed:.1f}s)")
        print(f"    stderr: {result.stderr[:500]}")
        return False
    print(f" done ({elapsed:.1f}s)")
    return True


def run_megaseg(raw_path, structure, out_path, gpu=0):
    ckpt = CHECKPOINT_DIR / "megaseg" / "dynunet.ckpt"
    if not ckpt.exists():
        print(f"  [MegaSeg] checkpoint not found: {ckpt}")
        return False
    code = textwrap.dedent(f"""\
        import torch, numpy as np, tifffile
        from monai.inferers import SlidingWindowInferer
        from monai.networks.nets import DynUNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kernels = [[3,3,3]]*4
        strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2]]
        net = DynUNet(spatial_dims=3, in_channels=1, out_channels=1,
                      kernel_size=kernels, strides=strides,
                      upsample_kernel_size=strides[1:])
        ckpt = torch.load("{ckpt}", map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {{k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}}
        if not state:
            state = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state, strict=False)
        net.to(device).eval()

        img = tifffile.imread("{raw_path}").astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        inferer = SlidingWindowInferer(roi_size=(32,128,128), sw_batch_size=4, overlap=0.5)
        with torch.no_grad():
            pred = inferer(tensor, net)
        pred = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        print(f"Saved {{pred.shape}}")
    """)
    return run_subprocess(ENVS["benchmark"], code, "MegaSeg", gpu)


def run_individual(raw_path, structure, out_path, gpu=0):
    ckpt = CHECKPOINT_DIR / "individual" / f"{structure}.ckpt"
    if not ckpt.exists():
        print(f"  [Individual] checkpoint not found: {ckpt}")
        return False
    code = textwrap.dedent(f"""\
        import torch, numpy as np, tifffile
        from monai.inferers import SlidingWindowInferer
        from monai.networks.nets import DynUNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kernels = [[3,3,3]]*4
        strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2]]
        net = DynUNet(spatial_dims=3, in_channels=1, out_channels=1,
                      kernel_size=kernels, strides=strides,
                      upsample_kernel_size=strides[1:])
        ckpt = torch.load("{ckpt}", map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {{k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}}
        if not state:
            state = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state, strict=False)
        net.to(device).eval()

        img = tifffile.imread("{raw_path}").astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        inferer = SlidingWindowInferer(roi_size=(32,128,128), sw_batch_size=4, overlap=0.5)
        with torch.no_grad():
            pred = inferer(tensor, net)
        pred = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        print(f"Saved {{pred.shape}}")
    """)
    return run_subprocess(ENVS["benchmark"], code, "Individual", gpu)


def run_nnunet(raw_path, structure, sample_id, out_path, gpu=0):
    existing = MAIN_BENCHMARK / "outputs" / "eval" / "uint8" / "nnunet_3d_fullres" / f"{sample_id}.nii.gz"
    if existing.exists():
        code = textwrap.dedent(f"""\
            import nibabel as nib, numpy as np, tifffile
            nii = nib.load("{existing}")
            pred = nii.get_fdata().astype(np.uint8)
            pred = np.transpose(pred, (2, 1, 0))
            pred = (pred > 0).astype(np.uint8)
            tifffile.imwrite("{out_path}", pred)
            print(f"Converted from NIfTI: {{pred.shape}}")
        """)
        return run_subprocess(ENVS["nnunet"], code, "nnU-Net (cached)", gpu)

    nnunet_dir = MAIN_BENCHMARK / "nnUNet_results" / "Dataset001_MegaSeg" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    code = textwrap.dedent(f"""\
        import tempfile, shutil, subprocess, nibabel as nib, numpy as np, tifffile
        from pathlib import Path

        raw = tifffile.imread("{raw_path}")
        raw_nii = nib.Nifti1Image(np.transpose(raw, (2,1,0)).astype(np.float32), np.eye(4))

        tmpdir = Path(tempfile.mkdtemp())
        nib.save(raw_nii, tmpdir / "{sample_id}_0000.nii.gz")
        outdir = tmpdir / "out"

        subprocess.run([
            "nnUNetv2_predict",
            "-i", str(tmpdir), "-o", str(outdir),
            "-d", "001", "-c", "3d_fullres", "-f", "0",
            "--disable_tta",
        ], check=True, capture_output=True)

        pred_nii = nib.load(outdir / "{sample_id}.nii.gz")
        pred = np.transpose(pred_nii.get_fdata().astype(np.uint8), (2,1,0))
        pred = (pred > 0).astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        shutil.rmtree(tmpdir)
        print(f"Saved {{pred.shape}}")
    """)
    return run_subprocess(ENVS["nnunet"], code, "nnU-Net", gpu)


def run_classic(raw_path, structure, out_path):
    code = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, "{MAIN_BENCHMARK / 'scripts'}")
        from classic_workflows import load_workflow_function, load_image_3d
        import numpy as np, tifffile

        wf = load_workflow_function("{structure}")
        if wf is None:
            print("No workflow for {structure}")
            sys.exit(1)

        img = load_image_3d("{raw_path}")
        result = wf(img)
        if isinstance(result, tuple):
            result = result[0]
        pred = (result > 0).astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        print(f"Saved {{pred.shape}}")
    """)
    return run_subprocess(ENVS["benchmark"], code, "Classic")


def run_otsu(raw_path, out_path):
    code = textwrap.dedent(f"""\
        import numpy as np, tifffile
        from skimage.filters import threshold_otsu
        img = tifffile.imread("{raw_path}")
        thresh = threshold_otsu(img)
        pred = (img > thresh).astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        print(f"Saved {{pred.shape}}, thresh={{thresh}}")
    """)
    return run_subprocess(ENVS["benchmark"], code, "Otsu")


def run_cellpose(raw_path, out_path, gpu=0):
    code = textwrap.dedent(f"""\
        import numpy as np, tifffile
        from cellpose.models import CellposeModel
        model = CellposeModel(gpu=True, use_bfloat16=False)
        img = tifffile.imread("{raw_path}").astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        masks, _, _ = model.eval(img, z_axis=0, diameter=None, do_3D=True, channels=[0, 0])
        pred = (masks > 0).astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        print(f"Saved {{pred.shape}}")
    """)
    return run_subprocess(ENVS["cellpose"], code, "CellPose", gpu)


def run_stardist(raw_path, out_path, gpu=0):
    code = textwrap.dedent(f"""\
        import numpy as np, tifffile
        from stardist.models import StarDist3D
        from csbdeep.utils import normalize
        model = StarDist3D.from_pretrained("3D_demo")
        img = tifffile.imread("{raw_path}")
        img_norm = normalize(img, 1, 99.8)
        labels, _ = model.predict_instances(img_norm)
        pred = (labels > 0).astype(np.uint8)
        tifffile.imwrite("{out_path}", pred)
        print(f"Saved {{pred.shape}}")
    """)
    return run_subprocess(ENVS["stardist"], code, "StarDist", gpu)


def run_sam(raw_path, out_path, gpu=0):
    code = textwrap.dedent(f"""\
        import numpy as np, tifffile, torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint="{MAIN_BENCHMARK / 'envs' / 'microsam' / 'checkpoints' / 'sam_vit_b_01ec64.pth'}")
        sam.to("cuda").eval()
        mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86,
                                                    stability_score_thresh=0.92, min_mask_region_area=100)
        img = tifffile.imread("{raw_path}")
        img_f = img.astype(np.float32)
        img_f = ((img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8) * 255).astype(np.uint8)
        combined = np.zeros(img.shape, dtype=np.uint8)
        for z in range(img.shape[0]):
            slice_rgb = np.stack([img_f[z]]*3, axis=-1)
            masks = mask_generator.generate(slice_rgb)
            for m in masks:
                combined[z][m["segmentation"]] = 1
        tifffile.imwrite("{out_path}", combined)
        print(f"Saved {{combined.shape}}")
    """)
    return run_subprocess(ENVS["microsam"], code, "SAM", gpu)


def make_comparison_figure(raw_path, gt_path, pred_paths, out_fig, sample_id, structure):
    """Generate a max-projection comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = tifffile.imread(raw_path)
    raw_mip = raw.max(axis=0).astype(np.float32)
    raw_mip = (raw_mip - raw_mip.min()) / (raw_mip.max() - raw_mip.min() + 1e-8)

    gt = tifffile.imread(gt_path) if gt_path and Path(gt_path).exists() else None
    gt_mip = gt.max(axis=0) > 0 if gt is not None else None

    method_labels = ["Raw", "GT"] + [m for m in pred_paths]
    n_cols = len(method_labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    axes[0].imshow(raw_mip, cmap="gray")
    axes[0].set_title("Raw", fontsize=9)
    axes[0].axis("off")

    if gt_mip is not None:
        axes[1].imshow(gt_mip, cmap="gray")
    axes[1].set_title("GT", fontsize=9)
    axes[1].axis("off")

    for i, (method, pred_path) in enumerate(pred_paths.items()):
        ax = axes[i + 2]
        if Path(pred_path).exists():
            pred = tifffile.imread(pred_path)
            pred_mip = pred.max(axis=0) > 0
            ax.imshow(pred_mip, cmap="gray")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(method, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{structure} — {sample_id}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(str(out_fig).replace(".png", f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all SOTA methods on a single sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--structure", required=True, help="Structure name (e.g. ACTN1)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-id", help="Sample ID from test.csv (e.g. ACTN1_11347)")
    group.add_argument("--raw", help="Direct path to raw TIFF")
    parser.add_argument("--gt", default=None, help="Path to GT segmentation TIFF (auto-resolved if using --sample-id)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/qualitative/{structure}/{sample_id}/)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--skip", nargs="+", choices=ALL_METHODS, default=[], help="Methods to skip")
    parser.add_argument("--only", nargs="+", choices=ALL_METHODS, default=None, help="Only run these methods")
    parser.add_argument("--no-figure", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    structure = args.structure

    if args.sample_id:
        raw_path, gt_path = resolve_sample(structure, args.sample_id)
        sample_id = args.sample_id
    else:
        raw_path = args.raw
        gt_path = args.gt
        sample_id = Path(raw_path).stem

    if not Path(raw_path).exists():
        sys.exit(f"ERROR: raw file not found: {raw_path}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        MAIN_BENCHMARK / "results" / "qualitative" / structure / sample_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.only if args.only else [m for m in ALL_METHODS if m not in args.skip]

    print(f"Structure: {structure}")
    print(f"Sample:    {sample_id}")
    print(f"Raw:       {raw_path}")
    print(f"GT:        {gt_path}")
    print(f"Output:    {out_dir}")
    print(f"Methods:   {methods}")
    print(f"GPU:       {args.gpu}")
    print()

    pred_paths = {}
    for method in methods:
        out_path = out_dir / f"{sample_id}_{method}.tiff"
        pred_paths[method] = str(out_path)

        if out_path.exists():
            print(f"  [{method}] cached: {out_path}")
            continue

        if method == "megaseg":
            run_megaseg(raw_path, structure, out_path, args.gpu)
        elif method == "individual":
            run_individual(raw_path, structure, out_path, args.gpu)
        elif method == "nnunet":
            run_nnunet(raw_path, structure, sample_id, out_path, args.gpu)
        elif method == "classic":
            run_classic(raw_path, structure, out_path)
        elif method == "otsu":
            run_otsu(raw_path, out_path)
        elif method == "cellpose":
            run_cellpose(raw_path, out_path, args.gpu)
        elif method == "stardist":
            run_stardist(raw_path, out_path, args.gpu)
        elif method == "sam":
            run_sam(raw_path, out_path, args.gpu)

    if not args.no_figure:
        print("\nGenerating comparison figure...")
        fig_path = out_dir / f"{sample_id}_comparison.png"
        make_comparison_figure(raw_path, gt_path, pred_paths, fig_path, sample_id, structure)

    print("\nDone.")


if __name__ == "__main__":
    main()
