#!/usr/bin/env python
"""Generate Hydra eval configs for additional structures evaluation.

Creates predict-only configs for:
  - MegaSeg DynUNet (1 config)
  - 14 individual structure models (14 configs)

All configs use the same all.csv (all extracted channel-volumes from all 4
additional structures). Output predictions go to:
  outputs/eval/uint8/{megaseg/dynunet,individual}/additional_structures/{MODEL}/

Usage:
    python scripts/generate_additional_eval_configs.py [--dry-run]
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDIVIDUAL_MODELS = [
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NPM1",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]

# Template for MegaSeg DynUNet predict-only config
MEGASEG_TEMPLATE = """\
# @package _global_
defaults:
  - override /data: local/segmentation.yaml
  - override /model: local/segmentation_ms25_uint8.yaml
  - override /callbacks: default.yaml
  - override /trainer: gpu.yaml
  - override /logger: csv.yaml

tags: ["megaseg_eval", "uint8", "additional_structures"]
seed: 12345

checkpoint:
  ckpt_path: ${paths.checkpoint_dir}/megaseg/dynunet.ckpt

train: False
test: False
predict: True

experiment_name: megaseg_eval_additional_structures
run_name: dynunet_megaseg_additional_structures
source_col: file_path
target_col: split

spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 1000
  devices: [0]

model:
  save_dir: ${paths.log_dir}/eval/uint8/megaseg/dynunet/additional_structures

paths:
  output_dir: ${paths.log_dir}/eval/uint8/megaseg/dynunet/additional_structures

data:
  path: ${paths.root_dir}/data/additional_structures_eval_csvs/all.csv
  split_column: split
  just_inference: True
  subsample:
  batch_size: 1
  num_workers: 4
  _aux:
    patch_shape: [32, 128, 128]

callbacks:
  rich_progress_bar: null
  saving:
    _target_: cyto_dl.callbacks.ImageSaver
    save_dir: ${paths.log_dir}/eval/uint8/megaseg/dynunet/additional_structures
    stages: ["predict"]
"""


def _individual_config(model: str) -> str:
    """Generate individual model config YAML with proper Hydra interpolations."""
    return f"""\
# @package _global_
defaults:
  - override /data: local/segmentation.yaml
  - override /model: local/segmentation_ms25_uint8.yaml
  - override /callbacks: default.yaml
  - override /trainer: gpu.yaml
  - override /logger: csv.yaml

tags: ["individual_eval", "uint8", "additional_structures"]
seed: 12345

checkpoint:
  ckpt_path: ${{paths.checkpoint_dir}}/individual/{model}.ckpt

train: False
test: False
predict: True

experiment_name: individual_eval_additional_structures_{model}
run_name: dynunet_{model}_additional_structures
source_col: file_path
target_col: split

spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 1000
  devices: [0]

model:
  save_dir: ${{paths.log_dir}}/eval/uint8/individual/additional_structures/{model}

paths:
  output_dir: ${{paths.log_dir}}/eval/uint8/individual/additional_structures/{model}

data:
  path: ${{paths.root_dir}}/data/additional_structures_eval_csvs/all.csv
  split_column: split
  just_inference: True
  subsample:
  batch_size: 1
  num_workers: 4
  _aux:
    patch_shape: [32, 128, 128]

callbacks:
  rich_progress_bar: null
  saving:
    _target_: cyto_dl.callbacks.ImageSaver
    save_dir: ${{paths.log_dir}}/eval/uint8/individual/additional_structures/{model}
    stages: ["predict"]
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    configs_dir = root / "configs" / "experiment" / "eval" / "uint8"

    # --- MegaSeg config ---
    ms_dir = configs_dir / "megaseg" / "dynunet" / "additional_structures"
    ms_file = ms_dir / "all.yaml"

    if args.dry_run:
        print(f"[dry-run] Would write: {ms_file}")
    else:
        ms_dir.mkdir(parents=True, exist_ok=True)
        ms_file.write_text(MEGASEG_TEMPLATE)
        print(f"Wrote: {ms_file}")

    # --- Individual model configs ---
    ind_dir = configs_dir / "individual" / "additional_structures"
    for model in INDIVIDUAL_MODELS:
        ind_file = ind_dir / f"{model}.yaml"
        content = _individual_config(model)

        if args.dry_run:
            print(f"[dry-run] Would write: {ind_file}")
        else:
            ind_dir.mkdir(parents=True, exist_ok=True)
            ind_file.write_text(content)
            print(f"Wrote: {ind_file}")

    total = 1 + len(INDIVIDUAL_MODELS)
    print(
        f"\nGenerated {total} configs (1 MegaSeg + {len(INDIVIDUAL_MODELS)} individual)"
    )


if __name__ == "__main__":
    main()
