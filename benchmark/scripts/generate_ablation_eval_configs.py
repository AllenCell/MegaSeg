#!/usr/bin/env python3
"""Generate eval configs for ablation experiments.

Each ablation model (4-struct, 7-struct, 14-struct) is evaluated on all 14
individual structure test sets, plus the combined test_new set.

Usage:
    python scripts/generate_ablation_eval_configs.py

Checkpoint paths must be updated after training completes. The generated configs
use a placeholder that should be replaced with the actual best checkpoint path.
"""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = BENCHMARK_DIR / "configs" / "experiment" / "eval" / "ablation"

# All 14 structures (excluding LMNB1 and NPM1)
STRUCTURES = [
    "AAVS1",
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "MYH10",
    "NUP153",
    "RAB5A",
    "SEC61B",
    "SON",
    "ST6GAL1",
    "TJP1",
    "TOMM20",
]

ABLATIONS = {
    "4_structs": {
        "structures": ["CTNNB1", "MYH10", "ACTN1", "ATP2A2"],
        "label": "4-STRUCTS",
    },
    "7_structs": {
        "structures": [
            "CTNNB1",
            "RAB5A",
            "TOMM20",
            "MYH10",
            "ACTN1",
            "ST6GAL1",
            "ATP2A2",
        ],
        "label": "7-STRUCTS",
    },
    "14_structs": {
        "structures": STRUCTURES,
        "label": "14-STRUCTS",
    },
}


def per_structure_eval_config(ablation_name: str, struct: str) -> str:
    """Generate eval config for one ablation model tested on one structure."""
    return f"""\
# @package _global_
# Eval: ablation {ablation_name} model tested on {struct}
defaults:
  - override /data: local/segmentation_finetune.yaml
  - override /model: local/segmentation_ms25_float32.yaml
  - override /callbacks: eval.yaml
  - override /trainer: gpu.yaml
  - override /logger: csv.yaml

tags: ["ablation_eval", "{ablation_name}"]
seed: 12345

channel_name: "C:0"

checkpoint:
  ckpt_path: CHECKPOINT_PATH_PLACEHOLDER
  weights_only: true
  strict: true

experiment_name: ablation_eval_{ablation_name}_{struct}
run_name: dynunet_{ablation_name}_{struct}
source_col: raw
target_col: seg
spatial_dims: 3
raw_im_channels: 1

trainer:
  max_epochs: 1000
  devices: [0]

model:
  save_dir: ${{paths.log_dir}}/eval/ablation/{ablation_name}/{struct}

paths:
  output_dir: ${{paths.log_dir}}/eval/ablation/{ablation_name}/{struct}

data:
  path: ${{paths.data_dir}}/variance-100x/{struct}/
  cache_dir: ${{paths.cache_dir}}
  subsample:
  batch_size: 1
  num_workers: 1
  _aux:
    patch_shape: [32, 128, 128]

callbacks:
  saving:
    _target_: cyto_dl.callbacks.ImageSaver
    save_dir: ${{paths.log_dir}}/eval/ablation/{ablation_name}/{struct}
    stages: ["test", "predict"]
"""


def main():
    total = 0
    for ablation_name in ABLATIONS:
        ablation_dir = CONFIGS_DIR / ablation_name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Per-structure eval (test on all 14 structures)
        for struct in STRUCTURES:
            config_path = ablation_dir / f"{struct}.yaml"
            config_path.write_text(per_structure_eval_config(ablation_name, struct))
            total += 1

    print(f"Generated {total} ablation eval configs in {CONFIGS_DIR}")
    print(f"  {len(ABLATIONS)} ablations x {len(STRUCTURES)} structures = {total}")
    print()
    print("NOTE: Checkpoint paths contain 'CHECKPOINT_PATH_PLACEHOLDER'.")
    print("After training, update with actual checkpoint paths, e.g.:")
    print(
        "  sed -i 's|CHECKPOINT_PATH_PLACEHOLDER|/path/to/best.ckpt|g' configs/experiment/eval/ablation/4_structs/*.yaml"
    )


if __name__ == "__main__":
    main()
