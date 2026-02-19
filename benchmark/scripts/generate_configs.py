#!/usr/bin/env python3
"""Generate all portable experiment configs for megaseg-benchmark.

Generates per dtype (float32, uint8):
- 14 individual eval configs (individual model on its own test set)
- 14 individual test_new configs (individual model on test_new set)
- 60 megaseg backbone eval configs (4 backbones × 14 structs + 4 backbones × test_new)
- 14 parameterized cross-eval configs (individual model on any struct via test_struct param)
- experiment_matrix.csv (392 rows for cross-eval job runner)

Total: 204 YAML configs + 1 CSV
"""

import csv
import os
from pathlib import Path
from textwrap import dedent

# ── Registries ──────────────────────────────────────────────────────────────

STRUCTURES = [
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

DTYPES = ["float32", "uint8"]

MEGASEG_BACKBONES = {
    "dynunet": {
        "model_config_template": "local/segmentation_ms25_{dtype}.yaml",
        "trainer": "gpu.yaml",
        "checkpoint": "megaseg/dynunet.ckpt",
    },
    "attention_unet": {
        "model_config_template": "local/segmentation_ms25_attn_unet_{dtype}.yaml",
        "trainer": "ddp.yaml",
        "checkpoint": "megaseg/attention_unet.ckpt",
    },
    "swin_unetr": {
        "model_config_template": "local/segmentation_ms25_swin_unetr_{dtype}.yaml",
        "trainer": "ddp.yaml",
        "checkpoint": "megaseg/swin_unetr.ckpt",
    },
    "unet": {
        "model_config_template": "local/segmentation_ms25_unet_{dtype}.yaml",
        "trainer": "ddp.yaml",
        "checkpoint": "megaseg/unet.ckpt",
    },
}


# ── Templates ───────────────────────────────────────────────────────────────


def individual_eval_config(struct: str, dtype: str) -> str:
    return dedent(f"""\
    # @package _global_
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: local/segmentation_ms25_{dtype}.yaml
      - override /callbacks: default.yaml
      - override /trainer: gpu.yaml
      - override /logger: csv.yaml

    tags: ["individual_eval", "{dtype}"]
    seed: 12345

    checkpoint:
      ckpt_path: ${{paths.checkpoint_dir}}/individual/{struct}.ckpt

    experiment_name: individual_eval_{struct}
    run_name: dynUNet_{struct}
    source_col: raw
    target_col: seg
    spatial_dims: 3
    raw_im_channels: 1

    trainer:
      max_epochs: 1000
      devices: [0]

    model:
      save_dir: ${{paths.log_dir}}/eval/{dtype}/individual/ms13-{struct}

    paths:
      output_dir: ${{paths.log_dir}}/eval/{dtype}/individual/ms13-{struct}

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
        save_dir: ${{paths.log_dir}}/eval/{dtype}/individual/ms13-{struct}
        stages: ["test", "predict"]
    """)


def individual_test_new_config(struct: str, dtype: str) -> str:
    return dedent(f"""\
    # @package _global_
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: local/segmentation_ms25_{dtype}.yaml
      - override /callbacks: default.yaml
      - override /trainer: gpu.yaml
      - override /logger: csv.yaml

    tags: ["individual_test_new", "{dtype}"]
    seed: 12345

    checkpoint:
      ckpt_path: ${{paths.checkpoint_dir}}/individual/{struct}.ckpt

    experiment_name: individual_test_new_{struct}
    run_name: dynUNet_test_new_{struct}
    source_col: raw
    target_col: seg
    spatial_dims: 3
    raw_im_channels: 1

    trainer:
      max_epochs: 1000
      devices: [0]

    model:
      save_dir: ${{paths.log_dir}}/eval/{dtype}/individual_test_new/ms13-{struct}

    paths:
      output_dir: ${{paths.log_dir}}/eval/{dtype}/individual_test_new/ms13-{struct}

    data:
      path: ${{paths.data_dir}}/standardized_test/
      cache_dir: ${{paths.cache_dir}}
      subsample:
      batch_size: 1
      num_workers: 1
      _aux:
        patch_shape: [32, 128, 128]

    callbacks:
      saving:
        _target_: cyto_dl.callbacks.ImageSaver
        save_dir: ${{paths.log_dir}}/eval/{dtype}/individual_test_new/ms13-{struct}
        stages: ["test", "predict"]
    """)


def megaseg_eval_config(backbone: str, struct: str, dtype: str) -> str:
    bb = MEGASEG_BACKBONES[backbone]
    model_config = bb["model_config_template"].format(dtype=dtype)
    return dedent(f"""\
    # @package _global_
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: {model_config}
      - override /callbacks: default.yaml
      - override /trainer: {bb["trainer"]}
      - override /logger: csv.yaml

    tags: ["megaseg_eval", "{dtype}"]
    seed: 12345

    checkpoint:
      ckpt_path: ${{paths.checkpoint_dir}}/{bb["checkpoint"]}

    experiment_name: megaseg_eval_{backbone}_{struct}
    run_name: {backbone}_megaseg_{struct}
    source_col: raw
    target_col: seg
    spatial_dims: 3
    raw_im_channels: 1

    trainer:
      max_epochs: 1000
      devices: [0]

    model:
      save_dir: ${{paths.log_dir}}/eval/{dtype}/megaseg/{backbone}/{struct}

    paths:
      output_dir: ${{paths.log_dir}}/eval/{dtype}/megaseg/{backbone}/{struct}

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
        save_dir: ${{paths.log_dir}}/eval/{dtype}/megaseg/{backbone}/{struct}
        stages: ["test", "predict"]
    """)


def megaseg_test_new_config(backbone: str, dtype: str) -> str:
    bb = MEGASEG_BACKBONES[backbone]
    model_config = bb["model_config_template"].format(dtype=dtype)
    return dedent(f"""\
    # @package _global_
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: {model_config}
      - override /callbacks: default.yaml
      - override /trainer: {bb["trainer"]}
      - override /logger: csv.yaml

    tags: ["megaseg_test_new", "{dtype}"]
    seed: 12345

    checkpoint:
      ckpt_path: ${{paths.checkpoint_dir}}/{bb["checkpoint"]}

    experiment_name: megaseg_eval_{backbone}_test_new
    run_name: {backbone}_megaseg_test_new
    source_col: raw
    target_col: seg
    spatial_dims: 3
    raw_im_channels: 1

    trainer:
      max_epochs: 1000
      devices: [0]

    model:
      save_dir: ${{paths.log_dir}}/eval/{dtype}/megaseg/{backbone}/test_new

    paths:
      output_dir: ${{paths.log_dir}}/eval/{dtype}/megaseg/{backbone}/test_new

    data:
      path: ${{paths.data_dir}}/standardized_test/
      cache_dir: ${{paths.cache_dir}}
      subsample:
      batch_size: 1
      num_workers: 1
      _aux:
        patch_shape: [32, 128, 128]

    callbacks:
      saving:
        _target_: cyto_dl.callbacks.ImageSaver
        save_dir: ${{paths.log_dir}}/eval/{dtype}/megaseg/{backbone}/test_new
        stages: ["test", "predict"]
    """)


def cross_eval_config(model_struct: str, dtype: str) -> str:
    return dedent(f"""\
    # @package _global_
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: local/segmentation_ms25_{dtype}.yaml
      - override /callbacks: default.yaml
      - override /trainer: gpu.yaml
      - override /logger: csv.yaml

    tags: ["cross_eval", "{dtype}"]
    seed: 12345

    test_struct: ???

    checkpoint:
      ckpt_path: ${{paths.checkpoint_dir}}/individual/{model_struct}.ckpt

    experiment_name: cross_eval_{model_struct}_on_${{test_struct}}
    run_name: dynUNet_cross_{model_struct}_on_${{test_struct}}
    source_col: raw
    target_col: seg
    spatial_dims: 3
    raw_im_channels: 1

    trainer:
      max_epochs: 1000
      devices: [0]

    model:
      save_dir: ${{paths.log_dir}}/eval/{dtype}/cross_eval/ms13-{model_struct}/tested_on_${{test_struct}}

    paths:
      output_dir: ${{paths.log_dir}}/eval/{dtype}/cross_eval/ms13-{model_struct}/tested_on_${{test_struct}}

    data:
      path: ${{paths.data_dir}}/variance-100x/${{test_struct}}/
      cache_dir: ${{paths.cache_dir}}
      subsample:
      batch_size: 1
      num_workers: 1
      _aux:
        patch_shape: [32, 128, 128]

    callbacks:
      saving:
        _target_: cyto_dl.callbacks.ImageSaver
        save_dir: ${{paths.log_dir}}/eval/{dtype}/cross_eval/ms13-{model_struct}/tested_on_${{test_struct}}
        stages: ["test", "predict"]
    """)


# ── Generator ───────────────────────────────────────────────────────────────


def main():
    # Resolve project root (script lives in scripts/)
    project_root = Path(__file__).resolve().parent.parent
    configs_dir = project_root / "configs" / "experiment" / "eval"

    counts = {"individual": 0, "individual_test_new": 0, "megaseg": 0, "cross_eval": 0}

    for dtype in DTYPES:
        # 1. Individual eval configs (14 per dtype)
        for struct in STRUCTURES:
            path = configs_dir / dtype / "individual" / f"{struct}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(individual_eval_config(struct, dtype))
            counts["individual"] += 1

        # 2. Individual test_new configs (14 per dtype)
        for struct in STRUCTURES:
            path = configs_dir / dtype / "individual_test_new" / f"{struct}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(individual_test_new_config(struct, dtype))
            counts["individual_test_new"] += 1

        # 3. MegaSeg backbone eval configs (4 backbones × 14 structs + 4 × test_new = 60 per dtype)
        for backbone in MEGASEG_BACKBONES:
            for struct in STRUCTURES:
                path = configs_dir / dtype / "megaseg" / backbone / f"{struct}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(megaseg_eval_config(backbone, struct, dtype))
                counts["megaseg"] += 1
            # test_new for each backbone
            path = configs_dir / dtype / "megaseg" / backbone / "test_new.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(megaseg_test_new_config(backbone, dtype))
            counts["megaseg"] += 1

        # 4. Cross-eval configs (14 parameterized per dtype)
        for model_struct in STRUCTURES:
            dir_name = f"{model_struct}_model"
            path = configs_dir / dtype / "cross_eval" / dir_name / "cross_eval.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(cross_eval_config(model_struct, dtype))
            counts["cross_eval"] += 1

    # 5. Generate experiment_matrix.csv (392 rows)
    matrix_path = project_root / "scripts" / "experiment_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dtype", "model_struct", "test_struct", "config_path", "output_dir"]
        )
        for dtype in DTYPES:
            for model_struct in STRUCTURES:
                for test_struct in STRUCTURES:
                    config_path = (
                        f"eval/{dtype}/cross_eval/{model_struct}_model/cross_eval"
                    )
                    output_dir = f"outputs/eval/{dtype}/cross_eval/ms13-{model_struct}/tested_on_{test_struct}"
                    writer.writerow(
                        [dtype, model_struct, test_struct, config_path, output_dir]
                    )
                    row_count += 1

    total = sum(counts.values())
    print(f"Generated {total} experiment configs:")
    for kind, count in counts.items():
        print(f"  {kind}: {count}")
    print(f"Generated experiment_matrix.csv with {row_count} rows")


if __name__ == "__main__":
    main()
