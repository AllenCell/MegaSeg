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

# ── Fine-tuning experiment matrix (L-shaped: 7 experiments) ─────────────────
# All training was done outside the benchmark. These configs run predict passes
# using eval.py to generate predictions from existing fine-tuned checkpoints.
#
# Checkpoint root: /allen/aics/users/ashwin.samudre/EMT_image_analysis/
#   Colony_mask_training_inference/logs/train/runs/megaseg_finetune_lamin/

_FT_CKPT_ROOT = (
    "/allen/aics/users/ashwin.samudre/EMT_image_analysis/"
    "Colony_mask_training_inference/logs/train/runs/megaseg_finetune_lamin"
)

FINETUNE_LMNB1_EXPERIMENTS = {
    "data100_converged": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin/2026-01-15_15-55-59/checkpoints/epoch_738.ckpt",
        "data_pct": 100,
        "desc": "100% data, converged (~88 FT epochs)",
    },
    "data100_10epochs": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin_10epochs/2026-01-16_12-50-20/checkpoints/epoch_651.ckpt",
        "data_pct": 100,
        "desc": "100% data, 10 FT epochs",
    },
    "data100_50epochs": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin_50epochs/2026-01-16_12-50-41/checkpoints/epoch_698.ckpt",
        "data_pct": 100,
        "desc": "100% data, 50 FT epochs",
    },
    "data100_100epochs": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin_100epochs/2026-01-16_12-51-55/checkpoints/epoch_738.ckpt",
        "data_pct": 100,
        "desc": "100% data, 100 FT epochs",
    },
    "data10_converged": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin_10data/2026-01-16_15-36-40/checkpoints/epoch_748.ckpt",
        "data_pct": 10,
        "desc": "10% data (17 train, 5 val), converged",
    },
    "data20_converged": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin_20data/2026-01-16_15-33-55/checkpoints/epoch_734.ckpt",
        "data_pct": 20,
        "desc": "20% data (33 train, 8 val), converged",
    },
    "data50_converged": {
        "ckpt": f"{_FT_CKPT_ROOT}/dynUNet_st6_base_cytodl_ft_lamin_50data/2026-01-16_15-33-27/checkpoints/epoch_726.ckpt",
        "data_pct": 50,
        "desc": "50% data (82 train, 20 val), converged",
    },
}

# ── Templates ───────────────────────────────────────────────────────────────


def individual_eval_config(struct: str, dtype: str) -> str:
    return dedent(f"""\
    # @package _global_
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: local/segmentation_ms25_{dtype}.yaml
      - override /callbacks: eval.yaml
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
      - override /callbacks: eval.yaml
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
      - override /callbacks: eval.yaml
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
      - override /callbacks: eval.yaml
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
      - override /callbacks: eval.yaml
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


def finetune_lmnb1_predict_config(label: str, ckpt_path: str, data_pct: int) -> str:
    """Generate a predict config for a fine-tuned LMNB1 model.

    These configs are used with eval.py (not train.py) to run
    trainer.predict on the test set using an existing fine-tuned checkpoint.
    """
    return dedent(f"""\
    # @package _global_
    # Fine-tuned LMNB1 predict config: {label}
    # Run with: CYTODL_CONFIG_PATH=configs uv run python cyto-dl/cyto_dl/eval.py \\
    #           experiment=finetune/lmnb1/{label}
    defaults:
      - override /data: local/segmentation.yaml
      - override /model: local/segmentation_ms25_uint8.yaml
      - override /callbacks: eval.yaml
      - override /trainer: gpu.yaml
      - override /logger: csv.yaml

    tags: ["finetune", "lmnb1", "predict", "data{data_pct}"]
    seed: 12345

    checkpoint:
      ckpt_path: {ckpt_path}

    experiment_name: finetune_lmnb1_{label}
    run_name: dynunet_lmnb1_{label}_predict
    source_col: raw
    target_col: seg
    spatial_dims: 3
    raw_im_channels: 1

    trainer:
      devices: [0]

    model:
      save_dir: ${{paths.log_dir}}/finetune/lmnb1/{label}

    paths:
      output_dir: ${{paths.log_dir}}/finetune/lmnb1/{label}

    data:
      path: ${{paths.root_dir}}/data/lmnb1/test/
      cache_dir: ${{paths.cache_dir}}
      subsample:
      batch_size: 1
      num_workers: 1
      _aux:
        patch_shape: [32, 128, 128]

    callbacks:
      saving:
        _target_: cyto_dl.callbacks.ImageSaver
        save_dir: ${{paths.log_dir}}/finetune/lmnb1/{label}
        stages: ["test", "predict"]
    """)


# ── Generator ───────────────────────────────────────────────────────────────


def main():
    # Resolve project root (script lives in scripts/)
    project_root = Path(__file__).resolve().parent.parent
    configs_dir = project_root / "configs" / "experiment" / "eval"

    counts = {
        "individual": 0,
        "individual_test_new": 0,
        "megaseg": 0,
        "cross_eval": 0,
        "finetune": 0,
    }

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

    # 5. Fine-tuning predict configs (7 L-shaped experiments)
    finetune_dir = project_root / "configs" / "experiment" / "finetune" / "lmnb1"
    finetune_count = 0
    for label, exp_cfg in FINETUNE_LMNB1_EXPERIMENTS.items():
        path = finetune_dir / f"{label}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            finetune_lmnb1_predict_config(
                label=label,
                ckpt_path=exp_cfg["ckpt"],
                data_pct=exp_cfg["data_pct"],
            )
        )
        finetune_count += 1
    counts["finetune"] = finetune_count

    # 6. Generate experiment_matrix.csv (392 rows)
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
