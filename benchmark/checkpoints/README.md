# Checkpoint Provenance

## Current Location

All checkpoints are stored as regular files in this directory (not symlinks).
They are readable by all team members without requiring access to any individual
user's restricted directories.

Layout:
- `individual/` -- 14 structure-specific DynUNet checkpoints (~191 MB each)
- `megaseg/` -- 4 multi-structure backbone checkpoints (56--720 MB)
- `finetune/lmnb1/` -- 7 LMNB1 fine-tuned checkpoints (~191 MB each)

Hydra configs reference these via `${paths.checkpoint_dir}`, which resolves to
this directory.

## Individual Model Checkpoints (DynUNet backbone)

All trained on single structures from the megaseg_b64_deep_new_13-STRUCTS experiment series.

| File | Original Training Run (relative) | Epoch |
|------|----------------------------------|-------|
| `ACTN1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_ACTN1/dynUNet_st6_base_cytodl_ACTN1/2025-11-04_23-35-39` | 498 |
| `ATP2A2.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_ATP2A2/dynUNet_st6_base_cytodl_ATP2A2/2025-11-04_23-55-16` | 490 |
| `CTNNB1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_CTNNB1/dynUNet_st6_base_cytodl_CTNNB1/2025-11-05_00-02-40` | 403 |
| `DSP.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_DSP/dynUNet_st6_base_cytodl_DSP/2025-11-10_17-21-23` | 95 |
| `GJA1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_GJA1/dynUNet_st6_base_cytodl_GJA1/2025-11-05_00-09-22` | 120 |
| `MYH10.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_MYH10/dynUNet_st6_base_cytodl_MYH10/2025-11-05_00-09-40` | 204 |
| `NPM1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_NPM1/dynUNet_st6_base_cytodl_NPM1/2025-11-05_00-10-21` | 45 |
| `NUP153.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_NUP153/dynUNet_st6_base_cytodl_NUP153/2025-11-05_00-10-30` | 234 |
| `RAB5A.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_RAB5A/dynUNet_st6_base_cytodl_RAB5A/2025-11-05_00-10-56` | 182 |
| `SEC61B.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_SEC61B/dynUNet_st6_base_cytodl_SEC61B/2025-11-05_00-11-17` | 483 |
| `SON.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_SON/dynUNet_st6_base_cytodl_SON/2025-11-05_00-15-40` | 173 |
| `ST6GAL1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_ST6GAL1/dynUNet_st6_base_cytodl_ST6GAL1/2025-11-05_00-15-40` | 159 |
| `TJP1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_TJP1/dynUNet_st6_base_cytodl_TJP1/2025-11-05_00-16-07` | 102 |
| `TOMM20.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_TOMM20/dynUNet_st6_base_cytodl_TOMM20/2025-11-05_00-16-08` | 636 |

Original training base path (historical): `Colony_mask_training_inference/logs/train/runs/`

## MegaSeg Backbone Checkpoints

Multi-structure models trained on all 14 structures simultaneously.

| File | Backbone | Original Training Run | Epoch |
|------|----------|----------------------|-------|
| `dynunet.ckpt` | DynUNet | `megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/2024-07-15_00-39-47` | 650 |
| `attention_unet.ckpt` | Attention UNet | `megaseg_b64_deep_new_13-STRUCTS_retrain_attn_unet/dynUNet_st6_base_cytodl_retrain_attn_unet/2025-12-05_09-07-15` | 474 |
| `swin_unetr.ckpt` | Swin UNETR | `megaseg_b64_deep_new_13-STRUCTS_retrain_swin_unetr/dynUNet_st6_base_cytodl_retrain_swin_unetr/2025-11-18_11-04-58` | 285 |
| `unet.ckpt` | UNet | `megaseg_b64_deep_new_13-STRUCTS_retrain_unet/dynUNet_st6_base_cytodl_retrain_unet/2025-12-07_04-40-57` | 429 |

## LMNB1 Fine-Tune Checkpoints

Fine-tuned from MegaSeg-13 pretrained DynUNet (`megaseg/dynunet.ckpt`) on LMNB1 data.

| File | Description | Original Training Run |
|------|------------|----------------------|
| `lmnb1_data100_converged.ckpt` | 100% data, converged (~88 FT epochs) | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin/2026-01-15_15-55-59` |
| `lmnb1_data100_10epochs.ckpt` | 100% data, 10 FT epochs | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin_10epochs/2026-01-16_12-50-20` |
| `lmnb1_data100_50epochs.ckpt` | 100% data, 50 FT epochs | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin_50epochs/2026-01-16_12-50-41` |
| `lmnb1_data100_100epochs.ckpt` | 100% data, 100 FT epochs | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin_100epochs/2026-01-16_12-51-55` |
| `lmnb1_data10_converged.ckpt` | 10% data, converged | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin_10data/2026-01-16_15-36-40` |
| `lmnb1_data20_converged.ckpt` | 20% data, converged | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin_20data/2026-01-16_15-33-55` |
| `lmnb1_data50_converged.ckpt` | 50% data, converged | `megaseg_finetune_lamin/dynUNet_st6_base_cytodl_ft_lamin_50data/2026-01-16_15-33-27` |
