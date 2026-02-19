# Checkpoint Provenance

## Individual Model Checkpoints (DynUNet backbone)

All trained on single structures from the megaseg_b64_deep_new_13-STRUCTS experiment series.

| File | Original Path | Epoch |
|------|--------------|-------|
| `ACTN1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_ACTN1/dynUNet_st6_base_cytodl_ACTN1/2025-11-04_23-35-39/checkpoints/epoch_498.ckpt` | 498 |
| `ATP2A2.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_ATP2A2/dynUNet_st6_base_cytodl_ATP2A2/2025-11-04_23-55-16/checkpoints/epoch_490.ckpt` | 490 |
| `CTNNB1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_CTNNB1/dynUNet_st6_base_cytodl_CTNNB1/2025-11-05_00-02-40/checkpoints/epoch_403.ckpt` | 403 |
| `DSP.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_DSP/dynUNet_st6_base_cytodl_DSP/2025-11-10_17-21-23/checkpoints/epoch_095.ckpt` | 95 |
| `GJA1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_GJA1/dynUNet_st6_base_cytodl_GJA1/2025-11-05_00-09-22/checkpoints/epoch_120.ckpt` | 120 |
| `MYH10.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_MYH10/dynUNet_st6_base_cytodl_MYH10/2025-11-05_00-09-40/checkpoints/epoch_204.ckpt` | 204 |
| `NPM1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_NPM1/dynUNet_st6_base_cytodl_NPM1/2025-11-05_00-10-21/checkpoints/epoch_045.ckpt` | 45 |
| `NUP153.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_NUP153/dynUNet_st6_base_cytodl_NUP153/2025-11-05_00-10-30/checkpoints/epoch_234.ckpt` | 234 |
| `RAB5A.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_RAB5A/dynUNet_st6_base_cytodl_RAB5A/2025-11-05_00-10-56/checkpoints/epoch_182.ckpt` | 182 |
| `SEC61B.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_SEC61B/dynUNet_st6_base_cytodl_SEC61B/2025-11-05_00-11-17/checkpoints/epoch_483.ckpt` | 483 |
| `SON.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_SON/dynUNet_st6_base_cytodl_SON/2025-11-05_00-15-40/checkpoints/epoch_173.ckpt` | 173 |
| `ST6GAL1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_ST6GAL1/dynUNet_st6_base_cytodl_ST6GAL1/2025-11-05_00-15-40/checkpoints/epoch_159.ckpt` | 159 |
| `TJP1.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_TJP1/dynUNet_st6_base_cytodl_TJP1/2025-11-05_00-16-07/checkpoints/epoch_102.ckpt` | 102 |
| `TOMM20.ckpt` | `megaseg_b64_deep_new_13-STRUCTS_TOMM20/dynUNet_st6_base_cytodl_TOMM20/2025-11-05_00-16-08/checkpoints/epoch_636.ckpt` | 636 |

Individual checkpoint base path: `/allen/aics/users/ashwin.samudre/EMT_image_analysis/Colony_mask_training_inference/logs/train/runs/`

## MegaSeg Backbone Checkpoints

Multi-structure models trained on all 14 structures simultaneously.

| File | Backbone | Original Path | Epoch |
|------|----------|--------------|-------|
| `dynunet.ckpt` | DynUNet | `/allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/logs/train/runs/megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/2024-07-15_00-39-47/checkpoints/epoch_650.ckpt` | 650 |
| `attention_unet.ckpt` | Attention UNet | `megaseg_b64_deep_new_13-STRUCTS_retrain_attn_unet/dynUNet_st6_base_cytodl_retrain_attn_unet/2025-12-05_09-07-15/checkpoints/epoch_474.ckpt` | 474 |
| `swin_unetr.ckpt` | Swin UNETR | `megaseg_b64_deep_new_13-STRUCTS_retrain_swin_unetr/dynUNet_st6_base_cytodl_retrain_swin_unetr/2025-11-18_11-04-58/checkpoints/epoch_285.ckpt` | 285 |
| `unet.ckpt` | UNet | `megaseg_b64_deep_new_13-STRUCTS_retrain_unet/dynUNet_st6_base_cytodl_retrain_unet/2025-12-07_04-40-57/checkpoints/epoch_429.ckpt` | 429 |

Attention UNet, Swin UNETR, UNet base path: `/allen/aics/users/ashwin.samudre/EMT_image_analysis/Colony_mask_training_inference/logs/train/runs/`
