# MegaSeg
A repo to keep track of the ongoing efforts in MegaSeg

## MegaSeg ml experiments and figure generation : ONGOING work

This repo is currently private and thus uses the absolute path with `/allen` in the following instructions. This will be removed in the future once we add all the configs and other scripts within this repository.

The following path - `/allen/aics/lumenoid/inner_outer_seg/megaseg_exp/` contains all the configs to generate MegaSeg inference on individual `test.csv` as well as `test_new.csv` files. This directory also contains figure generation code in `figures/` subdirectory (Fig 2A is generated). Entire directory structure is explained in the `README.md` within the directory itself. each subdirectory also has its own readme file.

To run megaseg training or inference for any config in this directory, use `conda activate /allen/aics/users/ashwin.samudre/miniconda3/envs/acm-eval` and then run MegaSeg with command from `https://github.com/AllenCell/EMT_image_analysis/tree/sm_colony_mask_pred_test_v0/Colony_mask_training_inference` via setting the correct path to config files. NOTE: I use `python ../../cyto-dl/cyto_dl/eval.py instead of python -m cyto_dl.eval` when running inference code and similarly use `train.py` for training.
