#!/usr/bin/env python
"""Tests for evaluate_benchmark.py pure functions."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import evaluate_benchmark as eb
from evaluate_benchmark import (
    compute_metrics,
    binarize_prediction,
    binarize_gt,
    extract_structure_from_filename,
    EvalJob,
    generate_jobs,
    resolve_pred_dir,
    resolve_test_csv,
    derive_prediction_path,
)


class TestComputeMetrics:
    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=bool)
        result = compute_metrics(mask, mask)
        assert result["dice"] == 1.0
        assert result["iou"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["accuracy"] == 1.0

    def test_no_overlap(self):
        gt = np.zeros((10, 10), dtype=bool)
        gt[:5] = True
        pred = np.zeros((10, 10), dtype=bool)
        pred[5:] = True
        result = compute_metrics(gt, pred)
        assert result["dice"] == 0.0
        assert result["iou"] == 0.0

    def test_both_empty(self):
        empty = np.zeros((10, 10), dtype=bool)
        result = compute_metrics(empty, empty)
        assert result["dice"] == 1.0
        assert result["iou"] == 1.0

    def test_partial_overlap(self):
        gt = np.zeros((10, 10), dtype=bool)
        gt[:6] = True  # 60 pixels
        pred = np.zeros((10, 10), dtype=bool)
        pred[:4] = True  # 40 pixels, 40 overlap
        result = compute_metrics(gt, pred)
        assert result["dice"] == pytest.approx(2 * 40 / (2 * 40 + 0 + 20))
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(40 / 60)


class TestBinarize:
    def test_uint8_threshold(self):
        pred = np.array([0, 127, 128, 255], dtype=np.uint8)
        result = binarize_prediction(pred, "uint8")
        np.testing.assert_array_equal(result, [False, False, True, True])

    def test_float32_threshold(self):
        pred = np.array([0.0, 0.49, 0.5, 1.0], dtype=np.float32)
        result = binarize_prediction(pred, "float32")
        np.testing.assert_array_equal(result, [False, False, True, True])

    def test_binarize_gt(self):
        gt = np.array([0, 1, 128, 255], dtype=np.uint8)
        result = binarize_gt(gt)
        np.testing.assert_array_equal(result, [False, True, True, True])


class TestStructureExtraction:
    def test_simple(self):
        assert extract_structure_from_filename("DSP_3970_IM.tiff") == "DSP"

    def test_with_digits(self):
        assert extract_structure_from_filename("ST6GAL1_4567_IM.tiff") == "ST6GAL1"

    def test_multi_segment(self):
        assert extract_structure_from_filename("HIST1H2BJ_5105_IM.tiff") == "HIST1H2BJ"

    def test_slc(self):
        assert extract_structure_from_filename("SLC25A17_1234_IM.tiff") == "SLC25A17"

    def test_standard_format(self):
        assert extract_structure_from_filename("ACTN1_10917.tiff") == "ACTN1"

    def test_aavs1(self):
        assert extract_structure_from_filename("AAVS1_10231_IM.tiff") == "AAVS1"

    def test_seg_suffix(self):
        assert extract_structure_from_filename("ACTN1_10917_seg.tiff") == "ACTN1"


class TestGenerateJobs:
    def test_individual_jobs(self):
        jobs = generate_jobs("uint8", eval_type="individual")
        assert len(jobs) == 14
        assert all(j.eval_type == "individual" for j in jobs)
        assert all(j.backbone is None for j in jobs)
        assert all(j.test_set == "standard" for j in jobs)
        assert all(j.model_struct == j.test_struct for j in jobs)

    def test_individual_test_new_jobs(self):
        jobs = generate_jobs("uint8", eval_type="individual_test_new")
        assert len(jobs) == 14
        assert all(j.test_set == "test_new" for j in jobs)

    def test_megaseg_jobs(self):
        jobs = generate_jobs("uint8", eval_type="megaseg")
        assert len(jobs) == 60
        assert all(j.backbone is not None for j in jobs)

    def test_cross_eval_jobs(self):
        jobs = generate_jobs("uint8", eval_type="cross_eval")
        assert len(jobs) == 196
        assert all(j.backbone is None for j in jobs)

    def test_all_jobs(self):
        jobs = generate_jobs("uint8")
        assert len(jobs) == 284

    def test_filter_by_structure(self):
        jobs = generate_jobs("uint8", eval_type="individual", structure="ACTN1")
        assert len(jobs) == 1
        assert jobs[0].model_struct == "ACTN1"

    def test_filter_by_backbone(self):
        jobs = generate_jobs("uint8", eval_type="megaseg", backbone="dynunet")
        assert len(jobs) == 15


class TestPathResolution:
    def test_individual_pred_dir(self):
        root = Path("/project")
        job = EvalJob("individual", "uint8", None, "ACTN1", "ACTN1", "standard")
        assert (
            resolve_pred_dir(job, root)
            == root / "outputs/eval/uint8/individual/ms13-ACTN1/seg"
        )

    def test_individual_test_new_pred_dir(self):
        root = Path("/project")
        job = EvalJob(
            "individual_test_new", "uint8", None, "ACTN1", "ACTN1", "test_new"
        )
        assert (
            resolve_pred_dir(job, root)
            == root / "outputs/eval/uint8/individual_test_new/ms13-ACTN1/seg"
        )

    def test_megaseg_pred_dir(self):
        root = Path("/project")
        job = EvalJob("megaseg", "uint8", "dynunet", "ACTN1", "ACTN1", "standard")
        assert (
            resolve_pred_dir(job, root)
            == root / "outputs/eval/uint8/megaseg/dynunet/ACTN1/seg"
        )

    def test_megaseg_test_new_pred_dir(self):
        root = Path("/project")
        job = EvalJob("megaseg", "uint8", "dynunet", "test_new", "test_new", "test_new")
        assert (
            resolve_pred_dir(job, root)
            == root / "outputs/eval/uint8/megaseg/dynunet/test_new/seg"
        )

    def test_cross_eval_pred_dir(self):
        root = Path("/project")
        job = EvalJob("cross_eval", "uint8", None, "ACTN1", "DSP", "standard")
        assert (
            resolve_pred_dir(job, root)
            == root / "outputs/eval/uint8/cross_eval/ms13-ACTN1/tested_on_DSP/seg"
        )

    def test_standard_test_csv(self):
        root = Path("/project")
        job = EvalJob("individual", "uint8", None, "ACTN1", "ACTN1", "standard")
        assert (
            resolve_test_csv(job, root) == root / "../data/variance-100x/ACTN1/test.csv"
        )

    def test_test_new_csv(self):
        root = Path("/project")
        job = EvalJob(
            "individual_test_new", "uint8", None, "ACTN1", "ACTN1", "test_new"
        )
        assert (
            resolve_test_csv(job, root)
            == root / "../data/standardized_test/test_new.csv"
        )

    def test_derive_prediction_path_tif(self, tmp_path):
        seg_dir = tmp_path / "seg"
        seg_dir.mkdir()
        (seg_dir / "ACTN1_10917.tif").touch()
        result = derive_prediction_path("ACTN1_10917.tiff", seg_dir)
        assert result == seg_dir / "ACTN1_10917.tif"

    def test_derive_prediction_path_exact(self, tmp_path):
        seg_dir = tmp_path / "seg"
        seg_dir.mkdir()
        (seg_dir / "ACTN1_10917.tiff").touch()
        result = derive_prediction_path("ACTN1_10917.tiff", seg_dir)
        assert result == seg_dir / "ACTN1_10917.tiff"


def test_parse_args_plot(monkeypatch):
    """Verify plot subcommand parses correctly."""
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_benchmark.py", "plot", "--dtype", "uint8", "--metric", "dice"],
    )
    args = eb.parse_args()
    assert args.command == "plot"
    assert args.dtype == "uint8"
    assert args.metric == "dice"


def test_parse_args_plot_all_metrics(monkeypatch):
    """Verify plot subcommand with no --metric defaults to None (all metrics)."""
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_benchmark.py", "plot", "--dtype", "uint8"],
    )
    args = eb.parse_args()
    assert args.command == "plot"
    assert args.metric is None
