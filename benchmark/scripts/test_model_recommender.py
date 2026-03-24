#!/usr/bin/env python
"""Tests for model_recommender.py."""

import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model_recommender import (
    save_embeddings,
    load_embeddings,
    compute_centroids,
    find_nearest_centroid,
    leave_one_structure_out,
    concatenate_layers,
    compare_layer_configs,
    analyze_unseen_structures,
    plot_retrieval_confusion_matrix,
    plot_layer_comparison,
    plot_per_structure_accuracy,
    MEGASEG13_SEEN,
    MEGASEG13_UNSEEN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(
    structures=("A", "B", "C"),
    layers=("layer1", "layer2"),
    n_samples=5,
    n_features=8,
    rng=None,
):
    """Build synthetic embeddings dict for testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    emb = {}
    for s in structures:
        emb[s] = {}
        for l in layers:
            emb[s][l] = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    return emb


def _make_separable_embeddings(
    structures=("A", "B", "C"),
    layer="layer1",
    n_samples=20,
    n_features=16,
    separation=10.0,
    rng=None,
):
    """Build embeddings where each structure occupies a distinct region."""
    if rng is None:
        rng = np.random.default_rng(123)
    emb = {}
    for i, s in enumerate(structures):
        center = np.zeros(n_features)
        center[i % n_features] = separation
        samples = center + rng.standard_normal((n_samples, n_features)) * 0.1
        emb[s] = {layer: samples.astype(np.float32)}
    return emb


# ===================================================================
# Group 1 — Embedding Persistence
# ===================================================================


class TestEmbeddingPersistence:
    """save_embeddings / load_embeddings roundtrip."""

    def test_roundtrip(self):
        emb = _make_embeddings(structures=("X", "Y"), layers=("L1", "L2"))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "emb.npz"
            save_embeddings(emb, p)
            loaded = load_embeddings(p)

        assert set(loaded.keys()) == {"X", "Y"}
        for s in emb:
            assert set(loaded[s].keys()) == set(emb[s].keys())
            for l in emb[s]:
                np.testing.assert_array_almost_equal(loaded[s][l], emb[s][l])

    def test_creates_parent_dirs(self):
        emb = _make_embeddings(structures=("Z",), layers=("L",), n_samples=2)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a" / "b" / "c" / "emb.npz"
            save_embeddings(emb, p)
            assert p.exists()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_embeddings(Path("/tmp/definitely_does_not_exist_42.npz"))


# ===================================================================
# Group 2 — Centroid computation
# ===================================================================


class TestCentroids:
    """compute_centroids and find_nearest_centroid."""

    def test_shape(self):
        emb = _make_embeddings(
            structures=("A", "B"), layers=("L",), n_samples=10, n_features=4
        )
        centroids = compute_centroids(emb, "L")
        assert set(centroids.keys()) == {"A", "B"}
        assert centroids["A"].shape == (4,)

    def test_mean_value(self):
        """Centroid should equal the mean of samples."""
        emb = {"S": {"L": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}}
        centroids = compute_centroids(emb, "L")
        np.testing.assert_array_almost_equal(centroids["S"], [2.0, 3.0])

    def test_skip_missing_layer(self):
        emb = {
            "A": {"L1": np.ones((3, 2))},
            "B": {"L2": np.ones((3, 2))},  # missing L1
        }
        centroids = compute_centroids(emb, "L1")
        assert "A" in centroids
        assert "B" not in centroids


# ===================================================================
# Group 3 — Nearest centroid lookup
# ===================================================================


class TestNearestCentroid:
    """find_nearest_centroid with cosine and euclidean metrics."""

    def test_cosine_exact_match(self):
        centroids = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        query = np.array([1.0, 0.0, 0.0])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        assert result["nearest"] == "A"

    def test_euclidean(self):
        centroids = {
            "A": np.array([0.0, 0.0]),
            "B": np.array([10.0, 10.0]),
        }
        query = np.array([1.0, 1.0])
        result = find_nearest_centroid(query, centroids, metric="euclidean")
        assert result["nearest"] == "A"

    def test_sorted_ranking(self):
        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "C": np.array([-1.0, 0.0]),
        }
        query = np.array([1.0, 0.1])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        # A should be nearest, ranking sorted by distance ascending
        assert result["ranking"][0] == "A"
        dists = [result["distances"][s] for s in result["ranking"]]
        assert dists == sorted(dists)

    def test_confidence_range(self):
        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
        }
        query = np.array([1.0, 0.0])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_high_for_clear_match(self):
        """Orthogonal centroids with exact query should give high confidence."""
        centroids = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
        }
        query = np.array([1.0, 0.0, 0.0])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        assert result["confidence"] > 0.5


# ===================================================================
# Group 4 — Leave-one-structure-out validation
# ===================================================================


class TestLeaveOneOut:
    """leave_one_structure_out on separable synthetic data."""

    def test_high_accuracy_separable(self):
        emb = _make_separable_embeddings()
        result = leave_one_structure_out(emb, "layer1", metric="cosine")
        assert result["top1_accuracy"] > 0.9

    def test_returns_expected_keys(self):
        emb = _make_separable_embeddings(structures=("A", "B"))
        result = leave_one_structure_out(emb, "layer1")
        expected = {
            "top1_accuracy",
            "top3_accuracy",
            "per_structure",
            "confusion_matrix",
            "structures",
        }
        assert expected == set(result.keys())

    def test_per_structure_has_all(self):
        structs = ("A", "B", "C")
        emb = _make_separable_embeddings(structures=structs)
        result = leave_one_structure_out(emb, "layer1")
        assert set(result["per_structure"].keys()) == set(structs)
        for s in structs:
            info = result["per_structure"][s]
            assert "top1" in info
            assert "top3" in info
            assert "nearest_wrong" in info
            assert "n_samples" in info

    def test_confusion_matrix_shape(self):
        structs = ("A", "B", "C", "D")
        emb = _make_separable_embeddings(structures=structs)
        result = leave_one_structure_out(emb, "layer1")
        n = len(structs)
        assert result["confusion_matrix"].shape == (n, n)

    def test_subset_structures(self):
        emb = _make_separable_embeddings(structures=("A", "B", "C"))
        result = leave_one_structure_out(emb, "layer1", structures=["A", "B"])
        assert set(result["structures"]) == {"A", "B"}


# ===================================================================
# Group 5 — Multi-layer & unseen analysis
# ===================================================================


class TestMultiLayer:
    """concatenate_layers and compare_layer_configs."""

    def test_single_layer_identity(self):
        emb = _make_embeddings(layers=("L1",), n_features=4)
        out = concatenate_layers(emb, ["L1"])
        for s in emb:
            np.testing.assert_array_almost_equal(out[s]["concatenated"], emb[s]["L1"])

    def test_two_layer_concat_shape(self):
        emb = _make_embeddings(layers=("L1", "L2"), n_samples=3, n_features=4)
        out = concatenate_layers(emb, ["L1", "L2"])
        for s in emb:
            assert out[s]["concatenated"].shape == (3, 8)

    def test_two_layer_concat_values(self):
        emb = {
            "S": {
                "L1": np.array([[1.0, 2.0]], dtype=np.float32),
                "L2": np.array([[3.0, 4.0]], dtype=np.float32),
            }
        }
        out = concatenate_layers(emb, ["L1", "L2"])
        np.testing.assert_array_almost_equal(
            out["S"]["concatenated"], [[1.0, 2.0, 3.0, 4.0]]
        )

    def test_skip_structure_missing_layer(self):
        emb = {
            "A": {"L1": np.ones((2, 3)), "L2": np.ones((2, 3))},
            "B": {"L1": np.ones((2, 3))},  # missing L2
        }
        out = concatenate_layers(emb, ["L1", "L2"])
        assert "A" in out
        assert "B" not in out

    def test_compare_layer_configs(self):
        emb = _make_separable_embeddings(structures=("A", "B", "C"), layer="L1")
        # Add a second layer so we can have two configs
        rng = np.random.default_rng(99)
        for s in emb:
            emb[s]["L2"] = (
                emb[s]["L1"]
                + rng.standard_normal(emb[s]["L1"].shape).astype(np.float32) * 0.01
            )
        configs = {"single": ["L1"], "double": ["L1", "L2"]}
        results = compare_layer_configs(emb, configs, metric="cosine")
        assert set(results.keys()) == {"single", "double"}
        for cfg in results.values():
            assert "top1_accuracy" in cfg


class TestUnseenAnalysis:
    """analyze_unseen_structures."""

    def test_nearest_seen_correct(self):
        """Unseen structure near 'A' should be classified as 'A'."""
        rng = np.random.default_rng(7)
        emb = {}
        # Seen structures well separated
        emb["A"] = {
            "L": (np.array([10.0, 0.0]) + rng.standard_normal((10, 2)) * 0.1).astype(
                np.float32
            )
        }
        emb["B"] = {
            "L": (np.array([0.0, 10.0]) + rng.standard_normal((10, 2)) * 0.1).astype(
                np.float32
            )
        }
        # Unseen near A
        emb["U"] = {
            "L": (np.array([10.0, 0.0]) + rng.standard_normal((10, 2)) * 0.1).astype(
                np.float32
            )
        }

        result = analyze_unseen_structures(
            emb, "L", seen_structures=["A", "B"], unseen_structures=["U"]
        )
        assert "U" in result
        assert result["U"]["nearest_seen"] == "A"

    def test_returns_all_distances(self):
        rng = np.random.default_rng(8)
        emb = {
            "S1": {"L": rng.standard_normal((5, 4)).astype(np.float32)},
            "S2": {"L": rng.standard_normal((5, 4)).astype(np.float32)},
            "U1": {"L": rng.standard_normal((5, 4)).astype(np.float32)},
        }
        result = analyze_unseen_structures(
            emb, "L", seen_structures=["S1", "S2"], unseen_structures=["U1"]
        )
        info = result["U1"]
        assert "mean_distances" in info
        assert set(info["mean_distances"].keys()) == {"S1", "S2"}
        assert "sample_predictions" in info
        assert len(info["sample_predictions"]) == 5
        assert "consistency" in info
        assert "prediction_counts" in info

    def test_constants_defined(self):
        assert len(MEGASEG13_SEEN) == 13
        assert len(MEGASEG13_UNSEEN) == 3
        assert "LMNB1" in MEGASEG13_UNSEEN
        assert "TOMM20" in MEGASEG13_SEEN


# ===================================================================
# Group 6 — Visualization smoke tests
# ===================================================================


class TestPlots:
    """Smoke tests for visualization functions."""

    def test_plot_retrieval_confusion_matrix_creates_file(self):
        cm = np.array([[8, 1, 1], [0, 9, 1], [2, 0, 8]])
        structs = ["A", "B", "C"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cm.png"
            plot_retrieval_confusion_matrix(cm, structs, p)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_plot_layer_comparison_creates_file(self):
        results = {
            "config_a": {"top1_accuracy": 0.85, "top3_accuracy": 0.95},
            "config_b": {"top1_accuracy": 0.70, "top3_accuracy": 0.90},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "layer_cmp.png"
            plot_layer_comparison(results, p)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_plot_per_structure_accuracy_creates_file(self):
        per_struct = {
            "A": {"top1": 0.9, "top3": 1.0},
            "B": {"top1": 0.7, "top3": 0.85},
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "per_struct.png"
            plot_per_structure_accuracy(per_struct, p)
            assert p.exists()
            assert p.stat().st_size > 0
