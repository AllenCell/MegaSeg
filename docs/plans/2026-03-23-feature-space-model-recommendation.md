# Feature Space Model Recommendation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a nearest-centroid model recommendation system using MegaSeg's `downsamples.1` (128-dim) feature space, validated by leave-one-structure-out retrieval and multi-layer comparison, with a CLI tool for new user data.

**Architecture:** Extract GAP-pooled features from MegaSeg-13's `downsamples.1` layer for 57 test samples per structure (16 structures). Compute per-structure centroids. For a new query volume, extract its features, find the nearest centroid by cosine distance, and recommend the appropriate model. Validate via leave-one-structure-out cross-validation and analyze truly unseen structures (AAVS1, LMNB1, NPM1 — not in MegaSeg-13 training).

**Tech Stack:** Python 3, PyTorch, MONAI (DynUNet), scikit-learn, UMAP, numpy, pandas, matplotlib, seaborn, tifffile.

---

## Important Context

### Repository & Worktree
- **Project root (main):** `/allen/aics/rep_learn/MegaSeg/`
- **Feature analysis worktree:** `/allen/aics/rep_learn/MegaSeg-worktrees/cross-structure-features/` (branch: `cross-structure-features`)
- **Create new branch** `feature-space-recommendation` from `cross-structure-features` for this work
- **Venv:** `/allen/aics/rep_learn/MegaSeg/benchmark/.venv/` (shared across worktrees via symlink or path)
- **uv:** `/home/ashwin.samudre/.local/bin/uv`

### Existing Code to Reuse
- `benchmark/scripts/cross_structure_features.py` (1335 lines) — contains `FeatureExtractor`, `load_dynunet()`, `load_structure_samples()`, `load_volume()`, all analysis functions. **KEY FILE.**
- The `FeatureExtractor.extract()` method returns `Dict[str, np.ndarray]` mapping layer name → 1-D GAP vector (shape `(C,)`).
- Embeddings are accumulated as `Dict[str, Dict[str, np.ndarray]]` where `embeddings[structure][layer]` has shape `(n_samples, n_features)`.
- **No embedding persistence exists** — embeddings are computed in-memory and never saved to disk. This must be added.

### Key Constants
```python
PATCH_SHAPE = (32, 128, 128)
MEGASEG_CKPT = "/allen/aics/assay-dev/users/Suraj/megaSeg/Amil/cyto-dl/logs/train/runs/megaseg_b64_deep_new_13-STRUCTS/dynUNet_st6_base_cytodl/2024-07-15_00-39-47/checkpoints/epoch_650.ckpt"
DATA_ROOT = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x")
ALL_STRUCTURES = ["AAVS1","ACTN1","ATP2A2","CTNNB1","DSP","GJA1","LMNB1","MYH10","NPM1","NUP153","RAB5A","SEC61B","SON","ST6GAL1","TJP1","TOMM20"]
EXTRACT_LAYERS = ["input_block","downsamples.0","downsamples.1","downsamples.2","upsamples.2","upsamples.1"]
# Feature dims: input_block=32, downsamples.0=64, downsamples.1=128, downsamples.2=256, upsamples.2=64, upsamples.1=128
```

### MegaSeg-13 Training Structures (13 seen)
ACTN1, ATP2A2, CTNNB1, DSP, GJA1, MYH10, NUP153, RAB5A, SEC61B, SON, ST6GAL1, TJP1, TOMM20

### Unseen Structures (3 — NOT in MegaSeg-13 training)
AAVS1, LMNB1, NPM1

### Test Pattern
- Existing test file: `benchmark/scripts/test_evaluate_benchmark.py` (233 lines)
- Uses `pytest`, `numpy`, class-based test organization
- `sys.path.insert(0, str(Path(__file__).parent))` to import sibling scripts
- Run with: `cd benchmark/scripts && python -m pytest test_model_recommender.py -v`

### SLURM
- GPU partition: `aics_gpu`, `--mem=48G`, `--gres=gpu:1`, `--constraint="titanxp|1080ti"`
- Always `set --` before `source activate`
- Logs to `/home/ashwin.samudre/logs/`

---

## Task 1: Add Embedding Persistence to `cross_structure_features.py`

**Files:**
- Modify: `benchmark/scripts/cross_structure_features.py` (after line ~1117, end of step 2)
- Test: `benchmark/scripts/test_model_recommender.py` (new file)

**Why:** The current script computes embeddings in memory but never saves them. We need to persist embeddings as `.npz` files so the recommender can load them without re-running feature extraction (which takes ~30 min on GPU for 57 samples × 16 structures).

**Step 1: Write the test for save/load roundtrip**

Create `benchmark/scripts/test_model_recommender.py`:

```python
#!/usr/bin/env python
"""Tests for model_recommender.py and embedding persistence."""

import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestEmbeddingPersistence:
    """Test save/load of embeddings dict to .npz format."""

    def _make_embeddings(self):
        """Create a minimal embeddings dict for testing."""
        rng = np.random.RandomState(42)
        return {
            "ACTN1": {
                "downsamples.1": rng.randn(5, 128).astype(np.float32),
                "input_block": rng.randn(5, 32).astype(np.float32),
            },
            "TOMM20": {
                "downsamples.1": rng.randn(5, 128).astype(np.float32),
                "input_block": rng.randn(5, 32).astype(np.float32),
            },
        }

    def test_save_load_roundtrip(self):
        from model_recommender import save_embeddings, load_embeddings

        emb = self._make_embeddings()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embeddings.npz"
            save_embeddings(emb, path)
            loaded = load_embeddings(path)

        assert set(loaded.keys()) == set(emb.keys())
        for struct in emb:
            assert set(loaded[struct].keys()) == set(emb[struct].keys())
            for layer in emb[struct]:
                np.testing.assert_array_almost_equal(
                    loaded[struct][layer], emb[struct][layer]
                )

    def test_save_creates_parent_dirs(self):
        from model_recommender import save_embeddings

        emb = self._make_embeddings()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "embeddings.npz"
            save_embeddings(emb, path)
            assert path.exists()

    def test_load_nonexistent_raises(self):
        from model_recommender import load_embeddings

        with pytest.raises(FileNotFoundError):
            load_embeddings(Path("/nonexistent/path.npz"))
```

**Step 2: Run test to verify it fails**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestEmbeddingPersistence -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_recommender'`

**Step 3: Write `save_embeddings` and `load_embeddings` in new module**

Create `benchmark/scripts/model_recommender.py`:

```python
#!/usr/bin/env python
"""
Model recommendation via nearest-centroid retrieval in MegaSeg feature space.

Uses GAP-pooled activations from MegaSeg's encoder layers (primarily
downsamples.1, 128-dim) to build per-structure centroids and recommend
the closest known structure for new query volumes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias: {structure: {layer: (n_samples, n_features)}}
Embeddings = Dict[str, Dict[str, np.ndarray]]

# ──────────────────────────────────────────────────────────
# Embedding persistence
# ──────────────────────────────────────────────────────────

def save_embeddings(embeddings: Embeddings, path: Path) -> None:
    """Save embeddings dict to a single .npz file.

    Keys are encoded as '{structure}/{layer}' -> (n_samples, n_features).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for struct, layers in embeddings.items():
        for layer, arr in layers.items():
            flat[f"{struct}/{layer}"] = arr
    np.savez_compressed(str(path), **flat)
    logger.info("Saved embeddings to %s (%d arrays)", path, len(flat))


def load_embeddings(path: Path) -> Embeddings:
    """Load embeddings dict from .npz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(str(path))
    embeddings: Embeddings = {}
    for key in data.files:
        struct, layer = key.split("/", 1)
        if struct not in embeddings:
            embeddings[struct] = {}
        embeddings[struct][layer] = data[key]
    logger.info("Loaded embeddings from %s (%d structures)", path, len(embeddings))
    return embeddings
```

**Step 4: Run test to verify it passes**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestEmbeddingPersistence -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add benchmark/scripts/model_recommender.py benchmark/scripts/test_model_recommender.py
git commit -m "feat: add embedding save/load persistence for model recommender"
```

---

## Task 2: Core Centroid Computation and Nearest-Neighbor Retrieval

**Files:**
- Modify: `benchmark/scripts/model_recommender.py`
- Test: `benchmark/scripts/test_model_recommender.py`

**Step 1: Write tests for centroid computation and retrieval**

Append to `test_model_recommender.py`:

```python
class TestCentroids:
    """Test centroid computation from embeddings."""

    def test_compute_centroids_shape(self):
        from model_recommender import compute_centroids

        rng = np.random.RandomState(42)
        embeddings = {
            "ACTN1": {"downsamples.1": rng.randn(10, 128).astype(np.float32)},
            "TOMM20": {"downsamples.1": rng.randn(10, 128).astype(np.float32)},
        }
        centroids = compute_centroids(embeddings, layer="downsamples.1")
        assert set(centroids.keys()) == {"ACTN1", "TOMM20"}
        assert centroids["ACTN1"].shape == (128,)
        assert centroids["TOMM20"].shape == (128,)

    def test_centroid_is_mean(self):
        from model_recommender import compute_centroids

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        embeddings = {"A": {"layer": arr}}
        centroids = compute_centroids(embeddings, layer="layer")
        np.testing.assert_array_almost_equal(centroids["A"], [2.0, 3.0])

    def test_centroid_skips_missing_layer(self):
        from model_recommender import compute_centroids

        embeddings = {
            "A": {"layer_x": np.ones((5, 4), dtype=np.float32)},
            "B": {"layer_y": np.ones((5, 4), dtype=np.float32)},
        }
        centroids = compute_centroids(embeddings, layer="layer_x")
        assert "A" in centroids
        assert "B" not in centroids


class TestNearestCentroid:
    """Test nearest-centroid retrieval."""

    def test_nearest_cosine_exact_match(self):
        from model_recommender import find_nearest_centroid

        centroids = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        query = np.array([0.9, 0.1, 0.0])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        assert result["nearest"] == "A"
        assert result["distances"]["A"] < result["distances"]["B"]

    def test_nearest_euclidean(self):
        from model_recommender import find_nearest_centroid

        centroids = {
            "A": np.array([0.0, 0.0]),
            "B": np.array([10.0, 10.0]),
        }
        query = np.array([1.0, 1.0])
        result = find_nearest_centroid(query, centroids, metric="euclidean")
        assert result["nearest"] == "A"

    def test_returns_sorted_ranking(self):
        from model_recommender import find_nearest_centroid

        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.5, 0.5]),
            "C": np.array([0.0, 1.0]),
        }
        query = np.array([0.8, 0.2])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        assert len(result["ranking"]) == 3
        assert result["ranking"][0] == result["nearest"]
        # Distances should be sorted ascending
        dists = [result["distances"][s] for s in result["ranking"]]
        assert dists == sorted(dists)

    def test_confidence_between_0_and_1(self):
        from model_recommender import find_nearest_centroid

        centroids = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
        }
        query = np.array([0.7, 0.3])
        result = find_nearest_centroid(query, centroids, metric="cosine")
        assert 0.0 <= result["confidence"] <= 1.0
```

**Step 2: Run tests to verify they fail**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestCentroids test_model_recommender.py::TestNearestCentroid -v`
Expected: FAIL — `ImportError`

**Step 3: Implement centroid computation and retrieval**

Append to `model_recommender.py`:

```python
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# ──────────────────────────────────────────────────────────
# Centroid computation
# ──────────────────────────────────────────────────────────

def compute_centroids(
    embeddings: Embeddings, layer: str
) -> Dict[str, np.ndarray]:
    """Compute per-structure centroids (mean vectors) for a given layer.

    Args:
        embeddings: {structure: {layer: (n_samples, n_features)}}
        layer: which layer to compute centroids for

    Returns:
        {structure: centroid_vector} where centroid_vector has shape (n_features,)
    """
    centroids = {}
    for struct, layers in embeddings.items():
        if layer not in layers:
            logger.warning("Structure %s missing layer %s, skipping", struct, layer)
            continue
        centroids[struct] = layers[layer].mean(axis=0)
    return centroids


def find_nearest_centroid(
    query: np.ndarray,
    centroids: Dict[str, np.ndarray],
    metric: str = "cosine",
) -> Dict:
    """Find the nearest structure centroid to a query feature vector.

    Args:
        query: (n_features,) feature vector
        centroids: {structure: (n_features,)} centroid vectors
        metric: "cosine" or "euclidean"

    Returns:
        dict with keys:
            nearest: str — closest structure name
            distances: {structure: float} — distance to each centroid
            ranking: List[str] — structures sorted by distance (ascending)
            confidence: float — 0-1 score (1 - normalized distance gap)
    """
    structures = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[s] for s in structures])  # (n_structs, n_features)
    query_2d = query.reshape(1, -1)  # (1, n_features)

    if metric == "cosine":
        dists = cosine_distances(query_2d, centroid_matrix).ravel()
    elif metric == "euclidean":
        dists = euclidean_distances(query_2d, centroid_matrix).ravel()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    distances = {s: float(d) for s, d in zip(structures, dists)}
    ranking = sorted(structures, key=lambda s: distances[s])
    nearest = ranking[0]

    # Confidence: ratio of gap between 1st and 2nd nearest to max distance
    if len(ranking) >= 2:
        d1 = distances[ranking[0]]
        d2 = distances[ranking[1]]
        d_max = max(dists)
        if d_max > 0:
            confidence = float(np.clip((d2 - d1) / d_max, 0.0, 1.0))
        else:
            confidence = 1.0
    else:
        confidence = 1.0

    return {
        "nearest": nearest,
        "distances": distances,
        "ranking": ranking,
        "confidence": confidence,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestCentroids test_model_recommender.py::TestNearestCentroid -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add benchmark/scripts/model_recommender.py benchmark/scripts/test_model_recommender.py
git commit -m "feat: add centroid computation and nearest-centroid retrieval"
```

---

## Task 3: Leave-One-Structure-Out Validation

**Files:**
- Modify: `benchmark/scripts/model_recommender.py`
- Test: `benchmark/scripts/test_model_recommender.py`

**Step 1: Write tests for leave-one-out validation**

Append to `test_model_recommender.py`:

```python
class TestLeaveOneOut:
    """Test leave-one-structure-out validation."""

    def _make_separable_embeddings(self, n_structs=4, n_samples=10, n_features=8):
        """Create embeddings where each structure is a tight cluster far from others."""
        rng = np.random.RandomState(42)
        names = [f"S{i}" for i in range(n_structs)]
        embeddings = {}
        for i, name in enumerate(names):
            center = np.zeros(n_features, dtype=np.float32)
            center[i % n_features] = 10.0  # well-separated
            embeddings[name] = {
                "layer": center + rng.randn(n_samples, n_features).astype(np.float32) * 0.1
            }
        return embeddings, names

    def test_perfect_separation_gives_high_accuracy(self):
        from model_recommender import leave_one_structure_out

        emb, structs = self._make_separable_embeddings()
        results = leave_one_structure_out(emb, layer="layer", structures=structs)
        assert results["top1_accuracy"] > 0.9
        assert results["top3_accuracy"] >= results["top1_accuracy"]

    def test_returns_expected_keys(self):
        from model_recommender import leave_one_structure_out

        emb, structs = self._make_separable_embeddings(n_structs=3, n_samples=5)
        results = leave_one_structure_out(emb, layer="layer", structures=structs)
        assert "top1_accuracy" in results
        assert "top3_accuracy" in results
        assert "per_structure" in results
        assert "confusion_matrix" in results
        assert "structures" in results

    def test_per_structure_has_all_structures(self):
        from model_recommender import leave_one_structure_out

        emb, structs = self._make_separable_embeddings(n_structs=4, n_samples=5)
        results = leave_one_structure_out(emb, layer="layer", structures=structs)
        assert set(results["per_structure"].keys()) == set(structs)
        for s in structs:
            assert "top1" in results["per_structure"][s]
            assert "top3" in results["per_structure"][s]
            assert "nearest_wrong" in results["per_structure"][s]

    def test_confusion_matrix_shape(self):
        from model_recommender import leave_one_structure_out

        emb, structs = self._make_separable_embeddings(n_structs=4, n_samples=5)
        results = leave_one_structure_out(emb, layer="layer", structures=structs)
        n = len(structs)
        assert results["confusion_matrix"].shape == (n, n)
```

**Step 2: Run tests to verify they fail**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestLeaveOneOut -v`
Expected: FAIL — `ImportError`

**Step 3: Implement leave-one-structure-out**

Append to `model_recommender.py`:

```python
def leave_one_structure_out(
    embeddings: Embeddings,
    layer: str,
    structures: Optional[List[str]] = None,
    metric: str = "cosine",
) -> Dict:
    """Leave-one-structure-out cross-validation for nearest-centroid retrieval.

    For each structure: hold out ALL its samples, compute centroids from
    remaining structures, classify each held-out sample by nearest centroid.

    Args:
        embeddings: {structure: {layer: (n_samples, n_features)}}
        layer: which layer to evaluate
        structures: list of structures to include (default: all in embeddings)
        metric: distance metric ("cosine" or "euclidean")

    Returns:
        dict with keys:
            top1_accuracy: float — fraction of samples correctly classified
            top3_accuracy: float — fraction where true structure is in top 3
            per_structure: {struct: {top1: float, top3: float, nearest_wrong: str|None}}
            confusion_matrix: np.ndarray (n_structs, n_structs) — rows=true, cols=predicted
            structures: List[str] — ordered structure names (index into confusion matrix)
    """
    if structures is None:
        structures = sorted(
            s for s in embeddings if layer in embeddings[s]
        )

    n_structs = len(structures)
    struct_to_idx = {s: i for i, s in enumerate(structures)}
    cm = np.zeros((n_structs, n_structs), dtype=int)
    per_structure = {}
    total_correct_1 = 0
    total_correct_3 = 0
    total_samples = 0

    for held_out in structures:
        if layer not in embeddings[held_out]:
            continue

        # Build centroids from all OTHER structures
        other_emb = {
            s: embeddings[s]
            for s in structures
            if s != held_out and layer in embeddings[s]
        }
        centroids = compute_centroids(other_emb, layer)

        # Classify each held-out sample
        samples = embeddings[held_out][layer]  # (n_samples, n_features)
        n = samples.shape[0]
        correct_1 = 0
        correct_3 = 0
        predictions = []

        for i in range(n):
            result = find_nearest_centroid(samples[i], centroids, metric=metric)
            pred = result["nearest"]
            predictions.append(pred)
            top3 = result["ranking"][:3]

            if pred == held_out:
                correct_1 += 1
            if held_out in top3:
                correct_3 += 1

            # Update confusion matrix
            pred_idx = struct_to_idx.get(pred, -1)
            true_idx = struct_to_idx[held_out]
            if pred_idx >= 0:
                cm[true_idx, pred_idx] += 1

        # Most common wrong prediction
        wrong_preds = [p for p in predictions if p != held_out]
        if wrong_preds:
            from collections import Counter
            nearest_wrong = Counter(wrong_preds).most_common(1)[0][0]
        else:
            nearest_wrong = None

        per_structure[held_out] = {
            "top1": correct_1 / n if n > 0 else 0.0,
            "top3": correct_3 / n if n > 0 else 0.0,
            "nearest_wrong": nearest_wrong,
            "n_samples": n,
        }
        total_correct_1 += correct_1
        total_correct_3 += correct_3
        total_samples += n

    return {
        "top1_accuracy": total_correct_1 / total_samples if total_samples > 0 else 0.0,
        "top3_accuracy": total_correct_3 / total_samples if total_samples > 0 else 0.0,
        "per_structure": per_structure,
        "confusion_matrix": cm,
        "structures": structures,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestLeaveOneOut -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add benchmark/scripts/model_recommender.py benchmark/scripts/test_model_recommender.py
git commit -m "feat: add leave-one-structure-out validation for retrieval accuracy"
```

---

## Task 4: Multi-Layer Comparison

**Files:**
- Modify: `benchmark/scripts/model_recommender.py`
- Test: `benchmark/scripts/test_model_recommender.py`

**Step 1: Write tests for multi-layer concatenation and comparison**

Append to `test_model_recommender.py`:

```python
class TestMultiLayer:
    """Test multi-layer feature concatenation and comparison."""

    def test_concatenate_single_layer(self):
        from model_recommender import concatenate_layers

        embeddings = {
            "A": {"L1": np.ones((5, 4), dtype=np.float32)},
        }
        result = concatenate_layers(embeddings, layers=["L1"])
        assert result["A"]["concatenated"].shape == (5, 4)

    def test_concatenate_two_layers(self):
        from model_recommender import concatenate_layers

        embeddings = {
            "A": {
                "L1": np.ones((5, 4), dtype=np.float32),
                "L2": np.zeros((5, 8), dtype=np.float32),
            },
        }
        result = concatenate_layers(embeddings, layers=["L1", "L2"])
        assert result["A"]["concatenated"].shape == (5, 12)
        # First 4 dims should be ones, last 8 should be zeros
        np.testing.assert_array_equal(result["A"]["concatenated"][:, :4], 1.0)
        np.testing.assert_array_equal(result["A"]["concatenated"][:, 4:], 0.0)

    def test_compare_layer_configs(self):
        from model_recommender import compare_layer_configs

        rng = np.random.RandomState(42)
        n_structs, n_samples = 4, 10
        names = [f"S{i}" for i in range(n_structs)]
        embeddings = {}
        for i, name in enumerate(names):
            center = np.zeros(16, dtype=np.float32)
            center[i * 4:(i + 1) * 4] = 10.0
            embeddings[name] = {
                "L1": (center[:8] + rng.randn(n_samples, 8) * 0.1).astype(np.float32),
                "L2": (center[8:] + rng.randn(n_samples, 8) * 0.1).astype(np.float32),
            }

        configs = {
            "L1_only": ["L1"],
            "L2_only": ["L2"],
            "L1+L2": ["L1", "L2"],
        }
        results = compare_layer_configs(embeddings, configs, structures=names)
        assert set(results.keys()) == {"L1_only", "L2_only", "L1+L2"}
        for config_name, r in results.items():
            assert "top1_accuracy" in r
```

**Step 2: Run tests to verify they fail**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestMultiLayer -v`
Expected: FAIL

**Step 3: Implement multi-layer functions**

Append to `model_recommender.py`:

```python
# ──────────────────────────────────────────────────────────
# Multi-layer comparison
# ──────────────────────────────────────────────────────────

def concatenate_layers(
    embeddings: Embeddings, layers: List[str]
) -> Embeddings:
    """Concatenate features from multiple layers into a single 'concatenated' layer.

    Each structure gets a new layer key 'concatenated' with shape
    (n_samples, sum_of_features_across_layers).
    Only structures that have ALL specified layers are included.

    Args:
        embeddings: {structure: {layer: (n_samples, n_features)}}
        layers: list of layer names to concatenate

    Returns:
        New embeddings dict with 'concatenated' key per structure.
    """
    result: Embeddings = {}
    for struct, layer_data in embeddings.items():
        if all(l in layer_data for l in layers):
            arrays = [layer_data[l] for l in layers]
            # Verify all have the same number of samples
            n_samples = arrays[0].shape[0]
            if all(a.shape[0] == n_samples for a in arrays):
                result[struct] = {"concatenated": np.concatenate(arrays, axis=1)}
            else:
                logger.warning(
                    "Structure %s has mismatched sample counts across layers, skipping",
                    struct,
                )
    return result


def compare_layer_configs(
    embeddings: Embeddings,
    configs: Dict[str, List[str]],
    structures: Optional[List[str]] = None,
    metric: str = "cosine",
) -> Dict[str, Dict]:
    """Compare leave-one-out retrieval accuracy across different layer configurations.

    Args:
        embeddings: {structure: {layer: (n_samples, n_features)}}
        configs: {config_name: [layer_names]} — each config is a list of layers to concatenate
        structures: structures to include (default: all)
        metric: distance metric

    Returns:
        {config_name: leave_one_structure_out results dict}
    """
    results = {}
    for config_name, layers in configs.items():
        if len(layers) == 1:
            # Single layer — use directly
            layer = layers[0]
            r = leave_one_structure_out(
                embeddings, layer=layer, structures=structures, metric=metric
            )
        else:
            # Multi-layer — concatenate first
            concat_emb = concatenate_layers(embeddings, layers)
            r = leave_one_structure_out(
                concat_emb, layer="concatenated", structures=structures, metric=metric
            )
        results[config_name] = r
        logger.info(
            "Config '%s' (%s): top1=%.3f, top3=%.3f",
            config_name, "+".join(layers), r["top1_accuracy"], r["top3_accuracy"],
        )
    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestMultiLayer -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add benchmark/scripts/model_recommender.py benchmark/scripts/test_model_recommender.py
git commit -m "feat: add multi-layer concatenation and layer config comparison"
```

---

## Task 5: Unseen Structure Analysis

**Files:**
- Modify: `benchmark/scripts/model_recommender.py`
- Test: `benchmark/scripts/test_model_recommender.py`

**Step 1: Write tests**

Append to `test_model_recommender.py`:

```python
class TestUnseenAnalysis:
    """Test unseen structure analysis (structures not in model training)."""

    def test_analyze_unseen(self):
        from model_recommender import analyze_unseen_structures

        rng = np.random.RandomState(42)
        # "Seen" structures with distinct clusters
        embeddings = {
            "SEEN_A": {"L": (np.array([10, 0, 0]) + rng.randn(5, 3) * 0.1).astype(np.float32)},
            "SEEN_B": {"L": (np.array([0, 10, 0]) + rng.randn(5, 3) * 0.1).astype(np.float32)},
            "UNSEEN_X": {"L": (np.array([9, 1, 0]) + rng.randn(5, 3) * 0.1).astype(np.float32)},
        }
        seen = ["SEEN_A", "SEEN_B"]
        unseen = ["UNSEEN_X"]

        results = analyze_unseen_structures(
            embeddings, layer="L", seen_structures=seen, unseen_structures=unseen
        )
        assert "UNSEEN_X" in results
        assert results["UNSEEN_X"]["nearest_seen"] == "SEEN_A"
        assert "mean_distance" in results["UNSEEN_X"]
        assert "sample_predictions" in results["UNSEEN_X"]

    def test_unseen_returns_all_distances(self):
        from model_recommender import analyze_unseen_structures

        rng = np.random.RandomState(42)
        embeddings = {
            "S1": {"L": rng.randn(5, 4).astype(np.float32)},
            "S2": {"L": rng.randn(5, 4).astype(np.float32)},
            "U1": {"L": rng.randn(5, 4).astype(np.float32)},
        }
        results = analyze_unseen_structures(
            embeddings, layer="L", seen_structures=["S1", "S2"], unseen_structures=["U1"]
        )
        assert set(results["U1"]["mean_distances"].keys()) == {"S1", "S2"}
```

**Step 2: Run tests to verify they fail**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestUnseenAnalysis -v`
Expected: FAIL

**Step 3: Implement unseen structure analysis**

Append to `model_recommender.py`:

```python
# MegaSeg-13 training structures (13 seen)
MEGASEG13_SEEN = [
    "ACTN1", "ATP2A2", "CTNNB1", "DSP", "GJA1", "MYH10",
    "NUP153", "RAB5A", "SEC61B", "SON", "ST6GAL1", "TJP1", "TOMM20",
]
# Structures NOT in MegaSeg-13 training (3 unseen)
MEGASEG13_UNSEEN = ["AAVS1", "LMNB1", "NPM1"]


def analyze_unseen_structures(
    embeddings: Embeddings,
    layer: str,
    seen_structures: List[str],
    unseen_structures: List[str],
    metric: str = "cosine",
) -> Dict[str, Dict]:
    """Analyze where unseen structures land relative to seen structure centroids.

    For each unseen structure, compute the centroid of seen structures, then
    classify each unseen sample and report statistics.

    Args:
        embeddings: full embeddings dict
        layer: layer to use
        seen_structures: structures that were in model training
        unseen_structures: structures NOT in model training
        metric: distance metric

    Returns:
        {unseen_struct: {
            nearest_seen: str,
            mean_distance: float,
            mean_distances: {seen_struct: float},
            sample_predictions: List[str],
            consistency: float  (fraction of samples predicting the majority class)
        }}
    """
    # Build centroids from seen structures only
    seen_emb = {s: embeddings[s] for s in seen_structures if s in embeddings}
    centroids = compute_centroids(seen_emb, layer)

    results = {}
    for unseen in unseen_structures:
        if unseen not in embeddings or layer not in embeddings[unseen]:
            logger.warning("Unseen structure %s has no embeddings for layer %s", unseen, layer)
            continue

        samples = embeddings[unseen][layer]
        n = samples.shape[0]
        predictions = []
        all_distances = {s: [] for s in centroids}

        for i in range(n):
            r = find_nearest_centroid(samples[i], centroids, metric=metric)
            predictions.append(r["nearest"])
            for s, d in r["distances"].items():
                all_distances[s].append(d)

        from collections import Counter
        pred_counts = Counter(predictions)
        majority_class = pred_counts.most_common(1)[0][0]
        consistency = pred_counts[majority_class] / n

        mean_distances = {s: float(np.mean(ds)) for s, ds in all_distances.items()}
        nearest_seen = min(mean_distances, key=mean_distances.get)

        results[unseen] = {
            "nearest_seen": nearest_seen,
            "mean_distance": mean_distances[nearest_seen],
            "mean_distances": mean_distances,
            "sample_predictions": predictions,
            "consistency": consistency,
            "prediction_counts": dict(pred_counts),
        }

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestUnseenAnalysis -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add benchmark/scripts/model_recommender.py benchmark/scripts/test_model_recommender.py
git commit -m "feat: add unseen structure analysis against seen centroids"
```

---

## Task 6: Visualization — UMAP with Query Overlay and Retrieval Figures

**Files:**
- Modify: `benchmark/scripts/model_recommender.py`
- No unit tests for plotting functions (visual outputs), but include a smoke test.

**Step 1: Write smoke test for plot functions**

Append to `test_model_recommender.py`:

```python
class TestPlots:
    """Smoke tests for visualization functions."""

    def test_plot_retrieval_confusion_matrix_creates_file(self):
        from model_recommender import plot_retrieval_confusion_matrix

        cm = np.array([[8, 1, 1], [0, 9, 1], [2, 0, 8]])
        structures = ["A", "B", "C"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cm.png"
            plot_retrieval_confusion_matrix(cm, structures, out)
            assert out.exists()

    def test_plot_layer_comparison_creates_file(self):
        from model_recommender import plot_layer_comparison

        results = {
            "L1": {"top1_accuracy": 0.8, "top3_accuracy": 0.95},
            "L1+L2": {"top1_accuracy": 0.85, "top3_accuracy": 0.97},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "comparison.png"
            plot_layer_comparison(results, out)
            assert out.exists()
```

**Step 2: Run tests to verify they fail**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestPlots -v`
Expected: FAIL

**Step 3: Implement visualization functions**

Append to `model_recommender.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────

def plot_retrieval_confusion_matrix(
    cm: np.ndarray, structures: List[str], out_path: Path, title: str = "Retrieval Confusion Matrix"
) -> None:
    """Plot row-normalized confusion matrix for retrieval results."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=structures, yticklabels=structures, ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted (Nearest Centroid)")
    ax.set_ylabel("True Structure")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def plot_layer_comparison(
    results: Dict[str, Dict], out_path: Path, title: str = "Layer Configuration Comparison"
) -> None:
    """Bar chart comparing top-1 and top-3 accuracy across layer configs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = list(results.keys())
    top1 = [results[c]["top1_accuracy"] for c in configs]
    top3 = [results[c]["top3_accuracy"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, top1, width, label="Top-1 Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, top3, width, label="Top-3 Accuracy", color="#4CAF50")

    ax.set_xlabel("Layer Configuration")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved layer comparison to %s", out_path)


def plot_umap_with_query(
    embeddings: Embeddings,
    layer: str,
    structures: List[str],
    query_features: Optional[np.ndarray] = None,
    query_label: str = "Query",
    out_path: Optional[Path] = None,
    seen_structures: Optional[List[str]] = None,
    unseen_structures: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    """UMAP plot of all structure embeddings with optional query point overlay.

    Seen structures shown as filled circles, unseen as open triangles,
    query as a star marker.
    """
    import umap

    # Collect all feature vectors and labels
    all_features = []
    all_labels = []
    for s in structures:
        if s in embeddings and layer in embeddings[s]:
            arr = embeddings[s][layer]
            all_features.append(arr)
            all_labels.extend([s] * arr.shape[0])
    if not all_features:
        logger.warning("No features to plot")
        return

    X = np.vstack(all_features)
    n_reference = X.shape[0]

    # If query provided, append it for joint UMAP projection
    if query_features is not None:
        X = np.vstack([X, query_features.reshape(1, -1)])
        all_labels.append(query_label)

    effective_neighbors = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=effective_neighbors,
        min_dist=min_dist, random_state=42,
    )
    coords = reducer.fit_transform(X)

    # Determine seen/unseen sets
    if seen_structures is None:
        seen_structures = structures
    if unseen_structures is None:
        unseen_structures = []
    seen_set = set(seen_structures)
    unseen_set = set(unseen_structures)

    # Build color palette
    unique_structs = sorted(set(s for s in structures if s in embeddings))
    palette = dict(zip(unique_structs, sns.color_palette("husl", len(unique_structs))))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot reference points
    for s in unique_structs:
        mask = np.array([l == s for l in all_labels[:n_reference]])
        if not mask.any():
            continue
        c = coords[:n_reference][mask]
        marker = "o" if s in seen_set else "^"
        facecolors = palette[s] if s in seen_set else "none"
        ax.scatter(c[:, 0], c[:, 1], c=np.array([palette[s]]),
                   marker=marker, s=60, label=s, alpha=0.7,
                   edgecolors=palette[s],
                   facecolors=[facecolors] if isinstance(facecolors, str) and facecolors == "none" else None)

    # Plot query point
    if query_features is not None:
        ax.scatter(coords[-1, 0], coords[-1, 1], c="red", marker="*",
                   s=300, zorder=10, label=query_label, edgecolors="black", linewidths=1)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
    ax.set_title(f"UMAP — {layer} (128-dim GAP features)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        logger.info("Saved UMAP plot to %s", out_path)
    plt.close(fig)


def plot_per_structure_accuracy(
    per_structure: Dict[str, Dict],
    out_path: Path,
    title: str = "Per-Structure Retrieval Accuracy",
) -> None:
    """Horizontal bar chart of per-structure top-1 and top-3 accuracy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    structs = sorted(per_structure.keys())
    top1 = [per_structure[s]["top1"] for s in structs]
    top3 = [per_structure[s]["top3"] for s in structs]

    y = np.arange(len(structs))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y - height / 2, top1, height, label="Top-1", color="#2196F3")
    ax.barh(y + height / 2, top3, height, label="Top-3", color="#4CAF50")
    ax.set_yticks(y)
    ax.set_yticklabels(structs)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-structure accuracy to %s", out_path)
```

**Step 4: Run tests to verify they pass**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py::TestPlots -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add benchmark/scripts/model_recommender.py benchmark/scripts/test_model_recommender.py
git commit -m "feat: add retrieval visualization (confusion matrix, layer comparison, UMAP overlay)"
```

---

## Task 7: CLI Main Pipeline

**Files:**
- Modify: `benchmark/scripts/model_recommender.py`

**Step 1: Write the `main()` function with argparse**

Append to `model_recommender.py`:

```python
import argparse
import json
import pandas as pd
import sys

# Import from cross_structure_features for model loading and feature extraction
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="MegaSeg feature-space model recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --build-centroids    Extract features from test data, compute centroids, save all artifacts
  --recommend          Recommend model for a query volume
  --validate           Run leave-one-structure-out validation
  --compare-layers     Compare retrieval accuracy across layer configurations
        """,
    )
    # Mode flags
    parser.add_argument("--build-centroids", action="store_true",
                        help="Build centroids from test data (requires GPU)")
    parser.add_argument("--validate", action="store_true",
                        help="Run leave-one-structure-out validation")
    parser.add_argument("--compare-layers", action="store_true",
                        help="Compare single-layer vs multi-layer retrieval")
    parser.add_argument("--recommend", type=str, default=None, metavar="VOLUME.tiff",
                        help="Path to query volume for model recommendation")

    # Shared options
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to embeddings .npz (load instead of extracting)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (default: MegaSeg-13)")
    parser.add_argument("--layer", type=str, default="downsamples.1",
                        help="Primary layer for retrieval (default: downsamples.1)")
    parser.add_argument("--n-samples", type=int, default=57,
                        help="Samples per structure for centroid building (default: 57)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                        help="Distance metric (default: cosine)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: benchmark results dir)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve paths
    from cross_structure_features import (
        load_dynunet, FeatureExtractor, load_structure_samples,
        ALL_STRUCTURES, EXTRACT_LAYERS, MEGASEG_CKPT, RESULTS_DIR,
        run_umap,
    )

    ckpt = args.checkpoint or MEGASEG_CKPT
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "model_recommendation"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or extract embeddings ──────────────────────
    if args.embeddings:
        logger.info("Loading embeddings from %s", args.embeddings)
        embeddings = load_embeddings(Path(args.embeddings))
    elif args.build_centroids or args.validate or args.compare_layers:
        logger.info("Extracting features from %d structures, %d samples each",
                     len(ALL_STRUCTURES), args.n_samples)
        model = load_dynunet(ckpt, args.device)
        extractor = FeatureExtractor(model, EXTRACT_LAYERS)

        embeddings = {}
        for struct in ALL_STRUCTURES:
            logger.info("Loading %s...", struct)
            samples = load_structure_samples(struct, n_samples=args.n_samples)
            if len(samples) < 2:
                logger.warning("Skipping %s (only %d samples)", struct, len(samples))
                continue
            feats_list = {layer: [] for layer in EXTRACT_LAYERS}
            for sample_id, volume in samples:
                feat = extractor.extract(volume, device=args.device)
                for layer in EXTRACT_LAYERS:
                    if layer in feat:
                        feats_list[layer].append(feat[layer])
            embeddings[struct] = {
                layer: np.stack(feats_list[layer]) for layer in EXTRACT_LAYERS
                if feats_list[layer]
            }
            logger.info("  %s: %d samples extracted", struct, len(samples))

        extractor.remove_hooks()
        del model
        if "torch" in sys.modules:
            import torch
            torch.cuda.empty_cache()

        # Save embeddings
        emb_path = out_dir / "embeddings.npz"
        save_embeddings(embeddings, emb_path)
        logger.info("Saved embeddings to %s", emb_path)
    else:
        embeddings = None

    # ── Build centroids mode ──────────────────────────
    if args.build_centroids and embeddings:
        centroids = compute_centroids(embeddings, args.layer)
        # Save centroids
        centroid_path = out_dir / "centroids.npz"
        np.savez_compressed(
            str(centroid_path),
            **{s: v for s, v in centroids.items()},
            _metadata=np.array([args.layer]),
        )
        logger.info("Saved %d centroids to %s", len(centroids), centroid_path)

        # UMAP of full reference set
        plot_umap_with_query(
            embeddings, args.layer, ALL_STRUCTURES,
            seen_structures=MEGASEG13_SEEN, unseen_structures=MEGASEG13_UNSEEN,
            out_path=figures_dir / "umap_reference_seen_unseen.png",
        )

    # ── Validate mode ─────────────────────────────────
    if args.validate and embeddings:
        logger.info("Running leave-one-structure-out validation on layer=%s", args.layer)

        # Validate on ALL 16 structures
        val_all = leave_one_structure_out(
            embeddings, args.layer, structures=ALL_STRUCTURES, metric=args.metric,
        )
        logger.info("ALL 16 structures: top1=%.3f, top3=%.3f",
                     val_all["top1_accuracy"], val_all["top3_accuracy"])

        # Validate on SEEN 13 only
        val_seen = leave_one_structure_out(
            embeddings, args.layer, structures=MEGASEG13_SEEN, metric=args.metric,
        )
        logger.info("SEEN 13 structures: top1=%.3f, top3=%.3f",
                     val_seen["top1_accuracy"], val_seen["top3_accuracy"])

        # Save results
        val_results = {
            "all_16": {
                "top1": val_all["top1_accuracy"],
                "top3": val_all["top3_accuracy"],
                "per_structure": {
                    s: {k: v for k, v in d.items()}
                    for s, d in val_all["per_structure"].items()
                },
            },
            "seen_13": {
                "top1": val_seen["top1_accuracy"],
                "top3": val_seen["top3_accuracy"],
                "per_structure": {
                    s: {k: v for k, v in d.items()}
                    for s, d in val_seen["per_structure"].items()
                },
            },
        }
        with open(out_dir / "validation_results.json", "w") as f:
            json.dump(val_results, f, indent=2, default=str)

        # Plots
        plot_retrieval_confusion_matrix(
            val_all["confusion_matrix"], val_all["structures"],
            figures_dir / "retrieval_cm_all16.png",
            title="Retrieval CM — All 16 Structures",
        )
        plot_retrieval_confusion_matrix(
            val_seen["confusion_matrix"], val_seen["structures"],
            figures_dir / "retrieval_cm_seen13.png",
            title="Retrieval CM — 13 Seen Structures",
        )
        plot_per_structure_accuracy(
            val_all["per_structure"],
            figures_dir / "per_structure_accuracy_all16.png",
            title="Per-Structure Retrieval — All 16",
        )

        # Unseen analysis
        unseen_results = analyze_unseen_structures(
            embeddings, args.layer,
            seen_structures=MEGASEG13_SEEN,
            unseen_structures=MEGASEG13_UNSEEN,
            metric=args.metric,
        )
        with open(out_dir / "unseen_analysis.json", "w") as f:
            json.dump(unseen_results, f, indent=2, default=str)

        # Print unseen summary
        for unseen, info in unseen_results.items():
            logger.info(
                "UNSEEN %s -> nearest seen: %s (dist=%.4f, consistency=%.1f%%)",
                unseen, info["nearest_seen"], info["mean_distance"],
                info["consistency"] * 100,
            )

    # ── Compare layers mode ───────────────────────────
    if args.compare_layers and embeddings:
        configs = {
            "downsamples.1 (128d)": ["downsamples.1"],
            "downsamples.0 (64d)": ["downsamples.0"],
            "input_block (32d)": ["input_block"],
            "downsamples.2 (256d)": ["downsamples.2"],
            "upsamples.2 (64d)": ["upsamples.2"],
            "upsamples.1 (128d)": ["upsamples.1"],
            "d0+d1 (192d)": ["downsamples.0", "downsamples.1"],
            "d0+d1+d2 (448d)": ["downsamples.0", "downsamples.1", "downsamples.2"],
            "all_healthy (416d)": ["input_block", "downsamples.0", "downsamples.1", "upsamples.2", "upsamples.1"],
        }
        layer_results = compare_layer_configs(
            embeddings, configs, structures=ALL_STRUCTURES, metric=args.metric,
        )

        # Save
        summary_rows = []
        for config_name, r in layer_results.items():
            summary_rows.append({
                "config": config_name,
                "top1_accuracy": r["top1_accuracy"],
                "top3_accuracy": r["top3_accuracy"],
            })
        df = pd.DataFrame(summary_rows).sort_values("top1_accuracy", ascending=False)
        df.to_csv(out_dir / "layer_comparison.csv", index=False)
        logger.info("\nLayer comparison:\n%s", df.to_string(index=False))

        plot_layer_comparison(layer_results, figures_dir / "layer_comparison.png")

    # ── Recommend mode ────────────────────────────────
    if args.recommend:
        from cross_structure_features import load_volume

        if embeddings is None and args.embeddings:
            embeddings = load_embeddings(Path(args.embeddings))
        if embeddings is None:
            logger.error("Need --embeddings or --build-centroids to provide reference data")
            sys.exit(1)

        centroids = compute_centroids(embeddings, args.layer)

        # Load query volume and extract features
        logger.info("Loading query volume: %s", args.recommend)
        query_vol = load_volume(args.recommend)

        model = load_dynunet(ckpt, args.device)
        extractor = FeatureExtractor(model, [args.layer])
        query_feat = extractor.extract(query_vol, device=args.device)
        extractor.remove_hooks()

        query_vector = query_feat[args.layer]
        result = find_nearest_centroid(query_vector, centroids, metric=args.metric)

        print(f"\n{'='*60}")
        print(f"  Model Recommendation for: {args.recommend}")
        print(f"{'='*60}")
        print(f"  Nearest structure:  {result['nearest']}")
        print(f"  Confidence:         {result['confidence']:.3f}")
        print(f"  Distance:           {result['distances'][result['nearest']]:.4f}")
        print(f"\n  Full ranking:")
        for i, s in enumerate(result['ranking'][:5], 1):
            print(f"    {i}. {s:12s}  (dist={result['distances'][s]:.4f})")
        print(f"{'='*60}\n")

        # UMAP with query overlay
        plot_umap_with_query(
            embeddings, args.layer, ALL_STRUCTURES,
            query_features=query_vector, query_label="Query",
            seen_structures=MEGASEG13_SEEN, unseen_structures=MEGASEG13_UNSEEN,
            out_path=figures_dir / "umap_query_recommendation.png",
        )
        logger.info("Recommendation complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Run all tests to make sure nothing broke**

Run: `cd benchmark/scripts && python -m pytest test_model_recommender.py -v`
Expected: All PASSED

**Step 3: Commit**

```bash
git add benchmark/scripts/model_recommender.py
git commit -m "feat: add CLI main pipeline with build/validate/compare/recommend modes"
```

---

## Task 8: Integrate Embedding Save into `cross_structure_features.py`

**Files:**
- Modify: `benchmark/scripts/cross_structure_features.py` (after line ~1117)

**Step 1: Add embedding save after feature extraction (step 2 of existing pipeline)**

In `cross_structure_features.py`, after the feature extraction loop (around line 1117), add:

```python
    # Save embeddings for downstream use (model recommendation)
    from model_recommender import save_embeddings
    emb_path = out_dir / "embeddings.npz"
    save_embeddings(embeddings, emb_path)
    logger.info("Saved embeddings to %s", emb_path)
```

This ensures that any run of the cross-structure analysis also persists embeddings for the recommender to load.

**Step 2: Verify cross_structure_features.py still imports cleanly**

Run: `cd benchmark/scripts && python -c "import cross_structure_features; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add benchmark/scripts/cross_structure_features.py
git commit -m "feat: persist embeddings to .npz after feature extraction"
```

---

## Task 9: SLURM Script and Full Pipeline Run

**Files:**
- Create: `benchmark/scripts/sbatch_model_recommender.sh`

**Step 1: Write SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=model-rec
#SBATCH --output=/home/ashwin.samudre/logs/%j.out
#SBATCH --error=/home/ashwin.samudre/logs/%j.err
#SBATCH --partition=aics_gpu
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="titanxp|1080ti"
#SBATCH --time=02:00:00

set --
source /allen/aics/rep_learn/MegaSeg/benchmark/.venv/bin/activate

SCRIPT_DIR="$(dirname "$0")"

echo "=== Step 1: Build centroids (extract 57 samples x 16 structures) ==="
python "$SCRIPT_DIR/model_recommender.py" \
    --build-centroids \
    --n-samples 57 \
    --device cuda

echo ""
echo "=== Step 2: Leave-one-structure-out validation ==="
python "$SCRIPT_DIR/model_recommender.py" \
    --validate \
    --embeddings "$SCRIPT_DIR/../results/cross_structure_features/model_recommendation/embeddings.npz" \
    --device cpu

echo ""
echo "=== Step 3: Multi-layer comparison ==="
python "$SCRIPT_DIR/model_recommender.py" \
    --compare-layers \
    --embeddings "$SCRIPT_DIR/../results/cross_structure_features/model_recommendation/embeddings.npz" \
    --device cpu

echo ""
echo "=== All done ==="
```

**Step 2: Make it executable and commit**

```bash
chmod +x benchmark/scripts/sbatch_model_recommender.sh
git add benchmark/scripts/sbatch_model_recommender.sh
git commit -m "feat: add SLURM script for model recommendation pipeline"
```

**Step 3: Submit the job**

```bash
sbatch benchmark/scripts/sbatch_model_recommender.sh
```

**Step 4: Monitor and verify output**

Check with `squeue -u ashwin.samudre` and inspect log output.

Expected outputs in `benchmark/results/cross_structure_features/model_recommendation/`:
- `embeddings.npz` — persisted embeddings (57 samples × 16 structures × 6 layers)
- `centroids.npz` — per-structure centroids for `downsamples.1`
- `validation_results.json` — leave-one-out accuracy (all 16 + seen 13)
- `unseen_analysis.json` — where AAVS1, LMNB1, NPM1 land
- `layer_comparison.csv` — accuracy table across 9 layer configs
- `figures/retrieval_cm_all16.png` — confusion matrix
- `figures/retrieval_cm_seen13.png` — confusion matrix (seen only)
- `figures/per_structure_accuracy_all16.png` — per-structure bar chart
- `figures/umap_reference_seen_unseen.png` — UMAP with seen/unseen distinction
- `figures/layer_comparison.png` — layer config comparison bar chart

**Step 5: Commit results**

```bash
git add benchmark/results/cross_structure_features/model_recommendation/
git commit -m "results: model recommendation validation — retrieval accuracy and layer comparison"
```

---

## Task 10: Create PR

**Step 1: Push branch and create PR**

```bash
git push -u origin feature-space-recommendation
gh pr create \
    --title "Feature space model recommendation system" \
    --body "$(cat <<'EOF'
## Summary
- Nearest-centroid model recommendation using MegaSeg's downsamples.1 (128-dim) feature space
- Leave-one-structure-out validation (16 structures, 57 samples each)
- Multi-layer comparison: single vs. concatenated layer retrieval accuracy
- Unseen structure analysis: AAVS1, LMNB1, NPM1 (not in MegaSeg-13 training)
- CLI tool with --build-centroids, --validate, --compare-layers, --recommend modes

## Key Results
- [FILL IN after SLURM job completes]
- Top-1 retrieval accuracy: X.X%
- Top-3 retrieval accuracy: X.X%

Closes #[TBD]
EOF
)"
```

---

## Expected Timeline

| Task | Estimated Time | GPU? |
|------|:-:|:-:|
| Task 1: Embedding persistence | 10 min | No |
| Task 2: Centroid + retrieval | 15 min | No |
| Task 3: Leave-one-out | 15 min | No |
| Task 4: Multi-layer comparison | 10 min | No |
| Task 5: Unseen analysis | 10 min | No |
| Task 6: Visualizations | 15 min | No |
| Task 7: CLI main pipeline | 15 min | No |
| Task 8: Integrate into cross_structure_features.py | 5 min | No |
| Task 9: SLURM run | 60-90 min (job) | Yes |
| Task 10: PR | 5 min | No |
| **Total** | **~2.5-3 hours** | |
