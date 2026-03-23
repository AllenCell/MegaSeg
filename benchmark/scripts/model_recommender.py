#!/usr/bin/env python
"""Model recommendation via nearest-centroid retrieval in MegaSeg feature space.

Provides utilities for:
  1. Saving/loading GAP-pooled embeddings (structure x layer)
  2. Computing per-structure centroids and nearest-centroid lookup
  3. Leave-one-structure-out cross-validation
  4. Multi-layer concatenation and comparison
  5. Unseen-structure analysis (structures not in the training set)

All distance computations use sklearn pairwise metrics (cosine or euclidean).
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Embeddings = Dict[str, Dict[str, np.ndarray]]
# {structure: {layer: (n_samples, n_features)}}

# ---------------------------------------------------------------------------
# Constants — MegaSeg-13 structure splits
# ---------------------------------------------------------------------------

MEGASEG13_SEEN = [
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

MEGASEG13_UNSEEN = ["AAVS1", "LMNB1", "NPM1"]


# ===================================================================
# Group 1 — Embedding persistence
# ===================================================================


def save_embeddings(embeddings: Embeddings, path: Path) -> None:
    """Save embeddings to ``.npz``.  Keys encoded as ``'{structure}/{layer}'``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: Dict[str, np.ndarray] = {}
    for structure, layers in embeddings.items():
        for layer, arr in layers.items():
            flat[f"{structure}/{layer}"] = arr
    np.savez_compressed(str(path), **flat)
    log.info("Saved embeddings with %d arrays to %s", len(flat), path)


def load_embeddings(path: Path) -> Embeddings:
    """Load embeddings from ``.npz``.  Raises ``FileNotFoundError`` if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    data = np.load(str(path))
    embeddings: Embeddings = {}
    for key in data.files:
        structure, layer = key.split("/", 1)
        if structure not in embeddings:
            embeddings[structure] = {}
        embeddings[structure][layer] = data[key]
    log.info("Loaded embeddings: %d structures from %s", len(embeddings), path)
    return embeddings


# ===================================================================
# Group 2 — Centroid computation
# ===================================================================


def compute_centroids(embeddings: Embeddings, layer: str) -> Dict[str, np.ndarray]:
    """Compute per-structure mean vectors for *layer*.

    Structures that do not contain *layer* are silently skipped.
    """
    centroids: Dict[str, np.ndarray] = {}
    for structure, layers in embeddings.items():
        if layer not in layers:
            log.info("Skipping %s — missing layer %s", structure, layer)
            continue
        centroids[structure] = layers[layer].mean(axis=0)
    return centroids


# ===================================================================
# Group 3 — Nearest centroid lookup
# ===================================================================


def _pairwise_distances(
    query: np.ndarray, targets: np.ndarray, metric: str
) -> np.ndarray:
    """Compute distances between a single query and multiple targets.

    Parameters
    ----------
    query : (n_features,) or (1, n_features)
    targets : (n_targets, n_features)
    metric : ``'cosine'`` or ``'euclidean'``

    Returns
    -------
    (n_targets,) distance array
    """
    q = query.reshape(1, -1)
    if metric == "cosine":
        return cosine_distances(q, targets).ravel()
    elif metric == "euclidean":
        return euclidean_distances(q, targets).ravel()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def find_nearest_centroid(
    query: np.ndarray,
    centroids: Dict[str, np.ndarray],
    metric: str = "cosine",
) -> Dict:
    """Find nearest centroid to *query*.

    Returns
    -------
    dict with keys:
        nearest : str — name of the closest structure
        distances : dict[str, float] — distance to every centroid
        ranking : list[str] — structures sorted by distance (ascending)
        confidence : float — ``(d2 - d1) / d_max`` clipped to [0, 1]
    """
    names = list(centroids.keys())
    targets = np.stack([centroids[n] for n in names])
    dists = _pairwise_distances(query, targets, metric)

    distances = {n: float(d) for n, d in zip(names, dists)}
    ranking = sorted(names, key=lambda n: distances[n])

    d1 = distances[ranking[0]]
    d2 = distances[ranking[1]] if len(ranking) > 1 else d1
    d_max = max(dists) if max(dists) > 0 else 1.0
    confidence = float(np.clip((d2 - d1) / d_max, 0.0, 1.0))

    return {
        "nearest": ranking[0],
        "distances": distances,
        "ranking": ranking,
        "confidence": confidence,
    }


# ===================================================================
# Group 4 — Leave-one-structure-out validation
# ===================================================================


def leave_one_structure_out(
    embeddings: Embeddings,
    layer: str,
    structures: Optional[List[str]] = None,
    metric: str = "cosine",
) -> Dict:
    """Leave-one-structure-out cross-validation.

    For each structure: hold out ALL its samples, build centroids from the
    remaining structures, and classify each held-out sample.

    Parameters
    ----------
    embeddings : nested dict ``{structure: {layer: array}}``
    layer : which layer to evaluate
    structures : subset of structures to use (default: all with *layer*)
    metric : ``'cosine'`` or ``'euclidean'``

    Returns
    -------
    dict with keys:
        top1_accuracy : float
        top3_accuracy : float
        per_structure : dict of per-structure stats
        confusion_matrix : np.ndarray (rows=true, cols=pred)
        structures : list[str] — ordered list used for confusion matrix
    """
    if structures is None:
        structures = [s for s in embeddings if layer in embeddings[s]]
    else:
        structures = [s for s in structures if layer in embeddings[s]]
    structures = sorted(structures)
    struct_idx = {s: i for i, s in enumerate(structures)}

    n_structs = len(structures)
    confusion = np.zeros((n_structs, n_structs), dtype=int)
    per_structure: Dict[str, Dict] = {}
    total_correct_top1 = 0
    total_correct_top3 = 0
    total_samples = 0

    for held_out in structures:
        # Build centroids for non-held-out structures from all their samples
        other_centroids = compute_centroids(
            {s: embeddings[s] for s in structures if s != held_out}, layer
        )

        samples = embeddings[held_out][layer]
        n_samples = samples.shape[0]
        top1_correct = 0
        top3_correct = 0
        nearest_wrong: Optional[str] = None
        min_wrong_dist = float("inf")

        for i in range(n_samples):
            # For the held-out structure, compute its centroid excluding
            # the current sample (leave-one-out within the structure).
            remaining = np.delete(samples, i, axis=0)
            centroids = dict(other_centroids)  # shallow copy
            if remaining.shape[0] > 0:
                centroids[held_out] = remaining.mean(axis=0)

            result = find_nearest_centroid(samples[i], centroids, metric)
            pred = result["nearest"]
            confusion[struct_idx[held_out], struct_idx[pred]] += 1

            if pred == held_out:
                top1_correct += 1
            else:
                # Track the most-confused wrong prediction
                d = result["distances"][pred]
                if d < min_wrong_dist:
                    min_wrong_dist = d
                    nearest_wrong = pred

            if held_out in result["ranking"][:3]:
                top3_correct += 1

        per_structure[held_out] = {
            "top1": top1_correct / n_samples if n_samples else 0.0,
            "top3": top3_correct / n_samples if n_samples else 0.0,
            "nearest_wrong": nearest_wrong,
            "n_samples": n_samples,
        }
        total_correct_top1 += top1_correct
        total_correct_top3 += top3_correct
        total_samples += n_samples

    top1_accuracy = total_correct_top1 / total_samples if total_samples else 0.0
    top3_accuracy = total_correct_top3 / total_samples if total_samples else 0.0

    log.info(
        "LOO validation (%s, %s): top1=%.3f  top3=%.3f  (%d structures, %d samples)",
        layer,
        metric,
        top1_accuracy,
        top3_accuracy,
        n_structs,
        total_samples,
    )

    return {
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "per_structure": per_structure,
        "confusion_matrix": confusion,
        "structures": structures,
    }


# ===================================================================
# Group 5 — Multi-layer concatenation & comparison
# ===================================================================


def concatenate_layers(embeddings: Embeddings, layers: List[str]) -> Embeddings:
    """Concatenate features from multiple layers into a ``'concatenated'`` key.

    Only structures that have **all** specified layers are included.
    """
    out: Embeddings = {}
    for structure, layer_dict in embeddings.items():
        if not all(l in layer_dict for l in layers):
            log.info("Skipping %s — missing one of %s", structure, layers)
            continue
        arrays = [layer_dict[l] for l in layers]
        out[structure] = {"concatenated": np.concatenate(arrays, axis=1)}
    return out


def compare_layer_configs(
    embeddings: Embeddings,
    configs: Dict[str, List[str]],
    structures: Optional[List[str]] = None,
    metric: str = "cosine",
) -> Dict[str, Dict]:
    """Compare leave-one-out accuracy across layer configurations.

    Parameters
    ----------
    embeddings : raw embeddings (multi-layer)
    configs : ``{config_name: [layer_names]}``
    structures : subset of structures (default: all)
    metric : ``'cosine'`` or ``'euclidean'``

    Returns
    -------
    ``{config_name: leave_one_structure_out results}``
    """
    results: Dict[str, Dict] = {}
    for name, layer_list in configs.items():
        log.info("Evaluating config '%s' with layers %s", name, layer_list)
        if len(layer_list) == 1:
            layer_key = layer_list[0]
            emb_to_use = embeddings
        else:
            emb_to_use = concatenate_layers(embeddings, layer_list)
            layer_key = "concatenated"
        results[name] = leave_one_structure_out(
            emb_to_use, layer_key, structures=structures, metric=metric
        )
    return results


# ===================================================================
# Unseen structure analysis
# ===================================================================


def analyze_unseen_structures(
    embeddings: Embeddings,
    layer: str,
    seen_structures: List[str],
    unseen_structures: List[str],
    metric: str = "cosine",
) -> Dict[str, Dict]:
    """Analyse how unseen structures map into the seen-structure feature space.

    For each unseen structure: build centroids from *seen* structures only,
    then classify every sample of the unseen structure.

    Returns
    -------
    ``{unseen_name: {nearest_seen, mean_distance, mean_distances,
    sample_predictions, consistency, prediction_counts}}``
    """
    # Build centroids from seen structures
    seen_emb = {s: embeddings[s] for s in seen_structures if s in embeddings}
    centroids = compute_centroids(seen_emb, layer)

    results: Dict[str, Dict] = {}
    for unseen in unseen_structures:
        if unseen not in embeddings or layer not in embeddings[unseen]:
            log.info("Skipping unseen %s — no data for layer %s", unseen, layer)
            continue

        samples = embeddings[unseen][layer]
        predictions: List[str] = []
        all_distances: List[Dict[str, float]] = []

        for i in range(samples.shape[0]):
            res = find_nearest_centroid(samples[i], centroids, metric)
            predictions.append(res["nearest"])
            all_distances.append(res["distances"])

        # Aggregate
        prediction_counts = dict(Counter(predictions))
        most_common = Counter(predictions).most_common(1)[0][0]

        # Mean distance to each seen structure across all samples
        mean_distances: Dict[str, float] = {}
        for s in centroids:
            mean_distances[s] = float(np.mean([d[s] for d in all_distances]))

        consistency = prediction_counts[most_common] / len(predictions)

        results[unseen] = {
            "nearest_seen": most_common,
            "mean_distance": mean_distances[most_common],
            "mean_distances": mean_distances,
            "sample_predictions": predictions,
            "consistency": consistency,
            "prediction_counts": prediction_counts,
        }
        log.info(
            "Unseen %s -> nearest seen: %s (consistency %.2f, mean_dist %.4f)",
            unseen,
            most_common,
            consistency,
            mean_distances[most_common],
        )

    return results
