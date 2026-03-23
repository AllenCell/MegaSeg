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

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

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


# ===================================================================
# Group 6 — Visualization functions
# ===================================================================


def plot_retrieval_confusion_matrix(
    cm: np.ndarray,
    structures: List[str],
    out_path: Path,
    title: str = "Retrieval Confusion Matrix",
) -> None:
    """Row-normalized heatmap of the retrieval confusion matrix."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        xticklabels=structures,
        yticklabels=structures,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted (Nearest Centroid)")
    ax.set_ylabel("True Structure")
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved retrieval confusion matrix: %s", out_path)


def plot_layer_comparison(
    results: Dict[str, Dict],
    out_path: Path,
    title: str = "Layer Configuration Comparison",
) -> None:
    """Grouped bar chart comparing top-1 and top-3 accuracy across layer configs."""
    config_names = list(results.keys())
    top1_vals = [results[c]["top1_accuracy"] for c in config_names]
    top3_vals = [results[c]["top3_accuracy"] for c in config_names]

    x = np.arange(len(config_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, top1_vals, width, label="Top-1", color="#2196F3")
    bars2 = ax.bar(x + width / 2, top3_vals, width, label="Top-3", color="#4CAF50")

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved layer comparison plot: %s", out_path)


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
    """UMAP scatter of all structure embeddings with optional query overlay."""
    import umap as umap_lib

    all_feats = []
    labels = []
    for struct in structures:
        if struct in embeddings and layer in embeddings[struct]:
            feats = embeddings[struct][layer]
            all_feats.append(feats)
            labels.extend([struct] * feats.shape[0])

    if query_features is not None:
        q = query_features.reshape(1, -1)
        all_feats.append(q)
        labels.append(query_label)

    X = np.vstack(all_feats)
    n_neighbors_eff = min(n_neighbors, X.shape[0] - 1)
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        random_state=42,
        metric="cosine",
    )
    coords = reducer.fit_transform(X)

    unique_structs = sorted(set(labels) - {query_label})
    n_colors = len(unique_structs)
    palette = sns.color_palette("husl", n_colors)
    color_map = {s: palette[i] for i, s in enumerate(unique_structs)}

    fig, ax = plt.subplots(figsize=(12, 10))

    for struct in unique_structs:
        mask = np.array([l == struct for l in labels])
        is_unseen = unseen_structures is not None and struct in unseen_structures

        if is_unseen:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=color_map[struct],
                label=struct,
                s=60,
                alpha=0.8,
                marker="^",
                facecolors="none",
                edgecolors=color_map[struct],
                linewidth=1.5,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=color_map[struct],
                label=struct,
                s=40,
                alpha=0.8,
            )

    # Plot query point
    if query_features is not None:
        q_mask = np.array([l == query_label for l in labels])
        ax.scatter(
            coords[q_mask, 0],
            coords[q_mask, 1],
            color="red",
            marker="*",
            s=300,
            label=query_label,
            zorder=10,
            edgecolors="k",
            linewidth=0.5,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_title(f"UMAP: {layer}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        log.info("Saved UMAP with query plot: %s", out_path)
    plt.close(fig)


def plot_per_structure_accuracy(
    per_structure: Dict[str, Dict],
    out_path: Path,
    title: str = "Per-Structure Retrieval Accuracy",
) -> None:
    """Horizontal bar chart of top-1 and top-3 per structure."""
    structs = sorted(per_structure.keys())
    top1_vals = [per_structure[s]["top1"] for s in structs]
    top3_vals = [per_structure[s]["top3"] for s in structs]

    y = np.arange(len(structs))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(6, len(structs) * 0.4)))
    ax.barh(y - height / 2, top1_vals, height, label="Top-1", color="#2196F3")
    ax.barh(y + height / 2, top3_vals, height, label="Top-3", color="#4CAF50")

    ax.set_yticks(y)
    ax.set_yticklabels(structs)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved per-structure accuracy plot: %s", out_path)


# ===================================================================
# CLI — Main pipeline
# ===================================================================


def main() -> None:
    """CLI entry point for model recommendation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Runtime imports from cross_structure_features (heavy deps)
    from cross_structure_features import (
        load_dynunet,
        FeatureExtractor,
        load_structure_samples,
        ALL_STRUCTURES,
        EXTRACT_LAYERS,
        MEGASEG_CKPT,
        RESULTS_DIR,
        load_volume,
    )

    parser = argparse.ArgumentParser(
        description="Model recommendation via nearest-centroid retrieval in MegaSeg feature space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode flags
    parser.add_argument(
        "--build-centroids",
        action="store_true",
        help="Extract features, compute centroids, save embeddings.npz and centroids.npz",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run leave-one-out validation on all-16 and seen-13 structures",
    )
    parser.add_argument(
        "--compare-layers",
        action="store_true",
        help="Compare 9 layer configurations and save results",
    )
    parser.add_argument(
        "--recommend",
        type=str,
        default=None,
        metavar="VOLUME.tiff",
        help="Recommend nearest structure for a query volume",
    )

    # Shared arguments
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        metavar="PATH",
        help="Load pre-extracted embeddings instead of extracting",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MEGASEG_CKPT,
        metavar="PATH",
        help="Model checkpoint (default: MEGASEG_CKPT)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="downsamples.1",
        help='Primary layer (default: "downsamples.1")',
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=57,
        help="Samples per structure (default: 57)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric (default: cosine)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Output directory (default: RESULTS_DIR/model_recommendation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Compute device (default: "cuda")',
    )

    args = parser.parse_args()

    # Resolve output directory
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else RESULTS_DIR / "model_recommendation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load or extract embeddings ----
    def get_embeddings() -> Embeddings:
        if args.embeddings:
            log.info("Loading pre-extracted embeddings from %s", args.embeddings)
            return load_embeddings(args.embeddings)

        log.info(
            "Extracting features from %d structures (%d samples each)...",
            len(ALL_STRUCTURES),
            args.n_samples,
        )
        model = load_dynunet(args.checkpoint, device=args.device)
        extractor = FeatureExtractor(model, EXTRACT_LAYERS)

        embeddings: Embeddings = {}
        for struct in ALL_STRUCTURES:
            log.info("  Extracting %s ...", struct)
            try:
                samples = load_structure_samples(struct, n_samples=args.n_samples)
            except FileNotFoundError as e:
                log.warning("  Skipping %s: %s", struct, e)
                continue

            layer_feats: Dict[str, List[np.ndarray]] = {l: [] for l in EXTRACT_LAYERS}
            for sid, vol in samples:
                feats = extractor.extract(vol, device=args.device)
                for ln, gap in feats.items():
                    layer_feats[ln].append(gap)

            struct_emb = {}
            for ln in EXTRACT_LAYERS:
                if layer_feats[ln]:
                    struct_emb[ln] = np.stack(layer_feats[ln])
            if struct_emb:
                embeddings[struct] = struct_emb
                log.info("  %s: %d samples", struct, len(samples))

        extractor.remove_hooks()
        return embeddings

    # ---- Mode: --build-centroids ----
    if args.build_centroids:
        log.info("=" * 60)
        log.info("  BUILD CENTROIDS")
        log.info("=" * 60)

        emb = get_embeddings()
        emb_path = out_dir / "embeddings.npz"
        save_embeddings(emb, emb_path)

        centroids = compute_centroids(emb, args.layer)
        centroid_emb: Embeddings = {
            s: {args.layer: c.reshape(1, -1)} for s, c in centroids.items()
        }
        centroid_path = out_dir / "centroids.npz"
        save_embeddings(centroid_emb, centroid_path)
        log.info("Saved centroids to %s", centroid_path)

        # Generate UMAP of reference set
        try:
            structures = sorted(emb.keys())
            plot_umap_with_query(
                emb,
                args.layer,
                structures,
                out_path=out_dir / "umap_reference_set.png",
                seen_structures=MEGASEG13_SEEN,
                unseen_structures=MEGASEG13_UNSEEN,
            )
        except Exception as e:
            log.warning("UMAP generation failed: %s", e)

        log.info("Build centroids complete. Output: %s", out_dir)

    # ---- Mode: --validate ----
    elif args.validate:
        log.info("=" * 60)
        log.info("  VALIDATE (Leave-One-Out)")
        log.info("=" * 60)

        emb = get_embeddings()

        # All 16 structures
        all_structs = sorted(emb.keys())
        log.info("Running LOO on all %d structures...", len(all_structs))
        result_all = leave_one_structure_out(
            emb, args.layer, structures=all_structs, metric=args.metric
        )

        # Seen 13 only
        seen = [s for s in MEGASEG13_SEEN if s in emb]
        log.info("Running LOO on %d seen structures...", len(seen))
        result_seen = leave_one_structure_out(
            emb, args.layer, structures=seen, metric=args.metric
        )

        # Save validation_results.json
        def _serialize_per_struct(ps):
            return {
                s: {
                    "top1": v["top1"],
                    "top3": v["top3"],
                    "nearest_wrong": v["nearest_wrong"],
                    "n_samples": v["n_samples"],
                }
                for s, v in ps.items()
            }

        validation_results = {
            "all_16": {
                "top1": result_all["top1_accuracy"],
                "top3": result_all["top3_accuracy"],
                "per_structure": _serialize_per_struct(result_all["per_structure"]),
            },
            "seen_13": {
                "top1": result_seen["top1_accuracy"],
                "top3": result_seen["top3_accuracy"],
                "per_structure": _serialize_per_struct(result_seen["per_structure"]),
            },
        }
        val_path = out_dir / "validation_results.json"
        val_path.write_text(json.dumps(validation_results, indent=2))
        log.info("Saved validation results: %s", val_path)

        # Unseen analysis
        unseen = [s for s in MEGASEG13_UNSEEN if s in emb]
        if unseen:
            unseen_results = analyze_unseen_structures(
                emb, args.layer, seen, unseen, metric=args.metric
            )
            # Serialize for JSON
            unseen_json = {}
            for s, info in unseen_results.items():
                unseen_json[s] = {
                    "nearest_seen": info["nearest_seen"],
                    "mean_distance": info["mean_distance"],
                    "consistency": info["consistency"],
                    "prediction_counts": info["prediction_counts"],
                    "mean_distances": info["mean_distances"],
                }
            unseen_path = out_dir / "unseen_analysis.json"
            unseen_path.write_text(json.dumps(unseen_json, indent=2))
            log.info("Saved unseen analysis: %s", unseen_path)

        # Plots: confusion matrices
        plot_retrieval_confusion_matrix(
            result_all["confusion_matrix"],
            result_all["structures"],
            out_dir / "confusion_matrix_all16.png",
            title="Retrieval Confusion Matrix (All 16)",
        )
        plot_retrieval_confusion_matrix(
            result_seen["confusion_matrix"],
            result_seen["structures"],
            out_dir / "confusion_matrix_seen13.png",
            title="Retrieval Confusion Matrix (Seen 13)",
        )

        # Per-structure accuracy plots
        plot_per_structure_accuracy(
            result_all["per_structure"],
            out_dir / "per_structure_accuracy_all16.png",
            title="Per-Structure Retrieval Accuracy (All 16)",
        )
        plot_per_structure_accuracy(
            result_seen["per_structure"],
            out_dir / "per_structure_accuracy_seen13.png",
            title="Per-Structure Retrieval Accuracy (Seen 13)",
        )

        log.info("Validation complete.")

    # ---- Mode: --compare-layers ----
    elif args.compare_layers:
        log.info("=" * 60)
        log.info("  COMPARE LAYER CONFIGURATIONS")
        log.info("=" * 60)

        emb = get_embeddings()

        configs = {
            "downsamples.1 (128d)": ["downsamples.1"],
            "downsamples.0 (64d)": ["downsamples.0"],
            "input_block (32d)": ["input_block"],
            "downsamples.2 (256d)": ["downsamples.2"],
            "upsamples.2 (64d)": ["upsamples.2"],
            "upsamples.1 (128d)": ["upsamples.1"],
            "d0+d1 (192d)": ["downsamples.0", "downsamples.1"],
            "d0+d1+d2 (448d)": ["downsamples.0", "downsamples.1", "downsamples.2"],
            "all_healthy (416d)": [
                "input_block",
                "downsamples.0",
                "downsamples.1",
                "upsamples.2",
                "upsamples.1",
            ],
        }

        results = compare_layer_configs(emb, configs, metric=args.metric)

        # Save CSV sorted by top1 descending
        rows = []
        for name, res in results.items():
            rows.append(
                {
                    "config": name,
                    "top1_accuracy": res["top1_accuracy"],
                    "top3_accuracy": res["top3_accuracy"],
                }
            )
        import pandas as pd

        df = pd.DataFrame(rows).sort_values("top1_accuracy", ascending=False)
        csv_path = out_dir / "layer_comparison.csv"
        df.to_csv(csv_path, index=False)
        log.info("Saved layer comparison CSV: %s", csv_path)

        # Summarise results for plot
        plot_results = {
            name: {
                "top1_accuracy": res["top1_accuracy"],
                "top3_accuracy": res["top3_accuracy"],
            }
            for name, res in results.items()
        }
        plot_layer_comparison(
            plot_results,
            out_dir / "layer_comparison.png",
        )

        log.info("Layer comparison complete.")

    # ---- Mode: --recommend ----
    elif args.recommend:
        log.info("=" * 60)
        log.info("  RECOMMEND — %s", args.recommend)
        log.info("=" * 60)

        emb = get_embeddings()

        # Load query volume and extract features
        log.info("Loading query volume: %s", args.recommend)
        query_vol = load_volume(args.recommend)

        model = load_dynunet(args.checkpoint, device=args.device)
        extractor = FeatureExtractor(model, EXTRACT_LAYERS)
        query_feats = extractor.extract(query_vol, device=args.device)
        extractor.remove_hooks()

        query_vec = query_feats[args.layer]
        centroids = compute_centroids(emb, args.layer)
        result = find_nearest_centroid(query_vec, centroids, metric=args.metric)

        # Print formatted recommendation
        vol_name = Path(args.recommend).name
        print("=" * 60)
        print(f"  Model Recommendation for: {vol_name}")
        print("=" * 60)
        print(f"  Nearest structure:  {result['nearest']}")
        print(f"  Confidence:         {result['confidence']:.4f}")
        print(f"  Distance:           {result['distances'][result['nearest']]:.4f}")
        print()
        print("  Full ranking:")
        for i, name in enumerate(result["ranking"][:5]):
            print(f"    {i + 1}. {name:<16} (dist={result['distances'][name]:.4f})")
        print("=" * 60)

        # UMAP with query overlay
        try:
            structures = sorted(emb.keys())
            plot_umap_with_query(
                emb,
                args.layer,
                structures,
                query_features=query_vec,
                query_label=f"Query ({vol_name})",
                out_path=out_dir / "umap_query_recommendation.png",
                seen_structures=MEGASEG13_SEEN,
                unseen_structures=MEGASEG13_UNSEEN,
            )
        except Exception as e:
            log.warning("UMAP generation failed: %s", e)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
