#!/usr/bin/env python
"""Cross-structure feature similarity analysis for MegaSeg DynUNet models.

Extracts intermediate layer activations from a general (multi-structure) model
for multiple organelle structures, then computes:
  1. CKA similarity matrices (structure x structure, per layer)
  2. Global-average-pooled embeddings with t-SNE / PCA / UMAP projections
  3. Cosine similarity heatmaps (structure x structure, per layer)
  4. Hierarchical clustering dendrograms per layer
  5. RSA: correlation of structure similarity matrices across layers
  6. Probing classifiers: logistic regression predicting structure identity
     from intermediate features (stratified k-fold CV)

Usage:
    python cross_structure_features.py [--n-samples 10] [--device cuda]

Outputs go to benchmark/results/cross_structure_features/.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCHMARK_DIR / "results" / "cross_structure_features"
DATA_ROOT = Path("/allen/aics/rep_learn/MegaSeg/data/variance-100x")

# MegaSeg pretrained (13-struct general model)
MEGASEG_CKPT = str(BENCHMARK_DIR / "checkpoints" / "megaseg" / "dynunet.ckpt")

# All 16 structures with available test data
ALL_STRUCTURES = [
    "AAVS1",
    "ACTN1",
    "ATP2A2",
    "CTNNB1",
    "DSP",
    "GJA1",
    "LMNB1",
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

# Layers to extract activations from (skip dead bottleneck)
EXTRACT_LAYERS = [
    "input_block",  # 32ch - shallow
    "downsamples.0",  # 64ch - mid-shallow
    "downsamples.1",  # 128ch - mid-level, healthy
    "downsamples.2",  # 256ch - deepest healthy encoder
    "upsamples.2",  # 64ch - decoder, mostly via skip
    "upsamples.1",  # 128ch - decoder, partial skip
]

# DynUNet architecture config (must match training)
DYNUNET_CONFIG = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    kernel_size=[3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2],
    upsample_kernel_size=[2, 2, 2, 2],
    res_block=True,
)

PATCH_SHAPE = (32, 128, 128)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_path(path: str) -> str:
    """Resolve a data path, stripping hash prefix if file not found.

    CSVs may reference files as ``{64-hex-hash}_{fov}.ome.tiff`` but the
    actual files on disk may lack the hash prefix.
    """
    p = Path(path)
    if p.exists():
        return path
    m = re.match(r"^[a-f0-9]{64}_(.+)$", p.name)
    if m:
        alt = p.parent / m.group(1)
        if alt.exists():
            return str(alt)
    raise FileNotFoundError(f"Cannot resolve: {path}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_dynunet(ckpt_path: str, device: str = "cuda") -> nn.Module:
    """Load a DynUNet model from a checkpoint."""
    from monai.networks.nets.dynunet import DynUNet

    model = DynUNet(**DYNUNET_CONFIG)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("state_dict", ckpt)
    # Strip common prefixes from cyto-dl checkpoints
    cleaned = {}
    for k, v in state.items():
        for prefix in ("model.", "net.", "backbone.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_egfp_channel_idx(path: str, channel_name: str = "EGFP") -> int:
    """Resolve EGFP channel index from OME-TIFF metadata."""
    import xml.etree.ElementTree as ET
    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.ome_metadata:
            raise ValueError(f"No OME metadata in {path}")
        root = ET.fromstring(tif.ome_metadata)
        for ns_uri in [
            "http://www.openmicroscopy.org/Schemas/OME/2016-06",
            "http://www.openmicroscopy.org/Schemas/OME/2015-01",
        ]:
            channels = root.findall(f".//{{{ns_uri}}}Channel")
            if channels:
                break
        else:
            channels = root.findall(".//Channel")

        for i, ch in enumerate(channels):
            if ch.attrib.get("Name") == channel_name:
                return i
    raise ValueError(f"Channel '{channel_name}' not found in {path}")


def load_volume(path: str) -> np.ndarray:
    """Load a volume as float32 ZYX with z-score normalization.

    Handles both:
      - OME-TIFF (multi-channel, needs EGFP extraction) from variance-100x
      - Single-channel TIFF from ablation data (14structs_rand)
    """
    import tifffile

    resolved = _resolve_path(path)
    img = tifffile.imread(resolved)

    # Multi-channel OME-TIFF: ZCYX
    if img.ndim == 4:
        egfp_idx = _get_egfp_channel_idx(resolved)
        img = img[:, egfp_idx, :, :]

    img = img.astype(np.float32)

    # Z-score normalization (matches training pipeline)
    mean_val = img.mean()
    std_val = img.std()
    if std_val > 0:
        img = (img - mean_val) / std_val
    return img


def load_structure_samples(
    structure: str, n_samples: int = 10
) -> List[Tuple[str, np.ndarray]]:
    """Load test samples for a structure.

    Tries two CSV locations:
      1. {DATA_ROOT}/{structure}/local/test_{structure.lower()}_local_cytodl.csv
         (OME-TIFF with hash-prefix paths)
      2. {DATA_ROOT}/{structure}/test.csv
         (single-channel ablation data)
    """
    # Try local CSV first (OME-TIFF format)
    local_csv = (
        DATA_ROOT / structure / "local" / f"test_{structure.lower()}_local_cytodl.csv"
    )
    ablation_csv = DATA_ROOT / structure / "test.csv"

    if local_csv.exists():
        csv_path = local_csv
    elif ablation_csv.exists():
        csv_path = ablation_csv
    else:
        raise FileNotFoundError(
            f"No test CSV found for {structure} at {local_csv} or {ablation_csv}"
        )

    df = pd.read_csv(csv_path)
    # Sample up to n_samples (deterministic: first N)
    df = df.head(n_samples)

    samples = []
    for _, row in df.iterrows():
        raw_path = str(row["raw"])
        sample_id = Path(raw_path).stem.split(".")[0]
        try:
            vol = load_volume(raw_path)
            samples.append((sample_id, vol))
        except Exception as e:
            log.warning("Skipping %s/%s: %s", structure, sample_id, e)

    return samples


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Hook-based intermediate activation extractor for DynUNet."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []

        for name in layer_names:
            module = dict(model.named_modules()).get(name)
            if module is None:
                log.warning("Layer %s not found in model", name)
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            self.features[name] = output.detach()

        return hook_fn

    def extract(
        self, volume: np.ndarray, device: str = "cuda"
    ) -> Dict[str, np.ndarray]:
        """Extract features from a center patch of the volume."""
        z, y, x = volume.shape
        pz, py, px = PATCH_SHAPE
        z0 = max(0, (z - pz) // 2)
        y0 = max(0, (y - py) // 2)
        x0 = max(0, (x - px) // 2)
        patch = volume[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]

        if patch.shape != PATCH_SHAPE:
            padded = np.zeros(PATCH_SHAPE, dtype=np.float32)
            sz, sy, sx = patch.shape
            padded[:sz, :sy, :sx] = patch
            patch = padded

        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
        self.features.clear()
        with torch.no_grad():
            self.model(tensor)

        # Global average pool each feature map to a 1-D vector
        result = {}
        for name, feat in self.features.items():
            # feat shape: (1, C, D, H, W)
            gap = feat.mean(dim=[0, 2, 3, 4]).cpu().numpy()  # (C,)
            result[name] = gap
        return result

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# CKA similarity
# ---------------------------------------------------------------------------


def _centering_matrix(n: int) -> np.ndarray:
    """H = I_n - (1/n) * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion (biased estimator)."""
    n = K.shape[0]
    H = _centering_matrix(n)
    KH = H @ K @ H
    LH = H @ L @ H
    return float(np.trace(KH @ LH)) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices (n_samples x n_features)."""
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0
    return float(hsic_kl / denom)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_cka_matrix(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    layer: str,
    structures: List[str],
) -> np.ndarray:
    """Compute CKA similarity matrix between structures at a given layer.

    Args:
        embeddings: {structure: {layer: (n_samples, n_features) array}}
        layer: which layer to compare
        structures: ordered list of structure names

    Returns:
        (n_structs, n_structs) CKA matrix
    """
    n = len(structures)
    cka_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            X = embeddings[structures[i]][layer]
            Y = embeddings[structures[j]][layer]
            val = linear_cka(X, Y)
            cka_mat[i, j] = val
            cka_mat[j, i] = val
    return cka_mat


def compute_cosine_matrix(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    layer: str,
    structures: List[str],
) -> np.ndarray:
    """Compute mean cosine similarity between structures at a given layer.

    For each pair, compute pairwise cosine between all sample embeddings
    and take the mean.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(structures)
    cos_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            X = embeddings[structures[i]][layer]  # (n_samples, n_features)
            Y = embeddings[structures[j]][layer]
            sim = cosine_similarity(X, Y)  # (n_i, n_j)
            val = float(sim.mean())
            cos_mat[i, j] = val
            cos_mat[j, i] = val
    return cos_mat


def run_tsne(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    layer: str,
    structures: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Run t-SNE on concatenated embeddings from all structures."""
    from sklearn.manifold import TSNE

    all_feats = []
    labels = []
    for struct in structures:
        feats = embeddings[struct][layer]
        all_feats.append(feats)
        labels.extend([struct] * feats.shape[0])

    X = np.vstack(all_feats)
    perp = min(30, X.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(X)
    return coords, labels


def run_pca(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    layer: str,
    structures: List[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Run PCA on concatenated embeddings."""
    from sklearn.decomposition import PCA

    all_feats = []
    labels = []
    for struct in structures:
        feats = embeddings[struct][layer]
        all_feats.append(feats)
        labels.extend([struct] * feats.shape[0])

    X = np.vstack(all_feats)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    return coords, labels, pca.explained_variance_ratio_


def run_umap(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    layer: str,
    structures: List[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Tuple[np.ndarray, List[str]]:
    """Run UMAP on concatenated embeddings from all structures."""
    import umap

    all_feats = []
    labels = []
    for struct in structures:
        feats = embeddings[struct][layer]
        all_feats.append(feats)
        labels.extend([struct] * feats.shape[0])

    X = np.vstack(all_feats)
    n_neighbors_eff = min(n_neighbors, X.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        random_state=42,
        metric="cosine",
    )
    coords = reducer.fit_transform(X)
    return coords, labels


def run_probing_classifier(
    embeddings: Dict[str, Dict[str, np.ndarray]],
    layer: str,
    structures: List[str],
    n_folds: int = 5,
) -> Dict:
    """Train a probing classifier (logistic regression) to predict structure from embeddings.

    Uses stratified k-fold cross-validation to evaluate how linearly separable
    structure identity is at each layer.

    Returns:
        Dict with keys:
          - 'accuracy_mean': mean CV accuracy
          - 'accuracy_std': std of CV accuracy
          - 'per_class_f1': dict of structure -> mean F1 score
          - 'confusion_matrix': (n_structs, n_structs) confusion matrix (summed over folds)
          - 'fold_accuracies': list of per-fold accuracies
          - 'structures': list of structure labels (in order used by classifier)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    all_feats = []
    labels = []
    for struct in structures:
        feats = embeddings[struct][layer]
        all_feats.append(feats)
        labels.extend([struct] * feats.shape[0])

    X = np.vstack(all_feats)
    y = np.array(labels)
    unique_labels = sorted(set(labels))

    # Encode labels as integers for sklearn
    label_to_idx = {s: i for i, s in enumerate(unique_labels)}
    y_int = np.array([label_to_idx[l] for l in y])

    n_classes = len(unique_labels)
    n_samples_per_class = min(np.bincount(y_int))

    # Adjust n_folds if too few samples per class
    effective_folds = min(n_folds, n_samples_per_class)
    if effective_folds < 2:
        log.warning(
            "Only %d samples per class at %s — skipping probing classifier",
            n_samples_per_class,
            layer,
        )
        return None

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
    scaler = StandardScaler()

    fold_accuracies = []
    fold_f1s = {s: [] for s in unique_labels}
    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    for train_idx, test_idx in skf.split(X, y_int):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        # Scale features
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        fold_acc = float(np.mean(y_pred == y_test))
        fold_accuracies.append(fold_acc)

        # Per-class F1
        f1s = f1_score(
            y_test, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
        )
        for i, s in enumerate(unique_labels):
            fold_f1s[s].append(float(f1s[i]))

        # Accumulate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
        cm_total += cm

    return {
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "per_class_f1": {s: float(np.mean(f1s)) for s, f1s in fold_f1s.items()},
        "confusion_matrix": cm_total,
        "fold_accuracies": fold_accuracies,
        "structures": unique_labels,
        "n_folds": effective_folds,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _short_labels(structures: List[str]) -> List[str]:
    """Shorter labels for heatmaps."""
    return [s[:6] for s in structures]


def plot_cka_heatmap(
    cka_mat: np.ndarray,
    structures: List[str],
    layer: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = _short_labels(structures)
    sns.heatmap(
        cka_mat,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"CKA Similarity: {layer}", fontsize=14)
    ax.set_xlabel("Structure")
    ax.set_ylabel("Structure")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved CKA heatmap: %s", out_path)


def plot_cosine_heatmap(
    cos_mat: np.ndarray,
    structures: List[str],
    layer: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = _short_labels(structures)
    sns.heatmap(
        cos_mat,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Cosine Similarity (mean): {layer}", fontsize=14)
    ax.set_xlabel("Structure")
    ax.set_ylabel("Structure")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cosine heatmap: %s", out_path)


def plot_tsne(
    coords: np.ndarray,
    labels: List[str],
    layer: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    unique = sorted(set(labels))
    cmap = matplotlib.colormaps["tab20"].resampled(len(unique))
    for i, struct in enumerate(unique):
        mask = np.array([l == struct for l in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=cmap(i),
            label=struct,
            s=40,
            alpha=0.8,
            edgecolors="k",
            linewidth=0.3,
        )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
    ax.set_title(f"t-SNE: {layer}", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved t-SNE plot: %s", out_path)


def plot_pca(
    coords: np.ndarray,
    labels: List[str],
    var_ratio: np.ndarray,
    layer: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 10))
    unique = sorted(set(labels))
    cmap = matplotlib.colormaps["tab20"].resampled(len(unique))
    for i, struct in enumerate(unique):
        mask = np.array([l == struct for l in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=cmap(i),
            label=struct,
            s=40,
            alpha=0.8,
            edgecolors="k",
            linewidth=0.3,
        )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
    ax.set_title(
        f"PCA: {layer} (var={var_ratio[0]:.1%}+{var_ratio[1]:.1%})", fontsize=14
    )
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved PCA plot: %s", out_path)


def plot_umap(
    coords: np.ndarray,
    labels: List[str],
    layer: str,
    out_path: Path,
):
    """Plot UMAP embeddings colored by structure."""
    fig, ax = plt.subplots(figsize=(12, 10))
    unique = sorted(set(labels))
    cmap = matplotlib.colormaps["tab20"].resampled(len(unique))
    for i, struct in enumerate(unique):
        mask = np.array([l == struct for l in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=cmap(i),
            label=struct,
            s=40,
            alpha=0.8,
            edgecolors="k",
            linewidth=0.3,
        )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
    ax.set_title(f"UMAP: {layer}", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved UMAP plot: %s", out_path)


def plot_probing_accuracy(
    results: Dict[str, Dict],
    layer_names: List[str],
    out_path: Path,
):
    """Bar chart of probing classifier accuracy per layer.

    Args:
        results: {layer_name: probing result dict from run_probing_classifier}
        layer_names: ordered list of layer names
        out_path: save path
    """
    layers = [l for l in layer_names if l in results and results[l] is not None]
    if not layers:
        return

    means = [results[l]["accuracy_mean"] for l in layers]
    stds = [results[l]["accuracy_std"] for l in layers]
    short_names = [
        l.replace("downsamples.", "enc")
        .replace("upsamples.", "dec")
        .replace("input_block", "inp")
        for l in layers
    ]

    n_classes = len(results[layers[0]]["structures"])
    chance = 1.0 / n_classes

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layers))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="k", alpha=0.85
    )
    ax.axhline(
        chance, color="red", linestyle="--", linewidth=1, label=f"Chance ({chance:.1%})"
    )
    ax.axhline(1.0, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title(
        f"Probing Classifier Accuracy by Layer ({n_classes}-class, {results[layers[0]]['n_folds']}-fold CV)",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved probing accuracy chart: %s", out_path)


def plot_probing_f1(
    results: Dict[str, Dict],
    layer_names: List[str],
    out_path: Path,
):
    """Heatmap of per-class F1 score across layers.

    Rows = structures, Columns = layers.
    """
    layers = [l for l in layer_names if l in results and results[l] is not None]
    if not layers:
        return

    structures = results[layers[0]]["structures"]
    short_layers = [
        l.replace("downsamples.", "enc")
        .replace("upsamples.", "dec")
        .replace("input_block", "inp")
        for l in layers
    ]

    # Build matrix: (n_structs, n_layers)
    f1_mat = np.zeros((len(structures), len(layers)))
    for j, layer in enumerate(layers):
        for i, struct in enumerate(structures):
            f1_mat[i, j] = results[layer]["per_class_f1"].get(struct, 0.0)

    fig, ax = plt.subplots(
        figsize=(max(8, len(layers) * 1.5), max(8, len(structures) * 0.5))
    )
    sns.heatmap(
        f1_mat,
        xticklabels=short_layers,
        yticklabels=structures,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Probing Classifier: Per-Structure F1 Score by Layer", fontsize=13)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Structure")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved probing F1 heatmap: %s", out_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    structures: List[str],
    layer: str,
    out_path: Path,
):
    """Plot confusion matrix for a probing classifier at a specific layer."""
    # Normalize by row (true class)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    labels = _short_labels(structures)
    sns.heatmap(
        cm_norm,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    short_layer = (
        layer.replace("downsamples.", "enc")
        .replace("upsamples.", "dec")
        .replace("input_block", "inp")
    )
    ax.set_title(f"Confusion Matrix (normalized): {short_layer}", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix: %s", out_path)


def plot_dendrogram(
    cos_mat: np.ndarray,
    structures: List[str],
    layer: str,
    out_path: Path,
):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    # Convert similarity to distance
    dist_mat = 1.0 - cos_mat
    np.fill_diagonal(dist_mat, 0)
    # Ensure symmetry
    dist_mat = (dist_mat + dist_mat.T) / 2
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        Z,
        labels=structures,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=0.3 * max(Z[:, 2]),
    )
    ax.set_title(f"Hierarchical Clustering: {layer}", fontsize=14)
    ax.set_ylabel("Distance (1 - cosine sim)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved dendrogram: %s", out_path)


def plot_rsa(
    rsa_mat: np.ndarray,
    layer_names: List[str],
    out_path: Path,
):
    """Plot RSA: correlation of structure similarity across layers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    short = [
        l.replace("downsamples.", "enc")
        .replace("upsamples.", "dec")
        .replace("input_block", "inp")
        for l in layer_names
    ]
    sns.heatmap(
        rsa_mat,
        xticklabels=short,
        yticklabels=short,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title("RSA: Layer-wise Structure Similarity Correlation", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved RSA plot: %s", out_path)


def plot_summary_grid(
    cka_matrices: Dict[str, np.ndarray],
    structures: List[str],
    out_path: Path,
):
    """Multi-panel CKA heatmaps for all layers in one figure."""
    n_layers = len(cka_matrices)
    cols = 3
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    labels = _short_labels(structures)

    for idx, (layer, mat) in enumerate(cka_matrices.items()):
        ax = axes_flat[idx]
        sns.heatmap(
            mat,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            square=True,
            cbar=False,
            annot_kws={"fontsize": 6},
        )
        short_name = (
            layer.replace("downsamples.", "enc")
            .replace("upsamples.", "dec")
            .replace("input_block", "inp")
        )
        ax.set_title(short_name, fontsize=11)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(len(cka_matrices), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Cross-Structure CKA Similarity (MegaSeg Pretrained)", fontsize=16, y=1.02
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved summary grid: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Cross-structure feature analysis")
    parser.add_argument(
        "--n-samples", type=int, default=10, help="Number of test samples per structure"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=MEGASEG_CKPT,
        help="Model checkpoint to analyze",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=None,
        help="Structures to analyze (default: all 16)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="megaseg_pretrained",
        help="Label for this model in output filenames",
    )
    args = parser.parse_args()

    structures = args.structures or ALL_STRUCTURES
    out_dir = RESULTS_DIR / args.label
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Cross-Structure Feature Similarity Analysis")
    log.info("Model: %s", args.label)
    log.info("Checkpoint: %s", args.checkpoint)
    log.info("Structures: %s", structures)
    log.info("Samples per structure: %d", args.n_samples)
    log.info("Layers: %s", EXTRACT_LAYERS)
    log.info("Output: %s", out_dir)
    log.info("=" * 70)

    # ---- 1. Load model ----
    t0 = time.time()
    log.info("Loading model...")
    model = load_dynunet(args.checkpoint, device=args.device)
    log.info("Model loaded in %.1fs", time.time() - t0)

    # ---- 2. Extract features for each structure ----
    extractor = FeatureExtractor(model, EXTRACT_LAYERS)

    # embeddings[structure][layer] = (n_samples, n_features)
    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    loaded_structures = []

    for struct in structures:
        log.info("--- Loading %s ---", struct)
        t1 = time.time()
        try:
            samples = load_structure_samples(struct, n_samples=args.n_samples)
        except FileNotFoundError as e:
            log.warning("Skipping %s: %s", struct, e)
            continue

        if len(samples) < 2:
            log.warning(
                "Skipping %s: only %d samples loaded (need >= 2)", struct, len(samples)
            )
            continue

        # Extract features for each sample
        layer_feats: Dict[str, List[np.ndarray]] = {l: [] for l in EXTRACT_LAYERS}
        for sid, vol in samples:
            feats = extractor.extract(vol, device=args.device)
            for layer_name, gap_vec in feats.items():
                layer_feats[layer_name].append(gap_vec)

        # Stack into matrices
        struct_embeddings = {}
        for layer_name in EXTRACT_LAYERS:
            if layer_feats[layer_name]:
                struct_embeddings[layer_name] = np.stack(layer_feats[layer_name])

        if struct_embeddings:
            embeddings[struct] = struct_embeddings
            loaded_structures.append(struct)
            n_loaded = len(layer_feats[EXTRACT_LAYERS[0]])
            log.info("  %s: %d samples, %.1fs", struct, n_loaded, time.time() - t1)

    extractor.remove_hooks()
    del model
    torch.cuda.empty_cache()

    # Save embeddings for downstream use (model recommendation)
    from model_recommender import save_embeddings as _save_emb

    emb_path = out_dir / "embeddings.npz"
    _save_emb(embeddings, emb_path)
    log.info("Saved embeddings to %s", emb_path)

    log.info("=" * 70)
    log.info("Loaded %d / %d structures", len(loaded_structures), len(structures))
    if len(loaded_structures) < 2:
        log.error("Need at least 2 structures for comparison. Exiting.")
        return

    # ---- 3. Compute similarity matrices ----
    log.info("Computing CKA and cosine similarity matrices...")
    cka_matrices = {}
    cos_matrices = {}

    for layer in EXTRACT_LAYERS:
        # Check all structures have this layer
        valid = [s for s in loaded_structures if layer in embeddings[s]]
        if len(valid) < 2:
            continue

        cka_mat = compute_cka_matrix(embeddings, layer, valid)
        cos_mat = compute_cosine_matrix(embeddings, layer, valid)
        cka_matrices[layer] = cka_mat
        cos_matrices[layer] = cos_mat

        # Save as CSV
        df_cka = pd.DataFrame(cka_mat, index=valid, columns=valid)
        df_cka.to_csv(out_dir / f"cka_{layer.replace('.', '_')}.csv")
        df_cos = pd.DataFrame(cos_mat, index=valid, columns=valid)
        df_cos.to_csv(out_dir / f"cosine_{layer.replace('.', '_')}.csv")

        log.info(
            "  %s: CKA range [%.3f, %.3f], cosine range [%.3f, %.3f]",
            layer,
            cka_mat[~np.eye(len(valid), dtype=bool)].min(),
            cka_mat[~np.eye(len(valid), dtype=bool)].max(),
            cos_mat[~np.eye(len(valid), dtype=bool)].min(),
            cos_mat[~np.eye(len(valid), dtype=bool)].max(),
        )

    # ---- 4. RSA: correlate structure similarity across layers ----
    log.info("Computing RSA (layer-to-layer similarity correlation)...")
    layer_list = [l for l in EXTRACT_LAYERS if l in cka_matrices]
    n_layers = len(layer_list)
    rsa_mat = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            # Upper triangle of similarity matrix (excluding diagonal)
            mask = np.triu(np.ones_like(cka_matrices[layer_list[i]], dtype=bool), k=1)
            v1 = cka_matrices[layer_list[i]][mask]
            v2 = cka_matrices[layer_list[j]][mask]
            if v1.std() > 0 and v2.std() > 0:
                rsa_mat[i, j] = float(np.corrcoef(v1, v2)[0, 1])
            else:
                rsa_mat[i, j] = 0.0

    df_rsa = pd.DataFrame(rsa_mat, index=layer_list, columns=layer_list)
    df_rsa.to_csv(out_dir / "rsa_layer_correlation.csv")

    # ---- 5. Generate plots ----
    log.info("Generating plots...")

    for layer in layer_list:
        valid = [s for s in loaded_structures if layer in embeddings[s]]
        safe_name = layer.replace(".", "_")

        # CKA heatmap
        plot_cka_heatmap(
            cka_matrices[layer],
            valid,
            layer,
            fig_dir / f"cka_{safe_name}.png",
        )

        # Cosine heatmap
        plot_cosine_heatmap(
            cos_matrices[layer],
            valid,
            layer,
            fig_dir / f"cosine_{safe_name}.png",
        )

        # t-SNE
        try:
            coords, labels = run_tsne(embeddings, layer, valid)
            plot_tsne(coords, labels, layer, fig_dir / f"tsne_{safe_name}.png")
        except Exception as e:
            log.warning("t-SNE failed for %s: %s", layer, e)

        # PCA
        try:
            coords, labels, var_ratio = run_pca(embeddings, layer, valid)
            plot_pca(coords, labels, var_ratio, layer, fig_dir / f"pca_{safe_name}.png")
        except Exception as e:
            log.warning("PCA failed for %s: %s", layer, e)

        # UMAP
        try:
            coords, labels = run_umap(embeddings, layer, valid)
            plot_umap(coords, labels, layer, fig_dir / f"umap_{safe_name}.png")
        except Exception as e:
            log.warning("UMAP failed for %s: %s", layer, e)

        # Dendrogram
        try:
            plot_dendrogram(
                cos_matrices[layer], valid, layer, fig_dir / f"dendro_{safe_name}.png"
            )
        except Exception as e:
            log.warning("Dendrogram failed for %s: %s", layer, e)

    # RSA plot
    plot_rsa(rsa_mat, layer_list, fig_dir / "rsa_layer_correlation.png")

    # Summary grid (CKA across all layers)
    plot_summary_grid(
        {l: cka_matrices[l] for l in layer_list},
        loaded_structures,
        fig_dir / "cka_summary_grid.png",
    )

    # ---- 6. Probing classifiers ----
    log.info("Running probing classifiers (structure identity from embeddings)...")
    probing_results = {}
    for layer in layer_list:
        valid = [s for s in loaded_structures if layer in embeddings[s]]
        safe_name = layer.replace(".", "_")
        try:
            result = run_probing_classifier(embeddings, layer, valid)
            if result is not None:
                probing_results[layer] = result
                log.info(
                    "  %s: accuracy=%.1f%% (+/- %.1f%%), %d-fold CV",
                    layer,
                    result["accuracy_mean"] * 100,
                    result["accuracy_std"] * 100,
                    result["n_folds"],
                )
                # Confusion matrix per layer
                plot_confusion_matrix(
                    result["confusion_matrix"],
                    result["structures"],
                    layer,
                    fig_dir / f"probing_cm_{safe_name}.png",
                )
        except Exception as e:
            log.warning("Probing classifier failed for %s: %s", layer, e)

    # Summary plots for probing
    if probing_results:
        plot_probing_accuracy(
            probing_results, layer_list, fig_dir / "probing_accuracy.png"
        )
        plot_probing_f1(probing_results, layer_list, fig_dir / "probing_f1_heatmap.png")

        # Save probing results as CSV
        probing_rows = []
        for layer, res in probing_results.items():
            row = {
                "layer": layer,
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "n_folds": res["n_folds"],
            }
            for s, f1 in res["per_class_f1"].items():
                row[f"f1_{s}"] = f1
            probing_rows.append(row)
        df_probing = pd.DataFrame(probing_rows)
        df_probing.to_csv(out_dir / "probing_classifier_results.csv", index=False)
        log.info(
            "Saved probing results CSV: %s", out_dir / "probing_classifier_results.csv"
        )

    # ---- 7. Summary report ----
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("Structures analyzed: %s", loaded_structures)
    for layer in layer_list:
        valid = [s for s in loaded_structures if layer in embeddings[s]]
        off_diag = cka_matrices[layer][~np.eye(len(valid), dtype=bool)]
        log.info(
            "  %s: CKA mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            layer,
            off_diag.mean(),
            off_diag.std(),
            off_diag.min(),
            off_diag.max(),
        )

    # Save summary
    summary_lines = [
        f"Cross-Structure Feature Similarity Analysis",
        f"Model: {args.label}",
        f"Checkpoint: {args.checkpoint}",
        f"Structures: {loaded_structures}",
        f"Samples per structure: {args.n_samples}",
        f"",
    ]
    for layer in layer_list:
        valid = [s for s in loaded_structures if layer in embeddings[s]]
        off_diag = cka_matrices[layer][~np.eye(len(valid), dtype=bool)]
        cos_off = cos_matrices[layer][~np.eye(len(valid), dtype=bool)]
        probing_str = ""
        if layer in probing_results:
            pr = probing_results[layer]
            probing_str = f" | Probing acc={pr['accuracy_mean']:.1%} (+/- {pr['accuracy_std']:.1%})"
        summary_lines.append(
            f"{layer}: CKA mean={off_diag.mean():.3f} std={off_diag.std():.3f} "
            f"| Cosine mean={cos_off.mean():.3f} std={cos_off.std():.3f}"
            f"{probing_str}"
        )

    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    log.info("Results saved to %s", out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
