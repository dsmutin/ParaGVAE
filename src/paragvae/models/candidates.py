"""Binning candidates that sit beside the GCN.

``colour_assignment`` and ``published_colouring`` read the tensor colour
mask. ``feature_kmeans`` clusters features and ignores the graph.
``vaegcn_kmeans`` and ``vaegcn_agglomerative`` cluster a GCN embedding.
``patch_transformer_gcn`` mixes feature slices, then trains the GCN on that
matrix. None of these copies an evaluation taxid into a colour.
"""

from __future__ import annotations

import numpy as np

from paragvae.models.arch import train_architecture
from paragvae.score import cluster_embedding
from paragvae.train import TrainResult, _zscore


def colour_assignment(node_colors: np.ndarray) -> np.ndarray:
    """One bin per node: the first maximum colour column.

    A row of zeros is its own bin, one past the last colour. Several positive
    colours keep the lowest index among the maxima.
    """
    colors = np.asarray(node_colors)
    if colors.ndim != 2:
        raise ValueError(f"node colours must be a matrix, got shape {colors.shape}")
    if colors.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if colors.shape[1] == 0:
        raise ValueError("node colours have no columns")
    blank = np.asarray(colors).sum(axis=1) == 0
    bins = np.argmax(colors, axis=1).astype(np.int64)
    bins[blank] = colors.shape[1]
    return bins


def published_colouring(cgt: object) -> np.ndarray:
    """Bin nodes by the colour mask already stored on the tensor."""
    return colour_assignment(np.asarray(cgt.node_colors))


def feature_kmeans(features: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Cluster node features. No graph and no GCN."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"features must be a non-empty matrix, got shape {matrix.shape}")
    return cluster_embedding(matrix, labels, "kmeans", seed)


def vaegcn_kmeans(embedding: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """K-means on a GCN embedding. ``k`` is the number of evaluation labels."""
    return cluster_embedding(embedding, labels, "kmeans", seed)


def vaegcn_agglomerative(embedding: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Average-linkage clustering of a GCN embedding."""
    return cluster_embedding(embedding, labels, "agglomerative", seed)


def patch_tokens(features: np.ndarray, patch: int = 4) -> np.ndarray:
    """Mix feature slices with one attention block. No learned weights."""
    if isinstance(patch, bool) or not isinstance(patch, int) or patch < 1:
        raise ValueError("patch must be a positive integer")
    matrix = _zscore(np.asarray(features, dtype=np.float32))
    n_rows, width = matrix.shape
    if n_rows == 0:
        raise ValueError("features are empty")
    pad = (patch - width % patch) % patch
    if pad:
        matrix = np.pad(matrix, ((0, 0), (0, pad)))
    tokens = matrix.reshape(n_rows, -1, patch)
    scale = np.float32(np.sqrt(patch))
    scores = np.matmul(tokens, np.swapaxes(tokens, -1, -2)) / scale
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, tokens).reshape(n_rows, -1).astype(np.float32)


def patch_transformer_gcn(
    *,
    features: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    patch: int = 4,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    lr: float = 0.05,
) -> TrainResult:
    """Train the GCN on patch-mixed features of the same graph."""
    mixed = patch_tokens(features, patch)
    return train_architecture(
        features=mixed,
        indptr=indptr,
        indices=indices,
        architecture="gcn",
        n_layers=2,
        loss="standard",
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        lr=lr,
    )
