"""Mandatory: the added encoders train and the biological gradient is finite."""

from __future__ import annotations

import numpy as np
import pytest

from paragvae.arch import train_architecture
from paragvae.models.candidates import colour_assignment, patch_transformer_gcn
from paragvae.bioloss import BioTargets, biological_gradient
from paragvae.cluster_methods import cluster_vamb

pytestmark = pytest.mark.mandatory


def _ring(n: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([index % 2 for index in range(n)], dtype=np.int64)
    rng = np.random.default_rng(0)
    features = labels[:, None].astype(np.float32) + 0.05 * rng.normal(size=(n, 4)).astype(np.float32)
    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.array([(index + 1) % n for index in range(n)], dtype=np.int32)
    # Store both directions so the undirected graph is a ring.
    rows = np.repeat(np.arange(n), 2)
    indices = np.array([(index + 1) % n if slot == 0 else (index - 1) % n for index in range(n) for slot in range(2)], dtype=np.int32)
    indptr = np.arange(0, 2 * n + 1, 2, dtype=np.int32)
    return features, indptr, indices, labels


def test_each_architecture_returns_a_finite_embedding():
    features, indptr, indices, _labels = _ring()
    names = ("gcn", "sage", "gat", "transformer", "joint", "mixhop", "jknet", "gps", "unet", "han", "diffpool")
    for architecture in names:
        result = train_architecture(
            features=features,
            indptr=indptr,
            indices=indices,
            architecture=architecture,
            n_layers=2 if architecture != "transformer" else 1,
            loss="standard",
            max_epochs=3,
            patience=2,
            latent=8,
            hidden=8,
            seed=0,
            lr=0.05,
        )
        assert result.embedding.shape[0] == features.shape[0]
        assert np.isfinite(result.embedding).all()


def test_three_layer_gcn_runs():
    features, indptr, indices, _labels = _ring()
    result = train_architecture(
        features=features,
        indptr=indptr,
        indices=indices,
        architecture="gcn",
        n_layers=3,
        max_epochs=2,
        patience=2,
        latent=8,
        hidden=8,
        seed=0,
    )
    assert result.embedding.shape[1] == 8


def test_biological_gradient_is_finite_and_same_class_step_shrinks_distance():
    latent = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    targets = BioTargets(
        taxon_index=np.array([0, 0, 1]),
        node_weight=np.ones(3),
        abundance=np.array([0.5, 0.5]),
    )
    loss, grad = biological_gradient(latent, targets, alpha=1.0, beta=0.0, seed=0, n_same=4, n_diff=4)
    assert np.isfinite(loss)
    assert np.isfinite(grad).all()
    stepped = latent - 0.1 * grad
    before = np.sum((latent[0] - latent[1]) ** 2)
    after = np.sum((stepped[0] - stepped[1]) ** 2)
    assert after < before


def test_colour_assignment_and_patch_gcn_run():
    features, indptr, indices, _labels = _ring()
    colors = np.eye(features.shape[0], 2, dtype=np.uint8)
    bins = colour_assignment(colors)
    assert bins.shape == (features.shape[0],)
    result = patch_transformer_gcn(
        features=features,
        indptr=indptr,
        indices=indices,
        max_epochs=2,
        patience=2,
        seed=0,
    )
    assert np.isfinite(result.embedding).all()


def test_vamb_separates_two_opposite_blobs():
    rng = np.random.default_rng(0)
    left = rng.normal(0.0, 0.01, size=(16, 4)).astype(np.float32)
    right = rng.normal(0.0, 0.01, size=(16, 4)).astype(np.float32)
    left[:, 0] += 1.0
    right[:, 1] += 1.0
    labels = cluster_vamb(np.vstack([left, right]))
    assert len(set(labels[:16].tolist())) == 1
    assert len(set(labels[16:].tolist())) == 1
    assert labels[0] != labels[16]
