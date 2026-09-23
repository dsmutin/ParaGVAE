"""Mandatory: contrastive loss is InfoNCE, with a gradient that matches a finite difference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.native import train_gcn_native
from paragvae.train import infonce_loss

pytestmark = pytest.mark.mandatory


def test_infonce_matches_a_hand_value() -> None:
    """One positive dot of 1 against one negative dot of 0."""
    latent = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    positive = np.array([[0, 1]])
    negatives = np.array([[2]])
    loss, _grad = infonce_loss(latent, positive, negatives)
    expected = np.log(np.exp(1.0) + np.exp(0.0)) - 1.0
    assert loss == pytest.approx(expected)


def test_infonce_gradient_matches_finite_difference() -> None:
    """d(loss)/d(z) agrees with a central difference on one coordinate."""
    latent = np.array([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.7], [0.0, -0.2]], dtype=np.float64)
    positive = np.array([[0, 1], [2, 3]])
    negatives = np.array([[2, 3], [0, 1]])
    _loss, grad = infonce_loss(latent, positive, negatives)
    step = 1e-5
    latent[0, 0] += step
    plus, _ = infonce_loss(latent, positive, negatives)
    latent[0, 0] -= 2 * step
    minus, _ = infonce_loss(latent, positive, negatives)
    numeric = (plus - minus) / (2 * step)
    assert grad[0, 0] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_native_contrastive_fit(tmp_path: Path) -> None:
    """The C++ trainer runs InfoNCE rather than refusing the loss code."""
    n = 8
    rows: list[int] = []
    cols: list[int] = []
    for node in range(n):
        nxt = (node + 1) % n
        rows.extend((node, nxt))
        cols.extend((nxt, node))
    order = np.argsort(rows, kind="mergesort")
    rows_arr = np.asarray(rows)[order]
    cols_arr = np.asarray(cols)[order]
    counts = np.bincount(rows_arr, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    features = np.ones((n, 2), dtype=np.float32)
    fit = train_gcn_native(
        features=features,
        indptr=indptr,
        indices=cols_arr,
        different_pairs=None,
        loss="contrastive",
        max_epochs=2,
        patience=2,
        latent=4,
        hidden=4,
        seed=1,
        lr=0.05,
        work=tmp_path / "job",
    )
    assert fit.embedding.shape == (n, 4)
    assert np.isfinite(fit.best_val_loss)
