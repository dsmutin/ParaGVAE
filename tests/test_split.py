"""Mandatory: validation edges are undirected and absent from the operator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.native import train_gcn_native
from paragvae.train import normalized_from_edges, split_edges, undirected_edges

pytestmark = pytest.mark.mandatory


def _ring(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Both directions of a cycle, which is how the assembly bundles store edges."""
    rows: list[int] = []
    cols: list[int] = []
    for node in range(n):
        nxt = (node + 1) % n
        rows.extend((node, nxt))
        cols.extend((nxt, node))
    order = np.argsort(rows, kind="mergesort")
    rows_arr = np.asarray(rows, dtype=np.int64)[order]
    cols_arr = np.asarray(cols, dtype=np.int64)[order]
    counts = np.bincount(rows_arr, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, cols_arr


def test_reverse_edges_are_one_edge() -> None:
    """A bidirectional ring of 12 nodes has 12 undirected edges, not 24."""
    indptr, indices = _ring(12)
    pairs, _weights = undirected_edges(indptr, indices)
    assert pairs.shape == (12, 2)
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_held_out_edges_are_not_training_edges() -> None:
    """The train and validation sets of an undirected split are disjoint."""
    indptr, indices = _ring(12)
    pairs, weights = undirected_edges(indptr, indices)
    train, _tw, val, _vw = split_edges(pairs, weights, seed=0)
    train_set = {tuple(edge) for edge in train.tolist()}
    val_set = {tuple(edge) for edge in val.tolist()}
    assert train_set.isdisjoint(val_set)
    assert len(val_set) == 12 // 5
    operator = normalized_from_edges(train, np.ones(train.shape[0], dtype=np.float32), 12)
    operator_edges, _weights = undirected_edges(operator.indptr, operator.indices)
    operator_set = {tuple(edge) for edge in operator_edges.tolist()}
    assert operator_set == train_set


def test_native_fit_on_a_ring(tmp_path: Path) -> None:
    """The C++ trainer accepts the train/validation job layout."""
    indptr, indices = _ring(12)
    features = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
    fit = train_gcn_native(
        features=features,
        indptr=indptr,
        indices=indices,
        different_pairs=None,
        loss="standard",
        max_epochs=3,
        patience=3,
        latent=4,
        hidden=4,
        seed=0,
        work=tmp_path / "job",
    )
    assert fit.embedding.shape == (12, 4)
    assert np.isfinite(fit.embedding).all()
