"""Mandatory: coloured-graph node and multi-label edge heads."""

from __future__ import annotations

import numpy as np
import pytest

from paragvae.heads import fit_coloured_graph
from paragvae.train import undirected_edges

pytestmark = pytest.mark.mandatory


def _csr(n: int, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Both directions of each undirected edge, stored as CSR."""
    rows: list[int] = []
    cols: list[int] = []
    for src, dst in edges:
        rows.extend((src, dst))
        cols.extend((dst, src))
    row_arr = np.asarray(rows, dtype=np.int64)
    col_arr = np.asarray(cols, dtype=np.int64)
    order = np.argsort(row_arr, kind="mergesort")
    row_arr = row_arr[order]
    col_arr = col_arr[order]
    counts = np.bincount(row_arr, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, col_arr


def _ring(n: int) -> tuple[np.ndarray, np.ndarray]:
    edges = [(node, (node + 1) % n) for node in range(n)]
    return _csr(n, edges)


def test_node_head_learns_a_ring() -> None:
    """Class-indicator features on a ring are fit above 0.7 train accuracy."""
    n = 20
    labels = np.arange(n) % 2
    features = np.zeros((n, 2), dtype=np.float64)
    features[np.arange(n), labels] = 1.0
    features += np.random.default_rng(0).normal(0.0, 0.05, size=features.shape)
    colors = np.zeros((n, 0), dtype=np.uint8)
    indptr, indices = _ring(n)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:16] = True
    result = fit_coloured_graph(
        features,
        colors,
        indptr,
        indices,
        mode="node",
        node_labels=labels,
        train_mask=train_mask,
        n_node_classes=2,
        max_epochs=40,
        patience=8,
        lr=0.05,
        seed=0,
    )
    assert result.node_prob is not None
    assert result.edge_prob is None
    assert result.node_prob.shape == (n, 2)
    pairs, _weights = undirected_edges(indptr, indices)
    assert np.array_equal(result.edge_index, pairs)
    predicted = result.node_prob.argmax(axis=1)
    accuracy = float(np.mean(predicted[train_mask] == labels[train_mask]))
    assert accuracy > 0.7


def test_edge_head_is_multilabel() -> None:
    """Six undirected edges, two classes, and a row with two 1s.

    Edge features are a scaled copy of that multi-hot target. The
    sigmoid can score both classes above one half on the same edge.
    """
    n = 4
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    indptr, indices = _csr(n, edges)
    pairs, _weights = undirected_edges(indptr, indices)
    assert pairs.shape[0] == 6
    by_edge = {
        (0, 1): (1.0, 1.0),
        (0, 2): (1.0, 0.0),
        (0, 3): (0.0, 1.0),
        (1, 2): (1.0, 1.0),
        (1, 3): (0.0, 0.0),
        (2, 3): (1.0, 1.0),
    }
    edge_labels = np.array([by_edge[(int(src), int(dst))] for src, dst in pairs], dtype=np.float64)
    assert np.any(np.all(edge_labels == 1.0, axis=1))
    order = {(int(src), int(dst)): row for row, (src, dst) in enumerate(pairs)}
    arc_features = np.zeros((indices.shape[0], 2), dtype=np.float64)
    rows = np.repeat(np.arange(n), np.diff(indptr))
    for cursor, (src, dst) in enumerate(zip(rows, indices, strict=True)):
        lo, hi = (int(src), int(dst)) if src < dst else (int(dst), int(src))
        arc_features[cursor] = 5.0 * edge_labels[order[(lo, hi)]]
    features = np.ones((n, 2), dtype=np.float64)
    colors = np.zeros((n, 0), dtype=np.uint8)
    edge_train = np.ones(pairs.shape[0], dtype=bool)
    result = fit_coloured_graph(
        features,
        colors,
        indptr,
        indices,
        mode="edge",
        edge_features=arc_features,
        edge_labels=edge_labels,
        edge_train_mask=edge_train,
        max_epochs=40,
        patience=8,
        hidden=4,
        lr=0.05,
        seed=0,
    )
    assert result.node_prob is None
    assert result.edge_prob is not None
    assert result.edge_prob.shape == (6, 2)
    assert np.all(result.edge_prob > 0.0)
    assert np.all(result.edge_prob < 1.0)
    both = np.all(edge_labels == 1.0, axis=1)
    assert np.any(np.all(result.edge_prob[both] > 0.5, axis=1))


def test_both_modes_return_both_arrays() -> None:
    """One shared fit on the ring returns node and edge probabilities."""
    n = 20
    labels = np.arange(n) % 2
    features = np.eye(2, dtype=np.float64)[labels]
    colors = np.zeros((n, 1), dtype=np.float64)
    colors[:, 0] = labels
    indptr, indices = _ring(n)
    pairs, _weights = undirected_edges(indptr, indices)
    edge_labels = np.zeros((pairs.shape[0], 2), dtype=np.float64)
    edge_labels[:, 0] = 1.0
    edge_labels[::3, 1] = 1.0
    train_mask = np.ones(n, dtype=bool)
    train_mask[-4:] = False
    result = fit_coloured_graph(
        features,
        colors,
        indptr,
        indices,
        mode="both",
        node_labels=labels,
        edge_labels=edge_labels,
        train_mask=train_mask,
        max_epochs=8,
        patience=4,
        lr=0.05,
        seed=0,
    )
    assert result.node_prob is not None
    assert result.edge_prob is not None
    assert result.node_prob.shape == (n, 2)
    assert result.edge_prob.shape == (pairs.shape[0], 2)
    assert np.array_equal(result.edge_index, pairs)
    assert np.isfinite(result.node_prob).all()
    assert np.isfinite(result.edge_prob).all()


def test_isolated_nodes_return_finite_probabilities() -> None:
    """A graph with no edges still yields finite node probabilities."""
    n = 7
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    features = np.zeros((n, 2), dtype=np.float64)
    features[np.arange(n), labels] = 1.0
    colors = np.zeros((n, 0), dtype=np.uint8)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    result = fit_coloured_graph(
        features,
        colors,
        indptr,
        indices,
        mode="node",
        node_labels=labels,
        train_mask=np.ones(n, dtype=bool),
        max_epochs=30,
        patience=8,
        lr=0.05,
        seed=0,
    )
    assert result.node_prob is not None
    assert result.edge_prob is None
    assert result.node_prob.shape == (n, 2)
    assert np.isfinite(result.node_prob).all()
    assert result.edge_index.shape == (0, 2)
