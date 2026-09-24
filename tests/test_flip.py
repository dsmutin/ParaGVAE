"""Mandatory: line-graph flip, edge training, and node pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.flip import (
    cluster_embeddings,
    edge_line_graph,
    infer_nodes_from_edges,
    train_edge_graph,
)
from paragvae.train import undirected_edges

pytestmark = pytest.mark.mandatory


def _assert_csr(indptr: np.ndarray, indices: np.ndarray, n_nodes: int) -> None:
    """CSR pointer and column index describe ``n_nodes`` rows."""
    assert indptr.shape == (n_nodes + 1,)
    assert int(indptr[0]) == 0
    assert np.all(np.diff(indptr) >= 0)
    assert indices.shape == (int(indptr[-1]),)
    if indices.size:
        assert int(indices.min()) >= 0
        assert int(indices.max()) < n_nodes


def _bidirectional_ring(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Both directions of a cycle, which is how the assembly bundles store edges."""
    rows: list[int] = []
    cols: list[int] = []
    for node in range(n):
        nxt = (node + 1) % n
        rows.extend((node, nxt))
        cols.extend((nxt, node))
    order = np.argsort(np.asarray(rows), kind="mergesort")
    rows_arr = np.asarray(rows, dtype=np.int64)[order]
    cols_arr = np.asarray(cols, dtype=np.int64)[order]
    counts = np.bincount(rows_arr, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, cols_arr


def test_path_becomes_two_edge_nodes() -> None:
    """A bidirectional 3-node path is two edge-nodes joined by one edge."""
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    indices = np.array([1, 0, 2, 1], dtype=np.int64)
    node_features = np.array(
        [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]],
        dtype=np.float32,
    )
    edge_features = np.array([[1.0], [3.0], [10.0], [14.0]], dtype=np.float32)
    node_colors = np.array([[1, 0], [0, 0], [0, 4]], dtype=np.uint8)
    graph = edge_line_graph(
        indptr,
        indices,
        node_features,
        edge_features=edge_features,
        node_colors=node_colors,
        edge_colors=np.zeros((4, 0), dtype=np.uint8),
    )
    assert graph.n_original_nodes == 3
    assert graph.edge_index.tolist() == [[0, 1], [1, 2]]
    assert graph.features.shape == (2, 5)
    np.testing.assert_allclose(
        graph.features,
        [[2.0, 4.0, 2.0, 0.5, 0.0], [6.0, 8.0, 12.0, 0.0, 2.0]],
    )
    _assert_csr(graph.indptr, graph.indices, 2)
    pairs, _weights = undirected_edges(graph.indptr, graph.indices)
    assert pairs.tolist() == [[0, 1]]
    plain = edge_line_graph(indptr, indices, node_features)
    assert plain.features.shape == (2, 3)
    np.testing.assert_allclose(plain.features[:, :2], [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(plain.features[:, 2], [1.0, 1.0])
    empty_colors = edge_line_graph(
        indptr,
        indices,
        node_features,
        edge_features=edge_features,
        node_colors=np.zeros((3, 0), dtype=np.uint8),
        edge_colors=np.zeros((4, 0), dtype=np.uint8),
    )
    assert empty_colors.features.shape == (2, 3)


def test_self_loops_are_dropped() -> None:
    """A self-loop is not an edge-node and its feature is not averaged in."""
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 0], dtype=np.int64)
    node_features = np.zeros((2, 1), dtype=np.float32)
    edge_features = np.array([[100.0], [1.0], [3.0]], dtype=np.float32)
    graph = edge_line_graph(indptr, indices, node_features, edge_features=edge_features)
    assert graph.edge_index.tolist() == [[0, 1]]
    assert graph.features.shape == (1, 2)
    np.testing.assert_allclose(graph.features, [[0.0, 2.0]])
    _assert_csr(graph.indptr, graph.indices, 1)
    pairs, _weights = undirected_edges(graph.indptr, graph.indices)
    assert pairs.shape == (0, 2)


def test_isolated_node_gets_a_zero_row() -> None:
    """A node with no incident edge is a zero row of the pooled embedding."""
    edge_embedding = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    pooled = infer_nodes_from_edges(edge_embedding, edge_index, n_nodes=4)
    assert pooled.shape == (4, 2)
    np.testing.assert_allclose(pooled[0], [1.0, 2.0])
    np.testing.assert_allclose(pooled[1], [2.0, 4.0])
    np.testing.assert_allclose(pooled[2], [3.0, 6.0])
    np.testing.assert_allclose(pooled[3], [0.0, 0.0])


def test_rejects_csr_and_feature_length_mismatch() -> None:
    """CSR nnz and feature row counts must match. Nothing is padded."""
    features = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="CSR indices length"):
        edge_line_graph(
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
            features,
        )
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="node_features"):
        edge_line_graph(indptr, indices, np.ones((3, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="edge_features"):
        edge_line_graph(indptr, indices, features, edge_features=np.ones((1, 1), dtype=np.float32))


def test_train_edge_graph_on_a_ring(tmp_path: Path) -> None:
    """A 6-node ring yields one finite embedding row per undirected edge."""
    indptr, indices = _bidirectional_ring(6)
    features = np.arange(18, dtype=np.float32).reshape(6, 3)
    result = train_edge_graph(
        indptr,
        indices,
        features,
        max_epochs=2,
        patience=2,
        lr=0.05,
        seed=0,
        work=tmp_path / "job",
    )
    assert result.embedding.shape[0] == 6
    assert result.embedding.ndim == 2
    assert np.isfinite(result.embedding).all()


def test_cluster_embeddings_separates_two_groups() -> None:
    """Oracle ``k`` is passed through to k-means with ``n_init=5``."""
    embedding = np.array(
        [[0.0, 0.0], [0.1, -0.1], [5.0, 5.0], [5.1, 4.9]],
        dtype=np.float64,
    )
    labels = cluster_embeddings(embedding, n_groups=2, seed=0)
    assert labels.shape == (4,)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
