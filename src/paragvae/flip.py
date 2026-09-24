"""Optional line-graph flip of a node-coloured assembly graph.

Each undirected edge becomes a node. The usual early-stopped GCN trains on
that graph, and edge embeddings are averaged back onto the original nodes.
This is not a default benchmark arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from paragvae.native import train_gcn_native
from paragvae.train import TrainResult


@dataclass
class EdgeGraph:
    """Line graph whose nodes are the undirected edges of an assembly graph.

    ``edge_index`` stores the original endpoints with ``i < j``, in the same
    order as ``features``. ``indptr`` and ``indices`` are the CSR adjacency
    of the line graph: two rows are neighbours when those original edges
    share a vertex. ``n_original_nodes`` is the node count of the graph that
    was flipped.
    """

    indptr: np.ndarray
    indices: np.ndarray
    features: np.ndarray
    edge_index: np.ndarray
    n_original_nodes: int


def _require_csr(indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return int64 CSR arrays and the node count.

    ``indices`` must have one entry per stored edge (``indptr[-1]``).
    """
    pointer = np.asarray(indptr)
    columns = np.asarray(indices)
    if pointer.ndim != 1 or pointer.size < 1:
        raise ValueError("CSR indptr must be a 1-d array")
    if not np.issubdtype(pointer.dtype, np.integer):
        raise ValueError("CSR indptr must be an integer array")
    if columns.ndim != 1:
        raise ValueError("CSR indices must be a 1-d array")
    if not np.issubdtype(columns.dtype, np.integer):
        raise ValueError("CSR indices must be an integer array")
    pointer = pointer.astype(np.int64, copy=False)
    columns = columns.astype(np.int64, copy=False)
    if int(pointer[0]) != 0:
        raise ValueError("CSR indptr must start at 0")
    if np.any(np.diff(pointer) < 0):
        raise ValueError("CSR indptr must be non-decreasing")
    nnz = int(pointer[-1])
    if columns.shape[0] != nnz:
        raise ValueError(f"CSR indices length {columns.shape[0]} does not match indptr nnz {nnz}")
    n_nodes = int(pointer.shape[0] - 1)
    if columns.size and (int(columns.min()) < 0 or int(columns.max()) >= n_nodes):
        raise ValueError(f"CSR indices point outside 0..{n_nodes - 1}")
    return pointer, columns, n_nodes


def _feature_matrix(name: str, values: np.ndarray, n_rows: int) -> np.ndarray:
    """Return a 2-d numeric matrix with exactly ``n_rows`` rows.

    A 1-d array is treated as one column. Row counts are not padded.
    """
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-d or 2-d, got ndim {array.ndim}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if array.shape[0] != n_rows:
        raise ValueError(f"{name} has {array.shape[0]} rows, expected {n_rows}")
    return array


def _optional_colors(name: str, values: np.ndarray | None, n_rows: int) -> np.ndarray | None:
    """Return a colour matrix, or None when it is missing or has width 0."""
    if values is None:
        return None
    array = _feature_matrix(name, values, n_rows)
    if array.shape[1] == 0:
        return None
    return array


def _collapse_undirected(
    indptr: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop self-loops and collapse reverse pairs into ``i < j``.

    Returns ``(pairs, kept_positions, starts, n_nodes)``. ``kept_positions``
    indexes surviving CSR entries in group order. ``starts`` indexes the
    first of those entries for each undirected edge.
    """
    pointer, columns, n_nodes = _require_csr(indptr, indices)
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(pointer))
    if rows.shape[0] != columns.shape[0]:
        raise ValueError(f"CSR indices length {columns.shape[0]} does not match indptr nnz {rows.shape[0]}")
    keep = rows != columns
    kept_positions = np.flatnonzero(keep)
    if kept_positions.size == 0:
        empty_pairs = np.zeros((0, 2), dtype=np.int64)
        empty_starts = np.zeros(0, dtype=np.int64)
        return empty_pairs, kept_positions, empty_starts, n_nodes
    src = rows[keep]
    dst = columns[keep]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    order = np.lexsort((hi, lo))
    lo = lo[order]
    hi = hi[order]
    kept_positions = kept_positions[order]
    change = np.empty(lo.size, dtype=bool)
    change[0] = True
    change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    starts = np.flatnonzero(change)
    pairs = np.stack([lo[starts], hi[starts]], axis=1).astype(np.int64, copy=False)
    return pairs, kept_positions, starts, n_nodes


def _group_mean(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Mean of row groups. ``starts`` indexes the first row of each group."""
    width = int(values.shape[1])
    if values.shape[0] == 0:
        return np.zeros((0, width), dtype=np.float32)
    sums = np.add.reduceat(np.asarray(values, dtype=np.float64), starts, axis=0)
    counts = np.diff(np.append(starts, values.shape[0]))
    return (sums / counts[:, None]).astype(np.float32)


def _endpoint_mean(matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Mean of the two endpoint rows of each undirected edge."""
    width = int(matrix.shape[1])
    if pairs.shape[0] == 0:
        return np.zeros((0, width), dtype=np.float32)
    numeric = np.asarray(matrix, dtype=np.float64)
    return ((numeric[pairs[:, 0]] + numeric[pairs[:, 1]]) * 0.5).astype(np.float32)


def _line_graph_csr(pairs: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR adjacency of the line graph, with both directions stored."""
    n_edges = int(pairs.shape[0])
    if n_edges == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    edge_ids = np.arange(n_edges, dtype=np.int64)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([edge_ids, edge_ids])
    incidence = sparse.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float64), (rows, cols)),
        shape=(int(n_nodes), n_edges),
    )
    overlap = (incidence.T @ incidence).tocsr()
    overlap.setdiag(0)
    overlap.eliminate_zeros()
    overlap.sum_duplicates()
    if overlap.nnz:
        overlap.data = np.ones(overlap.nnz, dtype=np.float64)
    overlap.sort_indices()
    return np.asarray(overlap.indptr, dtype=np.int64), np.asarray(overlap.indices, dtype=np.int64)


def edge_line_graph(
    indptr: np.ndarray,
    indices: np.ndarray,
    node_features: np.ndarray,
    edge_features: np.ndarray | None = None,
    node_colors: np.ndarray | None = None,
    edge_colors: np.ndarray | None = None,
) -> EdgeGraph:
    """Build the line graph of an undirected assembly graph.

    Self-loops are dropped. A stored reverse pair ``(i, j)`` and ``(j, i)``
    becomes one node, with endpoints ordered ``i < j``. Endpoint node
    features are averaged. Every stored edge-feature row of that undirected
    edge is averaged, so both directions contribute when both exist. Node
    colours and edge colours are averaged the same way.

    Each new-node feature row is the concatenation of:

    - the mean of the two endpoint node features
    - the edge features, or a column of ones when ``edge_features`` is omitted
    - the mean endpoint node colours, as float, when that matrix has nonzero width
    - the edge colours, as float, when that matrix has nonzero width

    ``edge_features`` and ``edge_colors`` have one row per stored CSR entry,
    including self-loops. Self-loop rows are discarded after the length check.
    A colour matrix that is missing or has width 0 is omitted. CSR lengths
    and feature row counts must agree; nothing is padded.

    Two line-graph nodes are adjacent when the original edges share a vertex.
    """
    pairs, kept_positions, starts, n_nodes = _collapse_undirected(indptr, indices)
    nnz = int(np.asarray(indptr)[-1]) if np.asarray(indptr).size else 0
    nodes = _feature_matrix("node_features", node_features, n_nodes)
    edges = None if edge_features is None else _feature_matrix("edge_features", edge_features, nnz)
    node_color_block = _optional_colors("node_colors", node_colors, n_nodes)
    edge_color_block = _optional_colors("edge_colors", edge_colors, nnz)
    blocks = [_endpoint_mean(nodes, pairs)]
    if edges is None:
        blocks.append(np.ones((pairs.shape[0], 1), dtype=np.float32))
    else:
        blocks.append(_group_mean(edges[kept_positions], starts))
    if node_color_block is not None:
        blocks.append(_endpoint_mean(node_color_block, pairs))
    if edge_color_block is not None:
        blocks.append(_group_mean(edge_color_block[kept_positions], starts))
    features = np.hstack(blocks).astype(np.float32, copy=False)
    line_indptr, line_indices = _line_graph_csr(pairs, n_nodes)
    return EdgeGraph(
        indptr=line_indptr,
        indices=line_indices,
        features=features,
        edge_index=np.asarray(pairs, dtype=np.int64),
        n_original_nodes=n_nodes,
    )


def train_edge_graph(
    indptr: np.ndarray,
    indices: np.ndarray,
    node_features: np.ndarray,
    edge_features: np.ndarray | None = None,
    node_colors: np.ndarray | None = None,
    edge_colors: np.ndarray | None = None,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    seed: int,
    work: Path,
) -> TrainResult:
    """Train the early-stopped GCN on the line graph of ``indptr``.

    The fit uses ``loss="standard"``, latent size 16, and hidden size 32.
    ``different_pairs`` is None. ``max_epochs``, ``patience``, ``lr``, and
    ``seed`` are the caller's early-stopping arguments. ``work`` is the
    trainer job directory.

    The returned embedding has one row per undirected edge, in the same
    order as ``edge_line_graph(...).edge_index`` on the same inputs.
    """
    graph = edge_line_graph(
        indptr,
        indices,
        node_features,
        edge_features=edge_features,
        node_colors=node_colors,
        edge_colors=edge_colors,
    )
    if graph.features.shape[0] == 0:
        raise ValueError("line graph has no undirected edges to train on")
    return train_gcn_native(
        features=graph.features,
        indptr=graph.indptr,
        indices=graph.indices,
        different_pairs=None,
        edge_weight=None,
        loss="standard",
        max_epochs=max_epochs,
        patience=patience,
        latent=16,
        hidden=32,
        seed=seed,
        lr=lr,
        work=Path(work),
    )


def infer_nodes_from_edges(
    edge_embedding: np.ndarray,
    edge_index: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Map an edge embedding back to the original nodes.

    Each node row is the mean of the embeddings of undirected edges incident
    to that node. An isolated original node, with no incident edge, gets a
    zero row of the same width as ``edge_embedding``.
    """
    embedding = np.asarray(edge_embedding, dtype=np.float32)
    if embedding.ndim != 2:
        raise ValueError("edge_embedding must be 2-d")
    index = np.asarray(edge_index, dtype=np.int64)
    if index.ndim != 2 or index.shape[1] != 2:
        raise ValueError("edge_index must have shape (E, 2)")
    if index.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"edge_index has {index.shape[0]} rows, edge_embedding has {embedding.shape[0]}"
        )
    nodes = int(n_nodes)
    if nodes < 0:
        raise ValueError("n_nodes must be non-negative")
    if index.size and (int(index.min()) < 0 or int(index.max()) >= nodes):
        raise ValueError(f"edge_index endpoints must lie in 0..{nodes - 1}")
    pooled = np.zeros((nodes, embedding.shape[1]), dtype=np.float64)
    counts = np.zeros(nodes, dtype=np.float64)
    if index.shape[0]:
        for side in (0, 1):
            endpoints = index[:, side]
            np.add.at(pooled, endpoints, embedding)
            np.add.at(counts, endpoints, 1.0)
        present = counts > 0
        pooled[present] /= counts[present, None]
    return pooled.astype(np.float32)


def cluster_embeddings(embedding: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    """Cluster embedding rows with k-means.

    ``n_groups`` is the caller's oracle ``k``, the same count the node-level
    arms take from ground-truth genomes. ``KMeans`` uses ``n_init=5``.
    """
    from sklearn.cluster import KMeans

    matrix = np.asarray(embedding, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("embedding must be a non-empty 2-d array")
    groups = int(n_groups)
    if groups < 1 or groups > matrix.shape[0]:
        raise ValueError(f"n_groups must be between 1 and {matrix.shape[0]}")
    model = KMeans(n_clusters=groups, random_state=int(seed), n_init=5)
    return np.asarray(model.fit_predict(matrix))
