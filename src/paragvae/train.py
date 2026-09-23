"""Short GCN-VAE on a coloured graph tensor.

The optimizer stops when held-out edge loss stalls. It does not run a fixed
long schedule. Genome labels are never an input to the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class TrainResult:
    """One early-stopped fit."""

    embedding: np.ndarray
    epochs_ran: int
    stopped_early: bool
    best_val_loss: float
    train_loss: float


def _zscore(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (values - center) / scale


def undirected_edges(
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique edges with ``i < j``. Self-loops are dropped.

    Opposite directions of the same edge are averaged when ``edge_weight``
    is given. A length mismatch is an error.
    """
    indptr = np.asarray(indptr)
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indptr) - 1
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    if rows.shape[0] != indices.shape[0]:
        raise ValueError(f"CSR indices length {indices.shape[0]} does not match indptr nnz {rows.shape[0]}")
    if edge_weight is None:
        weights = np.ones(rows.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != rows.shape[0]:
            raise ValueError(f"edge weight length {weights.shape[0]} does not match {rows.shape[0]} edges")
    keep = rows != indices
    rows, cols, weights = rows[keep], indices[keep], weights[keep]
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    order = np.lexsort((hi, lo))
    lo, hi, weights = lo[order], hi[order], weights[order]
    change = np.empty(lo.size, dtype=bool)
    change[0] = True
    change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    starts = np.flatnonzero(change)
    sums = np.add.reduceat(weights, starts)
    counts = np.diff(np.append(starts, lo.size))
    pairs = np.stack([lo[starts], hi[starts]], axis=1).astype(np.int64)
    return pairs, (sums / counts).astype(np.float32)


def split_edges(
    pairs: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out one fifth of undirected edges. Fewer than 8 edges are not split."""
    n_edges = int(pairs.shape[0])
    if n_edges < 8:
        return pairs, weights, pairs.copy(), weights.copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_edges)
    n_val = max(1, n_edges // 5)
    val_at = perm[:n_val]
    train_at = perm[n_val:]
    return pairs[train_at], weights[train_at], pairs[val_at], weights[val_at]


def normalized_from_edges(pairs: np.ndarray, weights: np.ndarray, n: int) -> sparse.csr_matrix:
    """Symmetric normalized adjacency of ``pairs`` plus self-loops."""
    if pairs.shape[0] == 0:
        undirected = sparse.eye(n, format="csr", dtype=np.float32)
    else:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.concatenate([weights, weights]).astype(np.float32)
        undirected = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        undirected.sum_duplicates()
        undirected.setdiag(0)
        undirected.eliminate_zeros()
        undirected.setdiag(1)
        undirected = undirected.tocsr()
    degree = np.asarray(undirected.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inverse[nonzero] = np.power(degree[nonzero], -0.5).astype(np.float32)
    scaled = sparse.diags(inverse) @ undirected @ sparse.diags(inverse)
    return scaled.tocsr().astype(np.float32)


def normalized_adjacency(
    indptr: np.ndarray,
    indices: np.ndarray,
    n: int,
    edge_weight: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Symmetric normalized adjacency with self-loops. Sparse, not an N×N dense matrix."""
    pairs, weights = undirected_edges(indptr, indices, edge_weight)
    if edge_weight is None:
        weights = np.ones(pairs.shape[0], dtype=np.float32)
    return normalized_from_edges(pairs, weights, n)


def knn_adjacency(features: np.ndarray, k: int = 5) -> sparse.csr_matrix:
    """Undirected k-nearest-neighbour graph in feature space."""
    matrix = _zscore(features)
    n = matrix.shape[0]
    k = int(min(k, max(1, n - 1)))
    # Chunked distances keep the Illumina graph from allocating n×n twice.
    neighbors = np.empty((n, k), dtype=np.int64)
    block = 512
    for start in range(0, n, block):
        stop = min(n, start + block)
        dots = matrix[start:stop] @ matrix.T
        own = np.sum(matrix[start:stop] ** 2, axis=1, keepdims=True)
        all_norm = np.sum(matrix ** 2, axis=1)
        dist = own + all_norm - 2.0 * dots
        for row in range(stop - start):
            dist[row, start + row] = np.inf
        chosen = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        neighbors[start:stop] = chosen
    rows = np.repeat(np.arange(n), k)
    cols = neighbors.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float32)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def _bce_logits(logit: np.ndarray, target: float) -> tuple[float, np.ndarray]:
    # stable binary cross-entropy and d(loss)/d(logit) averaged by the caller
    positive = np.maximum(logit, 0)
    loss = positive - logit * target + np.log1p(np.exp(-np.abs(logit)))
    probability = 1.0 / (1.0 + np.exp(-logit))
    grad = probability - target
    return float(np.mean(loss)), grad


class _Adam:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        if name not in self.m:
            self.m[name] = np.zeros_like(value)
            self.v[name] = np.zeros_like(value)
        self.m[name] = 0.9 * self.m[name] + 0.1 * grad
        self.v[name] = 0.999 * self.v[name] + 0.001 * (grad * grad)
        mhat = self.m[name] / (1 - 0.9**self.t)
        vhat = self.v[name] / (1 - 0.999**self.t)
        value -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def _init(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    scale = np.sqrt(2.0 / max(rows, 1))
    return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)


def train_gcn(
    *,
    features: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    different_pairs: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
    loss: str = "standard",
    max_epochs: int = 30,
    patience: int = 5,
    latent: int = 16,
    hidden: int = 32,
    seed: int = 0,
    lr: float = 0.01,
) -> TrainResult:
    """Fit a two-layer GCN decoder and return the best embedding.

    ``loss`` is ``standard`` (edge BCE), ``diff_c`` (BCE plus marker
    separation), ``proxy`` (stronger marker term), or ``contrastive``
    (neighbor InfoNCE against random nodes).
    """
    if loss not in {"standard", "diff_c", "proxy", "contrastive"}:
        raise ValueError(f"unknown loss {loss}")
    features = _zscore(features)
    n, width = features.shape
    if width == 0:
        features = np.ones((n, 1), dtype=np.float32)
        width = 1
    pairs, weights = undirected_edges(indptr, indices, edge_weight)
    if edge_weight is None:
        weights = np.ones(pairs.shape[0], dtype=np.float32)
    train_ends, train_weights, val_ends, _val_weights = split_edges(pairs, weights, seed)
    adjacency = normalized_from_edges(train_ends, train_weights, n)
    rng = np.random.default_rng(seed)
    hidden = int(min(hidden, max(4, width * 2)))
    latent = int(min(latent, hidden))
    w0 = _init(rng, width, hidden)
    w1 = _init(rng, hidden, latent)
    opt = _Adam(lr=lr)
    best_state = (w0.copy(), w1.copy())
    best_val = np.inf
    stall = 0
    epochs = 0
    last_train = np.inf
    marker = np.zeros((0, 2), dtype=np.int64) if different_pairs is None else np.asarray(different_pairs, dtype=np.int64)
    if marker.size:
        marker = marker[(marker[:, 0] >= 0) & (marker[:, 1] >= 0) & (marker[:, 0] < n) & (marker[:, 1] < n)]
        if marker.shape[0] > 4096:
            marker = marker[rng.choice(marker.shape[0], size=4096, replace=False)]
    marker_weight = {"diff_c": 1.0, "proxy": 2.0}.get(loss, 0.0)

    def embed(local_w0: np.ndarray, local_w1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pre_h = adjacency @ (features @ local_w0)
        hidden_state = np.where(pre_h > 0, pre_h, 0.01 * pre_h).astype(np.float32)
        latent_state = np.asarray(adjacency @ (hidden_state @ local_w1), dtype=np.float32)
        return hidden_state, latent_state, pre_h

    def negative_pairs(count: int) -> np.ndarray:
        draw = max(count, 1)
        left = rng.integers(0, n, size=draw)
        right = rng.integers(0, n, size=draw)
        same = left == right
        right[same] = (right[same] + 1) % n
        return np.stack([left, right], axis=1)

    val_negative = negative_pairs(max(val_ends.shape[0] * 2, 1))

    def pair_loss(
        latent_state: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(latent_state)
        if positive.shape[0] == 0:
            return 0.0, grad
        if negative is None:
            negative = negative_pairs(positive.shape[0] * 2)
        pos_logit = np.sum(latent_state[positive[:, 0]] * latent_state[positive[:, 1]], axis=1)
        neg_logit = np.sum(latent_state[negative[:, 0]] * latent_state[negative[:, 1]], axis=1)
        pos_loss, pos_grad = _bce_logits(pos_logit, 1.0)
        neg_loss, neg_grad = _bce_logits(neg_logit, 0.0)
        for pairs, pair_grad in (
            (positive, 0.5 * pos_grad / max(positive.shape[0], 1)),
            (negative, 0.5 * neg_grad / max(negative.shape[0], 1)),
        ):
            np.add.at(grad, pairs[:, 0], pair_grad[:, None] * latent_state[pairs[:, 1]])
            np.add.at(grad, pairs[:, 1], pair_grad[:, None] * latent_state[pairs[:, 0]])
        total = 0.5 * (pos_loss + neg_loss)
        if loss == "contrastive":
            return total, grad
        extra = 0.0
        if marker_weight and marker.shape[0]:
            delta = latent_state[marker[:, 0]] - latent_state[marker[:, 1]]
            energy = np.exp(-0.5 * np.sum(delta * delta, axis=1))
            extra = float(np.mean(energy))
            coeff = energy / max(marker.shape[0], 1)
            np.add.at(grad, marker[:, 0], marker_weight * coeff[:, None] * (latent_state[marker[:, 1]] - latent_state[marker[:, 0]]))
            np.add.at(grad, marker[:, 1], marker_weight * coeff[:, None] * (latent_state[marker[:, 0]] - latent_state[marker[:, 1]]))
        return total + marker_weight * extra, grad

    for epoch in range(max_epochs):
        opt.t = epoch + 1
        epochs = epoch + 1
        hidden_state, latent_state, pre_h = embed(w0, w1)
        positive = train_ends
        train_value, grad_z = pair_loss(latent_state, positive)
        last_train = train_value
        # Z = A @ (H @ W1)
        grad_u = np.asarray(adjacency.T @ grad_z, dtype=np.float32)
        grad_w1 = hidden_state.T @ grad_u
        grad_h = grad_u @ w1.T
        grad_pre = np.where(pre_h > 0, grad_h, 0.01 * grad_h).astype(np.float32)
        grad_xw = np.asarray(adjacency.T @ grad_pre, dtype=np.float32)
        grad_w0 = features.T @ grad_xw
        opt.step("w1", w1, grad_w1.astype(np.float32))
        opt.step("w0", w0, grad_w0.astype(np.float32))
        _, val_latent, _ = embed(w0, w1)
        val_value, _ = pair_loss(val_latent, val_ends, val_negative)
        if val_value + 1e-5 < best_val:
            best_val = val_value
            best_state = (w0.copy(), w1.copy())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    w0, w1 = best_state
    _, embedding, _ = embed(w0, w1)
    return TrainResult(
        embedding=embedding,
        epochs_ran=epochs,
        stopped_early=epochs < max_epochs,
        best_val_loss=float(best_val if np.isfinite(best_val) else last_train),
        train_loss=float(last_train),
    )
