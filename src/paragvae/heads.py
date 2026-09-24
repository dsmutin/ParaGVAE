"""Node and multi-label edge heads on one totally coloured graph.

Isolated nodes are common on these assembly graphs, and a convolution
sends them no neighbour message. A residual MLP reads each node's own
features, colours, and degree so those nodes still reach the head.

Class imbalance, not degree, was the main failure mode: a majority
class absorbed the others. Node weights are ``1/sqrt(train count)`` so
that class does not own the loss. On some graphs the label boundary
mattered; the edge vector includes the absolute endpoint-state
difference because that is the feature that exposes a boundary. Depth
was a weak feature, so this module does not add a depth loss.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse

_LEAK = 0.01


class HeadFit(NamedTuple):
    """Probabilities from :func:`fit_coloured_graph`.

    ``node_prob`` has shape ``(N, K)`` or is None when ``mode`` is
    ``edge``. ``edge_prob`` has shape ``(E, C)`` or is None when
    ``mode`` is ``node``. ``edge_index`` has shape ``(E, 2)`` and lists
    the undirected edges ``i < j`` in the same order as ``edge_prob``.
    """

    node_prob: np.ndarray | None
    edge_prob: np.ndarray | None
    edge_index: np.ndarray


class _Adam:
    """Adam with the same moments as the coloured-graph trainer."""

    def __init__(self, lr: float) -> None:
        self.lr = float(lr)
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        if name not in self.m:
            self.m[name] = np.zeros_like(value)
            self.v[name] = np.zeros_like(value)
        self.m[name] = 0.9 * self.m[name] + 0.1 * grad
        self.v[name] = 0.999 * self.v[name] + 0.001 * (grad * grad)
        mhat = self.m[name] / (1.0 - 0.9**self.t)
        vhat = self.v[name] / (1.0 - 0.999**self.t)
        value -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def _zscore(matrix: np.ndarray) -> np.ndarray:
    """Column z-score. A constant column is centered and left unscaled."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[1] == 0:
        return values
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (values - center) / scale


def _init(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    scale = np.sqrt(2.0 / max(rows, 1))
    return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float64)


def _zeros(cols: int) -> np.ndarray:
    return np.zeros(cols, dtype=np.float64)


def _leaky(pre: np.ndarray) -> np.ndarray:
    return np.where(pre > 0.0, pre, _LEAK * pre)


def _leaky_grad(pre: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return np.where(pre > 0.0, grad, _LEAK * grad)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    out = np.empty_like(logits)
    positive = logits >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_neg = np.exp(logits[~positive])
    out[~positive] = exp_neg / (1.0 + exp_neg)
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-d, got shape {values.shape}")
    return values


def _validate_csr(n: int, indptr: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.asarray(indptr, dtype=np.int64).reshape(-1)
    cols = np.asarray(indices, dtype=np.int64).reshape(-1)
    if ptr.shape != (n + 1,):
        raise ValueError(f"indptr length {ptr.shape[0]} does not match {n} nodes")
    if ptr[0] != 0 or ptr[-1] != cols.shape[0]:
        raise ValueError(
            f"indptr must start at 0 and end at the index length {cols.shape[0]}, got {ptr[0]}..{ptr[-1]}"
        )
    if np.any(np.diff(ptr) < 0):
        raise ValueError("indptr must be non-decreasing")
    if cols.size:
        if int(cols.min()) < 0 or int(cols.max()) >= n:
            raise ValueError("CSR indices are outside the node range")
    return ptr, cols


def _collapse_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_features: np.ndarray | None,
    edge_colors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop self-loops and collapse reverse arcs to ``i < j``.

    Directed edge features and colours are averaged across the arcs of
    one undirected edge. ``directed_to_undirected`` is ``-1`` on
    self-loops and the undirected row otherwise.
    """
    n = int(indptr.shape[0] - 1)
    nnz = int(indices.shape[0])
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    if rows.shape[0] != nnz:
        raise ValueError(f"CSR indices length {nnz} does not match indptr nnz {rows.shape[0]}")
    if edge_features is None:
        features = np.zeros((nnz, 0), dtype=np.float64)
    else:
        features = _as_2d(edge_features, "edge_features").astype(np.float64, copy=False)
        if features.shape[0] != nnz:
            raise ValueError(f"edge_features has {features.shape[0]} rows, CSR has {nnz} arcs")
        if not np.isfinite(features).all():
            raise ValueError("edge_features must be finite")
    if edge_colors is None:
        colors = np.zeros((nnz, 0), dtype=np.float64)
    else:
        colors = _as_2d(edge_colors, "edge_colors").astype(np.float64, copy=False)
        if colors.shape[0] != nnz:
            raise ValueError(f"edge_colors has {colors.shape[0]} rows, CSR has {nnz} arcs")
        if not np.isfinite(colors).all():
            raise ValueError("edge_colors must be finite")
    mapping = np.full(nnz, -1, dtype=np.int64)
    keep = rows != indices
    if not np.any(keep):
        empty = np.zeros((0, 2), dtype=np.int64)
        degree = np.zeros(n, dtype=np.float64)
        return empty, features[:0], colors[:0], degree, mapping
    kept_at = np.flatnonzero(keep)
    lo = np.minimum(rows[keep], indices[keep])
    hi = np.maximum(rows[keep], indices[keep])
    order = np.lexsort((hi, lo))
    lo, hi = lo[order], hi[order]
    change = np.empty(lo.shape[0], dtype=bool)
    change[0] = True
    if lo.shape[0] > 1:
        change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    starts = np.flatnonzero(change)
    n_edges = int(starts.shape[0])
    group = np.cumsum(change, dtype=np.int64) - 1
    mapping[kept_at[order]] = group
    edge_index = np.stack([lo[starts], hi[starts]], axis=1).astype(np.int64)
    counts = np.diff(np.append(starts, lo.shape[0])).astype(np.float64)
    feat_kept = features[keep][order]
    color_kept = colors[keep][order]
    feat_sum = np.add.reduceat(feat_kept, starts, axis=0) if feat_kept.shape[1] else np.zeros((n_edges, 0))
    color_sum = np.add.reduceat(color_kept, starts, axis=0) if color_kept.shape[1] else np.zeros((n_edges, 0))
    undirected_features = feat_sum / counts[:, None]
    undirected_colors = color_sum / counts[:, None]
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(degree, edge_index[:, 0], 1.0)
    np.add.at(degree, edge_index[:, 1], 1.0)
    return edge_index, undirected_features, undirected_colors, degree, mapping


def _reduce_directed(matrix: np.ndarray, mapping: np.ndarray, n_edges: int) -> np.ndarray:
    """Max-reduce a CSR-aligned matrix onto undirected edges."""
    values = np.asarray(matrix, dtype=np.float64)
    flat = values.ndim == 1
    if flat:
        values = values.reshape(-1, 1)
    out = np.zeros((n_edges, values.shape[1]), dtype=np.float64)
    valid = mapping >= 0
    if np.any(valid):
        np.maximum.at(out, mapping[valid], values[valid])
    return out.ravel() if flat else out


def _align_supervised(
    matrix: np.ndarray,
    *,
    name: str,
    n_edges: int,
    n_arcs: int,
    mapping: np.ndarray,
) -> np.ndarray:
    """Accept undirected ``edge_index`` order, or CSR order when it differs."""
    values = np.asarray(matrix)
    if values.shape[0] == n_edges:
        return values
    if values.shape[0] == n_arcs and n_arcs != n_edges:
        return _reduce_directed(values, mapping, n_edges)
    raise ValueError(
        f"{name} has {values.shape[0]} rows; expected {n_edges} undirected edges"
        + (f" or {n_arcs} CSR arcs" if n_arcs != n_edges else "")
    )


def _normalized_adjacency(edge_index: np.ndarray, n: int) -> sparse.csr_matrix:
    """Symmetric normalised adjacency without self-loops.

    An isolated node has a zero row. It receives no message.
    """
    if edge_index.shape[0] == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    src = np.concatenate([edge_index[:, 0], edge_index[:, 1]])
    dst = np.concatenate([edge_index[:, 1], edge_index[:, 0]])
    data = np.ones(src.shape[0], dtype=np.float64)
    matrix = sparse.csr_matrix((data, (src, dst)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.zeros(n, dtype=np.float64)
    nonzero = degree > 0.0
    inverse[nonzero] = np.power(degree[nonzero], -0.5)
    scaled = sparse.diags(inverse) @ matrix @ sparse.diags(inverse)
    return scaled.tocsr()


def _node_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """``1/sqrt(train count)``. Missing classes get weight 0."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0.0
    weights[present] = 1.0 / np.sqrt(counts[present])
    return weights


def _positive_weights(targets: np.ndarray) -> np.ndarray:
    """``(1 - rate) / rate`` from the positive rate of each edge class."""
    if targets.shape[0] == 0:
        raise ValueError("edge class weights need at least one training edge")
    rate = np.mean(targets, axis=0)
    weights = np.ones(targets.shape[1], dtype=np.float64)
    usable = (rate > 0.0) & (rate < 1.0)
    weights[usable] = (1.0 - rate[usable]) / rate[usable]
    return weights


def _weighted_ce(logits: np.ndarray, labels: np.ndarray, class_weight: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted-mean cross-entropy and d(loss)/d(logits)."""
    n_rows = int(logits.shape[0])
    if n_rows == 0:
        return 0.0, np.zeros_like(logits)
    shift = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - shift)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    log_probs = logits - shift - np.log(np.sum(exp, axis=1, keepdims=True))
    picked = class_weight[labels]
    denom = float(np.sum(picked))
    if denom <= 0.0:
        raise ValueError("node class weights sum to zero on the supervised nodes")
    loss = float(np.sum(picked * -log_probs[np.arange(n_rows), labels]) / denom)
    grad = probs
    grad[np.arange(n_rows), labels] -= 1.0
    grad *= (picked / denom)[:, None]
    return loss, grad


def _weighted_bce(logits: np.ndarray, targets: np.ndarray, pos_weight: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean multi-label BCE. Positives are scaled by ``pos_weight``."""
    if logits.size == 0:
        return 0.0, np.zeros_like(logits)
    positive = np.logaddexp(0.0, -logits)
    negative = np.logaddexp(0.0, logits)
    weight = pos_weight.reshape(1, -1)
    per = weight * targets * positive + (1.0 - targets) * negative
    loss = float(np.mean(per))
    grad = (_sigmoid(logits) * (weight * targets + (1.0 - targets)) - weight * targets) / per.size
    return loss, grad


def _holdout(indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit indices and a held-out fifth of the same train indices.

    The caller must pass train-mask indices only. Eval indices are not
    scored. A single training index cannot be split and is used to fit.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("early stopping needs at least one training row")
    if indices.size == 1:
        return indices.copy(), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = indices[rng.permutation(indices.size)]
    n_val = max(1, int(indices.size // 5))
    if n_val >= shuffled.size:
        n_val = shuffled.size - 1
    return shuffled[n_val:], shuffled[:n_val]


def _edge_inputs(
    state: np.ndarray,
    edge_index: np.ndarray,
    edge_features: np.ndarray,
    edge_colors: np.ndarray,
) -> np.ndarray:
    """Concat endpoint states, their absolute difference, and edge channels.

    The absolute difference is the boundary feature. No depth term is added.
    """
    if edge_index.shape[0] == 0:
        width = state.shape[1] * 3 + edge_features.shape[1] + edge_colors.shape[1]
        return np.zeros((0, width), dtype=np.float64)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    parts = [state[src], state[dst], np.abs(state[src] - state[dst])]
    if edge_features.shape[1]:
        parts.append(edge_features)
    if edge_colors.shape[1]:
        parts.append(edge_colors)
    return np.concatenate(parts, axis=1)


def _prepare_node_labels(
    node_labels: np.ndarray | None,
    train_mask: np.ndarray,
    n_node_classes: int | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    if node_labels is None:
        raise ValueError("mode 'node' or 'both' requires node_labels")
    labels = np.asarray(node_labels)
    if labels.ndim != 1 or labels.shape[0] != train_mask.shape[0]:
        raise ValueError(f"node_labels must have shape ({train_mask.shape[0]},)")
    if not np.isfinite(labels.astype(np.float64)).all():
        raise ValueError("node_labels must be finite")
    supervised = labels[train_mask]
    if supervised.size == 0:
        raise ValueError("train_mask selects no nodes")
    if np.any(supervised < 0):
        raise ValueError("training node labels must be non-negative class ids")
    inferred = int(supervised.max()) + 1
    n_classes = inferred if n_node_classes is None else int(n_node_classes)
    if n_classes < inferred:
        raise ValueError(f"n_node_classes={n_classes} is smaller than a training label")
    if np.any(supervised >= n_classes):
        raise ValueError("training node labels are outside n_node_classes")
    weights = _node_class_weights(supervised.astype(np.int64), n_classes)
    return labels.astype(np.int64), n_classes, weights


def _prepare_edge_labels(
    edge_labels: np.ndarray | None,
    edge_train: np.ndarray,
    n_edge_classes: int | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    if edge_labels is None:
        raise ValueError("mode 'edge' or 'both' requires edge_labels")
    labels = _as_2d(edge_labels, "edge_labels").astype(np.float64, copy=False)
    if labels.shape[0] != edge_train.shape[0]:
        raise ValueError(
            f"edge_labels has {labels.shape[0]} rows, undirected graph has {edge_train.shape[0]}"
        )
    if labels.shape[1] == 0:
        raise ValueError("edge_labels needs at least one class")
    if not np.isfinite(labels).all():
        raise ValueError("edge_labels must be finite")
    if np.any(labels < 0.0) or np.any(labels > 1.0):
        raise ValueError("edge_labels must lie in [0, 1]; several 1s in one row are allowed")
    n_classes = labels.shape[1] if n_edge_classes is None else int(n_edge_classes)
    if n_classes != labels.shape[1]:
        raise ValueError(f"n_edge_classes={n_classes} does not match edge_labels width {labels.shape[1]}")
    if not np.any(edge_train):
        raise ValueError("edge train mask selects no edges")
    weights = _positive_weights(labels[edge_train])
    return labels, n_classes, weights


def fit_coloured_graph(
    node_features: np.ndarray,
    node_colors: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    *,
    mode: str,
    edge_features: np.ndarray | None = None,
    edge_colors: np.ndarray | None = None,
    node_labels: np.ndarray | None = None,
    edge_labels: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    edge_train_mask: np.ndarray | None = None,
    n_node_classes: int | None = None,
    n_edge_classes: int | None = None,
    max_epochs: int = 40,
    patience: int = 8,
    hidden: int = 16,
    lr: float = 0.05,
    seed: int = 0,
) -> HeadFit:
    """Fit a softmax node head, a multi-label edge head, or both.

    The backbone is two layers. A residual MLP reads the concatenation of
    z-scored node features, node colours as float, and degree. It does not
    use neighbours. A graph convolution reads the undirected normalised
    adjacency. The paths are summed before the node head, because an
    isolated node gets no convolution message.

    ``mode`` is ``node``, ``edge``, or ``both``. In ``both``, one backbone
    is shared and the two losses are added. The node loss is class-weighted
    cross-entropy on ``train_mask`` only (weight ``1/sqrt(train count)``).
    The edge loss is mean binary cross-entropy. Each class is weighted by
    its positive train rate, ``(1 - rate) / rate`` on the positive term.
    Several 1s in one edge row are valid. The edge readout is a sigmoid,
    not a softmax.

    Early stopping uses a held-out fifth of the train mask. Eval nodes and
    eval edges are never scored. ``edge_labels`` are in undirected
    ``edge_index`` order. A CSR-length label matrix is max-reduced onto
    those edges when its row count is not the undirected count.

    Returns ``node_prob`` ``(N, K)`` or None, ``edge_prob`` ``(E, C)`` or
    None, and ``edge_index`` ``(E, 2)``.
    """
    if mode not in {"node", "edge", "both"}:
        raise ValueError("mode must be 'node', 'edge', or 'both'")
    if int(max_epochs) < 1:
        raise ValueError("max_epochs must be positive")
    if int(patience) < 1:
        raise ValueError("patience must be positive")
    if float(lr) <= 0.0:
        raise ValueError("lr must be positive")
    if int(hidden) < 1:
        raise ValueError("hidden must be positive")
    features = _as_2d(node_features, "node_features").astype(np.float64, copy=False)
    colors = _as_2d(node_colors, "node_colors").astype(np.float64, copy=False)
    n = int(features.shape[0])
    if n == 0:
        raise ValueError("node_features must contain at least one node")
    if colors.shape[0] != n:
        raise ValueError(f"node_colors has {colors.shape[0]} rows, node_features has {n}")
    if not np.isfinite(features).all():
        raise ValueError("node_features must be finite")
    if not np.isfinite(colors).all():
        raise ValueError("node_colors must be finite")
    ptr, cols = _validate_csr(n, indptr, indices)
    edge_index, edge_attr, edge_col, degree, mapping = _collapse_csr(ptr, cols, edge_features, edge_colors)
    if train_mask is None:
        nodes_train = np.ones(n, dtype=bool)
    else:
        nodes_train = np.asarray(train_mask, dtype=bool).reshape(-1)
        if nodes_train.shape != (n,):
            raise ValueError(f"train_mask must have shape ({n},)")
    n_arcs = int(cols.shape[0])
    if edge_train_mask is None:
        if train_mask is None:
            edges_train = np.ones(edge_index.shape[0], dtype=bool)
        else:
            edges_train = nodes_train[edge_index[:, 0]] & nodes_train[edge_index[:, 1]]
    else:
        aligned = _align_supervised(
            np.asarray(edge_train_mask).reshape(-1) if np.asarray(edge_train_mask).ndim == 1 else edge_train_mask,
            name="edge_train_mask",
            n_edges=int(edge_index.shape[0]),
            n_arcs=n_arcs,
            mapping=mapping,
        )
        edges_train = np.asarray(aligned, dtype=bool).reshape(-1)
        if edges_train.shape != (edge_index.shape[0],):
            raise ValueError(f"edge_train_mask must have shape ({edge_index.shape[0]},) or ({n_arcs},)")
    if edge_labels is not None and np.asarray(edge_labels).shape[0] != edge_index.shape[0]:
        edge_labels = _align_supervised(
            edge_labels,
            name="edge_labels",
            n_edges=int(edge_index.shape[0]),
            n_arcs=n_arcs,
            mapping=mapping,
        )
    want_node = mode in {"node", "both"}
    want_edge = mode in {"edge", "both"}
    labels = None
    node_weights = None
    n_node = 0
    edge_targets = None
    edge_weights = None
    if want_node:
        labels, n_node, node_weights = _prepare_node_labels(node_labels, nodes_train, n_node_classes)
    if want_edge:
        if edge_index.shape[0] == 0:
            raise ValueError("edge mode needs at least one undirected edge")
        edge_targets, _n_edge, edge_weights = _prepare_edge_labels(edge_labels, edges_train, n_edge_classes)
    inputs = np.concatenate([_zscore(features), colors, degree.reshape(-1, 1)], axis=1)
    adjacency = _normalized_adjacency(edge_index, n)
    hidden = int(hidden)
    width = int(inputs.shape[1])
    rng = np.random.default_rng(int(seed))
    params: dict[str, np.ndarray] = {
        "Wr0": _init(rng, width, hidden),
        "br0": _zeros(hidden),
        "Wr1": _init(rng, hidden, hidden),
        "br1": _zeros(hidden),
        "Wg0": _init(rng, width, hidden),
        "Wg1": _init(rng, hidden, hidden),
    }
    if want_node:
        params["Wn"] = _init(rng, hidden, n_node)
        params["bn"] = _zeros(n_node)
    if want_edge:
        assert edge_targets is not None
        edge_width = hidden * 3 + edge_attr.shape[1] + edge_col.shape[1]
        params["We"] = _init(rng, edge_width, edge_targets.shape[1])
        params["be"] = _zeros(edge_targets.shape[1])
    fit_nodes = np.zeros(0, dtype=np.int64)
    val_nodes = np.zeros(0, dtype=np.int64)
    fit_edges = np.zeros(0, dtype=np.int64)
    val_edges = np.zeros(0, dtype=np.int64)
    if want_node:
        fit_nodes, val_nodes = _holdout(np.flatnonzero(nodes_train), int(seed))
    if want_edge:
        fit_edges, val_edges = _holdout(np.flatnonzero(edges_train), int(seed) + 1)

    def forward(local: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        pre_r = inputs @ local["Wr0"] + local["br0"]
        hidden_r = _leaky(pre_r)
        residual = hidden_r @ local["Wr1"] + local["br1"]
        projected = inputs @ local["Wg0"]
        msg = np.asarray(adjacency @ projected, dtype=np.float64)
        hidden_g = _leaky(msg)
        conv = np.asarray(adjacency @ (hidden_g @ local["Wg1"]), dtype=np.float64)
        state = residual + conv
        cache = {"pre_r": pre_r, "hidden_r": hidden_r, "msg": msg, "hidden_g": hidden_g, "state": state}
        return state, cache

    def backward(
        local: dict[str, np.ndarray],
        cache: dict[str, np.ndarray],
        grad_state: np.ndarray,
    ) -> dict[str, np.ndarray]:
        grads = {name: np.zeros_like(value) for name, value in local.items()}
        grads["Wr1"] += cache["hidden_r"].T @ grad_state
        grads["br1"] += grad_state.sum(axis=0)
        grad_hidden_r = _leaky_grad(cache["pre_r"], grad_state @ local["Wr1"].T)
        grads["Wr0"] += inputs.T @ grad_hidden_r
        grads["br0"] += grad_hidden_r.sum(axis=0)
        grad_conv_in = np.asarray(adjacency.T @ grad_state, dtype=np.float64)
        grads["Wg1"] += cache["hidden_g"].T @ grad_conv_in
        grad_hidden_g = _leaky_grad(cache["msg"], grad_conv_in @ local["Wg1"].T)
        grad_projected = np.asarray(adjacency.T @ grad_hidden_g, dtype=np.float64)
        grads["Wg0"] += inputs.T @ grad_projected
        return grads

    def objective(
        local: dict[str, np.ndarray],
        node_idx: np.ndarray,
        edge_idx: np.ndarray,
        *,
        need_grad: bool,
    ) -> tuple[float | None, dict[str, np.ndarray] | None]:
        state, cache = forward(local)
        total = 0.0
        terms = 0
        grad_state = np.zeros_like(state)
        head_grads = {name: np.zeros_like(value) for name, value in local.items()} if need_grad else None
        if want_node and node_idx.size:
            if labels is None or node_weights is None:
                raise ValueError("node loss requires labels and class weights")
            logits = state[node_idx] @ local["Wn"] + local["bn"]
            loss, grad_logits = _weighted_ce(logits, labels[node_idx], node_weights)
            total += loss
            terms += 1
            if need_grad and head_grads is not None:
                head_grads["Wn"] += state[node_idx].T @ grad_logits
                head_grads["bn"] += grad_logits.sum(axis=0)
                np.add.at(grad_state, node_idx, grad_logits @ local["Wn"].T)
        if want_edge and edge_idx.size:
            assert edge_targets is not None and edge_weights is not None
            built = _edge_inputs(state, edge_index, edge_attr, edge_col)
            logits = built[edge_idx] @ local["We"] + local["be"]
            loss, grad_logits = _weighted_bce(logits, edge_targets[edge_idx], edge_weights)
            total += loss
            terms += 1
            if need_grad and head_grads is not None:
                head_grads["We"] += built[edge_idx].T @ grad_logits
                head_grads["be"] += grad_logits.sum(axis=0)
                grad_in = grad_logits @ local["We"].T
                width_state = state.shape[1]
                src = edge_index[edge_idx, 0]
                dst = edge_index[edge_idx, 1]
                grad_src = grad_in[:, :width_state]
                grad_dst = grad_in[:, width_state : 2 * width_state]
                grad_abs = grad_in[:, 2 * width_state : 3 * width_state]
                sign = np.sign(state[src] - state[dst])
                np.add.at(grad_state, src, grad_src + grad_abs * sign)
                np.add.at(grad_state, dst, grad_dst - grad_abs * sign)
        if terms == 0:
            return None, None
        if not need_grad or head_grads is None:
            return total, None
        body = backward(local, cache, grad_state)
        for name, grad in body.items():
            head_grads[name] += grad
        return total, head_grads

    opt = _Adam(lr=float(lr))
    best_score = np.inf
    best = {name: value.copy() for name, value in params.items()}
    stall = 0
    for epoch in range(int(max_epochs)):
        opt.t = epoch + 1
        _train_loss, grads = objective(params, fit_nodes, fit_edges, need_grad=True)
        if grads is None:
            raise ValueError("the fit split has no supervised rows")
        for name, value in params.items():
            opt.step(name, value, grads[name])
        val_loss, _ = objective(params, val_nodes, val_edges, need_grad=False)
        score = val_loss if val_loss is not None else _train_loss
        assert score is not None
        if score + 1e-5 < best_score:
            best_score = float(score)
            best = {name: value.copy() for name, value in params.items()}
            stall = 0
        else:
            stall += 1
            if stall >= int(patience):
                break
    for name, value in params.items():
        value[...] = best[name]
    state, _cache = forward(params)
    node_prob = None
    edge_prob = None
    if want_node:
        node_prob = _softmax(state @ params["Wn"] + params["bn"])
    if want_edge:
        built = _edge_inputs(state, edge_index, edge_attr, edge_col)
        edge_prob = _sigmoid(built @ params["We"] + params["be"])
    return HeadFit(node_prob=node_prob, edge_prob=edge_prob, edge_index=edge_index)
