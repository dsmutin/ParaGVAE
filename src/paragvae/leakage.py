"""Optional semi-supervised label features diffused along the graph.

Only nodes marked in ``observed_mask`` contribute their true class. Unobserved
nodes start at zero and receive mass only from neighbours, decayed at each hop.
Leakage is a semi-supervised channel. Passing a mask that is True on the nodes
you score is label leakage into the test set and is not a valid benchmark.
The default benchmarks must not call this module, and they do not.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from paragvae.train import undirected_edges


def _require_decay(decay: float) -> float:
    """Return ``decay`` when it lies in ``[0, 1)``."""
    if isinstance(decay, (bool, np.bool_)):
        raise ValueError("decay must be in [0, 1)")
    if isinstance(decay, np.ndarray) and decay.shape != ():
        raise ValueError("decay must be in [0, 1)")
    try:
        value = float(decay)
    except (TypeError, ValueError) as exc:
        raise ValueError("decay must be in [0, 1)") from exc
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(f"decay must be in [0, 1), got {value}")
    return value


def _require_steps(steps: int) -> int:
    """Return ``steps`` when it is an integer of at least 1."""
    if isinstance(steps, (bool, np.bool_)) or isinstance(steps, np.ndarray):
        raise ValueError("steps must be an integer >= 1")
    if not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be an integer >= 1")
    count = int(steps)
    if count < 1:
        raise ValueError("steps must be an integer >= 1")
    return count


def _row_normalized_adjacency(indptr: np.ndarray, indices: np.ndarray, n: int) -> sparse.csr_matrix:
    """Row-normalise the undirected graph with self-loops removed.

    A stored reverse edge is one neighbour. An isolated node gets a zero row,
    so diffusion leaves its features unchanged.
    """
    pairs, _weights = undirected_edges(indptr, indices)
    if pairs.shape[0] == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    sources = np.concatenate([pairs[:, 0], pairs[:, 1]])
    targets = np.concatenate([pairs[:, 1], pairs[:, 0]])
    adjacency = sparse.csr_matrix(
        (np.ones(sources.shape[0], dtype=np.float64), (sources, targets)),
        shape=(n, n),
    )
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    inverse = np.zeros(n, dtype=np.float64)
    present = degree > 0.0
    inverse[present] = np.reciprocal(degree[present])
    return (sparse.diags(inverse) @ adjacency).tocsr()


def leaked_label_features(
    labels: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    observed_mask: np.ndarray,
    *,
    decay: float = 0.5,
    steps: int = 3,
) -> np.ndarray:
    """Diffuse observed class labels along neighbouring nodes.

    ``labels`` is an int64 vector of length ``n``. ``observed_mask`` is a bool
    vector of the same length and is the only set of nodes whose true class
    is written into the one-hot. Unobserved rows start at zero, including
    nodes whose label will later be scored. The one-hot width is
    ``max(label) + 1``.

    Starting from that one-hot, repeat ``steps`` times:

    ``h = observed_one_hot + decay * row_normalized_adjacency @ h``

    ``decay`` must lie in ``[0, 1)`` and ``steps`` must be an integer ``>= 1``.
    A negative label on an observed node raises ``ValueError``. Isolated nodes
    keep their observed one-hot, or zeros when they are unobserved. Inputs are
    not modified. The result is float32 with shape ``(n, n_classes)``.

    A mask that is True on nodes you score leaks labels into the test set and
    is not a valid benchmark. The default benchmarks do not call this function.
    """
    decay_value = _require_decay(decay)
    step_count = _require_steps(steps)

    label_vector = np.asarray(labels)
    if label_vector.ndim != 1 or label_vector.dtype != np.int64:
        raise ValueError("labels must be an int64 vector")
    mask = np.asarray(observed_mask)
    n = int(label_vector.shape[0])
    if mask.ndim != 1 or mask.shape[0] != n or mask.dtype != np.bool_:
        raise ValueError("observed_mask must be a bool vector with one entry per node")

    pointer = np.asarray(indptr)
    columns = np.asarray(indices)
    if pointer.ndim != 1 or pointer.shape[0] != n + 1 or not np.issubdtype(pointer.dtype, np.integer):
        raise ValueError("indptr must be an integer vector of length n + 1")
    if columns.ndim != 1 or not np.issubdtype(columns.dtype, np.integer):
        raise ValueError("indices must be an integer vector")
    if int(pointer[0]) != 0 or int(pointer[-1]) != int(columns.shape[0]):
        raise ValueError("indptr does not describe indices")
    if n > 0 and np.any(np.diff(np.asarray(pointer, dtype=np.int64)) < 0):
        raise ValueError("indptr must be non-decreasing")

    if n and bool(np.any(label_vector[mask] < 0)):
        raise ValueError("negative labels are invalid on observed nodes")
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    n_classes = int(label_vector.max()) + 1
    if n_classes <= 0:
        return np.zeros((n, 0), dtype=np.float32)

    observed = np.zeros((n, n_classes), dtype=np.float64)
    observed_nodes = np.flatnonzero(mask)
    if observed_nodes.size:
        observed[observed_nodes, label_vector[observed_nodes]] = 1.0

    adjacency = _row_normalized_adjacency(pointer, columns, n)
    hidden = observed
    for _ in range(step_count):
        hidden = observed + decay_value * (adjacency @ hidden)
    return np.ascontiguousarray(hidden, dtype=np.float32)


def concat_leakage(features: np.ndarray, leaked: np.ndarray) -> np.ndarray:
    """Stack leaked class channels beside existing node features.

    Both arrays must be 2-d and have the same number of rows. Inputs are not
    modified.
    """
    left = np.asarray(features)
    right = np.asarray(leaked)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("features and leaked labels must be 2-d")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"row count mismatch: features has {left.shape[0]} rows, leaked has {right.shape[0]}"
        )
    return np.hstack((left, right))
