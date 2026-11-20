"""Mandatory: shared Adam, z-score, and init stay bit-identical to the pre-move copies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.heads import fit_coloured_graph
from paragvae.optim import Adam, he_init, zscore
from paragvae.ssl import fit_supervised_gcn
from paragvae.train import train_gcn

pytestmark = pytest.mark.mandatory

GOLDENS = Path(__file__).resolve().parent / "goldens" / "optim"


def _matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(8, 4)).astype(np.float64)
    matrix[:, 2] = 3.0
    matrix[0, 0] = 1e-9
    return matrix


def _ring(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
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


def _adam_steps(dtype: type, lr: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    gen = np.random.default_rng(11)
    value = gen.normal(size=(4, 3)).astype(dtype)
    start = value.copy()
    opt = Adam(lr=lr)
    for t, scale in enumerate((1.0, -0.3, 2.5), start=1):
        opt.t = t
        grad = gen.normal(size=value.shape).astype(dtype) * scale
        opt.step("w", value, grad)
    return start, value


def _load(name: str) -> np.ndarray:
    return np.load(GOLDENS / f"{name}.npy")


def test_zscore_matches_pre_move_copies() -> None:
    """float32/1e-6, float64/1e-8, and float64/1e-6 match the arrays captured before the move."""
    matrix = _matrix()
    empty = np.zeros((5, 0), dtype=np.float64)
    assert np.array_equal(zscore(matrix, dtype=np.float32, eps=1e-6), _load("train_zscore"))
    assert np.array_equal(zscore(matrix, dtype=np.float64, eps=1e-8), _load("heads_zscore"))
    assert np.array_equal(zscore(matrix, dtype=np.float64, eps=1e-6), _load("ssl_zscore"))
    assert np.array_equal(zscore(empty, dtype=np.float64, eps=1e-8), _load("heads_zscore_empty"))
    assert zscore(empty, dtype=np.float32, eps=1e-6).shape == (5, 0)
    assert zscore(empty, dtype=np.float32, eps=1e-6).dtype == np.float32


def test_he_init_matches_pre_move_copies() -> None:
    """He initialisation keeps the old float32 and float64 bit patterns."""
    assert np.array_equal(he_init(np.random.default_rng(7), 6, 3, dtype=np.float32), _load("train_init"))
    assert np.array_equal(he_init(np.random.default_rng(7), 6, 3, dtype=np.float64), _load("heads_init"))
    assert np.array_equal(he_init(np.random.default_rng(7), 6, 3, dtype=np.float64), _load("ssl_init"))


def test_adam_matches_pre_move_copies() -> None:
    """Three Adam steps on a named parameter match the pre-move trainers and heads."""
    train_start, train_end = _adam_steps(np.float32)
    heads_start, heads_end = _adam_steps(np.float64)
    ssl_start, ssl_end = _adam_steps(np.float64)
    assert np.array_equal(train_start, _load("train_adam_start"))
    assert np.array_equal(train_end, _load("train_adam_end"))
    assert np.array_equal(heads_start, _load("heads_adam_start"))
    assert np.array_equal(heads_end, _load("heads_adam_end"))
    assert np.array_equal(ssl_start, _load("ssl_adam_start"))
    assert np.array_equal(ssl_end, _load("ssl_adam_end"))


def test_call_sites_match_pre_move_runs() -> None:
    """Fixed-seed train_gcn, fit_coloured_graph, and fit_supervised_gcn stay bit-identical."""
    indptr, indices = _ring(8)
    features = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    fit = train_gcn(
        features=features,
        indptr=indptr,
        indices=indices,
        loss="standard",
        max_epochs=3,
        patience=3,
        latent=4,
        hidden=4,
        seed=0,
        lr=0.05,
    )
    assert np.array_equal(fit.embedding, _load("train_gcn_embedding"))

    colors = np.zeros((8, 2), dtype=np.float64)
    colors[:4, 0] = 1
    colors[4:, 1] = 1
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    mask = np.ones(8, dtype=bool)
    mask[7] = False
    head = fit_coloured_graph(
        features.astype(np.float64),
        colors,
        indptr,
        indices,
        mode="node",
        node_labels=labels,
        train_mask=mask,
        n_node_classes=2,
        max_epochs=5,
        patience=5,
        hidden=4,
        lr=0.05,
        seed=0,
    )
    assert np.array_equal(head.node_prob, _load("heads_node_prob"))

    supervised = fit_supervised_gcn(
        features.astype(np.float64),
        indptr,
        indices,
        labels,
        mask,
        n_classes=2,
        max_epochs=5,
        patience=5,
        lr=0.05,
        seed=0,
    )
    assert np.array_equal(supervised.probabilities, _load("ssl_probabilities"))
    assert np.array_equal(supervised.embedding, _load("ssl_embedding"))
