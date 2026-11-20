"""Shared Adam, column z-score, and He initialisation.

Callers keep their numeric types. The grid trainer uses float32 and
``eps=1e-6``; the node and edge heads use float64 and ``eps=1e-8``; the
supervised SSL GCN uses float64 and ``eps=1e-6``.
"""

from __future__ import annotations

import numpy as np


class Adam:
    """Adam with the moments used by the Python trainers and heads."""

    def __init__(self, lr: float = 0.01) -> None:
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


def zscore(matrix: np.ndarray, *, dtype: np.dtype | type = np.float32, eps: float = 1e-6) -> np.ndarray:
    """Column z-score. A constant column is centered and left unscaled."""
    values = np.asarray(matrix, dtype=dtype)
    if values.ndim == 2 and values.shape[1] == 0:
        return values
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < eps] = 1.0
    return (values - center) / scale


def he_init(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """He initialisation, ``N(0, sqrt(2 / rows))``."""
    scale = np.sqrt(2.0 / max(rows, 1))
    return rng.normal(0.0, scale, size=(rows, cols)).astype(dtype)
