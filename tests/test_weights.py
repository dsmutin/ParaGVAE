"""Mandatory: assembly edge weights enter the operator, and a bad length fails."""

from __future__ import annotations

import numpy as np
import pytest

from paragvae.train import normalized_adjacency, undirected_edges

pytestmark = pytest.mark.mandatory


def _edge() -> tuple[np.ndarray, np.ndarray]:
    """Two nodes joined in both directions."""
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    return indptr, indices


def test_weight_length_must_match_nnz() -> None:
    """A weight vector shorter than the CSR is an error."""
    indptr, indices = _edge()
    with pytest.raises(ValueError, match="edge weight length"):
        undirected_edges(indptr, indices, np.ones(1, dtype=np.float32))


def test_weights_change_normalized_adjacency() -> None:
    """A heavier edge must not produce the same operator as weight 1."""
    indptr, indices = _edge()
    plain = normalized_adjacency(indptr, indices, 2)
    heavy = normalized_adjacency(indptr, indices, 2, edge_weight=np.array([4.0, 4.0], dtype=np.float32))
    assert not np.allclose(plain.toarray(), heavy.toarray())
