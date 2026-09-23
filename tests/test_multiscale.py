"""Mandatory: structural features ignore self-loops."""

from __future__ import annotations

import numpy as np
import pytest

from paragvae.study import _degree_and_clustering

pytestmark = pytest.mark.mandatory


def test_self_loops_are_not_triangles() -> None:
    """A two-node edge stored with self-loops has clustering coefficient 0."""
    indptr = np.array([0, 2, 4], dtype=np.int64)
    indices = np.array([0, 1, 0, 1], dtype=np.int64)
    features = _degree_and_clustering(indptr, indices)
    assert features[:, 0].tolist() == [1.0, 1.0]
    assert features[:, 1].tolist() == [0.0, 0.0]


def test_triangle_has_clustering_one() -> None:
    """Three mutual neighbours have clustering coefficient 1."""
    indptr = np.array([0, 2, 4, 6], dtype=np.int64)
    indices = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    features = _degree_and_clustering(indptr, indices)
    assert features[:, 1].tolist() == [1.0, 1.0, 1.0]
