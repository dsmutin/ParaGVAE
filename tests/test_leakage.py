"""Mandatory: label leakage uses only an observed-node mask."""

from __future__ import annotations

import numpy as np
import pytest

from paragvae.leakage import concat_leakage, leaked_label_features

pytestmark = pytest.mark.mandatory


def _path() -> tuple[np.ndarray, np.ndarray]:
    """Undirected path 0-1-2 stored in both directions."""
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    indices = np.array([1, 0, 2, 1], dtype=np.int64)
    return indptr, indices


def test_decay_zero_keeps_observed_one_hot() -> None:
    """With decay 0 the output is the observed one-hot and unobserved rows stay 0."""
    indptr, indices = _path()
    labels = np.array([0, 1, 0], dtype=np.int64)
    mask = np.array([True, False, True])
    labels_before = labels.copy()
    mask_before = mask.copy()
    indptr_before = indptr.copy()
    indices_before = indices.copy()
    leaked = leaked_label_features(labels, indptr, indices, mask, decay=0.0, steps=3)
    expected = np.array(
        [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    assert leaked.dtype == np.float32
    np.testing.assert_array_equal(leaked, expected)
    np.testing.assert_array_equal(labels, labels_before)
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(indptr, indptr_before)
    np.testing.assert_array_equal(indices, indices_before)


def test_one_step_on_a_path_reaches_only_the_neighbour() -> None:
    """Node 0 observed as class 0 reaches node 1 in one step and not node 2."""
    indptr, indices = _path()
    labels = np.array([0, 1, 2], dtype=np.int64)
    mask = np.array([True, False, False])
    leaked = leaked_label_features(labels, indptr, indices, mask, decay=0.5, steps=1)
    expected = np.array(
        [[1.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(leaked, expected)
    assert leaked[2].sum() == 0.0
    assert leaked[1].sum() > 0.0


def test_all_unobserved_stays_zero() -> None:
    """A mask that is entirely False contributes no class mass."""
    indptr, indices = _path()
    labels = np.array([0, 1, 2], dtype=np.int64)
    mask = np.zeros(3, dtype=bool)
    leaked = leaked_label_features(labels, indptr, indices, mask, decay=0.5, steps=4)
    assert leaked.shape == (3, 3)
    assert leaked.dtype == np.float32
    assert np.count_nonzero(leaked) == 0


def test_observed_negative_label_raises() -> None:
    """An observed node with label -1 is invalid. An unobserved -1 is not."""
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="negative labels"):
        leaked_label_features(
            np.array([-1, 0], dtype=np.int64),
            indptr,
            indices,
            np.array([True, False]),
        )
    leaked = leaked_label_features(
        np.array([0, -1], dtype=np.int64),
        indptr,
        indices,
        np.array([True, False]),
        decay=0.0,
    )
    np.testing.assert_array_equal(leaked, np.array([[1.0], [0.0]], dtype=np.float32))


def test_two_unobserved_neighbours_do_not_copy_their_labels() -> None:
    """True labels of two unobserved neighbours never become a one-hot spike."""
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([1, 0], dtype=np.int64)
    labels = np.array([0, 1], dtype=np.int64)
    mask = np.array([False, False])
    leaked = leaked_label_features(labels, indptr, indices, mask, decay=0.5, steps=3)
    assert leaked.shape == (2, 2)
    assert leaked.dtype == np.float32
    assert np.count_nonzero(leaked) == 0
    assert leaked[0, 0] == 0.0
    assert leaked[1, 1] == 0.0


def test_decay_and_steps_are_bounded() -> None:
    """decay lies in [0, 1) and steps is an integer of at least 1."""
    indptr, indices = _path()
    labels = np.array([0, 0, 0], dtype=np.int64)
    mask = np.array([True, False, False])
    with pytest.raises(ValueError):
        leaked_label_features(labels, indptr, indices, mask, decay=1.0)
    with pytest.raises(ValueError):
        leaked_label_features(labels, indptr, indices, mask, decay=-0.1)
    with pytest.raises(ValueError):
        leaked_label_features(labels, indptr, indices, mask, steps=0)


def test_concat_leakage_matches_rows() -> None:
    """Horizontal stacking requires the same node count."""
    features = np.zeros((3, 2), dtype=np.float32)
    leaked = np.ones((3, 1), dtype=np.float32)
    stacked = concat_leakage(features, leaked)
    assert stacked.shape == (3, 3)
    assert stacked.dtype == np.float32
    np.testing.assert_array_equal(features, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="row count"):
        concat_leakage(features, np.ones((2, 1), dtype=np.float32))
