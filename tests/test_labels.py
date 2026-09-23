"""Mandatory: pickled VAEGbin label dicts become dense integer ids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.graphs import _check_row_counts, _label_vector

pytestmark = pytest.mark.mandatory


def test_dict_labels_follow_node_ids(tmp_path: Path) -> None:
    """String-keyed genome ids line up with the node name order."""
    path = tmp_path / "node_to_label.npy"
    np.save(path, {"b": 4, "a": 2})
    labels = _label_vector(path, ["a", "b", "missing"])
    assert labels.tolist() == [2, 4, -1]


def test_label_lookup_ignores_row_index(tmp_path: Path) -> None:
    """An integer key must not be treated as a stand-in for a missing node name."""
    path = tmp_path / "node_to_label.npy"
    np.save(path, {0: 5})
    labels = _label_vector(path, ["contig"])
    assert labels.tolist() == [-1]


def test_row_count_mismatch_is_an_error() -> None:
    """A feature matrix with the wrong number of nodes fails before training."""
    with pytest.raises(ValueError, match="node_features"):
        _check_row_counts("toy", 4, node_features=3)
