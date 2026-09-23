"""Mandatory: pickled VAEGbin label dicts become dense integer ids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.graphs import _label_vector

pytestmark = pytest.mark.mandatory


def test_dict_labels_follow_node_ids(tmp_path: Path) -> None:
    """String-keyed genome ids line up with the node name order."""
    path = tmp_path / "node_to_label.npy"
    np.save(path, {"b": 4, "a": 2})
    labels = _label_vector(path, ["a", "b", "missing"])
    assert labels.tolist() == [2, 4, -1]
