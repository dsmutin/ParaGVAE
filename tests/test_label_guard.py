"""Mandatory: evaluation taxids stay out of model inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.provenance_checks import (
    assert_labels_are_simulation_taxids,
    assert_training_inputs_hide_labels,
    simulation_tax_ids,
)

pytestmark = pytest.mark.mandatory


def test_simulation_tax_ids_keep_only_the_sim_role(tmp_path: Path):
    table = tmp_path / "accessions.tsv"
    table.write_text(
        "pair_id\trole\taccession\ttax_id\n"
        "a\tsim\tGCF_000000001.1\t10\n"
        "a\tdb\tGCF_000000002.1\t99\n",
        encoding="utf-8",
    )
    assert simulation_tax_ids(table) == {"GCF_000000001": 10}


def test_foreign_label_is_rejected():
    with pytest.raises(ValueError, match="not sim tax ids"):
        assert_labels_are_simulation_taxids(np.array([10, 11]), {"GCF_000000001": 10})


def test_feature_column_equal_to_the_label_is_rejected():
    labels = np.array([10, 20, 10])
    features = np.stack([labels, np.array([1, 2, 3])], axis=1).astype(np.float32)
    with pytest.raises(ValueError, match="features column 0"):
        assert_training_inputs_hide_labels(labels, features)


def test_unrelated_features_pass():
    labels = np.array([10, 20, 10])
    features = np.array([[100.0, 0.5, 1.0], [80.0, 0.4, 2.0], [90.0, 0.5, 1.0]], dtype=np.float32)
    assert_training_inputs_hide_labels(labels, features, colors=np.zeros((3, 0)))
