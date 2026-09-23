"""Mandatory: contig F1 does not depend on cluster id order."""

from __future__ import annotations

import numpy as np
import pytest

from paragvae.score import contig_f1, evaluate

pytestmark = pytest.mark.mandatory


def test_contig_f1_is_invariant_to_cluster_ids() -> None:
    """Two clusters that share a majority genome must be matched by overlap, not id order."""
    truth = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    predicted = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    renamed = np.where(predicted == 0, 9, predicted)
    assert contig_f1(truth, predicted) == pytest.approx(0.8)
    assert contig_f1(truth, renamed) == pytest.approx(contig_f1(truth, predicted))


def test_ari_ignores_unlabelled_nodes() -> None:
    """Label -1 is missing data, not a genome."""
    embedding = np.array([[0.0], [0.0], [5.0], [5.0], [9.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, -1])
    scores = evaluate(embedding, labels, "kmeans", seed=0)
    assert scores["ari"] == pytest.approx(1.0)
