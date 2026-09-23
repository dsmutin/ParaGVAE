"""Mandatory: the hypothesis grid stays finite and named."""

from __future__ import annotations

import pytest

import numpy as np

from paragvae.study import Arm, arm_is_redundant
from paragvae.suite import HYPOTHESES, arms_for

pytestmark = pytest.mark.mandatory


def test_hypothesis_folders_are_stable() -> None:
    """Research folders match the historical questions, one factor at a time."""
    assert HYPOTHESES == (
        "feature_source",
        "loss",
        "coloring",
        "graph_type",
        "multiscale",
        "clustering",
    )
    losses = [arm.arm for arm in arms_for("loss", max_epochs=2, patience=1)]
    assert losses == ["standard", "diff_c", "proxy", "contrastive"]
    assert all(arm.lr == 0.05 for arm in arms_for("loss", max_epochs=2, patience=1, lr=0.05))
    graphs = [arm.graph for arm in arms_for("graph_type", max_epochs=2, patience=1)]
    assert graphs == ["assembly", "knn_vae", "knn_kmer"]


def test_duplicate_knn_arm_is_redundant() -> None:
    """Identical VAE and k-mer matrices must not be trained as two graph types."""
    graph = type("Graph", (), {})()
    graph.vae_features = np.ones((4, 3), dtype=np.float32)
    graph.raw_features = np.ones((4, 3), dtype=np.float32)
    arm = Arm("graph_type", "knn_kmer", graph="knn_kmer")
    assert arm_is_redundant(graph, arm)
    graph.raw_features = np.zeros((4, 3), dtype=np.float32)
    assert not arm_is_redundant(graph, arm)
