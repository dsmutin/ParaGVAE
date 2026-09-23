"""Mandatory: the hypothesis grid stays finite and named."""

from __future__ import annotations

import pytest

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
    graphs = [arm.graph for arm in arms_for("graph_type", max_epochs=2, patience=1)]
    assert graphs == ["assembly", "knn_vae", "knn_kmer"]
