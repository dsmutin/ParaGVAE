"""Mandatory: study, registry, and DS scores match pre-unify evaluate goldens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.ds.analyze import _cluster
from paragvae.graphs import load_metametro_bubble
from paragvae.score import evaluate
from paragvae.study import Arm, run_arm

pytestmark = pytest.mark.mandatory

GOLDENS = Path(__file__).resolve().parent / "goldens" / "score"

_CASES = {
    "perfect": (np.array([0, 0, 1, 1, 2, 2]), np.array([5, 5, 7, 7, 8, 8])),
    "partial": (
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
        np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    ),
    "with_unlabelled": (np.array([0, 0, 1, 1, -1, -1]), np.array([0, 0, 1, 1, 2, 2])),
    "too_few_labelled": (np.array([0, -1, -1, -1]), np.array([0, 1, 1, 0])),
}


def _load(name: str) -> np.ndarray:
    return np.load(GOLDENS / f"{name}.npy")


def test_evaluate_matches_pre_move_unlabelled_fixture() -> None:
    """The public evaluate fixture, including label -1, is bit-identical."""
    embedding = np.array([[0.0], [0.0], [5.0], [5.0], [9.0]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, -1])
    scores = evaluate(embedding, labels, "kmeans", seed=0)
    assert scores["ari"] == _load("evaluate_unlabelled_ari")
    assert scores["f1"] == _load("evaluate_unlabelled_f1")
    assert np.array_equal(scores["predicted"], _load("evaluate_unlabelled_pred"))


def test_evaluate_matches_pre_move_study_and_ds_fixtures() -> None:
    """ARI and F1 for fixed label/prediction pairs match the captured study and DS paths."""
    embedding = np.zeros((1, 1), dtype=np.float64)
    for name, (labels, predicted) in _CASES.items():
        scores = evaluate(embedding, labels, "kmeans", seed=0, predicted=predicted)
        assert scores["ari"] == _load(f"study_{name}_ari")
        assert scores["f1"] == _load(f"study_{name}_f1")
        assert scores["ari"] == _load(f"ds_{name}_ari")
        assert scores["f1"] == _load(f"ds_{name}_f1")
        assert np.array_equal(scores["predicted"], predicted)


def test_evaluate_matches_pre_move_bubble_scores() -> None:
    """Scoring the captured bubble embedding does not change ARI or F1."""
    embedding = _load("bubble_embedding")
    labels = _load("bubble_labels")
    scores = evaluate(embedding, labels, "kmeans", seed=0)
    assert np.array_equal(scores["predicted"], _load("bubble_predicted"))
    assert scores["ari"] == _load("bubble_unrounded_ari")
    assert scores["f1"] == _load("bubble_unrounded_f1")
    clustered = _cluster(embedding, int(np.unique(labels).size), 0)
    analyzed = evaluate(embedding, labels, "kmeans", seed=0, predicted=clustered)
    assert np.array_equal(clustered, _load("bubble_ds_clusters"))
    assert analyzed["ari"] == _load("bubble_ds_ari")
    assert analyzed["f1"] == _load("bubble_ds_f1")


def test_run_arm_matches_pre_move_bubble_row(tmp_path: Path) -> None:
    """A short bubble arm still writes the same rounded ARI and F1."""
    graph = load_metametro_bubble()
    arm = Arm("feature_source", "vae_latent", max_epochs=2, patience=2, lr=0.05)
    row = run_arm(graph, arm, seed=0, work=tmp_path / "job", amber="false")
    assert row["ari"] == float(_load("run_arm_ari"))
    assert row["f1"] == float(_load("run_arm_f1"))
