"""Mandatory: the hypothesis grid can start without AMBER gold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from paragvae.graphs import load_metametro_bubble
from paragvae.study import Arm, amber_mode, run_arm
from paragvae.suite import load_graphs

pytestmark = pytest.mark.mandatory


def _toy_bundle(folder: Path) -> None:
    folder.mkdir(parents=True)
    adjacency = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    sparse.save_npz(folder / "adj_sparse.npz", adjacency)
    features = np.eye(3, 4, dtype=np.float32)
    np.save(folder / "node_features.npy", features)
    np.save(folder / "node_attributes_kmer.npy", features)
    np.save(folder / "node_attributes_depth.npy", np.ones(3, dtype=np.float32))
    np.save(folder / "node_names.npy", np.array(["c0", "c1", "c2"], dtype=object))
    np.save(folder / "node_to_label.npy", np.array([0, 0, 1], dtype=np.int64))


def _toy_config(root: Path, amber: object) -> dict:
    return {
        "vaegbin_data": str(root),
        "bundles": {"toy": "bundle"},
        "gold": {"toy": ""},
        "amber": amber,
        "seeds": [0],
        "max_epochs": 2,
        "patience": 1,
        "lr": 0.05,
    }


def test_grid_starts_without_gold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A toy bundle with empty gold paths still yields F1, ARI, and empty amber columns."""
    _toy_bundle(tmp_path / "bundle")
    monkeypatch.delenv("VAEGBIN_GOLD_TOY", raising=False)
    config = _toy_config(tmp_path, "auto")
    with pytest.warns(UserWarning, match="AMBER is disabled"):
        graphs = load_graphs(config)
    toy = next(graph for graph in graphs if graph.name == "toy")
    assert toy.gold is None
    arm = Arm("feature_source", "vae_latent", max_epochs=2, patience=1, lr=0.05)
    row = run_arm(toy, arm, seed=0, work=tmp_path / "job", amber=amber_mode(config.get("amber")))
    assert isinstance(row["f1"], float)
    assert isinstance(row["ari"], float)
    assert row["amber_f1"] == ""
    assert row["amber_ap"] == ""
    assert row["amber_ar"] == ""


def test_amber_true_without_gold_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """amber: true must fail before training when gold is missing."""
    _toy_bundle(tmp_path / "bundle")
    monkeypatch.delenv("VAEGBIN_GOLD_TOY", raising=False)
    with pytest.raises(ValueError, match="no AMBER gold standard"):
        load_graphs(_toy_config(tmp_path, True))
    graph = load_metametro_bubble()
    graph.gold = None
    arm = Arm("feature_source", "vae_latent", max_epochs=2, patience=1, lr=0.05)
    with pytest.raises(ValueError, match="no AMBER gold standard"):
        run_arm(graph, arm, seed=0, work=tmp_path / "job", amber="true")


def test_invalid_amber_mode_is_an_error() -> None:
    """The grid flag only accepts auto, true, or false."""
    with pytest.raises(ValueError, match="amber must be"):
        amber_mode("sometimes")
