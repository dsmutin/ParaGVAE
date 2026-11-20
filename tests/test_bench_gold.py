"""Mandatory: MetaMetro bench gold and the composition-only colouring arm."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from paragvae.graphs import (
    labels_from_gold_colors,
    load_metametro_bench,
    load_metametro_bubble,
    load_vaegbin_bundle,
)
from paragvae.metametro_path import ensure_metametro
from paragvae.study import Arm, _features, _topology, composition_color_matrix, run_arm

pytestmark = pytest.mark.mandatory


@pytest.fixture(scope="module")
def bubble_strain_2(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the in-process bench with the pinned MetaMetro."""
    ensure_metametro()
    from metametro.bench import build

    outdir = tmp_path_factory.mktemp("benches") / "bubble_strain_2"
    outdir.mkdir()
    build("bubble_strain_2", outdir=outdir)
    return outdir


def test_bench_gold_labels_are_non_empty(bubble_strain_2: Path) -> None:
    """Study labels come from load_bench_gold_cgt and are not an empty taxon layer."""
    ensure_metametro()
    from metametro.bench import load_bench_gold_cgt
    from metametro.errors import ContractError

    graph = load_metametro_bench(bubble_strain_2)
    gold_cgt = load_bench_gold_cgt(bubble_strain_2)
    expected = labels_from_gold_colors(gold_cgt.node_colors)
    assert graph.labels.shape[0] == graph.cgt.num_nodes
    assert int((graph.labels >= 0).sum()) > 0
    assert np.array_equal(graph.labels, expected)
    assert graph.gold is not None
    assert graph.bench_dir == bubble_strain_2
    with pytest.raises(ContractError, match="gold ground truth"):
        from metametro.bench import load_bench_cgt

        load_bench_cgt(bubble_strain_2, namespaces=["taxon"])


def test_coloring_arm_uses_only_composition(bubble_strain_2: Path) -> None:
    """cgt_colors sees the composition layer, not decaying or other namespaces."""
    ensure_metametro()
    from metametro.bench import load_bench_cgt, namespaces_for

    graph = load_metametro_bench(bubble_strain_2)
    composition = namespaces_for(("composition_kmeans",))
    comp = load_bench_cgt(bubble_strain_2, namespaces=composition)
    full = load_bench_cgt(bubble_strain_2)
    layer = composition_color_matrix(graph)
    assert layer.shape[1] == comp.node_colors.shape[1]
    assert layer.shape[1] < full.node_colors.shape[1]
    assert np.array_equal(layer, comp.node_colors)
    arm = Arm("coloring", "cgt_colors", colors=True, max_epochs=1, patience=1)
    indptr, indices = _topology(graph, arm)
    features = _features(graph, arm, indptr, indices)
    assert features.shape[1] == graph.vae_features.shape[1] + layer.shape[1]


def test_mock_and_vaegbin_skip_missing_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Graphs without a bench use k-means on raw features, not filter_colours."""
    ensure_metametro()
    from metametro.contracts.colour_filter import filter_colours
    from metametro.bench import load_bench_cgt, load_bench_cdbg, load_bench_cfa

    def _forbidden(*_args, **kwargs):
        if kwargs.get("namespaces"):
            raise AssertionError("must not request a namespace on a graph that may lack it")
        raise AssertionError("filter_colours / load_bench_* must not run on the fixture path")

    monkeypatch.setattr("metametro.contracts.colour_filter.filter_colours", _forbidden)
    monkeypatch.setattr("metametro.bench.select.filter_colours", _forbidden)
    monkeypatch.setattr("metametro.bench.load_bench_cgt", _forbidden)
    monkeypatch.setattr("metametro.bench.load_bench_cdbg", _forbidden)
    monkeypatch.setattr("metametro.bench.load_bench_cfa", _forbidden)
    monkeypatch.setattr("metametro.bench.select.load_bench_cgt", _forbidden)
    monkeypatch.setattr("metametro.bench.select.load_bench_cdbg", _forbidden)
    monkeypatch.setattr("metametro.bench.select.load_bench_cfa", _forbidden)
    # The imported aliases may already be bound; also patch the functions if used later.
    monkeypatch.setattr(filter_colours, "__call__", _forbidden, raising=False)

    bubble = load_metametro_bubble()
    assert bubble.bench_dir is None
    layer = composition_color_matrix(bubble)
    assert layer.shape == (bubble.cgt.num_nodes, 0)
    arm = Arm("coloring", "cgt_colors", colors=True, max_epochs=1, patience=1)
    indptr, indices = _topology(bubble, arm)
    from paragvae.graphs import composition_colors

    features = _features(bubble, arm, indptr, indices)
    fallback = composition_colors(bubble.raw_features, n_colors=8, seed=0)
    sample_width = int(np.asarray(bubble.cgt.node_colors).shape[1])
    assert sample_width > 0
    assert fallback.shape[1] != sample_width
    assert features.shape[1] == bubble.vae_features.shape[1] + fallback.shape[1]

    # A tiny VAEGbin-style bundle also has no composition layer.
    from scipy import sparse

    folder = tmp_path / "bundle"
    folder.mkdir()
    adjacency = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32))
    sparse.save_npz(folder / "adj_sparse.npz", adjacency)
    features = np.eye(3, 4, dtype=np.float32)
    np.save(folder / "node_features.npy", features)
    np.save(folder / "node_attributes_kmer.npy", features)
    np.save(folder / "node_attributes_depth.npy", np.ones(3, dtype=np.float32))
    np.save(folder / "node_names.npy", np.array(["c0", "c1", "c2"], dtype=object))
    np.save(folder / "node_to_label.npy", np.array([0, 0, 1], dtype=np.int64))
    toy = load_vaegbin_bundle(folder, name="toy")
    assert toy.bench_dir is None
    assert composition_color_matrix(toy).shape[1] == 0


def test_run_arm_warns_when_amber_gold_is_missing(tmp_path: Path) -> None:
    """amber: auto warns once and still scores ARI/F1 when labels exist."""
    graph = load_metametro_bubble()
    graph.gold = None
    arm = Arm("feature_source", "vae_latent", max_epochs=2, patience=1, lr=0.05)
    with pytest.warns(UserWarning, match="no AMBER gold standard"):
        row = run_arm(graph, arm, seed=0, work=tmp_path / "job", amber="auto")
    assert row["status"] == "ok"
    assert isinstance(row["ari"], float) and not math.isnan(row["ari"])
    assert row["amber_f1"] == ""


def test_run_arm_does_not_score_empty_truth_as_zero(tmp_path: Path) -> None:
    """Fewer than two usable labels yields no_labels and NaN, not ARI 0."""
    graph = load_metametro_bubble()
    graph.labels = np.full(graph.labels.shape, -1, dtype=np.int64)
    arm = Arm("feature_source", "vae_latent", max_epochs=2, patience=1, lr=0.05)
    with pytest.warns(UserWarning, match="fewer than two labelled nodes"):
        row = run_arm(graph, arm, seed=0, work=tmp_path / "empty", amber="false")
    assert row["status"] == "no_labels"
    assert math.isnan(row["ari"])
    assert math.isnan(row["f1"])
    assert row["ari"] != 0
