"""Mandatory: the colouring arm reads only composition_kmeans, never a foreign namespace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae import study
from paragvae.graphs import composition_colors, load_metametro_bench, load_metametro_bubble
from paragvae.metametro_path import ensure_metametro
from paragvae.study import Arm, _features, _topology, coloring_source, composition_color_matrix, run_arm

pytestmark = pytest.mark.mandatory

# Contig-style Kraken2 output for the four bubble_strain_2 nodes. A file
# stands in for the classifier, so no index and no network are needed.
KRAKEN = "C\tS\t10665\t40\t10665:10\nC\tA\t10760\t40\t10760:10\nC\tB\t10754\t40\t10754:10\nU\tT\t0\t40\t0:10\n"


def _build(root: Path, name: str, colourings: tuple[str, ...]) -> Path:
    ensure_metametro()
    from metametro.bench import build

    kraken = root / "nodes.kraken"
    kraken.write_text(KRAKEN, encoding="utf-8")
    outdir = root / name
    outdir.mkdir()
    build("bubble_strain_2", outdir=outdir, colourings=colourings, kraken_output=kraken)
    return outdir


@pytest.fixture(scope="module")
def foreign_bench(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """bubble_strain_2 with composition, decaying, and kraken2 colours."""
    root = tmp_path_factory.mktemp("foreign")
    return _build(root, "with_composition", ("composition_kmeans", "decaying", "kraken2"))


@pytest.fixture(scope="module")
def no_composition_bench(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """bubble_strain_2 with only a kraken2 colour layer."""
    root = tmp_path_factory.mktemp("no_composition")
    return _build(root, "kraken_only", ("kraken2",))


def _namespaces(outdir: Path) -> set[str]:
    from metametro.bench import load_bench_cdbg

    return {str(row["namespace"]) for row in load_bench_cdbg(outdir).colors}


def test_fixture_carries_foreign_namespaces(foreign_bench: Path, no_composition_bench: Path) -> None:
    assert {"composition", "decaying", "kraken2"} <= _namespaces(foreign_bench)
    assert _namespaces(no_composition_bench) == {"kraken2"}


def test_loader_keeps_only_composition_columns(foreign_bench: Path) -> None:
    """node_colors and color_ids hold the composition layer and nothing else."""
    from metametro.bench import load_bench_cdbg, load_bench_cgt, namespaces_for

    graph = load_metametro_bench(foreign_bench)
    composition = namespaces_for(("composition_kmeans",))
    comp_cdbg = load_bench_cdbg(foreign_bench, namespaces=composition)
    by_id = {int(row["color_id"]): str(row["namespace"]) for row in comp_cdbg.colors}
    assert {by_id[int(cid)] for cid in graph.cgt.color_ids} == set(composition)

    # The same columns, read off the full CGT by namespace.
    full_cdbg = load_bench_cdbg(foreign_bench)
    full_ns = {int(row["color_id"]): str(row["namespace"]) for row in full_cdbg.colors}
    full = load_bench_cgt(foreign_bench)
    keep = [col for col, cid in enumerate(full.color_ids) if full_ns[int(cid)] in composition]
    assert 0 < len(keep) < full.node_colors.shape[1]
    stored = np.asarray(graph.cgt.node_colors)
    assert stored.shape == (full.num_nodes, len(keep))
    assert np.array_equal(stored, np.asarray(full.node_colors)[:, keep])
    assert np.asarray(graph.cgt.edge_colors).shape[1] == len(keep)

    arm = Arm("coloring", "cgt_colors", colors=True, max_epochs=1, patience=1)
    assert coloring_source(graph, arm) == "composition_kmeans"
    indptr, indices = _topology(graph, arm)
    features = _features(graph, arm, indptr, indices)
    assert features.shape[1] == graph.vae_features.shape[1] + len(keep)
    assert np.array_equal(features[:, -len(keep) :], stored.astype(np.float32))


def test_bench_without_composition_does_not_fall_back(
    no_composition_bench: Path, tmp_path: Path
) -> None:
    """A bench that lacks composition_kmeans fails on the colouring arm, not k-means."""
    graph = load_metametro_bench(no_composition_bench)
    assert np.asarray(graph.cgt.node_colors).shape == (graph.cgt.num_nodes, 0)
    assert list(graph.cgt.color_ids) == []
    assert composition_color_matrix(graph).shape[1] == 0
    assert graph.bench_dir == no_composition_bench

    arm = Arm("coloring", "cgt_colors", colors=True, max_epochs=1, patience=1)
    with pytest.raises(ValueError, match="does not fall back to k-means"):
        coloring_source(graph, arm)
    indptr, indices = _topology(graph, arm)
    with pytest.raises(ValueError, match="does not fall back to k-means"):
        _features(graph, arm, indptr, indices)
    with pytest.raises(ValueError, match="does not fall back to k-means"):
        run_arm(graph, arm, seed=0, work=tmp_path / "job", amber="false")

    plain = Arm("coloring", "uncoloured", max_epochs=1, patience=1)
    assert coloring_source(graph, plain) == ""
    row = run_arm(graph, plain, seed=0, work=tmp_path / "plain", amber="false")
    assert row["coloring_source"] == ""


def test_graph_without_a_bench_uses_kmeans_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mock_cgt / VAEGbin graphs warn once and record coloring_source=kmeans_raw."""
    monkeypatch.setattr(study, "_WARNED", set())
    graph = load_metametro_bubble()
    assert graph.bench_dir is None
    assert composition_color_matrix(graph).shape[1] == 0

    arm = Arm("coloring", "cgt_colors", colors=True, max_epochs=1, patience=1)
    assert coloring_source(graph, arm) == "kmeans_raw"
    indptr, indices = _topology(graph, arm)
    with pytest.warns(UserWarning, match="coloring_source=kmeans_raw"):
        features = _features(graph, arm, indptr, indices)
    fallback = composition_colors(graph.raw_features, n_colors=8, seed=0)
    assert features.shape[1] == graph.vae_features.shape[1] + fallback.shape[1]
    assert np.array_equal(features[:, -fallback.shape[1] :], fallback.astype(np.float32))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _features(graph, arm, indptr, indices)
        row = run_arm(graph, arm, seed=0, work=tmp_path / "job", amber="false")
    assert row["coloring_source"] == "kmeans_raw"

    plain = Arm("coloring", "uncoloured", max_epochs=1, patience=1)
    assert coloring_source(graph, plain) == ""


def test_run_arm_records_composition_source(foreign_bench: Path, tmp_path: Path) -> None:
    graph = load_metametro_bench(foreign_bench)
    arm = Arm("coloring", "cgt_colors", colors=True, max_epochs=1, patience=1)
    row = run_arm(graph, arm, seed=0, work=tmp_path / "job", amber="false")
    assert row["coloring_source"] == "composition_kmeans"
