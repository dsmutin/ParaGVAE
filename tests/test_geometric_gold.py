"""Mandatory: geometric gold is read from ground_truth/, never from colors.tsv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.ds.analyze import route_cohesion
from paragvae.ds.load import (
    GEOMETRIC_IDENTITY_PREFIXES,
    bench_identity,
    geometric_bench_dir,
    load_geometric,
    sample_color_counts,
)
from paragvae.metametro_path import ensure_metametro

pytestmark = pytest.mark.mandatory


def _transit_graph():
    from metametro.transit_cfa import TransitRoute, transit_cfa

    stops = {"a": "Alpha", "b": "Beta", "c": "Gamma", "d": "Delta"}
    routes = [
        TransitRoute("bus2", "2", "bus"),
        TransitRoute("tram1", "1", "tram"),
        TransitRoute("bus9", "9", "bus"),
    ]
    trips = [("bus2", ["a", "b", "c"]), ("tram1", ["b", "c", "d"])]
    coordinates = {"a": (30.0, 59.0), "b": (30.1, 59.1), "c": (30.2, 59.2), "d": (30.3, 59.3)}
    return transit_cfa(stops, routes, trips, graph_id="toy_transit", k=5, coordinates=coordinates)


def _roxel_graph():
    from metametro.digraph_cfa import coloured_digraph_cfa

    nodes = [
        {"node_id": "n1", "longitude": "0.1", "latitude": "1.2"},
        {"node_id": "n2", "longitude": "0.2", "latitude": "1.3"},
        {"node_id": "n3", "longitude": "0.3", "latitude": "1.4"},
    ]
    edges = [
        {"source": "n1", "target": "n2", "type": "residential", "name": "Roxelweg"},
        {"source": "n1", "target": "n3", "type": "path", "name": "Pfad"},
    ]
    return coloured_digraph_cfa(
        nodes,
        edges,
        graph_id="toy_roxel",
        k=5,
        colour_columns=("type", "name"),
        keep_columns=("longitude", "latitude"),
    )


def _materialize(graph, outdir: Path, gold_namespaces: tuple[str, ...]) -> Path:
    ensure_metametro()
    from metametro.bench.colourings import BuildContext
    from metametro.bench.materialize import materialize

    materialize(graph, outdir, BuildContext(), gold_namespaces=gold_namespaces)
    return outdir


@pytest.fixture
def transit_bench(tmp_path: Path) -> Path:
    return _materialize(_transit_graph(), tmp_path / "spb", ("transport_type", "route"))


@pytest.fixture
def roxel_bench(tmp_path: Path) -> Path:
    return _materialize(_roxel_graph(), tmp_path / "roxel", ("type", "name"))


def _gold_namespaces(graph: dict) -> set[str]:
    return set(graph["gold"])


def test_sample_color_counts_ignores_non_sample_namespaces() -> None:
    colors = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    names = [("sample", "s1"), ("taxon", "t1"), ("sample", "s2")]
    assert np.array_equal(sample_color_counts(colors, names), np.array([2, 1, 1], dtype=np.float64))
    only_gold = [("transport_type", "bus"), ("route", "1"), ("type", "path")]
    assert np.array_equal(sample_color_counts(colors, only_gold), np.zeros(3, dtype=np.float64))


def test_transit_gold_is_read_from_ground_truth(transit_bench: Path) -> None:
    graph = load_geometric("spb_transit", outdir=transit_bench)
    assert (transit_bench / "ground_truth" / "transport_type_nodes.tsv").is_file()
    assert (transit_bench / "ground_truth" / "route_nodes.tsv").is_file()
    assert _gold_namespaces(graph) == {"transport_type", "route"}
    type_gold = graph["gold"]["transport_type"]
    route_gold = graph["gold"]["route"]
    assert type_gold["values"] == ("bus", "trolley", "tram")
    assert "bus:2" in route_gold["values"]
    assert type_gold["node_colors"].shape[0] == graph["features"].shape[0]
    assert route_gold["node_colors"].shape[0] == graph["features"].shape[0]
    assert graph["label_names"]
    assert graph["labels"].shape[0] == graph["features"].shape[0]
    n_modes = type_gold["node_colors"].sum(axis=1)
    n_routes = route_gold["node_colors"].sum(axis=1)
    assert n_modes.shape[0] == graph["features"].shape[0]
    assert n_routes.shape[0] == graph["features"].shape[0]
    scored = route_cohesion(
        route_gold["node_colors"],
        route_gold["names"],
        np.zeros(graph["features"].shape[0], dtype=np.int64),
        np.zeros((graph["features"].shape[0], 2)),
        min_stops=1,
    )
    assert not scored.empty
    assert set(scored["route"]) <= set(route_gold["values"])


def test_roxel_gold_is_type_and_name_from_ground_truth(roxel_bench: Path) -> None:
    graph = load_geometric("roxel", outdir=roxel_bench)
    assert (roxel_bench / "ground_truth" / "type_nodes.tsv").is_file()
    assert (roxel_bench / "ground_truth" / "name_nodes.tsv").is_file()
    assert _gold_namespaces(graph) == {"type", "name"}
    assert graph["labels"].shape[0] == graph["features"].shape[0]
    assert graph["label_names"]
    name_values = set(graph["gold"]["name"]["values"])
    assert {"Roxelweg", "Pfad"} <= name_values
    # Street names live in name gold / edge targets, not in leftover colour columns.
    color_values = {value for _namespace, value in graph["color_names"]}
    assert "Roxelweg" not in color_values and "Pfad" not in color_values


def test_old_colors_tsv_layout_raises_without_falling_back(tmp_path: Path) -> None:
    outdir = tmp_path / "old_spb"
    (outdir / "cdbg").mkdir(parents=True)
    (outdir / "cdbg" / "colors.tsv").write_text(
        "color_id\tnamespace\tvalue\n"
        "0\ttransport_type\tbus\n"
        "1\ttransport_type\ttram\n"
        "2\troute\tbus:2\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="ground_truth") as raised:
        load_geometric("spb_transit", outdir=outdir)
    message = str(raised.value)
    assert "colors.tsv" in message
    assert "0.23.0" in message
    assert "does not fall back" in message
    assert "transport_type" in message and "route" in message


def test_gold_never_appears_in_colouring_layers(transit_bench: Path, roxel_bench: Path) -> None:
    ensure_metametro()
    from metametro.bench import load_bench_cdbg, load_bench_cgt, namespaces_for
    from metametro.errors import ContractError

    gold = {
        transit_bench: {"transport_type", "route"},
        roxel_bench: {"type", "name"},
    }
    for folder, namespaces in gold.items():
        alias = "spb_transit" if folder is transit_bench else "roxel"
        graph = load_geometric(alias, outdir=folder)
        present = {namespace for namespace, _value in graph["color_names"]}
        assert not present & namespaces
        cdbg = load_bench_cdbg(folder)
        cdbg_ns = {str(row.get("namespace")) for row in (cdbg.colors or []) if row.get("namespace")}
        assert not cdbg_ns & namespaces
        for namespace in namespaces:
            with pytest.raises(ContractError, match="load_bench_gold"):
                load_bench_cgt(folder, namespaces=(namespace,))
        composition = namespaces_for(("composition_kmeans",))
        if composition and set(composition) <= cdbg_ns:
            coloured = load_bench_cgt(folder, namespaces=composition)
            by_id = {int(row["color_id"]): str(row["namespace"]) for row in cdbg.colors or []}
            kept = {by_id[int(cid)] for cid in coloured.color_ids}
            assert kept <= set(composition)
            assert not kept & namespaces


@pytest.mark.parametrize(
    "alias,bench",
    [("spb_transit", "spb_ground_transit"), ("roxel", "roxel")],
)
def test_published_identity_of_a_built_geometric_bench(alias: str, bench: str) -> None:
    """Deferred unless a real default build is present: this box has no Rscript or GTFS.

    When the bench exists at MetaMetro's default_outdir, the recorded identity
    must start with the prefix published in MetaMetro v0.24.0 docs/benchmarks.md
    (cfd1dcfb for spb_ground_transit, 3232245d for roxel). The full hash is
    not hard-coded here.
    """
    folder = geometric_bench_dir(alias)
    built = (folder / "manifest.yaml").is_file() or (folder / "identity.sha256").is_file()
    prefix = GEOMETRIC_IDENTITY_PREFIXES[bench]
    if not built:
        pytest.skip(
            f"{bench} is not built at {folder}; this box has no Rscript or GTFS to build it. "
            "When a default MetaMetro build is present, this test checks that "
            f"bench_identity(folder) starts with the published prefix {prefix}."
        )
    identity = bench_identity(folder)
    assert identity is not None
    assert identity.startswith(prefix)
