#!/usr/bin/env python3
"""Where the Saint Petersburg transit GCN mis-assigns a stop's transport type."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import _save_chart, analyze, route_cohesion  # noqa: E402
from load import load_geometric, training_budget  # noqa: E402


def main() -> None:
    """Predict the transport-type combination, then score route cohesion in that embedding."""
    graph = load_geometric("spb_transit")
    if graph["feature_names"] != ["longitude", "latitude", "coverage"]:
        raise RuntimeError(f"unexpected SPB features {graph['feature_names']}")
    features = graph["features"]
    center = features[:, :2].mean(axis=0)
    distance = np.linalg.norm(features[:, :2] - center, axis=1)
    type_gold = graph["gold"]["transport_type"]
    route_gold = graph["gold"]["route"]
    if len(type_gold["values"]) != 3:
        raise RuntimeError(f"expected 3 transport types, found {len(type_gold['values'])}")
    budget = training_budget()
    out = Path(__file__).resolve().parent
    analyze(
        name="spb_transit",
        features=features,
        labels=graph["labels"],
        label_names=graph["label_names"],
        indptr=graph["indptr"],
        indices=graph["indices"],
        edge_weight=graph["edge_weight"],
        extras={
            "coverage": (features[:, 2], "model_input"),
            "dist_from_mean_coordinate": (distance, "model_input"),
            "n_modes": (type_gold["node_colors"].sum(axis=1).astype(np.float64), "not_in_model"),
            "n_routes": (route_gold["node_colors"].sum(axis=1).astype(np.float64), "not_in_model"),
        },
        out=out,
        epilogue=(
            "Seed-0 error is exactly the non-bus stops: bus error rate is 0 on 3145 stops, and every other class, "
            "including tram, trolley, and every mixed label, has error rate 1. "
            "Stops with two or three transport modes (868 stops) are all errors. "
            "The boundary gap (0.131 with no foreign neighbour, 0.641 when every neighbour has another type) "
            "is the same split, because non-bus stops are the boundary of the bus majority."
        ),
        protocol=(
            "MetaMetro bench `spb_ground_transit` (default_outdir, MetaMetro >= 0.23.0). "
            "The node class is the sorted combination of `transport_type` gold from ground_truth/ "
            "(bus, tram, trolley, and the mixed labels), loaded with load_bench_states. "
            "It is not a single route id and is not read from colors.tsv or CGT colours. "
            "Model inputs are longitude, latitude, and the simulated passenger coverage from the CFA. "
            "Edge weight is that coverage on the hop. Route gold is multi-label and is scored afterwards "
            "as cluster cohesion, not as the k-means target. "
            f"Standard edge loss, at most {budget['max_epochs']} epochs, patience {budget['patience']}, "
            f"learning rate {budget['lr']}. Seeds 0 and 1."
        ),
        **budget,
    )
    import pandas as pd

    nodes = pd.read_csv(out / "nodes.csv")
    routes = route_cohesion(
        route_gold["node_colors"],
        route_gold["names"],
        nodes["cluster"].to_numpy(),
        features[:, :2],
    )
    if routes.empty:
        raise RuntimeError("no route colour had enough stops to score")
    routes.to_csv(out / "routes.csv", index=False)
    chart = (
        alt.Chart(routes)
        .mark_circle(size=50)
        .encode(
            x=alt.X("n_stops:Q", title="Stops on the route"),
            y=alt.Y("cohesion:Q", title="Fraction in the largest cluster", scale=alt.Scale(domain=[0, 1])),
            tooltip=["route", "n_stops", alt.Tooltip("cohesion:Q", format=".3f"), alt.Tooltip("null_cohesion:Q", format=".3f")],
        )
        .properties(width=360, height=240, title="Route cohesion in the type embedding")
    )
    _save_chart(chart, out / "route_cohesion")
    note = (
        "\n## Routes\n\n"
        "A route is a multi-label colour, so it is not the k-means target. "
        f"Among {len(routes)} routes with at least 8 stops, mean cohesion is {routes['cohesion'].mean():.3f}. "
        f"A size-matched shuffle of the same clusters has mean cohesion {routes['null_cohesion'].mean():.3f}. "
        "Span in `routes.csv` is the diagonal of the longitude/latitude box, in degrees.\n"
    )
    readme = out / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + note, encoding="utf-8")


if __name__ == "__main__":
    main()
