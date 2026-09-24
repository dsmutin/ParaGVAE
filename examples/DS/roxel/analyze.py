#!/usr/bin/env python3
"""Where the Roxel street GCN mis-assigns a vertex's road type."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import analyze  # noqa: E402
from load import load_geometric, training_budget  # noqa: E402


def main() -> None:
    """Predict road type from longitude and latitude. Node colours are empty."""
    graph = load_geometric("roxel")
    if graph["feature_names"] != ["longitude", "latitude"]:
        raise RuntimeError(f"unexpected Roxel features {graph['feature_names']}")
    if int(graph["node_colors"].sum()) != 0:
        raise RuntimeError("Roxel node colours were expected to be empty")
    folded = {}
    for name in graph["label_names"]:
        folded.setdefault(name.casefold(), []).append(name)
    split = [pair for pair in folded.values() if len(pair) > 1]
    features = graph["features"]
    center = features.mean(axis=0)
    distance = np.linalg.norm(features - center, axis=1)
    budget = training_budget()
    protocol = (
        "MetaMetro `data/work/roxel`, the sfnetworks Roxel street example. "
        "The node class is the road type. The node colour matrix is all zeros; street names are edge colours "
        "and are not the node target. Model inputs are longitude and latitude only. "
        f"Standard edge loss, at most {budget['max_epochs']} epochs, patience {budget['patience']}, "
        f"learning rate {budget['lr']}. Seeds 0 and 1."
    )
    if split:
        protocol += " The stored class names include a case split: " + ", ".join(
            " and ".join(pair) for pair in split
        ) + ". They are kept as separate classes."
    analyze(
        name="roxel",
        features=features,
        labels=graph["labels"],
        label_names=graph["label_names"],
        indptr=graph["indptr"],
        indices=graph["indices"],
        edge_weight=graph["edge_weight"],
        extras={"dist_from_mean_coordinate": (distance, "model_input")},
        out=Path(__file__).resolve().parent,
        protocol=protocol,
        **budget,
    )


if __name__ == "__main__":
    main()
