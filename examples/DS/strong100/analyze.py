#!/usr/bin/env python3
"""Where the Strong100 GCN puts a contig in the wrong majority genome."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import analyze, distance_to_class_mean  # noqa: E402
from load import load_vaegbin, training_budget  # noqa: E402


def main() -> None:
    """Train on the 32-d VAE latent and test depth, degree, and k-mer outliers."""
    graph = load_vaegbin("strong100")
    if graph["vae_equals_raw"]:
        raise RuntimeError("Strong100 was expected to have a VAE latent distinct from k-mer+depth")
    budget = training_budget()
    analyze(
        name="strong100",
        features=graph["features"],
        labels=graph["labels"],
        label_names=graph["label_names"],
        indptr=graph["indptr"],
        indices=graph["indices"],
        edge_weight=graph["edge_weight"],
        extras={
            "depth": (graph["raw"][:, -1], "not_in_model"),
            "kmer_distance_to_class_mean": (
                distance_to_class_mean(graph["raw"][:, :-1], graph["labels"]),
                "diagnostic_uses_labels",
            ),
        },
        out=Path(__file__).resolve().parent,
        protocol=(
            "Model inputs are the published 32-d VAE latent in `node_features.npy`, on the assembly CSR. "
            "Depth is `node_attributes_depth.npy` and is checked to be the last column of the raw matrix. "
            "The k-mer matrix is 136-d and is not a model input on this dataset. "
            f"Standard edge loss, at most {budget['max_epochs']} epochs, patience {budget['patience']}, "
            f"learning rate {budget['lr']}. Seeds 0 and 1."
        ),
        **budget,
    )


if __name__ == "__main__":
    main()
