#!/usr/bin/env python3
"""Where the ONT 1B GCN puts a contig in the wrong majority genome."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import analyze, distance_to_class_mean  # noqa: E402
from load import load_vaegbin, training_budget  # noqa: E402


def main() -> None:
    """Train on k-mer and depth. This bundle has no separate VAE latent."""
    graph = load_vaegbin("ont1b")
    if not graph["vae_equals_raw"]:
        raise RuntimeError("ONT 1B was expected to store the same matrix as VAE and raw features")
    budget = training_budget()
    analyze(
        name="ont1b",
        features=graph["features"],
        labels=graph["labels"],
        label_names=graph["label_names"],
        indptr=graph["indptr"],
        indices=graph["indices"],
        edge_weight=graph["edge_weight"],
        extras={
            "depth": (graph["raw"][:, -1], "model_input"),
            "kmer_distance_to_class_mean": (
                distance_to_class_mean(graph["raw"][:, :-1], graph["labels"]),
                "diagnostic_uses_labels",
            ),
        },
        out=Path(__file__).resolve().parent,
        protocol=(
            "This bundle has no separate 32-d VAE latent: `node_features.npy` equals k-mer (136-d) plus depth. "
            "The GCN therefore sees depth. The assembly graph is the CSR from the VAEGbin bundle. "
            f"Standard edge loss, at most {budget['max_epochs']} epochs, patience {budget['patience']}, "
            f"learning rate {budget['lr']}. Seeds 0 and 1."
        ),
        **budget,
    )


if __name__ == "__main__":
    main()
