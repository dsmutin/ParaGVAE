"""Persist tensors that later runs can reload without touching the assemblies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paragvae.graphs import StudyGraph
from paragvae.metametro_path import ensure_metametro

ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATES = ROOT / "intermediates"


def save_study(graph: StudyGraph, root: Path | None = None) -> Path:
    """Write features, labels, CSR, and a validated CGT directory."""
    ensure_metametro()
    from metametro.formats.cgt.io import dump_cgt

    dest = (root or INTERMEDIATES) / graph.name
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "vae_features.npy", graph.vae_features)
    np.save(dest / "raw_features.npy", graph.raw_features)
    np.save(dest / "labels.npy", graph.labels)
    np.save(dest / "different_pairs.npy", graph.different_pairs)
    np.save(dest / "node_colors.npy", np.asarray(graph.cgt.node_colors))
    dump_cgt(graph.cgt, dest / "cgt")
    return dest
