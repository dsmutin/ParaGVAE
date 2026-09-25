"""Graphs for the completion benchmarks.

``heldout_genera``, ``low75half``, and ``half100half`` are CDBG unitig graphs.
Evaluation labels are species ids from the assembly report. Kraken calls, when
the example stored them, are a separate training target for the biological
loss. ``phage_x10`` is the MetaMetro coloured tensor. It has no Kraken call
table, so the biological arm cannot run on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paragvae.bioloss import BioTargets, load_kraken_targets
from paragvae.graphs import StudyGraph
from paragvae.heldout import cdbg_dir, load_heldout_graph
from paragvae.metametro_path import ensure_metametro


def completion_paths(config: dict) -> dict[str, Path]:
    """Absolute paths of the four completion graphs from the dataset config."""
    block = config.get("completion")
    if not isinstance(block, dict):
        raise ValueError("configs/datasets.yaml is missing the completion paths")
    required = ("phage_x10", "heldout_genera", "low75half", "half100half")
    missing = [name for name in required if not block.get(name)]
    if missing:
        raise ValueError(f"completion paths are missing: {', '.join(missing)}")
    return {name: Path(block[name]) for name in required}


def load_completion(name: str, path: Path, metametro_src: str | Path) -> tuple[StudyGraph, BioTargets | None]:
    """Load one completion graph and its Kraken targets, if that file exists."""
    if name == "phage_x10":
        return _phage(path, metametro_src), None
    graph = load_heldout_graph(path, metametro_src)
    calls = path / "work" / "reprofile" / "kraken_calls.tsv"
    counts = path / "work" / "reprofile" / "kraken_counts.tsv"
    if not calls.is_file() or not counts.is_file():
        return graph, None
    ensure_metametro(metametro_src)
    from metametro.formats.cdbg.io import load_cdbg

    unitigs = sorted(load_cdbg(cdbg_dir(path)).unitigs, key=lambda unitig: unitig.unitig_id)
    if len(unitigs) != len(graph.labels):
        raise ValueError(f"{name} unitig count does not match the loaded graph")
    members = [[str(member) for member in unitig.members] for unitig in unitigs]
    lengths = np.asarray(graph.raw_features[:, 0], dtype=np.float64)
    return graph, load_kraken_targets(calls, counts, members, lengths)


def _phage(path: Path, metametro_src: str | Path) -> StudyGraph:
    folder = Path(path) / "cgt"
    required = ("indptr.npy", "indices.npy", "node_features.npy", "node_labels.npy", "edge_features.npy")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"phage CGT is missing {', '.join(missing)} under {folder}")
    ensure_metametro(metametro_src)
    from metametro.formats.cgt.io import load_cgt

    cgt = load_cgt(folder)
    labels = np.load(folder / "node_labels.npy").astype(np.int64)
    features = np.asarray(cgt.node_features, dtype=np.float32)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("phage labels and node features have different lengths")
    return StudyGraph(
        name="phage_x10",
        cgt=cgt,
        different_pairs=np.zeros((0, 2), dtype=np.int64),
        vae_features=features,
        raw_features=features.copy(),
        labels=labels,
        sequence_ids=[str(index) for index in range(features.shape[0])],
    )
