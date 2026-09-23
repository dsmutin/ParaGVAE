"""One-factor arms used by ``research/<hypothesis>/``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paragvae.graphs import StudyGraph, composition_colors
from paragvae.native import train_gcn_native
from paragvae.score import evaluate
from paragvae.train import knn_adjacency

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Arm:
    """One finite training setting."""

    hypothesis: str
    arm: str
    joint: bool = False
    loss: str = "standard"
    colors: bool = False
    graph: str = "assembly"
    multiscale: bool = False
    clustering: str = "kmeans"
    max_epochs: int = 25
    patience: int = 5
    lr: float = 0.05


def _degree_and_clustering(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Degree and local clustering coefficient, ignoring self-loops.

    Several VAEGbin graphs store a self-loop on every node. Counting that
    loop as a neighbour makes a path look like a triangle.
    """
    n = len(indptr) - 1
    neighbors: list[list[int]] = []
    edge_set: set[tuple[int, int]] = set()
    for node in range(n):
        start, stop = int(indptr[node]), int(indptr[node + 1])
        neigh: list[int] = []
        for col in np.asarray(indices[start:stop]).tolist():
            other = int(col)
            if other == node:
                continue
            neigh.append(other)
            edge_set.add((node, other))
        neighbors.append(neigh)
    degree = np.asarray([len(neigh) for neigh in neighbors], dtype=np.float32)
    clustering = np.zeros(n, dtype=np.float32)
    for node, neigh in enumerate(neighbors):
        deg = len(neigh)
        if deg < 2 or deg > 64:
            continue
        links = 0
        for i in range(deg):
            for j in range(i + 1, deg):
                a, b = neigh[i], neigh[j]
                if (a, b) in edge_set or (b, a) in edge_set:
                    links += 1
        clustering[node] = 2.0 * links / (deg * (deg - 1))
    return np.stack([degree, clustering], axis=1)


def same_feature_matrices(graph: StudyGraph) -> bool:
    """True when the VAE matrix and the raw matrix are the same array contents."""
    vae = np.asarray(graph.vae_features)
    raw = np.asarray(graph.raw_features)
    return vae.shape == raw.shape and np.array_equal(vae, raw)


def arm_is_redundant(graph: StudyGraph, arm: Arm) -> bool:
    """Skip a second kNN arm when it would rebuild the same graph."""
    return arm.graph == "knn_kmer" and same_feature_matrices(graph)


def _topology(graph: StudyGraph, arm: Arm) -> tuple[np.ndarray, np.ndarray]:
    cache = getattr(graph, "topology_cache", None)
    if cache is None:
        cache = {}
        graph.topology_cache = cache
    if arm.graph in cache:
        return cache[arm.graph]
    cgt = graph.cgt
    if arm.graph == "assembly":
        cached = np.asarray(cgt.indptr), np.asarray(cgt.indices)
        cache[arm.graph] = cached
        return cached
    if arm.graph == "knn_vae":
        adjacency = knn_adjacency(graph.vae_features, k=5)
        cached = adjacency.indptr.astype(np.int64), adjacency.indices.astype(np.int64)
        cache[arm.graph] = cached
        return cached
    if arm.graph == "knn_kmer":
        adjacency = knn_adjacency(graph.raw_features, k=5)
        cached = adjacency.indptr.astype(np.int64), adjacency.indices.astype(np.int64)
        cache[arm.graph] = cached
        return cached
    raise ValueError(arm.graph)


def assembly_edge_weight(graph: StudyGraph, arm: Arm, indptr: np.ndarray) -> np.ndarray | None:
    """Return assembly edge weights aligned with ``indptr``, or None for a kNN graph.

    A kNN graph has no stored weights. A length mismatch on an assembly graph
    is an error rather than a silent fallback to ones.
    """
    if arm.graph != "assembly":
        return None
    stored = np.asarray(graph.cgt.edge_features, dtype=np.float32)
    column = stored.reshape(-1) if stored.ndim == 1 else np.asarray(stored[:, 0], dtype=np.float32)
    expected = int(indptr[-1]) if len(indptr) else 0
    if column.shape[0] != expected:
        raise ValueError(f"{graph.name}: edge weight length {column.shape[0]} does not match {expected} CSR entries")
    return column


def _features(graph: StudyGraph, arm: Arm, indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    base = graph.raw_features if arm.joint else graph.vae_features
    parts = [np.asarray(base, dtype=np.float32)]
    if arm.colors:
        stored = np.asarray(graph.cgt.node_colors)
        if stored.ndim != 2 or stored.shape[1] == 0:
            stored = composition_colors(graph.raw_features, n_colors=8, seed=0)
        parts.append(stored.astype(np.float32))
    if arm.multiscale:
        parts.append(_degree_and_clustering(indptr, indices))
    return np.hstack(parts)


def run_arm(graph: StudyGraph, arm: Arm, seed: int, work: Path | None = None) -> dict[str, float | int | str | bool]:
    """Train until validation loss stalls, then score bins."""
    indptr, indices = _topology(graph, arm)
    features = _features(graph, arm, indptr, indices)
    started = time.perf_counter()
    job = work or (ROOT / "benchmark" / arm.hypothesis / "jobs" / f"{graph.name}_{arm.arm}_{seed}")
    fit = train_gcn_native(
        features=features,
        indptr=indptr,
        indices=indices,
        different_pairs=graph.different_pairs,
        edge_weight=assembly_edge_weight(graph, arm, indptr),
        loss=arm.loss,
        max_epochs=arm.max_epochs,
        patience=arm.patience,
        latent=16,
        hidden=32,
        seed=seed,
        lr=arm.lr,
        work=job,
    )
    scores = evaluate(fit.embedding, graph.labels, arm.clustering, seed)
    return {
        "dataset": graph.name,
        "hypothesis": arm.hypothesis,
        "arm": arm.arm,
        "seed": seed,
        "epochs_ran": fit.epochs_ran,
        "stopped_early": fit.stopped_early,
        "best_val_loss": round(fit.best_val_loss, 6),
        "train_loss": round(fit.train_loss, 6),
        "ari": round(scores["ari"], 6),
        "f1": round(scores["f1"], 6),
        "n_nodes": int(graph.cgt.num_nodes),
        "n_edges": int(indices.shape[0]),
        "seconds": round(time.perf_counter() - started, 3),
        "backend": "cpp",
    }
