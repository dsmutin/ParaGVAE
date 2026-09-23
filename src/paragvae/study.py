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


def _degree_and_clustering(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = len(indptr) - 1
    degree = np.diff(indptr).astype(np.float32)
    edge_set = {(int(row), int(col)) for row, col in zip(np.repeat(np.arange(n), degree.astype(int)), indices)}
    clustering = np.zeros(n, dtype=np.float32)
    for node in range(n):
        start, stop = int(indptr[node]), int(indptr[node + 1])
        neigh = indices[start:stop]
        deg = int(neigh.shape[0])
        if deg < 2 or deg > 64:
            continue
        links = 0
        for i in range(deg):
            for j in range(i + 1, deg):
                a, b = int(neigh[i]), int(neigh[j])
                if (a, b) in edge_set or (b, a) in edge_set:
                    links += 1
        clustering[node] = 2.0 * links / (deg * (deg - 1))
    return np.stack([degree, clustering], axis=1)


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
    job = work or (Path("benchmark") / arm.hypothesis / "jobs" / f"{graph.name}_{arm.arm}_{seed}")
    fit = train_gcn_native(
        features=features,
        indptr=indptr,
        indices=indices,
        different_pairs=graph.different_pairs,
        loss=arm.loss,
        max_epochs=arm.max_epochs,
        patience=arm.patience,
        latent=16,
        hidden=32,
        seed=seed,
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
