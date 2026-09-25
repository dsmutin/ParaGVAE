"""Benchmark the completion arms on the four named graphs.

One row is one architecture, loss, clustering, dataset, and seed. The GCN
embedding is scored with k-means, VAMB, and HDBSCAN. The other arms are
scored with k-means. A missing Kraken table or a missing ``hdbscan`` install
is written as ``status=missing`` and does not invent a score.

Run from the repository root:

    python research/run_registry.py
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path

import numpy as np

from paragvae.arch import train_architecture
from paragvae.cluster_methods import cluster_hdbscan, cluster_kmeans, cluster_vamb
from paragvae.completion import completion_paths, load_completion
from paragvae.flip import edge_line_graph, infer_nodes_from_edges
from paragvae.score import contig_f1
from paragvae.suite import ROOT, load_config
from sklearn.metrics import adjusted_rand_score

OUT = ROOT / "benchmark" / "registry"
FIELDS = (
    "dataset",
    "seed",
    "arm",
    "architecture",
    "n_layers",
    "loss",
    "clustering",
    "status",
    "f1",
    "ari",
    "epochs",
    "stopped_early",
    "n_nodes",
    "n_edges",
    "note",
)

# (arm, architecture, n_layers, loss, flip, clusterings)
ARMS = (
    ("gcn", "gcn", 2, "standard", False, ("kmeans", "vamb", "hdbscan")),
    ("sage", "sage", 2, "standard", False, ("kmeans",)),
    ("gat", "gat", 2, "standard", False, ("kmeans",)),
    ("gcn_layers3", "gcn", 3, "standard", False, ("kmeans",)),
    ("transformer", "transformer", 1, "standard", False, ("kmeans",)),
    ("proxy", "gcn", 2, "proxy", False, ("kmeans",)),
    ("contrastive", "gcn", 2, "contrastive", False, ("kmeans",)),
    ("biological", "gcn", 2, "biological", False, ("kmeans",)),
    ("joint", "joint", 2, "standard", False, ("kmeans",)),
    ("edge_flip", "gcn", 2, "standard", True, ("kmeans",)),
)


def main() -> None:
    _prefer_extracted_hdbscan()
    config = load_config()
    paths = completion_paths(config)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "results.csv"
    done = _done(destination)
    rows: list[dict] = []
    if destination.is_file():
        with destination.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    for name, path in paths.items():
        print(f"loading {name}", flush=True)
        graph, bio = load_completion(name, path, config["metametro_src"])
        labels = np.asarray(graph.labels)
        features = np.asarray(graph.raw_features, dtype=np.float32)
        indptr = np.asarray(graph.cgt.indptr)
        indices = np.asarray(graph.cgt.indices)
        edge_weight = np.asarray(graph.cgt.edge_features, dtype=np.float32)
        nnz = int(indptr[-1]) if len(indptr) else 0
        if edge_weight.ndim == 2 and edge_weight.shape[1] == 0:
            edge_weight = None
        elif edge_weight.ndim == 2:
            if edge_weight.shape[0] != nnz:
                raise ValueError(f"{name} edge features do not match the CSR")
            edge_weight = edge_weight[:, 0]
        elif edge_weight.size == 0:
            edge_weight = None
        else:
            edge_weight = edge_weight.reshape(-1)
        n_edges = int(indptr[-1]) if len(indptr) else 0
        for seed in config["seeds"]:
            for arm, architecture, n_layers, loss, flip, clusterings in ARMS:
                key = (name, int(seed), arm)
                if key in done and all((name, int(seed), arm, method) in done for method in clusterings):
                    continue
                note = ""
                if loss == "biological" and bio is None:
                    for method in clusterings:
                        rows.append(_missing(name, seed, arm, architecture, n_layers, loss, method, features.shape[0], n_edges, "no Kraken call table for this graph"))
                    _write(destination, rows)
                    continue
                try:
                    trained = _train(
                        features, indptr, indices, edge_weight, architecture, n_layers, loss, flip, bio, seed, config
                    )
                except Exception as exc:
                    traceback.print_exc()
                    for method in clusterings:
                        rows.append(_missing(name, seed, arm, architecture, n_layers, loss, method, features.shape[0], n_edges, f"{type(exc).__name__}: {exc}"))
                    _write(destination, rows)
                    continue
                usable = labels >= 0
                n_groups = int(np.unique(labels[usable]).size) if usable.any() else 2
                for method in clusterings:
                    try:
                        predicted = _cluster(trained.embedding, method, n_groups, int(seed))
                    except Exception as exc:
                        rows.append(_missing(name, seed, arm, architecture, n_layers, loss, method, features.shape[0], n_edges, f"{type(exc).__name__}: {exc}"))
                        continue
                    ari = 0.0 if int(usable.sum()) < 2 else float(adjusted_rand_score(labels[usable], predicted[usable]))
                    rows.append(
                        {
                            "dataset": name,
                            "seed": int(seed),
                            "arm": arm,
                            "architecture": architecture,
                            "n_layers": n_layers,
                            "loss": loss,
                            "clustering": method,
                            "status": "ok",
                            "f1": f"{contig_f1(labels, predicted):.6f}",
                            "ari": f"{ari:.6f}",
                            "epochs": trained.epochs_ran,
                            "stopped_early": int(trained.stopped_early),
                            "n_nodes": features.shape[0],
                            "n_edges": n_edges,
                            "note": note,
                        }
                    )
                _write(destination, rows)
                print(f"  {name} seed {seed} {arm} epochs {trained.epochs_ran}", flush=True)
    _chart(rows)
    print(f"wrote {destination}", flush=True)


def _train(features, indptr, indices, edge_weight, architecture, n_layers, loss, flip, bio, seed, config):
    train_features = features
    train_indptr = indptr
    train_indices = indices
    train_weight = edge_weight
    pooled = None
    if flip:
        weights = None if edge_weight is None else np.asarray(edge_weight, dtype=np.float32).reshape(-1, 1)
        flipped = edge_line_graph(indptr, indices, features, edge_features=weights)
        train_features = flipped.features
        train_indptr = flipped.indptr
        train_indices = flipped.indices
        train_weight = None
        pooled = flipped
    result = train_architecture(
        features=train_features,
        indptr=train_indptr,
        indices=train_indices,
        architecture=architecture,
        n_layers=n_layers,
        loss=loss,
        bio=bio,
        edge_weight=train_weight,
        max_epochs=int(config["max_epochs"]),
        patience=int(config["patience"]),
        seed=int(seed),
        lr=float(config["lr"]),
    )
    if pooled is not None:
        result.embedding = infer_nodes_from_edges(result.embedding, pooled.edge_index, pooled.n_original_nodes)
    return result


def _cluster(embedding: np.ndarray, method: str, n_groups: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        return cluster_kmeans(embedding, n_groups, seed)
    if method == "vamb":
        return cluster_vamb(embedding)
    if method == "hdbscan":
        return cluster_hdbscan(embedding)
    raise ValueError(method)


def _missing(dataset, seed, arm, architecture, n_layers, loss, method, n_nodes, n_edges, note) -> dict:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "arm": arm,
        "architecture": architecture,
        "n_layers": n_layers,
        "loss": loss,
        "clustering": method,
        "status": "missing",
        "f1": "",
        "ari": "",
        "epochs": "",
        "stopped_early": "",
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "note": note,
    }


def _done(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    found = set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            found.add((row["dataset"], int(row["seed"]), row["arm"], row["clustering"]))
            found.add((row["dataset"], int(row["seed"]), row["arm"]))
    return found


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _prefer_extracted_hdbscan() -> None:
    """Use the extracted conda-forge build when the env cannot import hdbscan.

    The home quota blocked ``conda install``. The package on disk is
    ``hdbscan 0.8.44`` (build ``py312h0ec8046_1``).
    """
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        extra = ROOT / ".." / "conda-pkgs" / "hdbscan-0.8.44-py312h0ec8046_1" / "lib" / "python3.12" / "site-packages"
        if extra.is_dir():
            import sys

            sys.path.insert(0, str(extra.resolve()))


def _chart(rows: list[dict]) -> None:
    from paragvae.plots import save_charts

    plotted = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        plotted.append(
            {
                "dataset": row["dataset"],
                "arm": f"{row['arm']}/{row['clustering']}",
                "f1": row["f1"],
                "ari": row["ari"],
                "epochs_ran": row["epochs"],
            }
        )
    if plotted:
        save_charts(plotted, OUT)


if __name__ == "__main__":
    main()
