"""Call the C++ epoch loop. Graph preparation stays in Python."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from paragvae.train import TrainResult, _zscore, normalized_from_edges, split_edges, undirected_edges

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cpp" / "gcn_train.cpp"
BINARY = ROOT / "cpp" / "gcn_train"

_LOSS = {"standard": 0, "diff_c": 1, "proxy": 2, "contrastive": 3}


def ensure_binary() -> Path:
    """Compile ``cpp/gcn_train.cpp`` when the binary is missing or older."""
    if BINARY.is_file() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return BINARY
    compiler = "g++"
    subprocess.run(
        [compiler, "-O3", "-std=c++17", "-o", str(BINARY), str(SOURCE)],
        check=True,
    )
    return BINARY


def train_gcn_native(
    *,
    features: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    different_pairs: np.ndarray | None,
    edge_weight: np.ndarray | None = None,
    loss: str,
    max_epochs: int,
    patience: int,
    latent: int,
    hidden: int,
    seed: int,
    lr: float,
    work: Path,
) -> TrainResult:
    """Run one early-stopped fit in the compiled trainer."""
    binary = ensure_binary()
    work.mkdir(parents=True, exist_ok=True)
    scaled = _zscore(features)
    if scaled.shape[1] == 0:
        scaled = np.ones((scaled.shape[0], 1), dtype=np.float32)
    pairs, weights = undirected_edges(indptr, indices, edge_weight)
    if edge_weight is None:
        weights = np.ones(pairs.shape[0], dtype=np.float32)
    train_edges, train_weights, val_edges, _val_weights = split_edges(pairs, weights, seed)
    operator = normalized_from_edges(train_edges, train_weights, scaled.shape[0])
    operator.sum_duplicates()
    operator.sort_indices()
    positives = np.asarray(train_edges, dtype=np.int32)
    validation = np.asarray(val_edges, dtype=np.int32)
    marker = np.zeros((0, 2), dtype=np.int32) if different_pairs is None else np.asarray(different_pairs, dtype=np.int64)
    if marker.size:
        n = scaled.shape[0]
        marker = marker[(marker[:, 0] >= 0) & (marker[:, 1] >= 0) & (marker[:, 0] < n) & (marker[:, 1] < n)]
        if marker.shape[0] > 4096:
            rng = np.random.default_rng(seed)
            marker = marker[rng.choice(marker.shape[0], size=4096, replace=False)]
    marker = np.asarray(marker, dtype=np.int32).reshape(-1, 2)
    (work / "indptr.i32").write_bytes(np.asarray(operator.indptr, dtype=np.int32).tobytes())
    (work / "indices.i32").write_bytes(np.asarray(operator.indices, dtype=np.int32).tobytes())
    (work / "values.f32").write_bytes(np.asarray(operator.data, dtype=np.float32).tobytes())
    (work / "features.f32").write_bytes(np.asarray(scaled, dtype=np.float32).tobytes())
    (work / "pos.i32").write_bytes(np.asarray(positives, dtype=np.int32).reshape(-1).tobytes())
    (work / "val.i32").write_bytes(np.asarray(validation, dtype=np.int32).reshape(-1).tobytes())
    (work / "mark.i32").write_bytes(marker.reshape(-1).tobytes())
    (work / "meta.txt").write_text(
        f"{scaled.shape[0]} {scaled.shape[1]} {operator.nnz} {positives.shape[0]} {validation.shape[0]} "
        f"{marker.shape[0]} {max_epochs} {patience} {seed} {hidden} {latent} {_LOSS[loss]} {lr}\n",
        encoding="utf-8",
    )
    subprocess.run([str(binary), str(work)], check=True)
    epochs_text, stopped_text, best_text, train_text = (work / "result.txt").read_text(encoding="utf-8").split()
    embedding = np.fromfile(work / "embedding.f32", dtype=np.float32).reshape(scaled.shape[0], -1)
    return TrainResult(
        embedding=embedding,
        epochs_ran=int(epochs_text),
        stopped_early=stopped_text == "1",
        best_val_loss=float(best_text),
        train_loss=float(train_text),
    )
