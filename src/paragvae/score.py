"""Binning scores that do not call AMBER.

Contig F1 assigns every predicted cluster the majority ground-truth genome
and then scores contig overlap. ARI is label-invariant. Neither number is
the paper's AMBER ``f1_score_seq``.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score


def contig_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Majority-label precision and recall, returned as their harmonic mean."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_ids = [label for label in np.unique(true_labels) if label >= 0]
    pred_ids = np.unique(pred_labels)
    if not true_ids or pred_ids.size == 0:
        return 0.0
    # Largest overlap first so the score does not depend on cluster ids.
    ranked = []
    for pred in pred_ids:
        members = true_labels[pred_labels == pred]
        if members.size == 0:
            continue
        values, counts = np.unique(members, return_counts=True)
        truth = int(values[np.argmax(counts)])
        if truth < 0:
            continue
        ranked.append((int(counts.max()), int(pred), truth))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    true_sizes = {int(label): int(np.sum(true_labels == label)) for label in true_ids}
    matched = 0
    seen: set[int] = set()
    for hit, _pred, truth in ranked:
        if truth in seen:
            continue
        seen.add(truth)
        matched += hit
    precision = matched / max(len(pred_labels), 1)
    recall = matched / max(sum(true_sizes.values()), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def cluster_embedding(embedding: np.ndarray, labels: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Cluster rows. ``k`` is the number of ground-truth genomes, shared by every arm."""
    matrix = np.asarray(embedding, dtype=np.float64)
    usable = np.asarray(labels) >= 0
    n_groups = int(np.unique(np.asarray(labels)[usable]).size)
    n_groups = max(2, min(n_groups, matrix.shape[0]))
    if method == "kmeans":
        model = KMeans(n_clusters=n_groups, random_state=seed, n_init=5)
        return model.fit_predict(matrix)
    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_groups, linkage="average")
        return model.fit_predict(matrix)
    raise ValueError(f"unknown clustering method {method}")


def evaluate(embedding: np.ndarray, labels: np.ndarray, method: str, seed: int) -> dict[str, float]:
    """Return ARI and contig F1 for one clustering of ``embedding``."""
    labels = np.asarray(labels)
    predicted = cluster_embedding(embedding, labels, method, seed)
    usable = labels >= 0
    if int(usable.sum()) < 2:
        ari = 0.0
    else:
        ari = float(adjusted_rand_score(labels[usable], predicted[usable]))
    return {"ari": ari, "f1": contig_f1(labels, predicted)}
