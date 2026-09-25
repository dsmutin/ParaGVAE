"""Post-hoc clusterings used by the May 2026 VAEGbin comparisons.

``kmeans`` is scikit-learn with the ground-truth number of groups.
``vamb`` is a NumPy port of the CPU iterative medoid in
``AIRI_may/vaegbin/genelib/vamb_clustering.py`` (RasmussenLab ``vamb/cluster.py``).
Rows stay aligned with their original indices. The archived snippet shuffled
the matrix and the index vector with two separate ``RandomState(0)`` streams.
``hdbscan`` calls the ``hdbscan`` package and assigns noise points to the
nearest cluster mean, as ``scripts/clustering/methods.py`` did.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.cluster import KMeans

# Normal density on [-0.075, 0.075], step 0.005, already scaled by that step
# in the archived module. Stored here before the 0.005 factor.
_PDF_RAW = np.asarray(
    [
        2.43432053e-11,
        9.13472041e-10,
        2.66955661e-08,
        6.07588285e-07,
        1.07697600e-05,
        1.48671951e-04,
        1.59837411e-03,
        1.33830226e-02,
        8.72682695e-02,
        4.43184841e-01,
        1.75283005e00,
        5.39909665e00,
        1.29517596e01,
        2.41970725e01,
        3.52065327e01,
        3.98942280e01,
        3.52065327e01,
        2.41970725e01,
        1.29517596e01,
        5.39909665e00,
        1.75283005e00,
        4.43184841e-01,
        8.72682695e-02,
        1.33830226e-02,
        1.59837411e-03,
        1.48671951e-04,
        1.07697600e-05,
        6.07588285e-07,
        2.66955661e-08,
        9.13472041e-10,
        2.43432053e-11,
    ],
    dtype=np.float64,
)
_DELTA_X = 0.005
_XMAX = 0.3
_DEFAULT_RADIUS = 0.06
_MEDOID_RADIUS = 0.05
_NORMALPDF = _DELTA_X * _PDF_RAW


def cluster_kmeans(embedding: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    """K-means. ``n_groups`` is the number of ground-truth classes."""
    matrix = np.asarray(embedding, dtype=np.float64)
    groups = max(2, min(int(n_groups), matrix.shape[0]))
    return KMeans(n_clusters=groups, random_state=seed, n_init=5).fit_predict(matrix).astype(np.int32)


def cluster_hdbscan(embedding: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """HDBSCAN. Noise labels are replaced by the nearest cluster mean."""
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is not installed in the paragvae environment") from exc
    matrix = np.asarray(embedding, dtype=np.float64)
    raw = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2).fit_predict(matrix)
    return _assign_noise(matrix, raw.astype(np.int32))


def _assign_noise(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    noise = out == -1
    if not noise.any():
        return _relabel(out)
    valid = ~noise
    if not valid.any():
        return np.zeros(len(out), dtype=np.int32)
    centroids = {int(lab): matrix[out == lab].mean(axis=0) for lab in set(out[valid].tolist())}
    for index in np.flatnonzero(noise):
        out[index] = min(centroids, key=lambda lab: np.linalg.norm(matrix[index] - centroids[lab]))
    return _relabel(out)


def _relabel(labels: np.ndarray) -> np.ndarray:
    mapping = {int(label): index for index, label in enumerate(sorted(set(int(x) for x in labels)))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)


def cluster_vamb(embedding: np.ndarray, *, maxsteps: int = 25, windowsize: int = 200, minsuccesses: int = 20) -> np.ndarray:
    """Iterative medoid bins. Every row receives the cluster that absorbed it."""
    matrix = np.asarray(embedding, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("VAMB clustering needs a non-empty 2-d embedding")
    if maxsteps < 1 or windowsize < 1 or not 1 <= minsuccesses <= windowsize:
        raise ValueError("VAMB maxsteps, windowsize, and minsuccesses are out of range")
    count = matrix.shape[0]
    order = np.random.RandomState(0).permutation(count)
    points = _normalize(matrix[order])
    labels = np.full(count, -1, dtype=np.int32)
    remaining = np.ones(count, dtype=bool)
    peak_valley = 0.1
    attempts: deque[bool] = deque(maxlen=windowsize)
    successes = 0
    rng = __import__("random").Random(0)
    seed = -1
    cluster_id = 0
    while remaining.any():
        active = np.flatnonzero(remaining)
        threshold = None
        distances = None
        medoid_local = 0
        success: bool | None = False
        while threshold is None:
            seed = (seed + 1) % active.size
            local = int(active[seed])
            medoid_local, distances = _wander(points, remaining, local, maxsteps, rng)
            histogram = np.histogram(distances[remaining], bins=int(np.ceil(_XMAX / _DELTA_X)), range=(0.0, _XMAX))[0]
            histogram = histogram.astype(np.float64)
            histogram[0] -= 1
            threshold, success = _find_threshold(histogram, peak_valley)
            if success is not None:
                if len(attempts) == attempts.maxlen:
                    successes -= int(attempts.popleft())
                successes += int(success)
                attempts.append(bool(success))
                if len(attempts) == windowsize and successes < minsuccesses:
                    peak_valley += 0.1
                    attempts.clear()
                    successes = 0
        assert distances is not None
        members = np.flatnonzero((distances <= threshold) & remaining)
        if members.size == 0:
            members = np.asarray([medoid_local], dtype=np.int64)
        labels[order[members]] = cluster_id
        remaining[members] = False
        cluster_id += 1
    if np.any(labels < 0):
        raise RuntimeError("VAMB clustering left a point unassigned")
    return labels


def _normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.array(matrix, dtype=np.float32, copy=True)
    zero = np.sum(values, axis=1) == 0
    values[zero] = 1.0 / values.shape[1]
    norms = np.linalg.norm(values, axis=1, keepdims=True) * np.sqrt(2.0)
    return values / norms


def _distances(matrix: np.ndarray, index: int) -> np.ndarray:
    dists = 0.5 - matrix @ matrix[index]
    dists[index] = 0.0
    return dists


def _wander(matrix: np.ndarray, remaining: np.ndarray, medoid: int, maxsteps: int, rng) -> tuple[int, np.ndarray]:
    futile = 0
    tried = {medoid}
    distances = _distances(matrix, medoid)
    cluster = np.flatnonzero((distances <= _MEDOID_RADIUS) & remaining)
    average = 0.0 if cluster.size <= 1 else float(distances[cluster].sum() / (cluster.size - 1))
    while cluster.size - len(tried) > 0 and futile < maxsteps:
        choices = [int(item) for item in cluster.tolist() if int(item) not in tried]
        if not choices:
            break
        sampled = int(rng.choice(choices))
        tried.add(sampled)
        sample_dist = _distances(matrix, sampled)
        sample_cluster = np.flatnonzero((sample_dist <= _MEDOID_RADIUS) & remaining)
        sample_avg = 0.0 if sample_cluster.size <= 1 else float(sample_dist[sample_cluster].sum() / (sample_cluster.size - 1))
        if sample_avg < average:
            medoid = sampled
            distances = sample_dist
            cluster = sample_cluster
            average = sample_avg
            futile = 0
            tried = {medoid}
        else:
            futile += 1
    return medoid, distances


def _smooth(histogram: np.ndarray) -> np.ndarray:
    pdf = _NORMALPDF
    densities = np.zeros(histogram.size + pdf.size - 1, dtype=np.float64)
    for index in range(histogram.size):
        densities[index : index + pdf.size] += pdf * histogram[index]
    return densities[15:-15]


def _find_threshold(histogram: np.ndarray, peak_valley_ratio: float) -> tuple[float | None, bool | None]:
    if float(histogram[:10].sum()) == 0.0:
        return 0.025, None
    densities = _smooth(histogram)
    peak_density = 0.0
    peak_over = False
    density_at_minimum = 0.0
    threshold = None
    success: bool | None = False
    delta = _XMAX / len(histogram)
    position = 0.0
    for density in densities:
        if not peak_over and density > peak_density:
            if position > 0.1:
                break
            peak_density = float(density)
        if not peak_over and density < 0.6 * peak_density:
            peak_over = True
            density_at_minimum = float(density)
        if peak_over and density > 1.5 * density_at_minimum:
            break
        if peak_over and density < density_at_minimum:
            density_at_minimum = float(density)
            if density < peak_valley_ratio * peak_density:
                threshold = position
                success = True
        position += delta
    if threshold is not None and threshold > 0.2 + peak_valley_ratio:
        threshold = None
        success = False
    if threshold is None and peak_valley_ratio > 0.55:
        return _DEFAULT_RADIUS, None
    return threshold, success
