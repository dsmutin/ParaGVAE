"""Build a MetaMetro coloured graph tensor from a VAEGbin bundle or a fixture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from paragvae.metametro_path import ensure_metametro


@dataclass
class StudyGraph:
    """CGT plus fields the tensor schema does not store.

    ``different_pairs`` are single-copy marker pairs from VAEGbin
    (``all_different.npy``). They are an auxiliary loss index, not colours
    and not ground-truth genome ids. ``vae_features`` are the published
    latent matrix. ``raw_features`` are k-mer and depth channels used only
    by the joint-training arm.
    """

    name: str
    cgt: object
    different_pairs: np.ndarray
    vae_features: np.ndarray
    raw_features: np.ndarray
    labels: np.ndarray


def _cgt_class():
    ensure_metametro()
    from metametro.formats.cgt.model import Cgt

    return Cgt


def _empty_colors(n_rows: int, width: int = 0) -> np.ndarray:
    return np.zeros((n_rows, width), dtype=np.uint8)


def cgt_from_csr(
    *,
    name: str,
    adjacency: sparse.spmatrix,
    node_features: np.ndarray,
    labels: np.ndarray,
    node_ids: list[str],
    edge_weight: np.ndarray | None = None,
    node_colors: np.ndarray | None = None,
    source: str,
) -> object:
    """Validate and return a CGT. Does not allocate a dense adjacency."""
    Cgt = _cgt_class()
    matrix = sparse.csr_matrix(adjacency)
    matrix.sum_duplicates()
    matrix.sort_indices()
    n = int(node_features.shape[0])
    if matrix.shape != (n, n):
        raise ValueError(f"adjacency shape {matrix.shape} does not match {n} nodes")
    features = np.asarray(node_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("node features must be 2-d")
    weights = np.ones(matrix.nnz, dtype=np.float32) if edge_weight is None else np.asarray(edge_weight, dtype=np.float32)
    if weights.shape != (matrix.nnz,):
        weights = np.ones(matrix.nnz, dtype=np.float32)
    colors = _empty_colors(n) if node_colors is None else np.asarray(node_colors, dtype=np.uint8)
    if colors.shape[0] != n:
        raise ValueError("node colour rows must match the node count")
    feature_names = [f"f{i}" for i in range(features.shape[1])]
    metadata = {
        "schema_version": "1.0",
        "num_nodes": n,
        "num_edges": int(matrix.nnz),
        "node_feature_names": feature_names,
        "edge_feature_names": ["weight"],
        "node_feature_dtype": "float32",
        "edge_feature_dtype": "float32",
        "topology": "csr",
        "contract": "cdbg_to_cgt",
        "contract_version": "1.0",
        "source": {"format": "vaegbin_bundle", "graph_id": name, "note": source},
    }
    mapping = [
        {"dense_id": i, "source_id": node_ids[i], "cfa_node_ids": [node_ids[i]]}
        for i in range(n)
    ]
    graph = Cgt(
        metadata=metadata,
        indptr=np.asarray(matrix.indptr, dtype=np.int64),
        indices=np.asarray(matrix.indices, dtype=np.int64),
        node_features=features,
        edge_features=weights.reshape(-1, 1),
        node_colors=colors,
        edge_colors=_empty_colors(int(matrix.nnz)),
        mapping=mapping,
        node_labels=np.asarray(labels, dtype=np.int64),
        edge_labels=None,
        color_ids=list(range(colors.shape[1])),
    )
    from metametro.formats.cgt.validator import validate_cgt

    validate_cgt(graph)
    return graph


def _label_vector(path: Path, node_ids: list[str]) -> np.ndarray:
    """Map VAEGbin ``node_to_label.npy`` (dict or vector) onto dense ids.

    The dict is pickled. Values that are already integers stay integers.
    Missing nodes are ``-1`` and are ignored by ARI only if every node is
    labelled; contig F1 skips negative labels.
    """
    raw = np.load(path, allow_pickle=True)
    if getattr(raw, "dtype", None) != object:
        return np.asarray(raw, dtype=np.int64).reshape(-1)
    payload = raw.item() if raw.shape == () else raw
    if not isinstance(payload, dict):
        return np.asarray(payload, dtype=np.int64).reshape(-1)
    vocab: dict[object, int] = {}
    codes = np.empty(len(node_ids), dtype=np.int64)
    for index, node_id in enumerate(node_ids):
        value = payload.get(node_id, payload.get(index))
        if isinstance(value, (int, np.integer)):
            codes[index] = int(value)
            continue
        if value is None:
            codes[index] = -1
            continue
        if value not in vocab:
            vocab[value] = len(vocab)
        codes[index] = vocab[value]
    return codes


def load_vaegbin_bundle(root: str | Path, name: str | None = None) -> StudyGraph:
    """Read a VAEGbin directory into a CGT.

    Assembly edges stay in CSR. Genome labels are evaluation-only and are
    stored on ``node_labels``. They are not copied into features or colours.
    """
    folder = Path(root)
    label = name or folder.name
    adjacency = sparse.load_npz(folder / "adj_sparse.npz").tocsr()
    weight_path = folder / "edge_weights.npy"
    edge_weight = np.load(weight_path).astype(np.float32).reshape(-1) if weight_path.is_file() else None
    vae = np.load(folder / "node_features.npy").astype(np.float32)
    kmer = np.load(folder / "node_attributes_kmer.npy").astype(np.float32)
    depth = np.load(folder / "node_attributes_depth.npy").astype(np.float32)
    if depth.ndim == 1:
        depth = depth.reshape(-1, 1)
    raw = np.hstack([kmer, depth]).astype(np.float32)
    names = np.load(folder / "node_names.npy", allow_pickle=True)
    node_ids = [str(item) for item in names.tolist()]
    labels = _label_vector(folder / "node_to_label.npy", node_ids)
    pairs_path = folder / "all_different.npy"
    if pairs_path.is_file():
        pairs = np.load(pairs_path)
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
    graph = cgt_from_csr(
        name=label,
        adjacency=adjacency,
        node_features=vae,
        labels=labels,
        node_ids=node_ids,
        edge_weight=edge_weight,
        node_colors=None,
        source="VAEGbin assembly graph; colours filled later by the colouring arm",
    )
    return StudyGraph(
        name=label,
        cgt=graph,
        different_pairs=pairs,
        vae_features=vae,
        raw_features=raw,
        labels=labels,
    )


def load_metametro_bubble() -> StudyGraph:
    """Coloured bubble fixture from MetaMetro (CFA colours already on the CGT)."""
    ensure_metametro()
    from metametro.fixtures import mock_cgt

    graph = mock_cgt()
    labels = np.asarray(graph.node_labels, dtype=np.int64)
    features = np.asarray(graph.node_features, dtype=np.float32)
    return StudyGraph(
        name="metametro_bubble",
        cgt=graph,
        different_pairs=np.zeros((0, 2), dtype=np.int64),
        vae_features=features,
        raw_features=features.copy(),
        labels=labels,
    )


def composition_colors(features: np.ndarray, n_colors: int = 8, seed: int = 0) -> np.ndarray:
    """Discrete composition colours. Fit ignores genome labels.

    Returns a uint8 one-hot matrix suitable for ``Cgt.node_colors``.
    """
    from sklearn.cluster import KMeans

    matrix = np.asarray(features, dtype=np.float64)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (matrix - matrix.mean(axis=0)) / scale
    n_colors = int(min(n_colors, max(2, scaled.shape[0])))
    model = KMeans(n_clusters=n_colors, random_state=seed, n_init=5)
    assigned = model.fit_predict(scaled)
    colors = np.zeros((scaled.shape[0], n_colors), dtype=np.uint8)
    colors[np.arange(scaled.shape[0]), assigned] = 1
    return colors
