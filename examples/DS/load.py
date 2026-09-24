"""Load the graphs used by the error analysis. Fail if a claimed column does not match."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
METAMETRO = Path("/mnt/tank/scratch/dsmutin/tools/my/metametro")
VAEGBIN = Path("/mnt/tank/scratch/dsmutin/partition-metagenomics/vaegbin/data")

BUNDLES = {
    "strong100": "strong100",
    "ont100m": "samovar10_ont100m_gfa",
    "ont1b": "samovar10_ont1b_gfa",
    "illumina": "samovar10_illumina_100m",
}


def training_budget() -> dict[str, float | int]:
    """Epoch cap, patience, and learning rate from ``configs/datasets.yaml``."""
    config = yaml.safe_load((REPO / "configs" / "datasets.yaml").read_text(encoding="utf-8"))
    return {
        "max_epochs": int(config["max_epochs"]),
        "patience": int(config["patience"]),
        "lr": float(config["lr"]),
    }


def _weights(array: np.ndarray, nnz: int) -> np.ndarray | None:
    array = np.asarray(array)
    if array.size == 0 or array.ndim == 2 and array.shape[1] == 0:
        return None
    column = array.reshape(-1) if array.ndim == 1 else np.asarray(array[:, 0])
    if column.shape[0] != nnz:
        raise ValueError(f"edge weight length {column.shape[0]} does not match {nnz} CSR entries")
    return column.astype(np.float32)


def load_vaegbin(name: str) -> dict:
    """Frozen VAE features, raw k-mer plus depth, and the cached assembly CSR."""
    if name not in BUNDLES:
        raise KeyError(name)
    cached = REPO / "intermediates" / name
    folder = VAEGBIN / BUNDLES[name]
    for path in (cached / "vae_features.npy", cached / "raw_features.npy", cached / "labels.npy", folder / "node_attributes_depth.npy", folder / "node_attributes_kmer.npy"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    raw = np.load(cached / "raw_features.npy")
    depth = np.load(folder / "node_attributes_depth.npy").reshape(-1)
    kmer = np.load(folder / "node_attributes_kmer.npy")
    if raw.shape[1] != kmer.shape[1] + 1:
        raise ValueError(f"{name}: raw width {raw.shape[1]} is not k-mer plus one depth column")
    if not np.allclose(raw[:, -1], depth):
        raise ValueError(f"{name}: the last raw column is not node_attributes_depth.npy")
    if not np.allclose(raw[:, :-1], kmer):
        raise ValueError(f"{name}: raw columns before the last are not node_attributes_kmer.npy")
    vae = np.load(cached / "vae_features.npy")
    labels = np.load(cached / "labels.npy").astype(np.int64)
    indptr = np.load(cached / "cgt" / "indptr.npy")
    indices = np.load(cached / "cgt" / "indices.npy")
    names = [str(i) for i in range(int(labels.max()) + 1)]
    return {
        "features": vae,
        "labels": labels,
        "label_names": names,
        "indptr": indptr,
        "indices": indices,
        "edge_weight": _weights(np.load(cached / "cgt" / "edge_features.npy"), int(indptr[-1])),
        "raw": raw,
        "vae_equals_raw": bool(vae.shape == raw.shape and np.array_equal(vae, raw)),
    }


def _fasta_lengths_gc(path: Path) -> dict[str, tuple[int, float]]:
    records: dict[str, tuple[int, float]] = {}
    name = None
    chunks: list[str] = []

    def store() -> None:
        if name is None:
            return
        sequence = "".join(chunks).upper()
        if not sequence:
            raise ValueError(f"{path}: {name} has an empty sequence")
        gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
        records[name] = (len(sequence), gc)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            store()
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    store()
    return records


def load_phage() -> dict:
    """T-phage CGT. The tensor feature is out_degree. Coverage and sequence come from the CFA."""
    root = METAMETRO / "data" / "work" / "phage_x10"
    cgt = root / "cgt"
    indptr = np.load(cgt / "indptr.npy")
    features = np.load(cgt / "node_features.npy")
    labels = np.load(cgt / "node_labels.npy").astype(np.int64)
    colors = np.load(cgt / "node_colors.npy")
    coverage_table = pd.read_csv(root / "coloured_cfa" / "nodes.tsv", sep="\t")
    coverage = {row.node_id: float(row.coverage) for row in coverage_table.itertuples(index=False)}
    sequences = _fasta_lengths_gc(root / "coloured_cfa" / "nodes.fna")
    mapping = pd.read_csv(cgt / "mapping.tsv", sep="\t")
    if len(mapping) != features.shape[0]:
        raise ValueError("phage mapping length does not match the node count")
    node_coverage = np.empty(len(mapping), dtype=np.float64)
    length = np.empty(len(mapping), dtype=np.float64)
    gc = np.empty(len(mapping), dtype=np.float64)
    for row in mapping.itertuples(index=False):
        cfa_id = str(row.cfa_node_ids).split(",")[0]
        if cfa_id not in coverage or cfa_id not in sequences:
            raise KeyError(f"phage node {row.dense_id} CFA id {cfa_id} has no coverage or sequence")
        node_coverage[int(row.dense_id)] = coverage[cfa_id]
        length[int(row.dense_id)], gc[int(row.dense_id)] = sequences[cfa_id]
    names = ["unclassified_or_mixed", "T1", "T3", "T4", "T5", "T7"]
    if int(labels.max()) + 1 != len(names):
        raise ValueError("phage label ids do not match the six documented classes")
    return {
        "features": features,
        "labels": labels,
        "label_names": names,
        "indptr": indptr,
        "indices": np.load(cgt / "indices.npy"),
        "edge_weight": _weights(np.load(cgt / "edge_features.npy"), int(indptr[-1])),
        "coverage": node_coverage,
        "length": length,
        "gc": gc,
        "n_sample_colors": colors.sum(axis=1).astype(np.float64),
    }


def _color_names(cgt: Path, cdbg_colors: Path) -> list[tuple[str, str]]:
    palette = pd.read_csv(cdbg_colors, sep="\t")
    by_id = {int(row.color_id): (str(row.namespace), str(row.value)) for row in palette.itertuples(index=False)}
    columns = pd.read_csv(cgt / "color_ids.tsv", sep="\t")
    names = []
    for row in columns.itertuples(index=False):
        if int(row.color_id) not in by_id:
            raise KeyError(f"colour id {row.color_id} is missing from {cdbg_colors}")
        names.append(by_id[int(row.color_id)])
    return names


def load_geometric(name: str) -> dict:
    """SPB transit or Roxel street CGT, including colour names."""
    if name not in {"spb_transit", "roxel"}:
        raise KeyError(name)
    folder = "spb_ground_transit" if name == "spb_transit" else "roxel"
    cgt = METAMETRO / "data" / "work" / folder / "states" / "cgt"
    meta = yaml.safe_load((cgt / "metadata.yaml").read_text(encoding="utf-8"))
    labels = np.load(cgt / "node_labels.npy").astype(np.int64)
    names = [str(item) for item in meta["node_class_names"]]
    if len(names) != int(labels.max()) + 1:
        raise ValueError(f"{name}: class names do not cover the label ids")
    indptr = np.load(cgt / "indptr.npy")
    return {
        "features": np.load(cgt / "node_features.npy"),
        "feature_names": [str(item) for item in meta["node_feature_names"]],
        "labels": labels,
        "label_names": names,
        "indptr": indptr,
        "indices": np.load(cgt / "indices.npy"),
        "edge_weight": _weights(np.load(cgt / "edge_features.npy"), int(indptr[-1])),
        "node_colors": np.load(cgt / "node_colors.npy"),
        "color_names": _color_names(cgt, cgt.parent / "cdbg" / "colors.tsv"),
    }
