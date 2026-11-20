"""Load the graphs used by the error analysis. Fail if a claimed column does not match."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]


def _config() -> dict:
    return yaml.safe_load((REPO / "configs" / "datasets.yaml").read_text(encoding="utf-8"))


def _vaegbin() -> Path:
    return Path(_config()["vaegbin_data"])


def _metametro_work(name: str) -> Path:
    from paragvae.metametro_path import work_dir

    return work_dir(name)

BUNDLES = {
    "strong100": "strong100",
    "ont100m": "samovar10_ont100m_gfa",
    "ont1b": "samovar10_ont1b_gfa",
    "illumina": "samovar10_illumina_100m",
}


def training_budget() -> dict[str, float | int]:
    """Epoch cap, patience, and learning rate from ``configs/datasets.yaml``."""
    config = _config()
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
    folder = _vaegbin() / BUNDLES[name]
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
    from paragvae.metametro_path import ensure_metametro, training_arrays

    ensure_metametro()
    from metametro import load_cgt

    _features, indptr, indices, edge_weight = training_arrays(load_cgt(cached / "cgt"))
    names = [str(i) for i in range(int(labels.max()) + 1)]
    return {
        "features": vae,
        "labels": labels,
        "label_names": names,
        "indptr": indptr,
        "indices": indices,
        "edge_weight": edge_weight,
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
    root = _metametro_work("phage_x10")
    cgt_dir = root / "cgt"
    from paragvae.metametro_path import training_arrays

    from metametro import load_cgt

    tensor = load_cgt(cgt_dir)
    features, indptr, indices, edge_weight = training_arrays(tensor)
    labels = np.asarray(tensor.node_labels, dtype=np.int64)
    colors = np.asarray(tensor.node_colors)
    coverage_table = pd.read_csv(root / "coloured_cfa" / "nodes.tsv", sep="\t")
    coverage = {row.node_id: float(row.coverage) for row in coverage_table.itertuples(index=False)}
    sequences = _fasta_lengths_gc(root / "coloured_cfa" / "nodes.fna")
    mapping = pd.read_csv(cgt_dir / "mapping.tsv", sep="\t")
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
    palette = _palette_path(cgt_dir)
    if palette is None:
        raise FileNotFoundError(
            f"phage CGT at {cgt_dir} has no colour palette, so n_sample_colors "
            "cannot count only the sample namespace"
        )
    return {
        "features": features,
        "labels": labels,
        "label_names": names,
        "indptr": indptr,
        "indices": indices,
        "edge_weight": edge_weight,
        "coverage": node_coverage,
        "length": length,
        "gc": gc,
        "n_sample_colors": sample_color_counts(colors, color_names_from_palette(palette, tensor.color_ids)),
    }


# Geometric benches and the gold namespaces load_geometric reads from ground_truth/.
# Signatures match MetaMetro v0.24.0 load_bench_states (docs/benchmarks.md):
# spb uses node_namespace=transport_type and edge_namespaces=(transport_type, route);
# roxel uses node_namespace=type and edge_namespaces=(name,).
GEOMETRIC = {
    "spb_transit": {
        "bench": "spb_ground_transit",
        "node_namespace": "transport_type",
        "edge_namespaces": ("transport_type", "route"),
    },
    "roxel": {
        "bench": "roxel",
        "node_namespace": "type",
        "edge_namespaces": ("name",),
    },
}

# Prefixes published in MetaMetro v0.24.0 docs/benchmarks.md for default
# builds with the GTFS mirror and the sfnetworks Roxel data. Full hashes
# are not stored here; a real build is required to check them.
GEOMETRIC_IDENTITY_PREFIXES = {
    "spb_ground_transit": "cfd1dcfb",
    "roxel": "3232245d",
}


def sample_color_counts(node_colors: np.ndarray, color_names: list[tuple[str, str]]) -> np.ndarray:
    """Return, per node, the number of colours in the ``sample`` namespace only."""
    colors = np.asarray(node_colors)
    if colors.ndim != 2:
        raise ValueError("node_colors must be a 2-d array")
    if colors.shape[1] != len(color_names):
        raise ValueError("colour columns do not match colour names")
    keep = [index for index, (namespace, _value) in enumerate(color_names) if namespace == "sample"]
    if not keep:
        return np.zeros(colors.shape[0], dtype=np.float64)
    return colors[:, keep].sum(axis=1).astype(np.float64)


def color_names_from_palette(palette_rows: list[dict] | Path, color_ids) -> list[tuple[str, str]]:
    """Map CGT colour ids to ``(namespace, value)`` using a colour dictionary."""
    if isinstance(palette_rows, Path):
        table = pd.read_csv(palette_rows, sep="\t")
        rows = table.to_dict(orient="records")
    else:
        rows = palette_rows
    by_id = {
        int(row["color_id"]): (str(row["namespace"]), str(row.get("value") or row["color_id"]))
        for row in rows
    }
    names = []
    for color_id in color_ids:
        if int(color_id) not in by_id:
            raise KeyError(f"colour id {color_id} is missing from the colour dictionary")
        names.append(by_id[int(color_id)])
    return names


def _palette_path(cgt_dir: Path) -> Path | None:
    for candidate in (cgt_dir.parent / "cdbg" / "colors.tsv", cgt_dir.parent.parent / "cdbg" / "colors.tsv"):
        if candidate.is_file():
            return candidate
    return None


def _gold_files(outdir: Path, namespace: str) -> bool:
    root = Path(outdir) / "ground_truth"
    return (root / f"{namespace}_nodes.tsv").is_file() and (root / f"{namespace}_edges.tsv").is_file()


def _missing_gold_error(name: str, outdir: Path, missing: list[str]) -> FileNotFoundError:
    needed = ", ".join(missing)
    return FileNotFoundError(
        f"{name}: gold {needed} is not under {Path(outdir) / 'ground_truth'}. "
        "MetaMetro 0.23.0 and newer store these namespaces in ground_truth/, "
        "not in colors.tsv or CGT colours. This loader does not fall back to colours."
    )


def geometric_bench_dir(name: str) -> Path:
    """Return the default MetaMetro benchbuild directory for a geometric example."""
    if name not in GEOMETRIC:
        raise KeyError(name)
    from paragvae.metametro_path import ensure_metametro

    ensure_metametro()
    from metametro.bench.paths import default_outdir
    from metametro.bench.registry import resolve

    return default_outdir(resolve(GEOMETRIC[name]["bench"]))


def bench_identity(outdir: Path) -> str | None:
    """Return the build identity of a bench directory, if it is recorded."""
    manifest = Path(outdir) / "manifest.yaml"
    if manifest.is_file():
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if data.get("identity"):
            return str(data["identity"])
    digest = Path(outdir) / "identity.sha256"
    if digest.is_file():
        return digest.read_text(encoding="utf-8").strip().split()[0] or None
    return None


def load_geometric(name: str, outdir: str | Path | None = None) -> dict:
    """Load a Saint Petersburg or Roxel bench, with gold from ``ground_truth/``.

    Uses ``metametro.bench.load_bench_states`` for features, node classes, and
    edge targets, and ``load_bench_gold`` / ``load_bench_gold_cgt`` for the
    gold matrices. Bench colours stay in ``node_colors``; gold never comes
    from ``colors.tsv``. Pass ``outdir`` to point at a benchbuild directory;
    the default is MetaMetro's ``default_outdir`` for the named bench.
    """
    if name not in GEOMETRIC:
        raise KeyError(name)
    spec = GEOMETRIC[name]
    needed = tuple(dict.fromkeys((spec["node_namespace"], *spec["edge_namespaces"])))
    folder = Path(outdir) if outdir is not None else geometric_bench_dir(name)
    missing = [namespace for namespace in needed if not _gold_files(folder, namespace)]
    if missing:
        raise _missing_gold_error(name, folder, missing)

    from paragvae.metametro_path import ensure_metametro, training_arrays

    ensure_metametro()
    from metametro.bench import load_bench_gold, load_bench_gold_cgt, load_bench_states

    states = load_bench_states(
        folder,
        node_namespace=spec["node_namespace"],
        edge_namespaces=spec["edge_namespaces"],
    )
    tensor = states.cgt
    features, indptr, indices, edge_weight = training_arrays(tensor)
    labels = np.asarray(tensor.node_labels, dtype=np.int64)
    names = [str(item) for item in states.node_class_names]
    if labels.size and len(names) != int(labels.max()) + 1:
        raise ValueError(f"{name}: class names do not cover the label ids")
    gold = {}
    for namespace in needed:
        labels_ns = load_bench_gold(folder, namespace)
        gold_cgt = load_bench_gold_cgt(folder, namespace)
        matrix = np.asarray(gold_cgt.node_colors)
        if matrix.shape[0] != features.shape[0]:
            raise ValueError(f"{name}: gold {namespace} is not aligned with the CGT nodes")
        gold[namespace] = {
            "values": labels_ns.values,
            "names": [(namespace, value) for value in labels_ns.values],
            "node_colors": matrix,
        }
    color_names = color_names_from_palette(list(states.cdbg.colors or []), tensor.color_ids)
    return {
        "features": features,
        "feature_names": [str(item) for item in tensor.metadata["node_feature_names"]],
        "labels": labels,
        "label_names": names,
        "indptr": indptr,
        "indices": indices,
        "edge_weight": edge_weight,
        "node_colors": np.asarray(tensor.node_colors),
        "color_names": color_names,
        "gold": gold,
        "edge_targets": np.asarray(states.edge_targets),
        "edge_label_names": list(states.edge_label_names),
        "bench_dir": folder,
    }
