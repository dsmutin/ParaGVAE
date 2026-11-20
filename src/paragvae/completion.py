"""Graphs for the completion benchmarks.

``heldout_genera``, ``low75half``, and ``half100half`` are CDBG unitig graphs.
Evaluation labels are species ids from the assembly report. Kraken calls, when
the example stored them, are a separate training target for the biological
loss. ``phage_x10`` is the MetaMetro coloured tensor. It has no Kraken call
table, so the biological arm cannot run on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from paragvae.bioloss import BioTargets, load_kraken_targets
from paragvae.graphs import StudyGraph, load_metametro_bench
from paragvae.heldout import cdbg_dir, load_heldout_graph
from paragvae.metametro_path import ensure_metametro


# phage_x10 is a built bench by contract; it must never be recorded as missing.
MUST_BE_BUILT = ("phage_x10",)
BENCH_ROOT_ENV = "PARAGVAE_BENCH_ROOT"


def completion_bench_root(config: dict | None = None, root: str | Path | None = None) -> Path | None:
    """Return the explicit MetaMetro checkout root, if one was given.

    Precedence: the ``root`` argument, then ``config['bench_root']``, then
    the ``PARAGVAE_BENCH_ROOT`` environment variable. There is no silent
    fallback that invents a root.
    """
    if root is not None:
        return Path(root)
    if config and config.get("bench_root"):
        return Path(str(config["bench_root"]))
    env = os.environ.get(BENCH_ROOT_ENV)
    if env:
        return Path(env)
    return None


def completion_paths(config: dict, *, bench_root: str | Path | None = None) -> dict[str, Path]:
    """Resolve the four completion graphs.

    A value that is a MetaMetro benchmark name (or a legacy alias) points at
    that benchbuild directory under ``bench_root`` (or the installed MetaMetro
    checkout). Any other value is a path, absolute or relative to the
    MetaMetro work directory. A path that does not exist is an error, with a
    hint to pass ``--bench-root`` or set ``PARAGVAE_BENCH_ROOT``. ``phage_x10``
    must be a built bench.
    """
    block = config.get("completion")
    if not isinstance(block, dict):
        raise ValueError("configs/datasets.yaml is missing the completion paths")
    required = ("phage_x10", "heldout_genera", "low75half", "half100half")
    missing = [name for name in required if not block.get(name)]
    if missing:
        raise ValueError(f"completion paths are missing: {', '.join(missing)}")
    from paragvae.metametro_path import ensure_metametro, work_dir

    ensure_metametro()
    from metametro.bench.paths import default_outdir
    from metametro.bench.registry import resolve
    from metametro.errors import ContractError

    root = completion_bench_root(config, bench_root)
    if root is not None and not root.is_dir():
        raise FileNotFoundError(
            f"bench root {root} does not exist; pass --bench-root or set {BENCH_ROOT_ENV}"
        )
    resolved = {}
    for name in required:
        raw = str(block[name])
        try:
            spec = resolve(raw)
        except ContractError:
            path = Path(raw)
            resolved[name] = path if path.is_absolute() else work_dir(raw)
            continue
        resolved[name] = default_outdir(spec, root=root) if root is not None else default_outdir(spec)
    for name, path in resolved.items():
        _require_existing_bench(name, path)
    return resolved


def _require_existing_bench(name: str, path: Path) -> None:
    """Raise if ``path`` is missing, or if a must-be-built bench is not built."""
    hint = f"Pass --bench-root or set {BENCH_ROOT_ENV} to a MetaMetro checkout that contains the bench."
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} was not found at {path}. {hint}")
    if name not in MUST_BE_BUILT:
        return
    status = _manifest_status(path)
    if status == "built":
        return
    if status is None:
        raise FileNotFoundError(
            f"{name} must be a built bench, but {path} has no readable manifest status. {hint}"
        )
    raise FileNotFoundError(
        f"{name} must be a built bench, but {path} has manifest status {status!r}. {hint}"
    )


def _manifest_status(root: Path) -> str | None:
    manifest = Path(root) / "manifest.yaml"
    if not manifest.is_file():
        return None
    try:
        from paragvae.metametro_path import ensure_metametro

        ensure_metametro()
        from metametro.tables import read_yaml

        data = read_yaml(manifest)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    status = str(data.get("status") or "").strip()
    return status or None


# Each example's ``run_example.py`` points at a different NCBI assembly report.
_REPORTS = {
    "heldout_genera": "assembly_data_report.jsonl",
    "low75half": "low75_assembly_data_report.jsonl",
    "half100half": "high100_assembly_data_report.jsonl",
}


def _assembly_report(name: str, example: Path) -> Path:
    filename = _REPORTS.get(name)
    if filename is None:
        raise ValueError(f"no assembly report is configured for {name}")
    path = example.resolve().parents[1] / "data" / "raw" / filename
    if not path.is_file():
        raise FileNotFoundError(f"assembly report for {name} is missing: {path}")
    return path


def load_completion(name: str, path: Path) -> tuple[StudyGraph, BioTargets | None]:
    """Load one completion graph and its Kraken targets, if that file exists."""
    if name == "phage_x10":
        return _phage(path), None
    report = _assembly_report(name, path)
    graph = load_heldout_graph(path, report=report)
    calls = path / "work" / "reprofile" / "kraken_calls.tsv"
    counts = path / "work" / "reprofile" / "kraken_counts.tsv"
    if not calls.is_file() or not counts.is_file():
        return graph, None
    ensure_metametro()
    from metametro.formats.cdbg.io import load_cdbg

    unitigs = sorted(load_cdbg(cdbg_dir(path)).unitigs, key=lambda unitig: unitig.unitig_id)
    if len(unitigs) != len(graph.labels):
        raise ValueError(f"{name} unitig count does not match the loaded graph")
    members = [[str(member) for member in unitig.members] for unitig in unitigs]
    lengths = np.asarray(graph.raw_features[:, 0], dtype=np.float64)
    return graph, load_kraken_targets(calls, counts, members, lengths)


def _phage(path: Path) -> StudyGraph:
    root = Path(path)
    if (root / "ground_truth" / "taxon_nodes.tsv").is_file():
        return load_metametro_bench(root, name="phage_x10")
    folder = root / "cgt"
    required = ("indptr.npy", "indices.npy", "node_features.npy", "node_labels.npy", "edge_features.npy")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"phage CGT is missing {', '.join(missing)} under {folder}")
    import warnings

    warnings.warn(
        "phage_x10 has no ground_truth/taxon_nodes.tsv; labels are read from the "
        "stored CGT node_labels.npy. Rebuild the bench and use load_bench_gold_cgt.",
        UserWarning,
        stacklevel=2,
    )
    ensure_metametro()
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
