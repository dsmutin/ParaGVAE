"""Benchmark the completion arms on the four named graphs.

One row is one architecture, loss, clustering, dataset, and seed. The GCN
embedding is scored with k-means, VAMB, and HDBSCAN. The other arms are
scored with k-means. A missing Kraken table or a missing ``hdbscan`` install
is written as ``status=missing`` and does not invent a score.

Run from the repository root:

    python research/run_registry.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

from paragvae.arch import train_architecture
from paragvae.cluster_methods import cluster_hdbscan, cluster_kmeans, cluster_vamb
from paragvae.completion import completion_paths, load_completion
from paragvae.metametro_path import training_arrays
from paragvae.provenance_checks import (
    assert_labels_are_simulation_taxids,
    assert_training_inputs_hide_labels,
    simulation_tax_ids,
)
from paragvae.flip import edge_line_graph, infer_nodes_from_edges
from paragvae.score import evaluate
from paragvae.suite import ROOT, load_config

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


def write_registry_provenance(
    dest: Path,
    config: dict,
    paths: dict[str, Path] | None = None,
    *,
    dirty: str | None = None,
) -> None:
    """Write a machine-generated ``provenance.txt`` for a completion run."""
    from paragvae.run_record import write_run_provenance

    footer = [
        "score: contig F1 and ARI from paragvae.score. Not AMBER f1_score_seq.",
        "hidden: 32",
        "latent: 16",
        f"hdbscan: {_hdbscan_record()}",
        (
            "labels: phage_x10 is the phage_species_5_x10 bench; labels are the "
            "5-species gold from load_bench_gold_cgt, and unitigs without gold "
            "are excluded from ARI and from contig F1 recall, but counted in the "
            "F1 precision denominator. The three assemblies score tax_id from the "
            "sim row of accessions.tsv, fixed before read generation. A contig "
            "receives that id when its alignment target is the simulated accession."
        ),
    ]
    if paths:
        footer.append("graphs: " + ", ".join(paths))
        for name, path in paths.items():
            footer.append(f"path_{name}: {path}")
            identity = _bench_identity(path)
            if identity:
                footer.append(f"identity_{name}: {identity}")
    payload = {key: value for key, value in config.items() if key not in {"amber", "bench_root"}}
    write_run_provenance(
        dest,
        command="python research/run_registry.py",
        config=payload,
        footer=footer,
        repo_root=ROOT,
        dirty=dirty,
    )


def _bench_identity(path: Path) -> str | None:
    digest = Path(path) / "identity.sha256"
    if digest.is_file():
        value = digest.read_text(encoding="utf-8").strip().split()[0]
        return value or None
    manifest = Path(path) / "manifest.yaml"
    if not manifest.is_file():
        return None
    try:
        from paragvae.metametro_path import ensure_metametro

        ensure_metametro()
        from metametro.tables import read_yaml

        data = read_yaml(manifest)
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("identity"):
        return None
    return str(data["identity"])


def _hdbscan_record() -> str:
    try:
        import hdbscan
    except ImportError:
        return "not installed"
    version = getattr(hdbscan, "__version__", "unknown")
    source = getattr(hdbscan, "__file__", "unknown")
    return f"{version} from {source}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the completion arms on the four named graphs.")
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=None,
        help="MetaMetro checkout that contains data/bench. Overrides PARAGVAE_BENCH_ROOT. "
        "Required when the installed MetaMetro checkout does not have the benches.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or [])
    _prefer_extracted_hdbscan()
    from paragvae.run_record import working_tree_dirty

    dirty = working_tree_dirty(ROOT)
    config = load_config()
    paths = completion_paths(config, bench_root=args.bench_root)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "results.csv"
    done = _done(destination)
    rows: list[dict] = []
    if destination.is_file():
        with destination.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    computed_records: list[dict] = []
    computed_ok = 0
    for name, path in paths.items():
        print(f"loading {name}", flush=True)
        if not _bench_allows_missing(path):
            damage = _built_bench_damage(name, path)
            if damage is not None:
                raise damage
        try:
            graph, bio = load_completion(name, path)
        except FileNotFoundError:
            reason = _missing_load_reason(name, path)
            if reason is None:
                raise
            print(f"  {name} missing: {reason}", flush=True)
            new_rows = _unloaded_rows(name, path, config["seeds"], reason, done)
            _upsert(rows, new_rows)
            computed_records.extend(_records_for(new_rows, dirty=dirty))
            _write(destination, rows)
            continue
        labels = np.asarray(graph.labels)
        if name != "phage_x10":
            assert_labels_are_simulation_taxids(labels, simulation_tax_ids(path / "accessions.tsv"))
        _tensor_features, indptr, indices, edge_weight = training_arrays(graph.cgt)
        features = np.asarray(graph.raw_features, dtype=np.float32)
        if features.shape[0] != _tensor_features.shape[0]:
            raise ValueError(f"{name} raw features do not match the CGT")
        n_edges = int(indptr[-1]) if len(indptr) else 0
        for seed in config["seeds"]:
            for arm, architecture, n_layers, loss, flip, clusterings in ARMS:
                key = (name, int(seed), arm)
                if key in done and all((name, int(seed), arm, method) in done for method in clusterings):
                    continue
                note = ""
                if loss == "biological":
                    for method in clusterings:
                        missing = _missing(
                            name, seed, arm, architecture, n_layers, loss, method, features.shape[0], n_edges,
                            "biological loss uses Kraken calls, not the pre-generation read taxid, and is not a training input",
                        )
                        _upsert(rows, missing)
                        computed_records.extend(_records_for([missing], dirty=dirty))
                    _write(destination, rows)
                    continue
                assert_training_inputs_hide_labels(
                    labels,
                    features,
                    colors=np.zeros((features.shape[0], 0), dtype=np.uint8),
                    edge_weight=edge_weight,
                )
                try:
                    trained = _train(
                        features, indptr, indices, edge_weight, architecture, n_layers, loss, flip, bio, seed, config
                    )
                except Exception as exc:
                    traceback.print_exc()
                    for method in clusterings:
                        missing = _missing(name, seed, arm, architecture, n_layers, loss, method, features.shape[0], n_edges, f"{type(exc).__name__}: {exc}")
                        _upsert(rows, missing)
                        computed_records.extend(_records_for([missing], dirty=dirty))
                    _write(destination, rows)
                    continue
                usable = labels >= 0
                n_groups = int(np.unique(labels[usable]).size) if usable.any() else 2
                for method in clusterings:
                    try:
                        predicted = _cluster(trained.embedding, method, n_groups, int(seed))
                    except Exception as exc:
                        missing = _missing(name, seed, arm, architecture, n_layers, loss, method, features.shape[0], n_edges, f"{type(exc).__name__}: {exc}")
                        _upsert(rows, missing)
                        computed_records.extend(_records_for([missing], dirty=dirty))
                        continue
                    scores = evaluate(trained.embedding, labels, method, int(seed), predicted=predicted)
                    ok_row = {
                        "dataset": name,
                        "seed": int(seed),
                        "arm": arm,
                        "architecture": architecture,
                        "n_layers": n_layers,
                        "loss": loss,
                        "clustering": method,
                        "status": "ok",
                        "f1": f"{float(scores['f1']):.6f}",
                        "ari": f"{float(scores['ari']):.6f}",
                        "epochs": trained.epochs_ran,
                        "stopped_early": int(trained.stopped_early),
                        "n_nodes": features.shape[0],
                        "n_edges": n_edges,
                        "note": note,
                    }
                    _upsert(rows, ok_row)
                    computed_records.extend(_records_for([ok_row], dirty=dirty))
                    computed_ok += 1
                _write(destination, rows)
                print(f"  {name} seed {seed} {arm} epochs {trained.epochs_ran}", flush=True)
    _chart(rows)
    if computed_records:
        from paragvae.run_record import merge_row_provenance

        merge_row_provenance(OUT / "row_provenance.csv", computed_records)
    if computed_ok:
        write_registry_provenance(OUT, config, paths, dirty=dirty)
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


def _records_for(rows, *, dirty: str) -> list[dict]:
    from paragvae.run_record import row_provenance_record

    if isinstance(rows, dict):
        rows = [rows]
    records = []
    for row in rows:
        records.append(
            row_provenance_record(
                row["dataset"],
                row["seed"],
                row["arm"],
                row["clustering"],
                repo_root=ROOT,
                dirty=dirty,
            )
        )
    return records


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


def _bench_manifest_status(root: Path) -> str | None:
    """Return ``manifest.yaml`` ``status``, or ``None`` if it cannot be read.

    Uses MetaMetro's ``read_yaml`` when the package is installed. There is no
    lighter public status-only API; ``check_bench_directory`` also validates
    the graphs and would itself fail on a damaged built bench.
    """
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


def _has_graph_artifacts(root: Path) -> bool:
    """True when the directory has a graph or identity, as in MetaMetro check.py."""
    folder = Path(root)
    return (
        (folder / "cfa").is_dir()
        or (folder / "cdbg").is_dir()
        or (folder / "cgt").is_dir()
        or (folder / "identity.sha256").is_file()
    )


def _bench_allows_missing(root: Path) -> bool:
    """True only for a clearly contract-only bench.

    Matches MetaMetro ``check_bench_directory``: ``status`` must be exactly
    ``contract`` and there must be no ``cfa/``, ``cdbg/``, ``cgt/``, or
    ``identity.sha256``. Every other status, including a typo, ``Built``,
    ``present``, ``missing``, an unreadable manifest, or ``contract`` with a
    graph or identity, is not allowed to become a missing row.
    """
    folder = Path(root)
    has_graph = _has_graph_artifacts(folder)
    if has_graph:
        return False
    status = _bench_manifest_status(folder)
    return status == "contract"


def _has_cdbg(root: Path) -> bool:
    """True when the bench has a graph directory (top-level or legacy path)."""
    direct = root / "cdbg"
    if (direct / "metadata.yaml").is_file():
        return True
    return (root / "work" / "reprofile" / "tocumg" / "cdbg").is_dir()


def _has_assembly_report(name: str, path: Path) -> bool:
    """True when the held-out assembly report exists, or the name does not use one."""
    from paragvae.completion import _REPORTS

    filename = _REPORTS.get(name)
    if filename is None:
        return True
    try:
        report = Path(path).resolve().parents[1] / "data" / "raw" / filename
    except IndexError:
        return False
    return report.is_file()


def _built_bench_damage(name: str, path: Path) -> FileNotFoundError | None:
    """Error for a built (or undeclared) bench that is missing a required part."""
    root = Path(path)
    parts: list[str] = []
    if not (root / "ground_truth").is_dir():
        parts.append("ground_truth/")
    if not _has_cdbg(root):
        parts.append("cdbg/metadata.yaml")
    if not _has_assembly_report(name, root):
        parts.append("assembly report")
    if not parts:
        return None
    status = _bench_manifest_status(root)
    if status == "built":
        prefix = f"{name} has manifest status built but is missing"
    else:
        prefix = f"{name} is not a contract-only bench and is missing"
    return FileNotFoundError(
        f"{prefix} {', '.join(parts)} at {root}. "
        "A missing row is allowed only when the bench is clearly contract-only "
        "(manifest status contract, and no cfa/, cdbg/, cgt/, or identity.sha256)."
    )


def _missing_load_reason(name: str, path: Path) -> str | None:
    """Return a precise reason if a failed load may be recorded as missing.

    A missing row is allowed only when the bench is clearly contract-only
    (manifest status ``contract`` and no graph or identity). Otherwise a
    lost ``cdbg/metadata.yaml``, assembly report, or ``ground_truth/``
    must stop the run.
    """
    root = Path(path)
    if not _bench_allows_missing(root):
        return None
    parts: list[str] = []
    if not (root / "ground_truth").is_dir():
        parts.append("no ground_truth")
    if not _has_cdbg(root):
        parts.append("no cdbg")
    if not _has_assembly_report(name, root):
        parts.append("no assembly report")
    if not parts:
        return None
    graph_parts = [part for part in parts if part != "no ground_truth"]
    if graph_parts:
        reason = "bench not built: " + ", ".join(graph_parts)
        if "no ground_truth" in parts:
            reason += ", no ground_truth"
        return reason
    return "no ground_truth"


def _unloaded_rows(name: str, path: Path, seeds, reason: str, done: set[tuple]) -> list[dict]:
    """``status=missing`` rows for every seed, arm, and clustering of a graph that cannot load."""
    rows = []
    for seed in seeds:
        for arm, architecture, n_layers, loss, _flip, clusterings in ARMS:
            for method in clusterings:
                if (name, int(seed), arm, method) in done:
                    continue
                rows.append(_missing(name, seed, arm, architecture, n_layers, loss, method, "", "", reason))
    return rows


def _done(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    found = set()
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            found.add((row["dataset"], int(row["seed"]), row["arm"], row["clustering"]))
            found.add((row["dataset"], int(row["seed"]), row["arm"]))
    return found


def _row_key(row: dict) -> tuple:
    return (row["dataset"], int(row["seed"]), row["arm"], row["clustering"])


def _upsert(rows: list[dict], new_rows: list[dict] | dict) -> None:
    """Replace an existing row with the same key, or append."""
    if isinstance(new_rows, dict):
        new_rows = [new_rows]
    index = {_row_key(row): i for i, row in enumerate(rows)}
    for row in new_rows:
        key = _row_key(row)
        if key in index:
            rows[index[key]] = row
        else:
            index[key] = len(rows)
            rows.append(row)


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
    main(sys.argv[1:])
