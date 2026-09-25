"""Load a held-out CFA/CDBG example as a coloured graph tensor.

Node features are length, GC, and out-degree. Species labels come from the
best PAF hit joined through the example's assembly report. They are stored
only as evaluation labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paragvae.graphs import StudyGraph
from paragvae.metametro_path import ensure_metametro

JOIN_FRACTION = 0.9


def cdbg_dir(example: Path) -> Path:
    """Return the CDBG directory for one held-out example."""
    return Path(example) / "work" / "reprofile" / "tocumg" / "cdbg"


def load_heldout_graph(
    example: str | Path,
    metametro_src: str | Path | None = None,
    report: str | Path | None = None,
) -> StudyGraph:
    """Load ``example`` (held-out genera or half strains).

    Raises ``FileNotFoundError`` when the CDBG directory is missing. A
    synthetic CFA/CDBG is not built here. Species ids are
    ``checkmSpeciesTaxId`` values already stored in the example assembly
    report. This function does not walk an NCBI taxdump.
    """
    root = Path(example)
    folder = cdbg_dir(root)
    if not folder.is_dir():
        raise FileNotFoundError(f"CDBG directory is missing: {folder}")
    report = Path(report) if report is not None else _assembly_report(root)
    paf = root / "work" / "reprofile" / "contigs.paf"
    for path in (report, paf):
        if not path.is_file():
            raise FileNotFoundError(f"required held-out file is missing: {path}")
    ensure_metametro(metametro_src)
    from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
    from metametro.formats.cdbg.io import load_cdbg
    from metametro.formats.cgt.validator import validate_cgt

    cdbg = load_cdbg(folder)
    unitigs = sorted(cdbg.unitigs, key=lambda unitig: unitig.unitig_id)
    if not unitigs:
        raise ValueError(f"{folder} has no unitigs")
    features = _node_features(cdbg, unitigs)
    labels = _species_labels(root, unitigs, paf, report)
    graph = cdbg_to_cgt(
        cdbg,
        node_features=features,
        node_feature_names=["length", "gc", "out_degree"],
        node_labels=labels,
    )
    validate_cgt(graph)
    return StudyGraph(
        name=root.name,
        cgt=graph,
        different_pairs=np.zeros((0, 2), dtype=np.int64),
        vae_features=features,
        raw_features=features.copy(),
        labels=labels,
        sequence_ids=[unitig.unitig_id for unitig in unitigs],
    )


def _assembly_report(example: Path) -> Path:
    """Assembly report used by ``examples/heldout_genera/run_example.py``."""
    return example.resolve().parents[1] / "data" / "raw" / "assembly_data_report.jsonl"


def _gc(sequence: str) -> float:
    text = sequence.upper()
    if not text:
        raise ValueError("unitig sequence is empty")
    return (text.count("G") + text.count("C")) / len(text)


def _node_features(cdbg: object, unitigs: list) -> np.ndarray:
    """Length and GC from the unitig FASTA, plus directed out-degree."""
    dense = {unitig.unitig_id: index for index, unitig in enumerate(unitigs)}
    degree = np.zeros(len(unitigs), dtype=np.float32)
    for link in cdbg.links:
        if link.source not in dense:
            raise ValueError(f"link {link.link_id} leaves a unitig that is not in the CDBG")
        degree[dense[link.source]] += 1.0
    rows = []
    for unitig in unitigs:
        sequence = str(unitig.sequence)
        rows.append((float(len(sequence)), _gc(sequence), float(degree[dense[unitig.unitig_id]])))
    return np.asarray(rows, dtype=np.float32)


def _species_by_accession(report: Path) -> dict[str, int]:
    """Map a versionless accession to the species id stored on that report row."""
    found: dict[str, int] = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        accession = str(record.get("accession") or record.get("currentAccession") or "")
        if not accession or "." not in accession:
            raise ValueError(f"assembly report row has no versioned accession: {accession!r}")
        species = (record.get("checkmInfo") or {}).get("checkmSpeciesTaxId")
        if species in (None, ""):
            raise ValueError(f"{accession} has no checkmSpeciesTaxId in {report}")
        key = accession.split(".", 1)[0]
        code = int(species)
        if key in found and found[key] != code:
            raise ValueError(f"accession key {key} maps to both {found[key]} and {code}")
        found[key] = code
    if not found:
        raise ValueError(f"{report} has no assembly rows")
    return found


def _best_hits(paf: Path) -> dict[str, str]:
    """Best PAF target per query, ranked by the matches column (index 9)."""
    best: dict[str, tuple[int, str]] = {}
    for line_number, line in enumerate(paf.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 12:
            raise ValueError(f"{paf} line {line_number} has fewer than 12 columns")
        query, target = parts[0], parts[5]
        matches = int(parts[9])
        previous = best.get(query)
        if previous is None or matches > previous[0]:
            best[query] = (matches, target)
    if not best:
        raise ValueError(f"{paf} has no alignments")
    return {query: target for query, (_matches, target) in best.items()}


def _species_labels(example: Path, unitigs: list, paf: Path, report: Path) -> np.ndarray:
    """Join PAF query ids to unitigs. Stop when fewer than 90% of nodes join."""
    species_of = _species_by_accession(report)
    hits = _best_hits(paf)
    query_species: dict[str, int] = {}
    unknown_targets: set[str] = set()
    for query, target in hits.items():
        species = species_of.get(target)
        if species is None:
            unknown_targets.add(target)
            continue
        query_species[query] = species
    labels = np.full(len(unitigs), -1, dtype=np.int64)
    joined = 0
    for index, unitig in enumerate(unitigs):
        members = [str(member) for member in unitig.members]
        species = {query_species[member] for member in members if member in query_species}
        if len(species) != 1:
            continue
        labels[index] = next(iter(species))
        joined += 1
    fraction = joined / len(unitigs)
    if fraction < JOIN_FRACTION:
        sample = ", ".join(sorted(unknown_targets)[:8]) or "none"
        raise ValueError(
            f"{example.name}: joined {joined} of {len(unitigs)} nodes "
            f"({fraction:.4f}); need at least {JOIN_FRACTION:.0%}. "
            f"Unmapped PAF targets (sample): {sample}"
        )
    return labels
