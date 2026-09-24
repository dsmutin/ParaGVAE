"""AMBER genome-binning stage.

Writes CAMI binning files and runs ``amber.py``. The reported score is
``f1_score_seq`` (average purity and completeness per sequence). Contig F1
stays a separate column.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_HEADER = "@Version:0.9.0\n@SampleID:SAMPLEID\n"


def write_binning_tsv(path: Path, sequence_ids: list[str], bin_ids: np.ndarray) -> None:
    """Write a CAMI binning file: one predicted bin id per sequence."""
    if len(sequence_ids) != len(bin_ids):
        raise ValueError(f"{len(sequence_ids)} sequence ids and {len(bin_ids)} bin ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER, "@@SEQUENCEID\tBINID\n"]
    for sequence_id, bin_id in zip(sequence_ids, bin_ids):
        lines.append(f"{sequence_id}\t{int(bin_id)}\n")
    path.write_text("".join(lines), encoding="utf-8")


def load_gold(path: Path) -> dict[str, tuple[str, int]]:
    """Map sequence id to ``(genome id, length)``.

    Accepts a CAMI gold file or a CSV with ``contig_id``, ``genome_id``, and
    ``length``. A missing length, a zero length, or a repeated sequence id is
    an error.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path} is empty")
    if text.lstrip().startswith("@") or "SEQUENCEID" in text.splitlines()[0]:
        return _gold_from_cami(text, path)
    return _gold_from_csv(text, path)


def _gold_from_cami(text: str, path: Path) -> dict[str, tuple[str, int]]:
    rows: dict[str, tuple[str, int]] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("@@"):
            header = line[2:].split("\t")
            continue
        if line.startswith("@"):
            continue
        if header is None:
            raise ValueError(f"{path} has no @@SEQUENCEID header")
        fields = dict(zip(header, line.split("\t")))
        sequence_id = fields.get("SEQUENCEID")
        genome = fields.get("BINID")
        length = fields.get("LENGTH", fields.get("_LENGTH"))
        if sequence_id is None or genome is None or length is None:
            raise ValueError(f"{path} row is missing SEQUENCEID, BINID, or LENGTH")
        _store_gold(rows, sequence_id, genome, length, path)
    if not rows:
        raise ValueError(f"{path} has no gold sequences")
    return rows


def _gold_from_csv(text: str, path: Path) -> dict[str, tuple[str, int]]:
    rows: dict[str, tuple[str, int]] = {}
    reader = csv.DictReader(text.splitlines())
    required = {"contig_id", "genome_id", "length"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"{path} must have columns contig_id, genome_id, length")
    for record in reader:
        _store_gold(rows, record["contig_id"], record["genome_id"], record["length"], path)
    if not rows:
        raise ValueError(f"{path} has no gold sequences")
    return rows


def _store_gold(rows: dict[str, tuple[str, int]], sequence_id: str, genome: str, length: str, path: Path) -> None:
    if sequence_id in rows:
        raise ValueError(f"{path} repeats sequence {sequence_id}")
    parsed = int(float(length))
    if parsed <= 0:
        raise ValueError(f"{path} sequence {sequence_id} has length {parsed}")
    if not str(genome):
        raise ValueError(f"{path} sequence {sequence_id} has an empty genome id")
    rows[sequence_id] = (str(genome), parsed)


def write_gold_tsv(path: Path, gold: dict[str, tuple[str, int]], sequence_ids: list[str]) -> None:
    """Write the CAMI gold file for ``sequence_ids``, in that order."""
    missing = [sequence_id for sequence_id in sequence_ids if sequence_id not in gold]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} sequences are absent from the gold standard, including {sample}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER, "@@SEQUENCEID\tBINID\tLENGTH\n"]
    for sequence_id in sequence_ids:
        genome, length = gold[sequence_id]
        lines.append(f"{sequence_id}\t{genome}\t{length}\n")
    path.write_text("".join(lines), encoding="utf-8")


def score_bins(
    sequence_ids: list[str],
    bin_ids: np.ndarray,
    gold: dict[str, tuple[str, int]],
    work: Path,
) -> dict[str, float]:
    """Run AMBER and return sequence-level precision, recall, and F1."""
    amber = shutil.which("amber.py")
    if amber is None:
        raise RuntimeError("amber.py is not on PATH. Install conda package cami-amber.")
    work.mkdir(parents=True, exist_ok=True)
    bins = work / "bins.tsv"
    gold_path = work / "gold_standard_genome.tsv"
    write_binning_tsv(bins, sequence_ids, bin_ids)
    write_gold_tsv(gold_path, gold, sequence_ids)
    out = work / "amber"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    done = subprocess.run(
        [amber, "-g", str(gold_path), "-l", "paragvae", "-o", str(out), "--silent", "--skip_gs", str(bins)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    results = out / "results.tsv"
    if not results.is_file():
        detail = (done.stderr or done.stdout or "").strip()
        raise RuntimeError(f"amber.py wrote no results.tsv (exit {done.returncode}). {detail}")
    return _read_results(results)


def _read_results(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    tools = [row for row in rows if "gold" not in str(row.get("Tool", "")).lower()]
    if len(tools) != 1:
        raise ValueError(f"{path} has {len(tools)} tool rows")
    row = tools[0]
    return {
        "amber_f1": float(row["f1_score_seq"]),
        "amber_ap": float(row["precision_avg_seq"]),
        "amber_ar": float(row["recall_avg_seq"]),
    }
