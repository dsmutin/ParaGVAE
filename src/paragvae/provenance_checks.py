"""Checks for evaluation taxids on simulated metagenomes.

The evaluation id of a simulated community is the ``tax_id`` on the ``sim``
row of the accession table. That table is fixed before reads are generated.
Kraken calls, CheckM ids, k-means colours, and fixture labels are not a
substitute.

Those ids may be used to score a model. They must not be a feature column,
an edge weight, or a colour of the graph that the model trains on.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def simulation_tax_ids(accessions: str | Path) -> dict[str, int]:
    """Map a versionless simulated accession to its pre-generation ``tax_id``."""
    path = Path(accessions)
    if not path.is_file():
        raise FileNotFoundError(f"accession table is missing: {path}")
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    if not rows or "role" not in rows[0] or "accession" not in rows[0] or "tax_id" not in rows[0]:
        raise ValueError(f"{path} must have role, accession, and tax_id columns")
    found: dict[str, int] = {}
    for row in rows:
        if row["role"] != "sim":
            continue
        accession = row["accession"].strip()
        if "." not in accession:
            raise ValueError(f"{path} sim accession has no version: {accession!r}")
        tax = row["tax_id"].strip()
        if not tax:
            raise ValueError(f"{accession} has no tax_id in {path}")
        key = accession.split(".", 1)[0]
        code = int(tax)
        if key in found and found[key] != code:
            raise ValueError(f"{key} has two sim tax ids in {path}")
        found[key] = code
    if not found:
        raise ValueError(f"{path} has no sim rows")
    return found


def assert_labels_are_simulation_taxids(labels: np.ndarray, allowed: dict[str, int]) -> None:
    """Fail unless every scored label is a pre-generation sim tax id."""
    if not allowed:
        raise ValueError("the pre-generation taxid table is empty")
    known = set(allowed.values())
    used = {int(label) for label in np.asarray(labels).tolist() if int(label) >= 0}
    if not used:
        raise ValueError("no node has a pre-generation taxid")
    foreign = sorted(used - known)
    if foreign:
        sample = ", ".join(str(item) for item in foreign[:8])
        raise ValueError(f"evaluation labels are not sim tax ids: {sample}")


def assert_training_inputs_hide_labels(
    labels: np.ndarray,
    features: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> None:
    """Fail if the evaluation label vector is copied into a model input."""
    target = np.asarray(labels)
    usable = target >= 0
    if int(usable.sum()) < 2:
        return
    _reject_matching_columns("features", np.asarray(features), target, usable)
    if colors is not None and np.asarray(colors).size:
        _reject_matching_columns("colors", np.asarray(colors), target, usable)
        values = {int(item) for item in np.unique(np.asarray(colors)) if int(item) >= 0}
        leaked = sorted(values & {int(label) for label in target[usable].tolist()})
        if leaked:
            raise ValueError(f"colour values include evaluation taxids: {leaked[:8]}")
    if edge_weight is not None and np.asarray(edge_weight).size:
        weight = np.asarray(edge_weight, dtype=np.float64).reshape(-1)
        if weight.shape[0] == target.shape[0]:
            _reject_matching_columns("edge_weight", weight.reshape(-1, 1), target, usable)


def _reject_matching_columns(name: str, matrix: np.ndarray, labels: np.ndarray, usable: np.ndarray) -> None:
    values = np.asarray(matrix)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != labels.shape[0]:
        return
    for column in range(values.shape[1]):
        channel = values[:, column]
        if channel.shape != labels.shape:
            continue
        if np.array_equal(channel[usable], labels[usable]):
            raise ValueError(f"{name} column {column} copies the evaluation labels")
