"""Biological auxiliary loss from the May BioLoss comparison.

The total is edge loss plus ``alpha`` times taxonomy consistency plus
``beta`` times ``1 - R²`` of read-weighted soft abundances against a Kraken
reference. Taxonomy ids and the abundance vector are inputs. They are not
read from the evaluation labels.

Same-taxon pairs are pulled together. Different-taxon pairs are pushed
until their squared distance reaches ``margin``. Prototypes used by the
abundance term are the current class means and do not receive gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BioTargets:
    """Kraken-derived targets for one graph.

    ``taxon_index`` is ``-1`` when the node has no classified taxon.
    ``abundance`` is a probability vector over the classified taxa, in the
    same order as those indices.
    """

    taxon_index: np.ndarray
    node_weight: np.ndarray
    abundance: np.ndarray


def load_kraken_targets(calls: Path, counts: Path, members: list[list[str]], lengths: np.ndarray) -> BioTargets:
    """Join Kraken contig calls onto unitigs through CFA member ids.

    A unitig is labelled only when every classified member agrees. Mixed or
    unclassified unitigs stay at ``-1`` and do not enter the taxonomy pairs.
    """
    call_path = Path(calls)
    count_path = Path(counts)
    if not call_path.is_file():
        raise FileNotFoundError(f"Kraken calls are missing: {call_path}")
    if not count_path.is_file():
        raise FileNotFoundError(f"Kraken counts are missing: {count_path}")
    called = _read_calls(call_path)
    if not called:
        raise ValueError(f"{call_path} has no classified sequences")
    if len(members) != len(lengths):
        raise ValueError("Kraken member rows and lengths do not match")
    raw = np.full(len(members), -1, dtype=np.int64)
    for index, nodes in enumerate(members):
        taxes = {called[node] for node in nodes if node in called}
        if len(taxes) == 1:
            raw[index] = next(iter(taxes))
    present = sorted({int(tax) for tax in raw.tolist() if tax >= 0})
    if not present:
        raise ValueError(f"no unitig in {call_path} received a single Kraken taxon")
    column = {tax: slot for slot, tax in enumerate(present)}
    taxon_index = np.full(len(members), -1, dtype=np.int64)
    for index, tax in enumerate(raw.tolist()):
        if tax >= 0:
            taxon_index[index] = column[int(tax)]
    abundance = np.zeros(len(present), dtype=np.float64)
    for tax, count in _read_counts(count_path).items():
        slot = column.get(int(tax))
        if slot is not None:
            abundance[slot] += count
    total = float(abundance.sum())
    if total <= 0:
        raise ValueError(f"{count_path} has no counts for the taxa on these unitigs")
    abundance /= total
    weight = np.asarray(lengths, dtype=np.float64)
    if np.any(weight < 0):
        raise ValueError("node weights must be non-negative")
    return BioTargets(taxon_index=taxon_index, node_weight=weight, abundance=abundance)


def biological_gradient(
    embedding: np.ndarray,
    targets: BioTargets,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    margin: float = 2.0,
    seed: int = 0,
    n_same: int = 256,
    n_diff: int = 256,
) -> tuple[float, np.ndarray]:
    """Return the auxiliary loss and its gradient with respect to ``embedding``."""
    if alpha < 0 or beta < 0:
        raise ValueError("biological loss weights must be non-negative")
    latent = np.asarray(embedding, dtype=np.float64)
    grad = np.zeros_like(latent)
    tax_loss, tax_grad = _taxonomy(latent, targets.taxon_index, margin=margin, seed=seed, n_same=n_same, n_diff=n_diff)
    r2_loss, r2_grad = _abundance(latent, targets)
    grad += alpha * tax_grad + beta * r2_grad
    return float(alpha * tax_loss + beta * r2_loss), grad.astype(np.float32)


def _read_calls(path: Path) -> dict[str, int]:
    found: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t")[:3] != ["seq_id", "status", "taxon_id"]:
        raise ValueError(f"{path} header must be seq_id, status, taxon_id")
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{path} line {line_number} has fewer than 3 columns")
        if parts[1] != "C":
            continue
        found[parts[0]] = int(parts[2])
    return found


def _read_counts(path: Path) -> dict[int, float]:
    totals: dict[int, float] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split("\t")[:3] != ["seq_id", "taxon_id", "count"]:
        raise ValueError(f"{path} header must be seq_id, taxon_id, count")
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{path} line {line_number} has fewer than 3 columns")
        tax = int(parts[1])
        if tax <= 0:
            continue
        totals[tax] = totals.get(tax, 0.0) + float(parts[2])
    return totals


def _taxonomy(
    latent: np.ndarray,
    taxon_index: np.ndarray,
    *,
    margin: float,
    seed: int,
    n_same: int,
    n_diff: int,
) -> tuple[float, np.ndarray]:
    grad = np.zeros_like(latent)
    groups: dict[int, list[int]] = {}
    for index, taxon in enumerate(np.asarray(taxon_index).tolist()):
        if int(taxon) >= 0:
            groups.setdefault(int(taxon), []).append(index)
    rng = np.random.default_rng(seed)
    same: list[tuple[int, int]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        draw = min(n_same, len(members) * 2)
        pick = rng.choice(members, size=(draw, 2), replace=True)
        for left, right in pick:
            if left != right:
                same.append((int(left), int(right)))
    taxa = list(groups)
    different: list[tuple[int, int]] = []
    if len(taxa) >= 2:
        while len(different) < n_diff:
            first, second = rng.choice(taxa, size=2, replace=False)
            different.append((int(rng.choice(groups[int(first)])), int(rng.choice(groups[int(second)]))))
    loss = 0.0
    if same:
        left = latent[[pair[0] for pair in same]]
        right = latent[[pair[1] for pair in same]]
        delta = left - right
        loss += float(np.mean(np.sum(delta * delta, axis=1)))
        coeff = (2.0 / len(same)) * delta
        np.add.at(grad, [pair[0] for pair in same], coeff)
        np.add.at(grad, [pair[1] for pair in same], -coeff)
    if different:
        left = latent[[pair[0] for pair in different]]
        right = latent[[pair[1] for pair in different]]
        delta = left - right
        distance = np.sum(delta * delta, axis=1)
        hinge = np.maximum(margin - distance, 0.0)
        loss += float(np.mean(hinge))
        active = hinge > 0
        coeff = np.zeros_like(delta)
        coeff[active] = (-2.0 / len(different)) * delta[active]
        np.add.at(grad, [pair[0] for pair in different], coeff)
        np.add.at(grad, [pair[1] for pair in different], -coeff)
    return loss, grad


def _abundance(latent: np.ndarray, targets: BioTargets) -> tuple[float, np.ndarray]:
    observed = targets.taxon_index >= 0
    if int(observed.sum()) < 2 or targets.abundance.size < 2:
        return 0.0, np.zeros_like(latent)
    means = []
    for slot in range(targets.abundance.size):
        rows = latent[targets.taxon_index == slot]
        means.append(rows.mean(axis=0) if rows.size else np.zeros(latent.shape[1]))
    prototypes = np.stack(means, axis=0)
    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    proto_norm[proto_norm < 1e-8] = 1.0
    prototypes = prototypes / proto_norm
    rows = latent[observed]
    row_norm = np.linalg.norm(rows, axis=1, keepdims=True)
    row_norm[row_norm < 1e-8] = 1.0
    normalized = rows / row_norm
    logits = normalized @ prototypes.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    probs = weights / weights.sum(axis=1, keepdims=True)
    node_weight = targets.node_weight[observed]
    mass = node_weight[:, None] * probs
    predicted = mass.sum(axis=0)
    total = float(predicted.sum()) + 1e-8
    predicted = predicted / total
    target = np.asarray(targets.abundance, dtype=np.float64)
    target = target / (float(target.sum()) + 1e-8)
    loss, grad_pred = _one_minus_r2(predicted, target)
    # Quotient s / sum(s): d(pred)/d(s) applied to grad_pred.
    grad_s = (grad_pred - float(grad_pred @ predicted)) / total
    grad_probs = node_weight[:, None] * grad_s
    grad_logits = probs * (grad_probs - np.sum(probs * grad_probs, axis=1, keepdims=True))
    grad_norm = grad_logits @ prototypes
    grad_rows = grad_norm / row_norm
    grad = np.zeros_like(latent)
    grad[observed] = grad_rows
    return loss, grad


def _one_minus_r2(predicted: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    left = predicted - predicted.mean()
    right = target - target.mean()
    num = float(left @ right)
    left_ss = float(left @ left)
    right_ss = float(right @ right)
    den = np.sqrt(left_ss * right_ss) + 1e-8
    correlation = num / den
    # d(r)/d(centered) then the mean-centering Jacobian is I - 1/n.
    grad_left = right / den - correlation * right_ss * left / (den * den)
    grad_pred = grad_left - grad_left.mean()
    r2 = float(np.clip(correlation * correlation, 0.0, 1.0))
    return 1.0 - r2, (-2.0 * correlation) * grad_pred
