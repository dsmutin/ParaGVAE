"""Mandatory: the AMBER stage writes CAMI files and reads f1_score_seq."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.amber import load_gold, score_bins, write_binning_tsv, write_gold_tsv

pytestmark = pytest.mark.mandatory


def test_perfect_bins_score_one(tmp_path: Path) -> None:
    """A binning that copies the gold genomes has AMBER sequence F1 of 1."""
    gold = {"a": ("g1", 10), "b": ("g1", 20), "c": ("g2", 5)}
    scores = score_bins(["a", "b", "c"], np.asarray([0, 0, 1]), gold, tmp_path)
    assert scores["amber_f1"] == pytest.approx(1.0)
    assert scores["amber_ap"] == pytest.approx(1.0)
    assert scores["amber_ar"] == pytest.approx(1.0)


def test_missing_sequence_is_an_error(tmp_path: Path) -> None:
    """A sequence that is not in the gold standard stops the stage."""
    gold_path = tmp_path / "gold.tsv"
    write_gold_tsv(gold_path, {"a": ("g1", 4)}, ["a"])
    loaded = load_gold(gold_path)
    with pytest.raises(ValueError, match="absent"):
        write_gold_tsv(tmp_path / "other.tsv", loaded, ["a", "missing"])


def test_binning_header(tmp_path: Path) -> None:
    """Predicted bins use the CAMI sequence and bin columns."""
    path = tmp_path / "bins.tsv"
    write_binning_tsv(path, ["contig_1"], np.asarray([3]))
    text = path.read_text(encoding="utf-8")
    assert "@@SEQUENCEID\tBINID\n" in text
    assert text.endswith("contig_1\t3\n")
