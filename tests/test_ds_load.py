"""Mandatory: DS phage loader checks for a colour palette."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.mandatory


def test_load_phage_requires_a_colour_palette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A phage CGT without colors.tsv must raise (G5)."""
    from paragvae.ds.load import load_phage

    root = tmp_path / "phage_x10"
    cgt = root / "cgt"
    cfa = root / "coloured_cfa"
    cgt.mkdir(parents=True)
    cfa.mkdir()
    ids = "\n".join(f"{i}\tnode{i}" for i in range(6))
    (cgt / "mapping.tsv").write_text("dense_id\tcfa_node_ids\n" + ids + "\n", encoding="utf-8")
    coverage_rows = "node_id\tcoverage\n" + "\n".join(f"node{i}\t1.0" for i in range(6)) + "\n"
    (cfa / "nodes.tsv").write_text(coverage_rows, encoding="utf-8")
    fasta = "".join(f">node{i}\nACGTACGT\n" for i in range(6))
    (cfa / "nodes.fna").write_text(fasta, encoding="utf-8")

    class Tensor:
        node_labels = np.arange(6, dtype=np.int64)
        node_colors = np.zeros((6, 1), dtype=np.int64)
        color_ids = [0]

    features = np.ones((6, 1), dtype=np.float32)
    indptr = np.arange(7, dtype=np.int64)
    indices = np.array([1, 0, 3, 2, 5, 4], dtype=np.int64)
    monkeypatch.setattr("paragvae.ds.load._metametro_work", lambda name: root)
    monkeypatch.setattr("metametro.load_cgt", lambda path: Tensor())
    monkeypatch.setattr(
        "paragvae.metametro_path.training_arrays",
        lambda tensor: (features, indptr, indices, None),
    )
    with pytest.raises(FileNotFoundError, match="colour palette"):
        load_phage()
