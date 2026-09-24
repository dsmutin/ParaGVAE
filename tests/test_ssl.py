"""Mandatory: SSL plan checks and a class-weighted supervised GCN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paragvae.ssl import (
    KRAKEN_COLOUR_FN,
    _cgt_after_kraken,
    _class_weights,
    fit_supervised_gcn,
    require_tools,
    ssl_plan,
)

pytestmark = pytest.mark.mandatory


def test_ssl_plan_missing_genome_dir(tmp_path: Path) -> None:
    """A missing genome directory fails before any process starts."""
    with pytest.raises(FileNotFoundError, match="genome_dir"):
        ssl_plan(tmp_path / "missing-genomes", tmp_path / "work", tmp_path / "kraken-db")


def test_ssl_plan_empty_genome_dir(tmp_path: Path) -> None:
    """An empty genome directory is not a usable input."""
    genomes = tmp_path / "genomes"
    genomes.mkdir()
    database = tmp_path / "kraken-db"
    database.mkdir()
    with pytest.raises(ValueError, match="empty"):
        ssl_plan(genomes, tmp_path / "work", database)


def test_ssl_plan_rejects_even_k(tmp_path: Path) -> None:
    """Assembly k must be odd and at least 15."""
    with pytest.raises(ValueError, match="odd"):
        ssl_plan(tmp_path / "genomes", tmp_path / "work", tmp_path / "kraken-db", k=20)


def test_require_tools_names_every_missing_executable() -> None:
    """One FileNotFoundError lists each missing binary, not only the first."""
    missing = (
        "paragvae-absent-samovar",
        "paragvae-absent-megahit",
        "paragvae-absent-kraken2",
    )
    with pytest.raises(FileNotFoundError) as caught:
        require_tools(samovar=missing[0], megahit=missing[1], kraken2=missing[2])
    text = str(caught.value)
    for name in missing:
        assert name in text


def _ring() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected 8-cycle as CSR."""
    n = 8
    rows: list[int] = []
    cols: list[int] = []
    for node in range(n):
        nxt = (node + 1) % n
        rows.extend((node, nxt))
        cols.extend((nxt, node))
    order = np.argsort(rows, kind="mergesort")
    rows_arr = np.asarray(rows)[order]
    cols_arr = np.asarray(cols)[order]
    counts = np.bincount(rows_arr, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    return indptr, cols_arr.astype(np.int64), np.ones(n, dtype=bool)


def _labeled_ring() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Six training nodes carry the label in their features; two eval nodes do not."""
    indptr, indices, mask = _ring()
    mask[6:] = False
    labels = np.array([0, 0, 0, 1, 1, 1, -1, 99], dtype=np.int64)
    rng = np.random.default_rng(0)
    features = np.zeros((8, 2), dtype=np.float32)
    features[0:3, 0] = 4.0
    features[3:6, 1] = 4.0
    features[:6] += rng.normal(0.0, 0.05, size=(6, 2)).astype(np.float32)
    features[6] = np.array([-4.0, -4.0], dtype=np.float32)
    features[7] = np.array([4.0, 4.0], dtype=np.float32)
    return features, indptr, indices, labels, mask


def test_class_weight_is_inverse_sqrt_count() -> None:
    """A class with four training nodes weighs half of a singleton class."""
    weights = _class_weights(np.array([0, 0, 0, 0, 1], dtype=np.int64), n_classes=2)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(1.0)
    assert weights.shape == (2,)


def test_kraken_colouring_stops_at_named_gap(tmp_path: Path) -> None:
    """FASTG is loaded with MetaMetro, then Kraken colouring names the missing function."""
    from paragvae.metametro_path import ensure_metametro

    try:
        ensure_metametro()
    except FileNotFoundError:
        pytest.skip("MetaMetro checkout is not on this machine")
    from metametro.errors import ContractError

    bad = tmp_path / "bad.fastg"
    bad.write_text("not a fastg\n", encoding="utf-8")
    plan = {
        "fastg": str(bad),
        "k": 21,
        "kraken_out": "kraken.out",
        "kraken_report": "kraken.report",
    }
    with pytest.raises(ContractError):
        _cgt_after_kraken(plan)
    fastg = tmp_path / "k21.fastg"
    fastg.write_text(">NODE_1_length_32_cov_1.0_ID_1;\n" + ("ACGT" * 8) + "\n", encoding="utf-8")
    plan["fastg"] = str(fastg)
    with pytest.raises(NotImplementedError) as caught:
        _cgt_after_kraken(plan)
    assert str(caught.value) == KRAKEN_COLOUR_FN


def test_supervised_gcn_beats_chance_on_the_train_mask() -> None:
    """The head beats chance on the six observed nodes. Eval nodes are not scored."""
    features, indptr, indices, labels, mask = _labeled_ring()
    fit = fit_supervised_gcn(
        features,
        indptr,
        indices,
        labels,
        mask,
        n_classes=2,
        max_epochs=40,
        patience=8,
        lr=0.05,
        seed=0,
    )
    assert fit.probabilities.shape == (8, 2)
    assert fit.embedding.shape[0] == 8
    assert np.isfinite(fit.probabilities).all()
    predicted = fit.probabilities.argmax(axis=1)
    accuracy = float(np.mean(predicted[mask] == labels[mask]))
    assert accuracy > 0.5
    other = labels.copy()
    other[6] = 4
    other[7] = -5
    again = fit_supervised_gcn(
        features,
        indptr,
        indices,
        other,
        mask,
        n_classes=2,
        max_epochs=40,
        patience=8,
        lr=0.05,
        seed=0,
    )
    assert np.allclose(fit.probabilities, again.probabilities)
    assert np.allclose(fit.embedding, again.embedding)


def test_supervised_loss_does_not_require_samovar(monkeypatch: pytest.MonkeyPatch) -> None:
    """The masked loss runs with no assembler and does not look up executables."""

    def _refuse_lookup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fit_supervised_gcn looked up an executable")

    monkeypatch.setattr("paragvae.ssl.shutil.which", _refuse_lookup)
    features, indptr, indices, labels, mask = _labeled_ring()
    fit = fit_supervised_gcn(
        features,
        indptr,
        indices,
        labels,
        mask,
        n_classes=2,
        max_epochs=2,
        patience=2,
        lr=0.05,
        seed=0,
    )
    assert fit.probabilities.shape == (8, 2)


def test_ssl_plan_argv_matches_metametro(tmp_path: Path) -> None:
    """Samovar and MEGAHIT argv come from MetaMetro. Kraken2 uses the stated flags."""
    from paragvae.metametro_path import ensure_metametro

    try:
        ensure_metametro()
    except FileNotFoundError:
        pytest.skip("MetaMetro checkout is not on this machine")
    from metametro.contracts.external import (
        contig2fastg_command,
        megahit_command,
        samovar_generate_command,
    )

    genomes = tmp_path / "genomes"
    genomes.mkdir()
    (genomes / "g.fasta").write_text(">g\nACGT\n", encoding="utf-8")
    database = tmp_path / "kraken-db"
    database.mkdir()
    work = tmp_path / "work"
    plan = ssl_plan(genomes, work, database, k=21, total_reads=200, seed=1)
    assert not work.exists()
    names = [stage["name"] for stage in plan["stages"]]
    assert names == ["samovar_generate", "iss", "megahit", "kraken2", "contig2fastg"]
    iss = Path(plan["iss_dir"])
    reads = Path(plan["reads"])
    assembly = Path(plan["assembly_dir"])
    contigs = Path(plan["contigs"])
    assert plan["stages"][0]["argv"] == samovar_generate_command(
        genomes,
        iss,
        n_samples=2,
        total_reads=200,
        host_fraction=0,
        seed=1,
        samovar="samovar",
    )
    assert plan["stages"][1]["argv"] == [
        "bash",
        str(iss / ".generate" / "generate.sh"),
        "--directory",
        str(iss),
    ]
    assert plan["stages"][2]["argv"] == megahit_command(
        reads,
        assembly,
        k=21,
        threads=4,
        megahit="megahit",
        min_count=2,
    )
    assert plan["stages"][3]["argv"] == [
        "kraken2",
        "--db",
        str(database),
        "--threads",
        "4",
        "--output",
        str(work / "kraken.out"),
        "--report",
        str(work / "kraken.report"),
        str(contigs),
    ]
    assert plan["stages"][4]["argv"] == contig2fastg_command(contigs, 21, toolkit="megahit_toolkit")
    assert plan["stages"][4]["stdout"] == str(work / "k21.fastg")
