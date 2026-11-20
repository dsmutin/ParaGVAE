"""Mandatory: a completion bench that was never built becomes status=missing rows, not a crash."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_registry import ARMS, _unloaded_rows  # noqa: E402

from paragvae.completion import load_completion  # noqa: E402

pytestmark = pytest.mark.mandatory


def _contract_only(tmp_path: Path) -> Path:
    """The layout of a ``status: contract`` bench: pin and abundance, no graph."""
    root = tmp_path / "bacteria_species_20_heldout" / "megahit" / "reads100000"
    (root / "contract").mkdir(parents=True)
    (root / "ground_truth").mkdir()
    (root / "contract" / "accessions.tsv").write_text("accession\n", encoding="utf-8")
    (root / "ground_truth" / "abundance.csv").write_text("accession,abundance\n", encoding="utf-8")
    (root / "manifest.yaml").write_text("bench: bacteria_species_20_heldout\nstatus: contract\n", encoding="utf-8")
    raw = root.parents[1] / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "assembly_data_report.jsonl").write_text("", encoding="utf-8")
    return root


def test_contract_only_bench_yields_missing_rows(tmp_path: Path) -> None:
    root = _contract_only(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        load_completion("heldout_genera", root)
    rows = _unloaded_rows("heldout_genera", root, [0, 1], caught.value, set())
    expected = 2 * sum(len(clusterings) for *_rest, clusterings in ARMS)
    assert len(rows) == expected
    assert {row["status"] for row in rows} == {"missing"}
    assert all(row["f1"] == "" and row["ari"] == "" for row in rows)
    assert all(row["note"].startswith("bench status contract; graph not built") for row in rows)
    assert all("cdbg" in row["note"] for row in rows)


def test_unloaded_rows_skip_done_keys(tmp_path: Path) -> None:
    root = _contract_only(tmp_path)
    done = {("heldout_genera", 0, "gcn", "kmeans")}
    rows = _unloaded_rows("heldout_genera", root, [0], FileNotFoundError("x"), done)
    keys = {(row["arm"], row["clustering"]) for row in rows}
    assert ("gcn", "kmeans") not in keys
    assert ("gcn", "vamb") in keys
