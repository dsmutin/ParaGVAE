"""Mandatory: an unbuilt completion bench becomes status=missing rows, not a crash."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_registry import (  # noqa: E402
    ARMS,
    _done,
    _missing_load_reason,
    _unloaded_rows,
    _write,
)

from paragvae.completion import load_completion  # noqa: E402

pytestmark = pytest.mark.mandatory


def _contract_only(tmp_path: Path, *, assembly_report: bool = True) -> Path:
    """The layout of a ``status: contract`` bench: pin and abundance, no graph."""
    root = tmp_path / "bacteria_species_20_heldout" / "megahit" / "reads100000"
    (root / "contract").mkdir(parents=True)
    (root / "ground_truth").mkdir()
    (root / "contract" / "accessions.tsv").write_text("accession\n", encoding="utf-8")
    (root / "ground_truth" / "abundance.csv").write_text("accession,abundance\n", encoding="utf-8")
    (root / "manifest.yaml").write_text("bench: bacteria_species_20_heldout\nstatus: contract\n", encoding="utf-8")
    if assembly_report:
        raw = root.parents[1] / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / "assembly_data_report.jsonl").write_text("", encoding="utf-8")
    return root


def _built_heldout(tmp_path: Path) -> Path:
    """A held-out bench that has cdbg/, ground_truth/, and an assembly report."""
    root = _contract_only(tmp_path, assembly_report=True)
    cdbg = root / "cdbg"
    cdbg.mkdir()
    (cdbg / "metadata.yaml").write_text("kind: cdbg\n", encoding="utf-8")
    return root


def _tiny_config() -> dict:
    return {"seeds": [0], "max_epochs": 1, "patience": 1, "lr": 0.05}


def _patch_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, paths: dict[str, Path]) -> None:
    import run_registry

    monkeypatch.setattr(run_registry, "OUT", tmp_path)
    monkeypatch.setattr(run_registry, "completion_paths", lambda config: paths)
    monkeypatch.setattr(run_registry, "load_config", _tiny_config)
    monkeypatch.setattr(run_registry, "_chart", lambda rows: None)


def test_contract_only_bench_yields_missing_rows(tmp_path: Path) -> None:
    root = _contract_only(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        load_completion("heldout_genera", root)
    assert "CDBG" in str(caught.value) or "cdbg" in str(caught.value).lower()
    reason = _missing_load_reason("heldout_genera", root)
    assert reason == "bench not built: no cdbg"
    rows = _unloaded_rows("heldout_genera", root, [0, 1], reason, set())
    expected = 2 * sum(len(clusterings) for *_rest, clusterings in ARMS)
    assert len(rows) == expected
    assert {row["status"] for row in rows} == {"missing"}
    assert all(row["f1"] == "" and row["ari"] == "" for row in rows)
    assert all(row["note"] == "bench not built: no cdbg" for row in rows)


def test_unloaded_rows_skip_done_keys(tmp_path: Path) -> None:
    root = _contract_only(tmp_path)
    done = {("heldout_genera", 0, "gcn", "kmeans")}
    rows = _unloaded_rows("heldout_genera", root, [0], "bench not built: no cdbg", done)
    keys = {(row["arm"], row["clustering"]) for row in rows}
    assert ("gcn", "kmeans") not in keys
    assert ("gcn", "vamb") in keys


def test_missing_reason_lists_absent_build_artifacts(tmp_path: Path) -> None:
    root = _contract_only(tmp_path, assembly_report=False)
    assert _missing_load_reason("heldout_genera", root) == "bench not built: no cdbg, no assembly report"


def test_missing_reason_no_ground_truth(tmp_path: Path) -> None:
    root = _contract_only(tmp_path)
    for child in (root / "ground_truth").iterdir():
        child.unlink()
    (root / "ground_truth").rmdir()
    # still no cdbg, report is present
    assert _missing_load_reason("heldout_genera", root) == "bench not built: no cdbg, no ground_truth"


def test_done_ignores_missing_rows(tmp_path: Path) -> None:
    destination = tmp_path / "results.csv"
    _write(
        destination,
        [
            {
                "dataset": "heldout_genera",
                "seed": 0,
                "arm": "gcn",
                "architecture": "gcn",
                "n_layers": 2,
                "loss": "standard",
                "clustering": "kmeans",
                "status": "missing",
                "f1": "",
                "ari": "",
                "epochs": "",
                "stopped_early": "",
                "n_nodes": "",
                "n_edges": "",
                "note": "stale",
            },
            {
                "dataset": "phage_x10",
                "seed": 0,
                "arm": "gcn",
                "architecture": "gcn",
                "n_layers": 2,
                "loss": "standard",
                "clustering": "kmeans",
                "status": "ok",
                "f1": "0.304348",
                "ari": "0.100221",
                "epochs": "27",
                "stopped_early": "1",
                "n_nodes": "983",
                "n_edges": "1270",
                "note": "",
            },
        ],
    )
    done = _done(destination)
    assert ("heldout_genera", 0, "gcn", "kmeans") not in done
    assert ("heldout_genera", 0, "gcn") not in done
    assert ("phage_x10", 0, "gcn", "kmeans") in done
    assert ("phage_x10", 0, "gcn") in done


def test_main_unbuilt_bench_writes_missing_rows_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() must catch FileNotFoundError for an unbuilt bench (R3)."""
    import run_registry

    root = _contract_only(tmp_path / "unbuilt")
    _patch_main(monkeypatch, tmp_path / "out", {"heldout_genera": root})
    run_registry.main()
    destination = tmp_path / "out" / "results.csv"
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    expected = sum(len(clusterings) for *_rest, clusterings in ARMS)
    assert len(rows) == expected
    assert {row["status"] for row in rows} == {"missing"}
    assert {row["note"] for row in rows} == {"bench not built: no cdbg"}


def test_main_built_bench_with_lost_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built bench missing a required file must stop, not become missing rows."""
    import run_registry

    root = _built_heldout(tmp_path / "built")
    assert _missing_load_reason("heldout_genera", root) is None
    with pytest.raises(FileNotFoundError, match="contigs.paf"):
        load_completion("heldout_genera", root)
    _patch_main(monkeypatch, tmp_path / "out", {"heldout_genera": root})
    with pytest.raises(FileNotFoundError, match="contigs.paf"):
        run_registry.main()
    destination = tmp_path / "out" / "results.csv"
    assert not destination.is_file()


def test_main_rerun_recomputes_missing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status=missing must not count as done, so a rerun rewrites the rows."""
    import run_registry

    root = _contract_only(tmp_path / "unbuilt")
    out = tmp_path / "out"
    destination = out / "results.csv"
    stale = []
    for arm, architecture, n_layers, loss, _flip, clusterings in ARMS:
        for method in clusterings:
            stale.append(
                {
                    "dataset": "heldout_genera",
                    "seed": 0,
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
                    "n_nodes": "",
                    "n_edges": "",
                    "note": "stale missing reason",
                }
            )
    out.mkdir()
    _write(destination, stale)
    _patch_main(monkeypatch, out, {"heldout_genera": root})
    run_registry.main()
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    assert len(rows) == len(stale)
    assert all(row["note"] == "bench not built: no cdbg" for row in rows)
    assert all(row["status"] == "missing" for row in rows)
