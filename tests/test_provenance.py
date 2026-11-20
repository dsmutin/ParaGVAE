"""Mandatory: learning rate is configured, and a run records its provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import metametro
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_hypothesis import write_provenance  # noqa: E402
from run_registry import write_registry_provenance  # noqa: E402

from paragvae.metametro_path import REQUIRED_METAMETRO  # noqa: E402

pytestmark = pytest.mark.mandatory


def _config() -> dict:
    return {
        "seeds": [0, 1],
        "max_epochs": 60,
        "patience": 8,
        "lr": 0.05,
        "vaegbin_data": "/data",
    }


def test_provenance_records_config(tmp_path: Path) -> None:
    """The sidecar names the learning rate and the MetaMetro checkout."""
    write_provenance(tmp_path, "loss", _config())
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "generated automatically; do not edit"
    assert "lr: 0.05" in text
    assert "command: python research/run_hypothesis.py loss" in text
    assert "numpy:" in text
    assert "metametro_version:" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text
    assert "metametro_commit:" in text
    assert "vaegbin_data: /data" in text
    assert "dirty:" in text


def test_provenance_records_installed_version_not_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """metametro_version is the installed package, not a copy of the pin."""
    monkeypatch.setattr("paragvae.metametro_path.ensure_metametro", lambda: "9.9.9")
    write_provenance(tmp_path, "loss", _config())
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert "metametro_version: 9.9.9" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text


def test_provenance_version_tracks_metametro_dunder_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patching metametro.__version__ (not ensure_metametro) must show up in the sidecar.

    ensure_metametro is left unpatched, so a mutation that returns the pin
    instead of metametro.__version__ fails when the two differ.
    """
    monkeypatch.setattr(metametro, "__version__", "9.8.7")
    write_provenance(tmp_path, "loss", _config())
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert f"metametro_version: 9.8.7" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text
    assert "metametro_version: 9.8.7" in text.split("metametro_pin")[0]


def test_registry_provenance_tracks_metametro_dunder_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry sidecar uses the same installed-version helper."""
    monkeypatch.setattr(metametro, "__version__", "9.8.7")
    write_registry_provenance(
        tmp_path,
        _config(),
        paths={"phage_x10": tmp_path / "phage"},
    )
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert "command: python research/run_registry.py" in text
    assert "metametro_version: 9.8.7" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text
    assert "path_phage_x10:" in text


def test_registry_provenance_records_real_install(tmp_path: Path) -> None:
    """Unpatched, the registry sidecar matches the installed package and pin."""
    write_registry_provenance(tmp_path, _config())
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert f"metametro_version: {metametro.__version__}" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text
    assert "metametro_commit: unknown" not in text
    assert "metametro_commit:" in text
    line = next(part for part in text.splitlines() if part.startswith("metametro_commit:"))
    commit = line.split(":", 1)[1].strip()
    assert commit != "unknown"
    assert len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit)
    assert text.splitlines()[0] == "generated automatically; do not edit"
    assert "hidden: 32" in text
    assert "latent: 16" in text
    assert "hdbscan:" in text
    assert "labels:" in text
    assert "load_bench_gold_cgt" in text
    assert "dirty:" in text


def test_ds_analyze_provenance_tracks_metametro_dunder_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake metametro.__version__ appears in the DS analyze provenance sidecar."""
    import numpy as np

    from paragvae.ds.analyze import analyze

    monkeypatch.setattr(metametro, "__version__", "9.8.7")
    features = np.array([[0.0], [1.0], [0.1], [1.1]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    indptr = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    indices = np.array([1, 0, 3, 2], dtype=np.int64)
    out = tmp_path / "ds"
    analyze(
        name="toy",
        features=features,
        labels=labels,
        label_names=["a", "b"],
        indptr=indptr,
        indices=indices,
        edge_weight=None,
        extras={},
        out=out,
        protocol="toy",
        max_epochs=1,
        patience=1,
        lr=0.05,
        seeds=(0,),
    )
    text = (out / "provenance.txt").read_text(encoding="utf-8")
    assert "command: python examples/DS/toy/analyze.py" in text
    assert "metametro_version: 9.8.7" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text
    assert "metametro_commit:" in text
    assert "paragvae_commit:" in text
    assert "metametro_version: 9.8.7" in text.split("metametro_pin")[0]


def test_generated_automatically_header_is_required(tmp_path: Path) -> None:
    """The machine-written sidecar starts with the generated-automatically header."""
    from paragvae.run_record import GENERATED_HEADER

    write_registry_provenance(tmp_path, _config())
    first = (tmp_path / "provenance.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first == GENERATED_HEADER
    assert first == "generated automatically; do not edit"


def test_paragvae_commit_is_this_repository_head(tmp_path: Path) -> None:
    """paragvae_commit must be git rev-parse HEAD of the ParaGVAE checkout, not MetaMetro."""
    import subprocess

    from paragvae.run_record import git_head

    write_registry_provenance(tmp_path, _config())
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    expected = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert f"paragvae_commit: {expected}" in text
    assert expected == git_head(ROOT)
    # It must not silently pick up the installed MetaMetro checkout.
    from paragvae.metametro_path import metametro_root

    mm = str(git_head(metametro_root()) or "")
    if mm and mm != expected:
        assert f"paragvae_commit: {mm}" not in text


def test_metametro_commit_rejects_arbitrary_text() -> None:
    from paragvae.run_record import recorded_git_commit

    assert recorded_git_commit("unknown", what="metametro_commit") == "unknown"
    digest = "f" * 40
    assert recorded_git_commit(digest, what="metametro_commit") == digest
    with pytest.raises(ValueError, match="40-character lowercase hex"):
        recorded_git_commit("0.24.0", what="metametro_commit")
    with pytest.raises(ValueError, match="40-character lowercase hex"):
        recorded_git_commit("not-a-commit", what="metametro_commit")


def test_write_rejects_a_non_hex_metametro_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("paragvae.run_record.installed_metametro_commit", lambda: "v0.24.0-pin")
    with pytest.raises(ValueError, match="metametro_commit"):
        write_registry_provenance(tmp_path, _config())


def test_hypothesis_sidecar_keeps_vaegbin_data(tmp_path: Path) -> None:
    write_provenance(tmp_path, "loss", _config())
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert "vaegbin_data: /data" in text
    dropped = {key: value for key, value in _config().items() if key != "vaegbin_data"}
    with pytest.raises(KeyError, match="vaegbin_data"):
        write_provenance(tmp_path, "loss", dropped)


def test_registry_sidecar_records_bench_identity(tmp_path: Path) -> None:
    phage = tmp_path / "phage"
    phage.mkdir()
    (phage / "identity.sha256").write_text(
        "78121c1b05a5814bfc17ac3b7650ed5223a0f584cfada24e9224b7ba0364564c\n",
        encoding="utf-8",
    )
    write_registry_provenance(tmp_path, _config(), paths={"phage_x10": phage})
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert "identity_phage_x10: 78121c1b05a5814bfc17ac3b7650ed5223a0f584cfada24e9224b7ba0364564c" in text
    assert "hdbscan:" in text
    assert "hidden: 32" in text
    assert "latent: 16" in text


def _stub_ok_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, paths: dict[str, Path], *, results=None) -> None:
    import numpy as np
    import run_registry
    from test_registry_missing import _patch_main

    features = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    indptr = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    indices = np.array([1, 0, 3, 2], dtype=np.int64)

    class Graph:
        def __init__(self) -> None:
            self.labels = labels
            self.raw_features = features
            self.cgt = object()

    class Fit:
        embedding = np.array([[0.0, 1.0], [0.1, 1.1], [1.0, 0.0], [1.1, 0.1]], dtype=np.float32)
        epochs_ran = 1
        stopped_early = True

    _patch_main(monkeypatch, tmp_path / "out", paths)
    monkeypatch.setattr(run_registry, "load_completion", lambda name, path: (Graph(), None))
    monkeypatch.setattr(run_registry, "training_arrays", lambda cgt: (features, indptr, indices, None))
    monkeypatch.setattr(run_registry, "assert_labels_are_simulation_taxids", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_registry, "simulation_tax_ids", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_registry, "assert_training_inputs_hide_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_registry, "_train", lambda *args, **kwargs: Fit())
    monkeypatch.setattr(run_registry, "_cluster", lambda embedding, method, n_groups, seed: labels.copy())
    monkeypatch.setattr(run_registry, "evaluate", lambda *args, **kwargs: {"f1": 0.5, "ari": 0.25})
    if results is not None:
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        from run_registry import _write

        _write(tmp_path / "out" / "results.csv", results)


def test_main_writes_provenance_when_it_computes_ok_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() must write provenance.txt when it computes ok rows (M7)."""
    import subprocess

    import run_registry
    from test_registry_missing import _built_heldout

    root = _built_heldout(tmp_path / "built")
    _stub_ok_run(monkeypatch, tmp_path, {"heldout_genera": root})
    run_registry.main([])
    text = (tmp_path / "out" / "provenance.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "generated automatically; do not edit"
    expected = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert f"paragvae_commit: {expected}" in text
    rows_file = tmp_path / "out" / "row_provenance.csv"
    assert rows_file.is_file()
    body = rows_file.read_text(encoding="utf-8")
    assert expected in body
    assert "heldout_genera" in body


def test_main_does_not_rewrite_provenance_when_no_ok_rows_are_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume that computes no new ok rows must leave provenance.txt alone."""
    import run_registry
    from test_registry_missing import ARMS, _built_heldout, _write

    root = _built_heldout(tmp_path / "built")
    existing = []
    for arm, architecture, n_layers, loss, _flip, clusterings in ARMS:
        for method in clusterings:
            existing.append(
                {
                    "dataset": "heldout_genera",
                    "seed": 0,
                    "arm": arm,
                    "architecture": architecture,
                    "n_layers": n_layers,
                    "loss": loss,
                    "clustering": method,
                    "status": "ok",
                    "f1": "0.500000",
                    "ari": "0.250000",
                    "epochs": "1",
                    "stopped_early": "1",
                    "n_nodes": "4",
                    "n_edges": "4",
                    "note": "",
                }
            )
    _stub_ok_run(monkeypatch, tmp_path, {"heldout_genera": root}, results=existing)
    sidecar = tmp_path / "out" / "provenance.txt"
    sidecar.write_text("SENTINEL\n", encoding="utf-8")
    before = (tmp_path / "out" / "results.csv").read_bytes()
    run_registry.main([])
    assert sidecar.read_text(encoding="utf-8") == "SENTINEL\n"
    assert (tmp_path / "out" / "results.csv").read_bytes() == before
