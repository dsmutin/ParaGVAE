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
    assert "lr: 0.05" in text
    assert "command: python research/run_hypothesis.py loss" in text
    assert "numpy:" in text
    assert "metametro_version:" in text
    assert f"metametro_pin: {REQUIRED_METAMETRO}" in text
    assert "metametro_commit:" in text


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
    # The editable 0.22.1 checkout is fdd0519…; accept a full or short hash.
    line = next(part for part in text.splitlines() if part.startswith("metametro_commit:"))
    commit = line.split(":", 1)[1].strip()
    assert commit != "unknown"
    assert len(commit) >= 7


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

