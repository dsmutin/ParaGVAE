"""Mandatory: completion paths distinguish an unknown bench name from a missing outdir."""

from __future__ import annotations

from pathlib import Path

import pytest
from metametro.errors import ContractError

from paragvae.completion import completion_paths

pytestmark = pytest.mark.mandatory

_NAMES = {
    "phage_x10": "phage_species_5_x10",
    "heldout_genera": "bacteria_species_20_heldout",
    "low75half": "3domain_genus_75_half",
    "half100half": "4domain_family_100_half",
}


def _patch_metametro(monkeypatch: pytest.MonkeyPatch, *, resolve, default_outdir) -> None:
    monkeypatch.setattr("paragvae.metametro_path.ensure_metametro", lambda: None)
    monkeypatch.setattr("metametro.bench.registry.resolve", resolve)
    monkeypatch.setattr("metametro.bench.paths.default_outdir", default_outdir)


def test_unknown_bench_name_is_treated_as_a_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ContractError from resolve means the value is a path, not a bench name."""
    absolute = tmp_path / "custom_graph"
    work = tmp_path / "work"
    config = {
        "completion": {
            **_NAMES,
            "phage_x10": str(absolute),
        }
    }

    def resolve(_raw: str) -> object:
        raise ContractError([f"unknown benchmark: {_raw}"])

    def default_outdir(_spec: object) -> Path:
        raise AssertionError("default_outdir must not run when resolve fails")

    def work_dir(name: str) -> Path:
        return work / name

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    monkeypatch.setattr("paragvae.metametro_path.work_dir", work_dir)
    paths = completion_paths(config)
    assert paths["phage_x10"] == absolute
    assert paths["heldout_genera"] == work / "bacteria_species_20_heldout"
    assert paths["low75half"] == work / "3domain_genus_75_half"
    assert paths["half100half"] == work / "4domain_family_100_half"


def test_default_outdir_error_is_not_a_path_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """ContractError from default_outdir, such as a missing checkout, must propagate."""

    def resolve(raw: str) -> object:
        return {"name": raw}

    def default_outdir(_spec: object) -> Path:
        raise ContractError(["no checkout, pass --outdir"])

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    with pytest.raises(ContractError, match="pass --outdir"):
        completion_paths({"completion": dict(_NAMES)})
