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
    absolute.mkdir()
    (absolute / "manifest.yaml").write_text("status: built\n", encoding="utf-8")
    for name in ("bacteria_species_20_heldout", "3domain_genus_75_half", "4domain_family_100_half"):
        (work / name).mkdir(parents=True)
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


def _write_built(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.yaml").write_text("status: built\n", encoding="utf-8")
    return path


def test_missing_bench_directory_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolved bench that is not on disk must not become missing rows."""

    def resolve(raw: str) -> object:
        return {"name": raw}

    def default_outdir(_spec: object, root=None) -> Path:
        return tmp_path / "absent" / "data" / "bench" / "nowhere"

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    with pytest.raises(FileNotFoundError, match="PARAGVAE_BENCH_ROOT"):
        completion_paths({"completion": dict(_NAMES)})


def test_phage_x10_must_be_a_built_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """phage_x10 with no built manifest is an error, even if the directory exists."""

    def resolve(raw: str) -> object:
        return {"name": raw}

    def default_outdir(spec: object, root=None) -> Path:
        name = spec["name"] if isinstance(spec, dict) else "x"
        path = tmp_path / "root" / "data" / "bench" / name
        path.mkdir(parents=True, exist_ok=True)
        if name != "phage_species_5_x10":
            (path / "manifest.yaml").write_text("status: contract\n", encoding="utf-8")
        else:
            (path / "manifest.yaml").write_text("status: contract\n", encoding="utf-8")
        return path

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    with pytest.raises(FileNotFoundError, match="phage_x10 must be a built bench"):
        completion_paths({"completion": dict(_NAMES)})


def test_bench_root_selects_the_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--bench-root / PARAGVAE_BENCH_ROOT is passed to default_outdir."""
    chosen = tmp_path / "explicit"

    def resolve(raw: str) -> object:
        return {"name": raw, "raw": raw}

    seen: list = []

    def default_outdir(spec: object, root=None) -> Path:
        seen.append(root)
        raw = spec["raw"] if isinstance(spec, dict) else "x"
        return _write_built(chosen / "data" / "bench" / raw / "assembler" / "props")

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    chosen.mkdir()
    paths = completion_paths({"completion": dict(_NAMES)}, bench_root=chosen)
    assert seen and all(Path(root) == chosen for root in seen)
    assert paths["phage_x10"] == chosen / "data" / "bench" / "phage_species_5_x10" / "assembler" / "props"


def test_missing_explicit_bench_root_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metametro(
        monkeypatch,
        resolve=lambda raw: {"name": raw},
        default_outdir=lambda spec, root=None: tmp_path / "unused",
    )
    with pytest.raises(FileNotFoundError, match="bench root"):
        completion_paths({"completion": dict(_NAMES)}, bench_root=tmp_path / "no-such-root")
