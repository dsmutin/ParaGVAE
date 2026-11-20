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


def test_phage_x10_with_no_manifest_status_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A phage bench with a missing or unreadable manifest status must raise (B2)."""

    def resolve(raw: str) -> object:
        return {"name": raw}

    def default_outdir(spec: object, root=None) -> Path:
        name = spec["name"] if isinstance(spec, dict) else "x"
        path = tmp_path / "root" / "data" / "bench" / name
        path.mkdir(parents=True, exist_ok=True)
        if name != "phage_species_5_x10":
            (path / "manifest.yaml").write_text("status: contract\n", encoding="utf-8")
        return path

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    with pytest.raises(FileNotFoundError, match="no readable manifest status"):
        completion_paths({"completion": dict(_NAMES)})


def test_phage_x10_with_unreadable_manifest_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(raw: str) -> object:
        return {"name": raw}

    def default_outdir(spec: object, root=None) -> Path:
        name = spec["name"] if isinstance(spec, dict) else "x"
        path = tmp_path / "root" / "data" / "bench" / name
        path.mkdir(parents=True, exist_ok=True)
        if name != "phage_species_5_x10":
            (path / "manifest.yaml").write_text("status: contract\n", encoding="utf-8")
        else:
            (path / "manifest.yaml").write_text(": this is not yaml {", encoding="utf-8")
        return path

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    with pytest.raises(FileNotFoundError, match="no readable manifest status"):
        completion_paths({"completion": dict(_NAMES)})


def test_missing_heldout_path_says_was_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing held-out path must fail on existence, not later on a manifest (B3)."""

    def resolve(raw: str) -> object:
        return {"name": raw}

    def default_outdir(spec: object, root=None) -> Path:
        name = spec["name"] if isinstance(spec, dict) else "x"
        path = tmp_path / "root" / "data" / "bench" / name
        if name == "phage_species_5_x10":
            return _write_built(path)
        return path

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    with pytest.raises(FileNotFoundError, match="was not found"):
        completion_paths({"completion": dict(_NAMES)})


def test_env_bench_root_is_used_when_cli_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PARAGVAE_BENCH_ROOT is used when --bench-root is not passed (B5)."""
    from paragvae.completion import BENCH_ROOT_ENV, completion_paths

    chosen = tmp_path / "from-env"
    chosen.mkdir()
    seen: list = []

    def resolve(raw: str) -> object:
        return {"name": raw, "raw": raw}

    def default_outdir(spec: object, root=None) -> Path:
        seen.append(root)
        raw = spec["raw"] if isinstance(spec, dict) else "x"
        return _write_built(chosen / "data" / "bench" / raw / "assembler" / "props")

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    monkeypatch.delenv(BENCH_ROOT_ENV, raising=False)
    monkeypatch.setenv(BENCH_ROOT_ENV, str(chosen))
    paths = completion_paths({"completion": dict(_NAMES)})
    assert seen and all(Path(root) == chosen for root in seen)
    assert paths["phage_x10"] == chosen / "data" / "bench" / "phage_species_5_x10" / "assembler" / "props"


def test_cli_bench_root_wins_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--bench-root wins over PARAGVAE_BENCH_ROOT and config['bench_root'] (B6)."""
    from paragvae.completion import BENCH_ROOT_ENV, completion_bench_root, completion_paths

    cli = tmp_path / "from-cli"
    env = tmp_path / "from-env"
    cfg = tmp_path / "from-config"
    for folder in (cli, env, cfg):
        folder.mkdir()
    monkeypatch.setenv(BENCH_ROOT_ENV, str(env))
    assert completion_bench_root({"bench_root": str(cfg)}, cli) == cli
    assert completion_bench_root({"bench_root": str(cfg)}, None) == cfg
    monkeypatch.delenv(BENCH_ROOT_ENV, raising=False)
    # when only the env is set
    monkeypatch.setenv(BENCH_ROOT_ENV, str(env))
    assert completion_bench_root(None, None) == env

    seen: list = []

    def resolve(raw: str) -> object:
        return {"name": raw, "raw": raw}

    def default_outdir(spec: object, root=None) -> Path:
        seen.append(root)
        raw = spec["raw"] if isinstance(spec, dict) else "x"
        return _write_built(cli / "data" / "bench" / raw / "assembler" / "props")

    _patch_metametro(monkeypatch, resolve=resolve, default_outdir=default_outdir)
    monkeypatch.setenv(BENCH_ROOT_ENV, str(env))
    paths = completion_paths({"completion": dict(_NAMES), "bench_root": str(cfg)}, bench_root=cli)
    assert seen and all(Path(root) == cli for root in seen)
    assert paths["phage_x10"].is_dir()


def test_main_passes_bench_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main(['--bench-root', path]) must forward that path to completion_paths (B7)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
    import run_registry
    from test_registry_missing import _contract_only

    seen: dict = {}

    def fake_paths(config, *, bench_root=None):
        seen["bench_root"] = bench_root
        return {"heldout_genera": _contract_only(tmp_path / "unbuilt")}

    monkeypatch.setattr(run_registry, "completion_paths", fake_paths)
    monkeypatch.setattr(run_registry, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_registry, "load_config", lambda: {"seeds": [0], "max_epochs": 1, "patience": 1, "lr": 0.05})
    monkeypatch.setattr(run_registry, "_chart", lambda rows: None)
    chosen = tmp_path / "explicit-root"
    chosen.mkdir()
    run_registry.main(["--bench-root", str(chosen)])
    assert seen["bench_root"] == chosen
