"""Mandatory: every machine-readable MetaMetro pin matches PIN in the CI script."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from paragvae.metametro_path import REQUIRED_METAMETRO

pytestmark = pytest.mark.mandatory


def _pin(repo_root: Path) -> str:
    """Return PIN from scripts/ci_install_metametro.sh without the leading v."""
    text = (repo_root / "scripts" / "ci_install_metametro.sh").read_text(encoding="utf-8")
    found = re.findall(r"^PIN=v(\d+\.\d+\.\d+)$", text, flags=re.MULTILINE)
    assert len(found) == 1, found
    return found[0]


def test_required_metametro_matches_pin(repo_root: Path) -> None:
    assert REQUIRED_METAMETRO == _pin(repo_root)


def test_pyproject_dependency_matches_pin(repo_root: Path) -> None:
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = [dep for dep in data["project"]["dependencies"] if dep.startswith("metametro")]
    assert deps == [f"metametro>={_pin(repo_root)}"]


def test_environment_comment_matches_pin(repo_root: Path) -> None:
    first = (repo_root / "environment.yml").read_text(encoding="utf-8").splitlines()[0]
    assert first == f"# metametro>={_pin(repo_root)} is required and is not a conda package."
