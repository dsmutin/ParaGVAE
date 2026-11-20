"""Write a machine-generated provenance sidecar for a benchmark run."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from paragvae.metametro_path import REQUIRED_METAMETRO, metametro_root

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def git_head(folder: Path) -> str | None:
    """Return the commit at ``folder``, or ``None`` if it is not a git checkout."""
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    head = done.stdout.strip()
    return head or None


def installed_metametro_version() -> str:
    """Return the installed MetaMetro package version (``metametro.__version__``)."""
    from paragvae import metametro_path

    return metametro_path.ensure_metametro()


def installed_metametro_commit() -> str:
    """Return the MetaMetro commit that provided the install, or ``unknown``.

    The commit is read from the wheel's ``direct_url.json`` ``vcs_info`` when
    present, otherwise from ``git rev-parse`` on an editable checkout or on the
    ``file://`` path in ``direct_url.json``. It is never invented.
    """
    from_metadata = _commit_from_direct_url()
    if from_metadata:
        return from_metadata
    try:
        root = metametro_root()
    except (FileNotFoundError, ImportError, OSError):
        return "unknown"
    head = git_head(root)
    return head if head else "unknown"


def _commit_from_direct_url() -> str | None:
    try:
        raw = importlib.metadata.distribution("metametro").read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, FileNotFoundError, OSError):
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    commit = (data.get("vcs_info") or {}).get("commit_id")
    if commit:
        return str(commit)
    url = str(data.get("url") or "")
    if url.startswith("file://"):
        return git_head(Path(url.removeprefix("file://")))
    return None


def write_run_provenance(
    dest: Path,
    command: str,
    config: dict,
    *,
    header: list[str] | None = None,
    footer: list[str] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Write ``provenance.txt`` under ``dest`` for one run.

    ``metametro_version`` is the installed package, from
    ``installed_metametro_version``. ``metametro_pin`` is the pin. A commit
    that cannot be determined is recorded as ``unknown``.
    """
    import numpy
    import sklearn

    compiler_bin = os.environ.get("CXX") or "g++"
    compiler = subprocess.run([compiler_bin, "--version"], capture_output=True, text=True)
    compiler_line = (
        compiler.stdout.splitlines()[0]
        if compiler.returncode == 0 and compiler.stdout
        else "g++ not found"
    )
    lines = list(header or [])
    lines.extend(
        [
            f"command: {command}",
            f"python: {platform.python_version()}",
            f"numpy: {numpy.__version__}",
            f"scikit-learn: {sklearn.__version__}",
            f"compiler: {compiler_line}",
            f"seeds: {config['seeds']}",
            f"max_epochs: {config['max_epochs']}",
            f"patience: {config['patience']}",
            f"lr: {config['lr']}",
        ]
    )
    if "amber" in config:
        from paragvae.study import amber_mode

        lines.append(f"amber: {amber_mode(config.get('amber', 'auto'))}")
    lines.extend(
        [
            f"metametro_version: {installed_metametro_version()}",
            f"metametro_pin: {REQUIRED_METAMETRO}",
            f"metametro_commit: {installed_metametro_commit()}",
            f"paragvae_commit: {git_head(repo_root or _PACKAGE_ROOT) or 'unknown'}",
        ]
    )
    lines.extend(footer or [])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
