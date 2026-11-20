"""Write a machine-generated provenance sidecar for a benchmark run."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

from paragvae.metametro_path import REQUIRED_METAMETRO, metametro_root

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

GENERATED_HEADER = "generated automatically; do not edit"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ROW_PROVENANCE_FIELDS = (
    "dataset",
    "seed",
    "arm",
    "clustering",
    "paragvae_commit",
    "dirty",
    "metametro_version",
    "metametro_commit",
    "timestamp",
    "source",
)

# Paths that count as code when recording dirty. Run outputs under
# benchmark/ and examples/ are excluded so writing results cannot set the flag.
CODE_DIRTY_PATHS = (
    "src",
    "research",
    "configs",
    "scripts",
    "tests",
    "pyproject.toml",
    "VERSION",
)


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


def recorded_git_commit(value: str, *, what: str) -> str:
    """Accept a 40-character lowercase hex digest, or ``unknown``."""
    text = str(value)
    if text == "unknown" or COMMIT_RE.fullmatch(text):
        return text
    raise ValueError(f"{what} must be a 40-character lowercase hex digest or 'unknown', not {value!r}")


def working_tree_dirty(folder: Path, *, paths: tuple[str, ...] = CODE_DIRTY_PATHS) -> str:
    """Return ``true``, ``false``, or ``unknown`` for uncommitted code changes.

    Only ``paths`` are inspected (code, not ``benchmark/`` or ``examples/``
    outputs). A git error is recorded as ``unknown``, not as clean.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "status", "--porcelain", "--", *paths],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if done.returncode != 0:
        return "unknown"
    return "true" if _porcelain_dirty(done) else "false"


def _porcelain_dirty(done: subprocess.CompletedProcess) -> bool:
    return bool(done.stdout.strip())


def row_provenance_record(
    dataset: str,
    seed: object,
    arm: str,
    clustering: str,
    *,
    repo_root: Path | None = None,
    dirty: str | None = None,
) -> dict[str, str]:
    """Return one per-row provenance record for the current tree.

    ``dirty`` should be computed once at the start of the run. If it is
    omitted, the code paths of ``repo_root`` are inspected now.
    """
    root = repo_root or _PACKAGE_ROOT
    if dirty is None:
        dirty = working_tree_dirty(root)
    if dirty not in {"true", "false", "unknown"}:
        raise ValueError(f"dirty must be true, false, or unknown, not {dirty!r}")
    return {
        "dataset": str(dataset),
        "seed": str(int(seed)),
        "arm": str(arm),
        "clustering": str(clustering),
        "paragvae_commit": recorded_git_commit(git_head(root) or "unknown", what="paragvae_commit"),
        "dirty": dirty,
        "metametro_version": installed_metametro_version(),
        "metametro_commit": recorded_git_commit(installed_metametro_commit(), what="metametro_commit"),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "recorded",
    }


def merge_row_provenance(path: Path, new_rows: list[dict]) -> None:
    """Update ``path`` with ``new_rows``, leaving other rows unchanged."""
    path = Path(path)
    existing: dict[tuple[str, str, str, str], dict] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], row["seed"], row["arm"], row["clustering"])
                existing[key] = {field: row.get(field, "") for field in ROW_PROVENANCE_FIELDS}
    for row in new_rows:
        record = {field: str(row.get(field, "")) for field in ROW_PROVENANCE_FIELDS}
        key = (record["dataset"], record["seed"], record["arm"], record["clustering"])
        existing[key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_PROVENANCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow(existing[key])


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
    dirty: str | None = None,
) -> None:
    """Write ``provenance.txt`` under ``dest`` for one run.

    ``metametro_version`` is the installed package, from
    ``installed_metametro_version``. ``metametro_pin`` is the pin. A commit
    that cannot be determined is recorded as ``unknown``. ``dirty`` should be
    computed once at the start of the run over code paths only.
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
    if dirty is None:
        dirty = working_tree_dirty(repo_root or _PACKAGE_ROOT)
    if dirty not in {"true", "false", "unknown"}:
        raise ValueError(f"dirty must be true, false, or unknown, not {dirty!r}")
    root = repo_root or _PACKAGE_ROOT
    lines = [GENERATED_HEADER]
    lines.extend(header or [])
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
            f"metametro_commit: {recorded_git_commit(installed_metametro_commit(), what='metametro_commit')}",
            f"paragvae_commit: {recorded_git_commit(git_head(root) or 'unknown', what='paragvae_commit')}",
            f"dirty: {dirty}",
        ]
    )
    lines.extend(footer or [])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
