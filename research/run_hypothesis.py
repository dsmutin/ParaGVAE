#!/usr/bin/env python3
"""Run one hypothesis folder: ``python research/run_hypothesis.py feature_source``."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paragvae.cache import save_study  # noqa: E402
from paragvae.plots import save_charts  # noqa: E402
from paragvae.study import arm_is_redundant, run_arm  # noqa: E402
from paragvae.suite import arms_for, load_config, load_graphs  # noqa: E402


def _git_head(folder: Path) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "not a git checkout"
    return done.stdout.strip()


def write_provenance(dest: Path, hypothesis: str, config: dict) -> None:
    """Record the command, package versions, and checkouts next to a benchmark."""
    import numpy
    import sklearn

    compiler_bin = os.environ.get("CXX") or "g++"
    compiler = subprocess.run([compiler_bin, "--version"], capture_output=True, text=True)
    compiler_line = compiler.stdout.splitlines()[0] if compiler.returncode == 0 and compiler.stdout else "g++ not found"
    from paragvae.metametro_path import REQUIRED_METAMETRO, ensure_metametro, metametro_root

    ensure_metametro()
    lines = [
        f"hypothesis: {hypothesis}",
        f"command: python research/run_hypothesis.py {hypothesis}",
        f"python: {platform.python_version()}",
        f"numpy: {numpy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"compiler: {compiler_line}",
        f"seeds: {config['seeds']}",
        f"max_epochs: {config['max_epochs']}",
        f"patience: {config['patience']}",
        f"lr: {config['lr']}",
        f"metametro_version: {REQUIRED_METAMETRO}",
        f"metametro_commit: {_git_head(metametro_root())}",
        f"paragvae_commit: {_git_head(ROOT)}",
        f"vaegbin_data: {config['vaegbin_data']}",
    ]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(hypothesis: str, graphs=None, config=None) -> Path:
    """Write ``research/<hypothesis>/results.csv`` and return that path."""
    config = config or load_config()
    if graphs is None:
        graphs = load_graphs(config)
        for graph in graphs:
            save_study(graph)
    arms = arms_for(hypothesis, int(config["max_epochs"]), int(config["patience"]), float(config["lr"]))
    rows = []
    bench = ROOT / "benchmark" / hypothesis
    skipped = []
    for graph in graphs:
        for arm in arms:
            if arm_is_redundant(graph, arm):
                skipped.append({"dataset": graph.name, "arm": arm.arm, "reason": "same features as knn_vae"})
                print(f"{graph.name} {arm.arm} skipped: same features as knn_vae", flush=True)
                continue
            for seed in config["seeds"]:
                job = bench / "jobs" / f"{graph.name}_{arm.arm}_{seed}"
                row = run_arm(graph, arm, int(seed), work=job)
                rows.append(row)
                print(
                    f"{row['dataset']} {row['arm']} seed={row['seed']} "
                    f"epochs={row['epochs_ran']} f1={row['f1']} ari={row['ari']} amber_f1={row['amber_f1']}",
                    flush=True,
                )
    out = ROOT / "research" / hypothesis
    out.mkdir(parents=True, exist_ok=True)
    bench.mkdir(parents=True, exist_ok=True)
    if skipped:
        with (bench / "skipped.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "arm", "reason"])
            writer.writeheader()
            writer.writerows(skipped)
    if not rows:
        raise RuntimeError(f"no arms ran for {hypothesis}")
    dest = out / "results.csv"
    fields = list(rows[0])
    for path in (dest, bench / "results.csv"):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    save_charts(rows, bench)
    write_provenance(bench, hypothesis, config)
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_hypothesis.py <hypothesis>")
    print(main(sys.argv[1]))
