#!/usr/bin/env python3
"""Run one hypothesis folder: ``python research/run_hypothesis.py joint_training``."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paragvae.cache import save_study  # noqa: E402
from paragvae.plots import write_charts  # noqa: E402
from paragvae.study import run_arm  # noqa: E402
from paragvae.suite import arms_for, load_config, load_graphs  # noqa: E402


def main(hypothesis: str, graphs=None, config=None) -> Path:
    """Write ``research/<hypothesis>/results.csv`` and return that path."""
    config = config or load_config()
    if graphs is None:
        graphs = load_graphs(config)
        for graph in graphs:
            save_study(graph)
    arms = arms_for(hypothesis, int(config["max_epochs"]), int(config["patience"]))
    rows = []
    bench = ROOT / "benchmark" / hypothesis
    for graph in graphs:
        for arm in arms:
            for seed in config["seeds"]:
                job = bench / "jobs" / f"{graph.name}_{arm.arm}_{seed}"
                row = run_arm(graph, arm, int(seed), work=job)
                rows.append(row)
                print(
                    f"{row['dataset']} {row['arm']} seed={row['seed']} "
                    f"epochs={row['epochs_ran']} f1={row['f1']} ari={row['ari']}",
                    flush=True,
                )
    out = ROOT / "research" / hypothesis
    out.mkdir(parents=True, exist_ok=True)
    bench.mkdir(parents=True, exist_ok=True)
    dest = out / "results.csv"
    fields = list(rows[0])
    for path in (dest, bench / "results.csv"):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    try:
        write_charts(rows, bench)
    except Exception as error:  # noqa: BLE001 — keep the CSV if the chart spec fails
        (bench / "chart_error.txt").write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_hypothesis.py <hypothesis>")
    print(main(sys.argv[1]))
