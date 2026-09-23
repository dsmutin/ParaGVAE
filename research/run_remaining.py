#!/usr/bin/env python3
"""Finish hypotheses after feature_source and redraw that chart."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from paragvae.plots import write_charts  # noqa: E402
from paragvae.suite import HYPOTHESES, load_config, load_graphs  # noqa: E402
from run_hypothesis import main  # noqa: E402


def _redraw(name: str) -> None:
    path = ROOT / "benchmark" / name / "results.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    write_charts(rows, path.parent)


def run() -> None:
    """Reload cached graphs and run every hypothesis except the finished one."""
    _redraw("feature_source")
    print("redrew feature_source", flush=True)
    config = load_config()
    graphs = load_graphs(config)
    for name in HYPOTHESES:
        if name == "feature_source":
            continue
        print(f"=== {name} ===", flush=True)
        print(main(name, graphs=graphs, config=config), flush=True)


if __name__ == "__main__":
    run()
