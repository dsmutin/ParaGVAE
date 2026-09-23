#!/usr/bin/env python3
"""Run every hypothesis once the graphs are in memory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paragvae.cache import save_study  # noqa: E402
from paragvae.suite import HYPOTHESES, load_config, load_graphs  # noqa: E402
from run_hypothesis import main  # noqa: E402


def run() -> None:
    """Load each graph once, then execute the six hypothesis benchmarks."""
    config = load_config()
    graphs = load_graphs(config)
    for graph in graphs:
        print(f"cached {graph.name} nodes={graph.cgt.num_nodes}", flush=True)
        save_study(graph)
    for name in HYPOTHESES:
        print(f"=== {name} ===", flush=True)
        print(main(name, graphs=graphs, config=config), flush=True)


if __name__ == "__main__":
    run()
