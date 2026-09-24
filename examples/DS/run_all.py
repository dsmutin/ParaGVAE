#!/usr/bin/env python3
"""Run every error analysis and write the cross-graph index."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = ("strong100", "ont100m", "ont1b", "illumina", "phage", "spb_transit", "roxel")


def _index() -> None:
    rows = []
    for name in EXAMPLES:
        summary = json.loads((ROOT / name / "summary.json").read_text(encoding="utf-8"))
        rows.append(summary)
    lines = [
        "# Where the GCN is wrong",
        "",
        "Each folder trains the standard early-stopped GCN used in the benchmarks and marks a node wrong when its class is not the majority class of its cluster. Charts are Altair HTML and PNG.",
        "",
        "| Example | Nodes | Isolated | Error rate | Stable error | Contig F1 | ARI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| [{row['name']}]({row['name']}/README.md) | {row['n']} | {row['n_isolated']} | "
            f"{row['error_rate']:.3f} | {row['stable_error_rate']:.3f} | {row['f1_seed0']:.3f} | {row['ari_seed0']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Stable error counts nodes that are wrong for both seeds. The per-folder README has the degree bins, the label-boundary bins, and the Spearman table.",
            "",
            "Assembly graphs are Strong100, ONT 100M, ONT 1B, and Illumina. MetaMetro examples are the T-phage graph, the Saint Petersburg ground-transit graph, and the Roxel street graph.",
            "",
            "## What the errors line up with",
            "",
            "Class size is the strongest Spearman row on every graph except the phage graph, where sample colour is stronger and is itself a stand-in for class. Small classes are absorbed into the majority cluster. That is a property of the label frequencies and of k-means, not of degree.",
            "",
            "A label boundary (a neighbour with a different ground-truth class) raises the error rate on Strong100 (0.232 if every neighbour agrees, 0.702 if every neighbour disagrees; 396 nodes have a neighbour), on Roxel (0.370 vs 0.838), and on the Saint Petersburg stops (0.131 vs 0.641). On Illumina the boundary bins go the other way (0.794 vs 0.572), and inside the three largest genomes the rates sit between 0.46 and 0.57, so the boundary is not where that graph fails. ONT 100M has only 30 nodes with a neighbour, and the boundary correlation there does not beat a shuffle (permutation p 0.274).",
            "",
            "Degree role (isolated, tip, path, branch) does not mark a shared failure mode. Strong100 is mostly isolated (456 of 852) and those contigs are wrong 0.570 of the time, about the same as tips and paths. ONT 100M is 353 isolated nodes out of 383. ONT 1B and Illumina are connected and still wrong on about half or more of every degree bin.",
            "",
            "Depth is not the failure mode. On Strong100 it is not a model input and rho is -0.015 (permutation p 0.632). Where the GCN does see depth, rho is -0.051 on ONT 100M, -0.170 on ONT 1B, and -0.051 on Illumina. The Illumina p-value is small because there are 5140 nodes; the association is still tiny.",
            "",
            "K-mer distance to the contig's own genome mean (a diagnostic that uses the labels) has rho 0.264, 0.165, 0.379, and 0.195 on Strong100, ONT 100M, ONT 1B, and Illumina. Composition outliers of a genome are somewhat harder. On Strong100 the model never sees that k-mer matrix.",
            "",
            "The phage tensor gives the GCN only `out_degree`. T4 is almost entirely correct (error 0.023) and T5 and T7 are entirely wrong, even though T4 and T5 have nearly the same mean degree. Coverage, GC, and length correlations follow the genome, which the model cannot see. See `phage/README.md`.",
            "",
            "Saint Petersburg k-means returns bus and nothing else: the error rate 0.312 is the fraction of non-bus stops. Route colours are not the training target. Across 529 routes with at least 8 stops, the fraction of stops that share one cluster averages 0.582, against 0.350 for a size-matched shuffle of the same clusters. Routes are tighter than chance and are still split. See `spb_transit/README.md`.",
            "",
            "Roxel predicts road type from longitude and latitude. The node colour matrix is empty. Street names are edge colours and are not scored. The stored names `unclassified` and `Unclassified` are kept as two classes. Errors concentrate where a vertex meets a different road type.",
            "",
        ]
    )
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Execute each `analyze.py`, then write `examples/DS/README.md`."""
    for name in EXAMPLES:
        print(f"=== {name} ===", flush=True)
        runpy.run_path(str(ROOT / name / "analyze.py"), run_name="__main__")
    _index()


if __name__ == "__main__":
    sys.exit(main())
