#!/usr/bin/env python3
"""Where the T-phage GCN puts a unitig in the wrong majority genome."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import analyze  # noqa: E402
from load import load_phage, training_budget  # noqa: E402


def main() -> None:
    """Train on out_degree only. Coverage, length, and GC are diagnostics from the CFA."""
    graph = load_phage()
    budget = training_budget()
    analyze(
        name="phage",
        features=graph["features"],
        labels=graph["labels"],
        label_names=graph["label_names"],
        indptr=graph["indptr"],
        indices=graph["indices"],
        edge_weight=graph["edge_weight"],
        extras={
            "coverage": (graph["coverage"], "not_in_model"),
            "length": (graph["length"], "not_in_model"),
            "gc": (graph["gc"], "not_in_model"),
            "n_sample_colors": (graph["n_sample_colors"], "not_in_model"),
        },
        out=Path(__file__).resolve().parent,
        epilogue=(
            "The Spearman rows for sample colour, GC, coverage, and degree mostly track which genome a unitig belongs to. "
            "T4 (389 unitigs, mean degree 0.19) has error rate 0.023. T5 (177 unitigs, mean degree 0.23) has error rate 1. "
            "Degree does not separate those two. T5 and T7 are wrong on every unitig. "
            "Of the 339 unitigs seen in both samples, 260 are T4 and 76 are unclassified; those are the classes the clustering gets right. "
            "Among T1, T3, T5, and T7, a second sample colour is almost absent, so the colour correlation is not an independent cause."
        ),
        protocol=(
            "MetaMetro `data/work/phage_x10`. The CGT node feature is only `out_degree`. "
            "Coverage is the coloured-CFA node column. Length and GC are computed from `nodes.fna`. "
            "Sample colours are the two sample columns and are not concatenated into the model. "
            "Label 0 is the unclassified or mixed class recorded in MetaMetro `docs/baseline-run.md`; "
            "the other labels are T1, T3, T4, T5, and T7. "
            f"Standard edge loss, at most {budget['max_epochs']} epochs, patience {budget['patience']}, "
            f"learning rate {budget['lr']}. Seeds 0 and 1."
        ),
        **budget,
    )


if __name__ == "__main__":
    main()
