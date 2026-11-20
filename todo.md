# paragvae features

Scaffold checklist. Check a box only after mandatory tests pass.

- [x] Baseline CLI and pipeline (`status=baseline`)
- [x] CGT trainer, hypothesis benchmarks, version 0.1.0 (baseline CLI kept)
- [x] AMBER ``f1_score_seq`` stage on binning output (version 0.2.0)
- [x] Optional AMBER on the hypothesis grid (``amber`` flag, version 0.7.3)
- [x] Shared Adam, z-score, and init in ``paragvae.optim`` (version 0.7.4)
- [x] ``completion_paths`` only treats a failed ``resolve`` as a filesystem path (version 0.7.5)
- [x] Shared ``score.evaluate`` path; removed ``vaegcn_*`` and ``flip.cluster_embeddings`` (version 0.7.6)
- [x] MetaMetro v0.20.1 pin; bench gold from ``load_bench_gold_cgt``; composition-only colouring arm (version 0.7.7)
- [x] MetaMetro v0.21.0 pin (version 0.7.8)
- [x] MetaMetro v0.22.0 pin (version 0.7.9)
- [x] MetaMetro pin has one source, ``PIN`` in ``scripts/ci_install_metametro.sh``; consistency test (version 0.7.10)
- [x] Colouring arm ignores foreign namespaces (kraken2, decaying); k-means fallback warns and sets ``coloring_source`` (version 0.7.11)
- [x] MetaMetro v0.22.1 pin (version 0.7.12)
- [x] ``benchmark/registry/results.csv`` recomputed: ``phage_x10`` on the rebuilt bench gold; ``heldout_genera``, ``low75half``, ``half100half`` are ``status=missing`` until their graphs are built (version 0.7.13)
- [x] Provenance records the installed MetaMetro version and the pin separately (version 0.7.14)
- [x] Edge line-graph flip (train and infer)
- [x] Decaying label leakage from an observed-node mask
- [x] SSL plan plus supervised class-weighted GCN
- [x] Coloured-graph node and multi-label edge heads (version 0.3.0)
- [x] Composition colours from MetaMetro k-means; ToCUMG colour selection
- [x] GraphSAGE, GAT, 3-layer GCN, GCN-Transformer, joint VAE-GCN, biological loss, VAMB and HDBSCAN (version 0.4.0)
