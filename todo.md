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
- [x] MetaMetro v0.24.0 pin (version 0.7.19)
- [x] Geometric gold from ``ground_truth/`` via ``load_bench_states`` / ``load_bench_gold``; no ``colors.tsv`` fallback; ``n_sample_colors`` counts only ``sample``; DS analyze writes ``provenance.txt`` through ``run_record`` (version 0.8.0)
- [x] Registry missing rows only when the bench manifest status is not ``built``; a built bench with a lost file raises (version 0.8.1)
- [x] Registry per-row provenance, no sidecar overwrite on a resume with no new ok rows, explicit ``--bench-root`` / ``PARAGVAE_BENCH_ROOT``, and a truthful historical attribution of the 22 phage rows (version 0.8.2)
- [x] Dirty computed once on code paths (``unknown`` on git error); historical row provenance marked reconstructed; missing rows only for contract-only benches; DS numbers labelled pre-0.23 (version 0.8.3)
- [x] ``benchmark/registry/results.csv``: ``phage_x10`` recomputed on the rebuilt bench gold (version 0.7.13).
- [ ] heldout_genera, low75half, half100half recompute — #1197
- [x] Registry missing-row policy and F1 wording for unlabelled nodes (version 0.7.16). A built bench may no longer be recorded as missing; only a clearly contract-only bench may (versions 0.8.1, 0.8.3).
- [x] Provenance records the installed MetaMetro version and the pin separately (version 0.7.14)
- [x] Provenance reads `metametro.__version__` unmocked and `run_registry` writes `benchmark/registry/provenance.txt` (version 0.7.17)
- [x] Colouring arm: ``kmeans_raw`` plus warning for graphs without a bench; a bench without composition fails (version 0.7.15)
- [x] Colouring arm: ``coloring_source=none`` for uncoloured rows; no warning on the standard ``kmeans_raw`` path; colourings rule matches the code (version 0.7.18)
- [ ] Recompute ``research/*/results.csv`` so rows include ``coloring_source`` (#1201; needs the VAEGbin bundle)
- [x] Edge line-graph flip (train and infer)
- [x] Decaying label leakage from an observed-node mask
- [x] SSL plan plus supervised class-weighted GCN
- [x] Coloured-graph node and multi-label edge heads (version 0.3.0)
- [x] Composition colours from MetaMetro k-means; ToCUMG colour selection
- [x] GraphSAGE, GAT, 3-layer GCN, GCN-Transformer, joint VAE-GCN, biological loss, VAMB and HDBSCAN (version 0.4.0)
