# MetaMetro data structures

ParaGVAE does not invent a second graph schema.

| Brief name | MetaMetro object | Role |
|---|---|---|
| Colouring step (called TCA in the project brief) | CFA colour dictionary, then `Cgt.node_colors` / `Cgt.edge_colors` (`uint8`) | Discrete colours only |
| Training tensor | CGT (`indptr`, `indices`, `node_features`, `edge_features`) | The GCN reads this and nothing else |

Genome labels sit on `Cgt.node_labels` for scoring. They are not features, colours, or loss targets.

## Schema gap (no code change)

CGT schema 1.0 stores colours as a `uint8` presence matrix, not as probabilities. Kraken probability vectors from the May 2026 runs do not fit that column. This reproduction therefore uses:

- MetaMetro bubble: sample colours already written by `cdbg_to_cgt`
- VAEGbin bundles: an 8-way k-means of k-mer composition, stored only in the colouring arm as extra channels

No MetaMetro source file was edited. A later schema can add an optional `float32` colour channel if probabilistic colours need to live beside the `uint8` mask.

## Loader

`paragvae.graphs.cgt_from_csr` builds a CGT from a VAEGbin CSR bundle and calls `validate_cgt`. Assembly topology stays sparse. The training code never allocates a dense `N×N` parameter matrix; the normalized operator is a SciPy CSR matrix. Assembly runs use `edge_features` as the edge weight. A weight vector whose length is not the CSR nnz is an error.
