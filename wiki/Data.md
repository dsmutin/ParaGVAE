# Data

paraGVAE does not invent a second graph schema. Paths live in `configs/datasets.yaml`. Assemblies are not stored in git.

| Brief name | MetaMetro object | Role |
|------------|------------------|------|
| Colouring step (called TCA in the project brief) | CFA colour dictionary, then `Cgt.node_colors` / `Cgt.edge_colors` (`uint8`) | Discrete colours only |
| Training tensor | CGT (`indptr`, `indices`, `node_features`, `edge_features`) | The GCN reads this |

`paragvae.graphs.cgt_from_csr` builds a CGT from a VAEGbin CSR bundle and calls `validate_cgt`. A weight vector whose length is not the CSR nnz is an error.

## Graphs in the grid

| Name | What it is |
|------|------------|
| `metametro_bubble` | MetaMetro mock CGT (sample colours already on the tensor) |
| `strong100` | Strong100 Flye bundle. The only graph whose 32-d VAE latent differs from k-mer+depth |
| `ont100m` | `samovar10_ont100m_gfa` |
| `ont1b` | `samovar10_ont1b_gfa` |
| `illumina` | `samovar10_illumina_100m` |

On the bubble, both ONT graphs, and Illumina, `vae_features.npy` and the raw matrix are the same array. A feature-source comparison is informative only on Strong100.

Cached copies, including the CGT directory, are under `intermediates/<dataset>/`.

## Schema gap

CGT schema 1.0 stores colours as a `uint8` presence matrix, not as probabilities. Kraken probability vectors from the May 2026 runs do not fit that column. No MetaMetro source file was edited.

This grid therefore uses:

- the bubble's CFA sample colours
- elsewhere, an 8-way k-means of k-mer composition, written only into the colouring arm

## ONT 100M assembly

That assembly graph has 23 undirected edges besides self-loops, and 353 of 383 nodes are isolated. An assembly-versus-kNN gap on this graph is not a test of a connected assembly.
