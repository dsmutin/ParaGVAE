# Hypotheses

Each folder changes one factor. Seeds are 0 and 1. Charts and tables are under `benchmark/<hypothesis>/`.

| Folder | Question | Arms |
|--------|----------|------|
| `feature_source` | Frozen 32-d VAE latent versus raw k-mer and depth. This is not end-to-end VAE+GCN training | `vae_latent`, `raw_features` |
| `loss` | Unsupervised objective | `standard` (edge BCE), `diff_c` (marker weight 1), `proxy` (marker weight 2), `contrastive` (InfoNCE, 8 negatives) |
| `coloring` | Concatenate CGT colour channels | `uncoloured`, `cgt_colors` |
| `graph_type` | Assembly edges versus 5-NN. `knn_kmer` is skipped when it would copy `knn_vae` | `assembly`, `knn_vae`, `knn_kmer` |
| `multiscale` | Degree and local clustering coefficient. Self-loops are not neighbours | `latent_only`, `degree_clustering` |
| `clustering` | Same embedding, two clusterers | `kmeans`, `agglomerative` (average linkage) |

`knn_kmer` is skipped for the bubble, both ONT graphs, and Illumina. The skip list is `benchmark/graph_type/skipped.csv`. Strong100 keeps both kNN arms.

The numerical comparison is [Findings](Findings).
