# Reproduction protocol

Historical May 2026 runs lived in `AIRI_may` and mixed three scores (internal contig F1, AMBER `f1_score_seq`, taxonomy macro-F1). This repository repeats the **direction** of each hypothesis on MetaMetro tensors with one shared score:

- embedding from a 2-layer GCN compiled in `cpp/gcn_train.cpp`, early-stopped on held-out undirected edges. Epoch cap, patience, and learning rate come from `configs/datasets.yaml` (60, 8, and 0.05). Those edges are removed from the propagation operator. A stored reverse edge is the same edge, not a second sample. Each run writes `benchmark/<hypothesis>/provenance.txt`.
- bins from k-means unless the clustering hypothesis says otherwise
- `k` = number of ground-truth genomes (same `k` for every arm)
- contig F1 = majority-genome overlap; ARI from scikit-learn
- AMBER `f1_score_seq` is a later stage on the same bins (`amber_f1`, `amber_ap`, `amber_ar`). It does not replace contig F1

VAMB is not re-run. Contig F1 will not match `results_final_summary.md`. AMBER is the sequence-level score from `amber.py`.

## Arms

| Folder | Question | Comparison |
|---|---|---|
| `research/feature_source` | Frozen VAE latent versus raw k-mer+depth. Not end-to-end joint training | `vae_latent` vs `raw_features` |
| `research/loss` | Unsupervised objective | `standard`, `diff_c`, `proxy`, `contrastive` (InfoNCE, 8 negatives) |
| `research/coloring` | Add CGT colour channels | `uncoloured` vs `cgt_colors` |
| `research/graph_type` | Assembly edges vs feature kNN. `knn_kmer` is skipped when it duplicates `knn_vae` | `assembly`, `knn_vae`, `knn_kmer` |
| `research/multiscale` | Degree and local clustering, self-loops excluded | `latent_only` vs `degree_clustering` |
| `research/clustering` | Same embedding, two clusterers | `kmeans` vs `agglomerative` |

`diff_c` / `proxy` use `all_different.npy` (single-copy marker pairs). They do not use genome ids. The bubble fixture has no marker pairs, so those two losses match `standard` there.

## Data

`configs/datasets.yaml` points at the MetaMetro checkout and the VAEGbin bundles `strong100`, `samovar10_ont100m_gfa`, `samovar10_ont1b_gfa`, and `samovar10_illumina_100m`.

```bash
conda activate paragvae
python research/run_hypothesis.py feature_source
```

Seeds are 0 and 1.
