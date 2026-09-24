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

## Edge line-graph flip

`paragvae.flip` is an optional variant. It is not a default benchmark arm, and `research/run_hypothesis.py` does not call it.

`edge_line_graph` drops self-loops and treats a stored reverse pair as one undirected edge. Each undirected edge becomes a node of the line graph, and two of those nodes are adjacent when the original edges share a vertex. New-node features are the mean of the endpoint features, the edge features (or a column of ones), and any node or edge colour blocks that have nonzero width. `train_edge_graph` fits the same early-stopped GCN used elsewhere (`standard` loss, latent size 16, hidden size 32) on that line graph. `infer_nodes_from_edges` maps the edge embedding back to the original nodes by averaging incident edge rows. An original node with no incident edge is a zero row.

## Label leakage (optional)

`paragvae.leakage.leaked_label_features` is an optional semi-supervised channel. It writes a one-hot only on an observed-node mask and diffuses that mass along neighbouring nodes with a decay. Unobserved nodes start at zero. `concat_leakage` stacks those channels beside existing node features.

Passing a mask that is True on the nodes you score is label leakage into the test set and is not a valid benchmark. The default benchmarks must not call this, and they do not. It is not one of the arms above.

## Data

`configs/datasets.yaml` points at the MetaMetro checkout and the VAEGbin bundles `strong100`, `samovar10_ont100m_gfa`, `samovar10_ont1b_gfa`, and `samovar10_illumina_100m`.

```bash
conda activate paragvae
python research/run_hypothesis.py feature_source
```

Seeds are 0 and 1.
