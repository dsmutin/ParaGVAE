# Methods compared

Contig F1 matches clusters to genomes by overlap. ARI is computed on labels at least 0. Neither number is AMBER `f1_score_seq`. K-means uses the number of those labels as `k`. VAMB and HDBSCAN do not. Training stops when held-out edge loss stalls, with at most 60 epochs, patience 8, and learning rate 0.05.

## Completion run

`research/run_registry.py` trains each arm on `phage_x10`, `heldout_genera`, `low75half`, and `half100half`. The table is the mean contig F1 of seeds 0 and 1. Rows are in `benchmark/registry/results.csv`. The chart is `benchmark/registry/f1.html`.

| Arm | What it changes | phage_x10 | heldout_genera | low75half | half100half |
|---|---|---:|---:|---:|---:|
| gcn / kmeans | 2-layer GCN, edge BCE | 0.523 | 0.172 | 0.290 | 0.208 |
| gcn / vamb | same embedding, iterative medoid | 0.523 | 0.382 | 0.639 | 0.769 |
| gcn / hdbscan | same embedding, HDBSCAN, noise assigned to the nearest mean | 0.387 | 0.019 | 0.031 | 0.038 |
| sage | GraphSAGE blocks, then a dense map | 0.518 | 0.236 | 0.506 | 0.205 |
| gat | one-head neighbour attention, then a dense map | 0.527 | 0.206 | 0.369 | 0.191 |
| gcn_layers3 | GCN with 3 layers | 0.492 | 0.178 | 0.310 | 0.228 |
| transformer | global attention, residual, one GCN block | 0.516 | 0.280 | 0.279 | 0.252 |
| proxy | marker weight 2 on top of edge BCE | 0.523 | 0.172 | 0.290 | 0.208 |
| contrastive | InfoNCE against 8 negatives | 0.516 | 0.145 | 0.209 | 0.208 |
| biological | edge BCE plus Kraken taxonomy consistency and abundance R² | — | 0.146 | 0.319 | 0.149 |
| joint | linear VAE whose sample is the GCN input; embedding is the concatenation | 0.570 | 0.193 | 0.319 | 0.199 |
| edge_flip | line graph, then the 2-layer GCN, pooled back to nodes | 0.411 | 0.429 | 0.479 | — |

`phage_x10` has no Kraken call table, so the biological arm is `status=missing`. `half100half` stores 0 edges, so the line graph has no nodes and `edge_flip` is `status=missing`.

These four graphs have no single-copy marker pairs. `proxy` is then the same loss as `gcn`, and the F1 values match. `half100half` also has no edges, so every message-passing arm there is a function of the node features alone. The VAMB score of 0.769 on that graph is the medoid clustering of that feature embedding, not a message that crossed an edge. `heldout_genera` has 24 stored edges on 6439 nodes, so a high `edge_flip` F1 there is not evidence of long-range message passing.

The May graph transformer used 4 attention heads. This run uses 1 head. The joint model is a linear encoder and decoder plus a 2-layer GCN, not the TensorFlow VAE from `AIRI_may/scripts/end2end`.

## One-factor grid already in this repository

`research/run_hypothesis.py` compares one change at a time on the MetaMetro bubble and the four VAEGbin bundles (Strong100, ONT 100M, ONT 1B, Illumina). The measured tables are `docs/findings.md` and `benchmark/<hypothesis>/summary.csv`.

| Hypothesis | Arms |
|---|---|
| feature_source | frozen VAE latent versus raw k-mer and depth. This does not train the VAE. On ONT and Illumina the two matrices are the same. |
| loss | `standard` edge BCE, `diff_c` (marker weight 1), `proxy` (marker weight 2), `contrastive` (InfoNCE, 8 negatives) |
| coloring | no colours versus the CGT `uint8` colours |
| graph_type | assembly graph, kNN on the VAE latent, kNN on k-mers. The k-mer kNN is skipped when that matrix equals the VAE matrix. |
| multiscale | latent only versus latent plus degree and clustering coefficient |
| clustering | k-means versus average-linkage agglomerative clustering |

`paragvae.leakage`, `paragvae.ssl`, and `paragvae.heads` are in the package. They are not rows of this completion table. Leakage adds decayed labels from an observed-node mask. SSL plans samovar, MEGAHIT, and Kraken2 and then trains a class-weighted supervised GCN. Heads predict node classes, multi-label edge classes, or both.

## May 2026 archive, not rerun here

The archive is `/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/AIRI_may`. Its internal F1, AMBER `f1_score_seq`, and taxonomic macro-F1 are different quantities. Some early AMBER rows were later marked invalid in that archive. This repository does not reproduce those AMBER numbers.

| Archive comparison | What was compared | Where |
|---|---|---|
| GCN, GraphSAGE, GAT | `gnn_type` on Strong100 | `configs/finetune/runs/arch_sage.json`, `arch_gat.json` |
| Depth | 1 layer versus 3 | `configs/finetune/runs/layers_1.json`, `layers_3.json` |
| Joint VAE+GCN | frozen pipeline versus joint fine-tune | `configs/end2end/runs/` |
| Biological loss | edge loss versus taxonomy consistency plus abundance R² | `scripts/bioloss/` |
| Clustering | VAMB versus k-means versus HDBSCAN on a fixed embedding | `scripts/clustering/methods.py` |
| Historical DIFF-C and proxy | graph BCE plus a single-copy penalty, not the marker weights 1 and 2 used here | `scripts/bioloss/bio_losses.py` |
| GCN-Transformer and other wrappers | transformer, Graph U-Net, MixHop, GPS, JKNet, HAN, DiffPool, against `full_multiscale_500` | `configs/sota/runs/` |
| Graph construction | assembly with top-5 neighbours, coverage, k-mer, GC | archive graph configs |
| Colouring | taxon PCA and Kraken probabilities, plus `conflict_resolve` without a new training run | colored-suite configs |

Graph U-Net, MixHop, GPS, JKNet, HAN, and DiffPool are not implemented in this repository.
