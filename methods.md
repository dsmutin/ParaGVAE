# Methods compared

Contig F1 matches clusters to genomes by overlap. ARI is computed only on labels at least 0. Contig F1 excludes those unlabelled nodes from recall, but counts them in the precision denominator. Neither number is AMBER `f1_score_seq`. K-means uses the number of those labels as `k`. VAMB and HDBSCAN do not. Training stops when held-out edge loss stalls, with at most 60 epochs, patience 8, and learning rate 0.05.

## Completion run

`research/run_registry.py` trains each arm on `phage_x10`, `heldout_genera`, `low75half`, and `half100half`. The table is the mean contig F1 of seeds 0 and 1. Rows are in `benchmark/registry/results.csv`. The chart is `benchmark/registry/f1.html`.

| Arm | What it changes | phage_x10 | heldout_genera | low75half | half100half |
|---|---|---:|---:|---:|---:|
| gcn / kmeans | 2-layer GCN, edge BCE | 0.292 | — | — | — |
| gcn / vamb | same embedding, iterative medoid | 0.021 | — | — | — |
| gcn / hdbscan | same embedding, HDBSCAN, noise assigned to the nearest mean | 0.240 | — | — | — |
| sage | GraphSAGE blocks, then a dense map | 0.452 | — | — | — |
| gat | one-head neighbour attention, then a dense map | 0.504 | — | — | — |
| gcn_layers3 | GCN with 3 layers | 0.451 | — | — | — |
| transformer | global attention, residual, one GCN block | 0.406 | — | — | — |
| proxy | marker weight 2 on top of edge BCE | 0.292 | — | — | — |
| contrastive | InfoNCE against 8 negatives | 0.408 | — | — | — |
| biological | edge BCE plus Kraken taxonomy consistency and abundance R². Kraken is a prediction, so this arm is not a valid use of the pre-generation taxid and is not trained anymore | — | — | — | — |
| joint | linear VAE whose sample is the GCN input; embedding is the concatenation | 0.378 | — | — | — |
| edge_flip | line graph, then the 2-layer GCN, pooled back to nodes | 0.455 | — | — | — |

The table was recomputed in version 0.7.13 against MetaMetro v0.22.1. `phage_x10` is the `phage_species_5_x10` build (identity `78121c1b…564c`). Its gold is 5 phage species from `load_bench_gold_cgt`; 80 of 983 unitigs have no gold. ARI excludes them; contig F1 excludes them from recall but counts them in the precision denominator. Before 0.7.13 those 80 unitigs were a scored class and the input was a stored CGT, so the old phage numbers are not comparable. `heldout_genera`, `low75half`, and `half100half` are not built on this box (no `cdbg/` or assembly report; #1197), so every row is `status=missing` with that reason; no score is carried over. The biological arm is `status=missing` everywhere.

The completion trainer does not read colours. Assembly node features are length, GC, and out-degree. The phage tensor features are the bench k-mer composition and coverage (137 columns). Edge weights are the first edge-feature column when that matrix has a column; the phage matrix has width 0, so those edges are unweighted. Gold taxon on a MetaMetro bench is read with `load_bench_gold_cgt`, not from a colour layer. Held-out assembly labels still come from the accession table and PAF. Colour palettes are not a GCN input. The scored id for the three assemblies is `tax_id` on the `sim` row of `accessions.tsv`. A contig gets that id when the alignment target is the simulated accession. Phage labels are the species gold of the bench.

These graphs have no single-copy marker pairs. `proxy` is then the same loss as `gcn`, and the F1 values match. In the earlier assembly builds `half100half` stored no edges and `heldout_genera` 24 edges on 6439 nodes, so scores there would not be evidence of message passing.

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
