# Hypothesis checks on CGT tensors

Score: contig F1 and ARI after k-means, unless the clustering folder says otherwise. `k` is the number of ground-truth genomes. This is not AMBER `f1_score_seq`, and VAMB is not rerun. Means are over seeds 0 and 1. Contig F1 matches a genome to the cluster with the largest overlap, so the score does not depend on cluster ids.

The GCN is trained on undirected edges. One fifth are held out and removed from the propagation operator. Learning rate 0.05, at most 60 epochs, patience 8. Illumina fits hit that cap, so those rows are a shared budget. Provenance for each folder is `benchmark/<hypothesis>/provenance.txt`.

Charts: `benchmark/<hypothesis>/f1.html`.

## Feature source

Frozen VAE latent versus raw k-mer and depth. This is not end-to-end VAE+GCN training. Only Strong100 has two different matrices. There the contig F1 values are 0.312 (latent) and 0.314 (raw). The gap is smaller than the seed scatter. The latent arm has the higher ARI (0.293 vs 0.265) and stopped near epoch 41, while the raw arm hit the 60-epoch cap. On ONT 100M, ONT 1B, Illumina, and the bubble the two matrices are identical, so the arms match.

## Loss

`standard` is edge BCE. `diff_c` and `proxy` add the single-copy marker push with weights 1 and 2. `contrastive` is InfoNCE of each training edge against 8 random nodes. The bubble has no marker pairs, so the first three losses match (contig F1 0.750).

No loss leads on every dataset:

| Dataset | Highest contig F1 | Notes |
|---|---|---|
| Strong100 | diff_c 0.328 | standard 0.312, contrastive 0.302 |
| ONT 100M | standard 0.320 | seed scatter ±0.077; proxy 0.202 is lower |
| ONT 1B | diff_c 0.433 | standard 0.354, proxy 0.287 |
| Illumina | contrastive 0.199 | all four arms are at or near the epoch cap; ARI stays near 0.06 |

## Colouring

Discrete colours (CFA sample colours on the bubble; 8-way k-mer composition elsewhere) are concatenated only in the colour arm.

The effect is not the same on every graph. Strong100 stays at 0.313 vs 0.312. ONT 100M falls from 0.320 to 0.202. ONT 1B rises from 0.354 to 0.421 and Illumina from 0.165 to 0.199, while ARI does not rise on either. The bubble is unchanged at 0.750.

## Graph type

`knn_kmer` is skipped when it would copy `knn_vae`. That happens for the bubble, both ONT graphs, and Illumina (`benchmark/graph_type/skipped.csv`). Strong100 keeps both kNN arms; assembly contig F1 is 0.312, k-mer kNN 0.298, latent kNN 0.290.

ONT 100M's assembly graph has 23 undirected edges besides self-loops, and 353 of 383 nodes are isolated. Its kNN contig F1 is 0.333 vs assembly 0.320. That is not a comparison against a connected assembly graph. ONT 1B kNN is 0.381 vs assembly 0.354. Illumina kNN is 0.228 vs assembly 0.165, with a lower ARI (0.037 vs 0.061).

## Multiscale

Degree and local clustering coefficient, with self-loops ignored. They lower Strong100 (0.281 vs 0.312) and ONT 100M (0.248 vs 0.320). They raise ONT 1B (0.469 vs 0.354). Illumina moves from 0.165 to 0.188 and stays at the epoch cap. The bubble colour of this arm is unstable across the two seeds (mean 0.750, standard deviation 0.250).

## Clustering

Average-linkage agglomerative clustering uses the same embedding and the same `k`. VAMB is not in this check.

On Strong100, agglomerative contig F1 is 0.388 and ARI 0.365, against k-means 0.312 and 0.293. On ONT 100M, k-means is higher on both (0.320 / 0.139 vs 0.228 / 0.036). ONT 1B contig F1 is tied near 0.35, and k-means ARI is higher (0.130 vs 0.042). Illumina agglomerative F1 is 0.215 vs 0.165, with ARI 0.012 vs 0.061.
