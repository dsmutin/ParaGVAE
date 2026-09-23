# Findings

Means over seeds 0 and 1. The score is [contig F1 and ARI](How-we-score). Charts: `benchmark/<hypothesis>/f1.html`.

## Feature source

Only Strong100 has two different matrices. Contig F1 is 0.312 for the frozen latent and 0.314 for the raw matrix. The gap is smaller than the seed scatter. The latent arm has the higher ARI (0.293 vs 0.265) and stopped near epoch 41. The raw arm hit the 60-epoch cap. On the other four graphs the matrices are identical, so the arms match.

## Loss

No loss leads on every dataset. The bubble's first three losses match at contig F1 0.750.

| Dataset | Highest contig F1 | Beside it |
|---------|-------------------|-----------|
| Strong100 | diff_c 0.328 | standard 0.312, contrastive 0.302 |
| ONT 100M | standard 0.320 | seed scatter ±0.077; proxy 0.202 is lower |
| ONT 1B | diff_c 0.433 | standard 0.354, proxy 0.287 |
| Illumina | contrastive 0.199 | all four arms are at or near the epoch cap; ARI stays near 0.06 |

## Colouring

Strong100 stays at 0.313 vs 0.312. ONT 100M falls from 0.320 to 0.202. ONT 1B rises from 0.354 to 0.421 and Illumina from 0.165 to 0.199. ARI does not rise on those two. The bubble stays at 0.750.

## Graph type

Strong100 assembly contig F1 is 0.312, k-mer kNN 0.298, latent kNN 0.290.

ONT 100M kNN is 0.333 vs assembly 0.320. That assembly is almost empty: 23 undirected edges besides self-loops, 353 of 383 nodes isolated. ONT 1B kNN is 0.381 vs assembly 0.354. Illumina kNN is 0.228 vs assembly 0.165, with a lower ARI (0.037 vs 0.061).

## Multiscale

Degree and local clustering coefficient lower Strong100 (0.281 vs 0.312) and ONT 100M (0.248 vs 0.320). They raise ONT 1B (0.469 vs 0.354). Illumina moves from 0.165 to 0.188 and stays at the epoch cap. The bubble arm is unstable across the two seeds (mean 0.750, standard deviation 0.250).

## Clustering

On Strong100, average-linkage contig F1 is 0.388 and ARI 0.365, against k-means 0.312 and 0.293. On ONT 100M, k-means is higher on both (0.320 / 0.139 vs 0.228 / 0.036). ONT 1B contig F1 is tied near 0.35, and k-means ARI is higher (0.130 vs 0.042). Illumina agglomerative F1 is 0.215 vs 0.165, with ARI 0.012 vs 0.061.
