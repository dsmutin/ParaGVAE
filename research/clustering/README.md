# Clustering

The same early-stopped embedding, then k-means or average-linkage agglomerative clustering. Both use `k` equal to the number of ground-truth genomes. VAMB is not in this check.

Agglomerative clustering is higher on Strong100 (contig F1 0.388, ARI 0.365 vs 0.312 and 0.293). K-means is higher on ONT 100M. ONT 1B contig F1 is tied and k-means keeps the higher ARI (0.130 vs 0.042). Illumina agglomerative F1 is 0.215 vs 0.165, with ARI 0.012 vs 0.061.

Numbers and the chart: `benchmark/clustering/`.
