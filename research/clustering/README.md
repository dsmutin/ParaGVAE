# Clustering

Numbers in this folder predate the contig-F1 matching fix and are not valid until the rerun.

The same early-stopped embedding, then k-means or average-linkage agglomerative clustering. Both use `k` equal to the number of ground-truth genomes. VAMB is not in this check.

K-means keeps a higher ARI on ONT 1B (0.275 vs 0.065) and on Illumina (0.081 vs 0.010). Contig F1 on Illumina is higher for agglomerative clustering (0.219 vs 0.174).

Numbers and the chart: `benchmark/clustering/`.
