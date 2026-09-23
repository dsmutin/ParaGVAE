# Graph type

Assembly CSR versus a 5-nearest-neighbour graph. `knn_kmer` is omitted when the k-mer matrix equals the VAE matrix; see `benchmark/graph_type/skipped.csv`.

Strong100 assembly contig F1 is 0.312, above both kNN arms. ONT 100M has 23 undirected edges besides self-loops and 353 isolated nodes, so its assembly-versus-kNN gap (0.320 vs 0.333) is not a test of a connected assembly. ONT 1B kNN is 0.381 vs assembly 0.354. Illumina kNN is 0.228 vs assembly 0.165, with lower ARI.

Numbers and the chart: `benchmark/graph_type/`.
