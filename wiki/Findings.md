# Findings

The numeric comparison lives in the code repository at `docs/findings.md`, with charts under `benchmark/`. These numbers are from the rerun after the scoring and split fixes. The score is contig F1 plus ARI, not AMBER, and VAMB was not rerun.

- On Strong100 the raw matrix and the frozen latent are tied on contig F1 (0.314 vs 0.312). Other datasets use the same matrix for both arms.
- No loss wins on every dataset. InfoNCE is the highest arm on Illumina and not on the ONT graphs.
- Colour channels leave Strong100 unchanged, lower ONT 100M, and raise contig F1 on ONT 1B and Illumina without raising ARI.
- ONT 100M has 23 undirected assembly edges and 353 isolated nodes. Its kNN comparison is not against a connected assembly graph.
- Degree and clustering coefficient help ONT 1B and lower Strong100 and ONT 100M. Self-loops are excluded.
- Agglomerative clustering beats k-means on Strong100. K-means keeps higher ARI on the ONT graphs.
