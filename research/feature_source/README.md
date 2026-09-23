# Feature source

Frozen VAE latent versus raw k-mer and depth. This is not end-to-end joint training.

Only Strong100 has two different matrices. Mean contig F1 is 0.312 for the latent and 0.314 for the raw features. The latent arm has higher ARI (0.293 vs 0.265). Other datasets store the same array in both roles, so the arms match.

Numbers and the chart: `benchmark/feature_source/`.
