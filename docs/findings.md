# Hypothesis checks on CGT tensors

The numbers below were scored before the contig-F1 matching fix and are not valid. They will be replaced by a rerun.



Score: contig F1 and ARI after k-means, unless the clustering folder says otherwise. `k` is the number of ground-truth genomes. This is not AMBER `f1_score_seq`, and VAMB is not rerun. Means are over seeds 0 and 1. Several Strong100 and Illumina fits hit the 60-epoch cap while validation loss was still falling, so those rows are a shared budget, not a fully stopped optimum.

Charts: `benchmark/<hypothesis>/f1.html`.

## Joint training

Only Strong100 has a 32-d VAE latent distinct from k-mer+depth. There, training on the raw matrix scored higher contig F1 than the frozen latent (0.313 vs 0.253). On ONT 100M, ONT 1B, Illumina, and the MetaMetro bubble the two matrices are the same file contents, so the arms match exactly. The May 2026 AMBER test did not find a significant gain for joint fine-tuning; this protocol is a different score and only separates the arms on Strong100.

## Loss

Marker pairs exist for the VAEGbin bundles and are absent on the bubble, so `standard`, `diff_c`, and `proxy` match there. On the real graphs no loss wins everywhere:

| Dataset | Best arm | Contig F1 |
|---|---|---|
| Strong100 | proxy 0.256, standard 0.253 | essentially tied |
| ONT 100M | proxy 0.302 | above standard 0.214 |
| ONT 1B | diff_c 0.481 | above standard 0.448 |
| Illumina | standard 0.174 | contrastive 0.141 is lower |

`contrastive` is the weak arm on the larger graphs. The May result that standard DIFF-C won an AMBER bake-off is not repeated as a large gap under this contig F1.

## Colouring

Discrete colours (CFA sample colours on the bubble; 8-way k-mer composition elsewhere, stored as `uint8` and concatenated only in the colour arm) do not raise contig F1. Strong100 0.217 vs 0.253 uncoloured. ONT 1B 0.211 vs 0.448. Illumina 0.148 vs 0.174.

## Graph type

Assembly edges beat a 5-NN feature graph on ONT 1B (0.448 vs 0.201) and on Strong100 ARI (0.301 vs about 0.26). ONT 100M is the exception: kNN contig F1 is 0.266 vs assembly 0.214. Illumina assembly is only slightly higher (0.174 vs 0.163). Where `node_features.npy` equals the k-mer matrix, `knn_vae` and `knn_kmer` are the same graph.

## Multiscale

Degree and local clustering coefficient help Illumina (0.217 vs 0.174) and ONT 1B slightly (0.469 vs 0.448). They lower Strong100 (0.213 vs 0.253). Extra structural channels are not a uniform upgrade.

## Clustering

Average-linkage agglomerative clustering does not replace k-means. ONT 1B k-means contig F1 is 0.448 vs 0.374, and ARI drops from 0.275 to 0.065. Illumina agglomerative F1 is higher (0.219 vs 0.174) while ARI falls (0.010 vs 0.081). The historical requirement for VAMB is not retested here.
