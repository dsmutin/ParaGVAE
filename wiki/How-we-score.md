# How we score

Two numbers, both from `paragvae.score`. Neither is AMBER `f1_score_seq`. VAMB is not in this repository.

| Score | What it measures |
|-------|------------------|
| Contig F1 | Each predicted cluster is matched to the ground-truth genome with the largest overlap, largest clusters first, and each genome is used once. Precision and recall of that overlap, then the harmonic mean. |
| ARI | Adjusted Rand index from scikit-learn. It does not depend on label ids. |

`k` is the number of ground-truth genomes with a non-negative label, and it is at least 2. Every arm of a hypothesis uses that same `k`.

Contig F1 does not depend on cluster ids. A high contig F1 with a low ARI means the overlap can look acceptable while the partition still disagrees with the genomes.

## What is not claimed

The May 2026 archive mixed internal contig F1, AMBER, and taxonomy macro-F1. Absolute numbers here will not match that archive. The grid repeats the direction of each question under this score.

Genome labels do not enter the loss. Marker losses use `all_different.npy` (single-copy marker pairs). The bubble has no marker pairs, so `standard`, `diff_c`, and `proxy` match there.
