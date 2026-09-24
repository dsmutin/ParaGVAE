# How we score

Contig F1 and ARI come from `paragvae.score`. AMBER `f1_score_seq` is a separate stage, `paragvae.amber`, on the same bins. VAMB is not in this repository.

| Score | What it measures |
|-------|------------------|
| Contig F1 | Each predicted cluster is matched to the ground-truth genome with the largest overlap, largest clusters first, and each genome is used once. Precision and recall of that overlap, then the harmonic mean. |
| ARI | Adjusted Rand index from scikit-learn. It does not depend on label ids. |

`k` is the number of ground-truth genomes with a non-negative label, and it is at least 2. Every arm of a hypothesis uses that same `k`.

Contig F1 does not depend on cluster ids. A high contig F1 with a low ARI means the overlap can look acceptable while the partition still disagrees with the genomes.

## AMBER

After clustering, the stage writes a CAMI binning file and runs `amber.py` (package `cami-amber`). Columns on each results row:

| Column | AMBER field |
|--------|-------------|
| `amber_ap` | `precision_avg_seq` |
| `amber_ar` | `recall_avg_seq` |
| `amber_f1` | `f1_score_seq` |

Gold sequences and lengths come from `configs/datasets.yaml`. A missing sequence or a non-positive length stops the run. The bubble fixture uses unitig sequence lengths from MetaMetro. The chart is `benchmark/<hypothesis>/amber_f1.html`.

## What is not claimed

The May 2026 archive mixed internal contig F1, AMBER, and taxonomy macro-F1. Absolute numbers here will not match that archive. The grid repeats the direction of each question under this score.

Genome labels do not enter the loss. Marker losses use `all_different.npy` (single-copy marker pairs). The bubble has no marker pairs, so `standard`, `diff_c`, and `proxy` match there.
