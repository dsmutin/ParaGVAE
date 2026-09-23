# Findings

The numeric comparison lives in the code repository at `docs/findings.md`, with charts under `benchmark/`.

Short version, contig F1, two seeds, not AMBER:

- Joint training can be compared only on Strong100, where the raw matrix beat the frozen 32-d latent (0.313 vs 0.253) inside a 60-epoch cap.
- No loss wins on every dataset. Contrastive is the weak arm on the larger graphs.
- Adding CGT colour channels does not raise contig F1.
- Assembly edges beat feature kNN on ONT 1B. ONT 100M is the exception.
- Degree and clustering coefficient help Illumina and not Strong100.
- K-means keeps higher ARI than average-linkage on the ONT graphs. VAMB was not rerun.
