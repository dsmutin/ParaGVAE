# Loss

`standard` is edge BCE. `diff_c` and `proxy` add the marker push (weights 1 and 2). `contrastive` is InfoNCE against 8 random nodes per training edge.

No loss leads everywhere. diff_c is highest on Strong100 (0.328) and ONT 1B (0.433). Standard is highest on ONT 100M (0.320), where the seed scatter is ±0.077. Contrastive is highest on Illumina (0.199), and those fits hit the 60-epoch cap. The bubble has no marker pairs, so the first three losses match at 0.750.

Numbers and the chart: `benchmark/loss/`.
