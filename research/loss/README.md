# Loss

Numbers in this folder predate the contig-F1 matching fix and are not valid until the rerun.

`standard` edge BCE, `diff_c` and `proxy` add the single-copy marker push (weights 1 and 2), `contrastive` uses neighbour pairs.

No single loss leads on every dataset. Proxy is highest on ONT 100M (0.302). `diff_c` is highest on ONT 1B (0.481). Strong100 standard and proxy are tied near 0.25. Contrastive is lower on the larger graphs. The bubble has no marker pairs, so the first three losses match.

Numbers and the chart: `benchmark/loss/`.
