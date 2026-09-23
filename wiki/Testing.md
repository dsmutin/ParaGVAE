# Testing

## Data

| Path | Role |
|------|------|
| `examples/toy/data/` | scaffold CLI output |
| `intermediates/<dataset>/cgt/` | cached MetaMetro CGT |
| `configs/datasets.yaml` | paths to MetaMetro and the VAEGbin bundles |

External assemblies are not stored in git. The cache is what a later run reloads.

## Checks

1. `pytest -m mandatory` — version, CLI, hypothesis names, label-dict alignment
2. `python examples/toy/run.py` — baseline JSON
3. `python research/run_all.py` — six hypothesis benchmarks, two seeds, early stopping

The binning score in the benchmarks is contig F1 plus ARI. It is not AMBER.
