# Testing

## Test data

| Path | Role |
|------|------|
| `examples/toy/data/` | JSON written by `examples/toy/run.py` |
| `tests/` | Contract tests. No large assemblies in git |
| `intermediates/<dataset>/cgt/` | Cached MetaMetro CGT for a later reload |
| `configs/datasets.yaml` | MetaMetro checkout and the four VAEGbin bundle folders |

External assemblies stay outside the repository.

## Integrative testing

1. Mandatory contracts: `pytest -m mandatory` (version file, CLI JSON, hypothesis names, label-dict alignment).
2. Toy: `python examples/toy/run.py`. The JSON `status` is `baseline`.
3. Required CI on every push and pull request: `required-tests.yml`.
4. Full suite on a release or a manual run: `full-tests.yml` (mandatory, optional, and the toy).
5. Hypothesis grid: `python research/run_all.py`. Two seeds, early stopping, one provenance file per hypothesis.

The grid score is contig F1 plus ARI. See [How we score](How-we-score).
