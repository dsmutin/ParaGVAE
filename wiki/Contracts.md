# Contracts

Change a row only together with the test that locks it.

## Implementation table

| Contract | Implementation |
|----------|----------------|
| Single version source (`VERSION`) | `paragvae.__version__` reads `VERSION` |
| CLI `--version` and JSON run | baseline (`paragvae.cli.main`). `status` stays `baseline` |
| Pipeline result keys `status`, `ok`, `input_path` | baseline (`paragvae.baseline.run_pipeline`) |
| Conda-only install (`environment.yml`) | env file + `PYTHONPATH=src` |
| Mandatory pytest on every commit | `pytest -m mandatory` |
| Optional pytest on release / manual | `pytest -m optional` |
| Toy example runs the tool | `examples/toy/run.py` |
| English docs on public APIs | module and function docstrings |
| No `git push` unless the human asks | rule `no-push` |
| Colouring lives in CFA tables and CGT `uint8` matrices | `paragvae.graphs`; brief name TCA is this step. See [Data](Data) |
| GCN training reads the CGT tensor | `cpp/gcn_train` via `paragvae.native`. No dense `N×N` adjacency |
| Hypothesis checks are one-factor and early-stopped | `research/<hypothesis>/` via `paragvae.suite` |
| Binning score is contig F1 and ARI | `paragvae.score` |
| AMBER `f1_score_seq` is a separate output stage | `paragvae.amber.score_bins` |

Genome labels are used for the score and for choosing `k`. They are not features and they do not enter the loss.
