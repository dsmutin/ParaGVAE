# Contracts

Scaffold contracts from `/start-dev`. Add tool-specific rows when the product spec is known. Change a row only together with tests.

## Implementation table

| Contract | Implementation |
|----------|----------------|
| Single version source (`VERSION`) | baseline (`paragvae.__version__` reads `VERSION`) |
| CLI `--version` and JSON run | baseline (`paragvae.cli.main`) |
| Pipeline result keys `status`, `ok`, `input_path` | baseline (`paragvae.baseline.run_pipeline`) |
| Conda-only install (`environment.yml`) | baseline (env file + `PYTHONPATH=src`) |
| Mandatory pytest on every commit | baseline (`pytest -m mandatory`) |
| Optional pytest on release / manual | baseline (`pytest -m optional` placeholder `pass`) |
| Toy example runs the tool | baseline (`examples/toy/run.py`) |
| English docs on public APIs | baseline (module/function docstrings) |
| No `git push` unless the human asks | baseline (rule `no-push`) |
| Colouring lives in CFA colour tables and CGT `uint8` colour matrices | `docs/metametro.md` (brief name TCA = this step) |
| GCN training reads the CGT tensor only | `paragvae.train.train_gcn` |
| Hypothesis checks are one-factor and early-stopped | `research/<hypothesis>/` via `paragvae.suite` |
| Binning score is contig F1 and ARI, not AMBER | `paragvae.score` |
