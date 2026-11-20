# Contributing to paragvae

All development follows this guide. Project rules require it (`follow-contributing`).

English is required for every public function, class, module, CLI flag, data file, and user-facing document. Other languages or missing documentation are not allowed.

## Architecture

```
src/paragvae/          the tool
  models/              encoders, heads, SSL, leakage, line-graph flip, binning candidates
  ds/                  error analysis (loading and the tables/charts it writes)
  cli, baseline, graphs, train, study
tests/                 mandatory vs optional pytest
examples/toy/          end-to-end run of the baseline CLI
examples/DS/           thin entry points; the analysis code is src/paragvae/ds
cite/                  BibTeX for integrated third-party tools
agents/                portable rules and skills (any IDE)
```

New methods and new analysis code go under `src/paragvae/models/` or `src/paragvae/ds/`. Do not add a second package tree or a sibling checkout for a method that belongs in this tool. Benchmark result tables are not part of this layout; do not start a new results tree beside `src/`.

CLI (`paragvae.cli`) calls the baseline pipeline and returns `{status, ok, input_path}`.

Contracts, test data, and integrative testing are on the [GitHub wiki](https://github.com/dsmutin/ParaGVAE/wiki). There is no `wiki/` tree in this repository.

Replace baseline bodies with real implementations. Keep the documented return keys until you change the contract on the [wiki Contracts page](https://github.com/dsmutin/ParaGVAE/wiki/Contracts) and the tests together.

## Testing architecture

| Kind | Marker | Command | When |
|------|--------|---------|------|
| Required | `mandatory` | `pytest -m mandatory` | every commit; GitHub Action `required-tests` |
| Optional | `optional` | `pytest` (all) | release or workflow_dispatch; Action `full-tests` |
| Examples | — | `python examples/toy/run.py` | full CI; after features that touch the CLI |
| Vignettes | — | any `vignettes/` or extra `examples/*` | full CI when those files exist |

Do not mark a contract test `optional`. Optional tests are slow, extra, or nice-to-have.

After **any new feature**, run the **mandatory** suite (and toy if the CLI changed) before you stop.

## Feature checklist (`todo.md`)

Track work in `todo.md` (checkboxes). One line per feature or fix. Check it off only when mandatory tests pass. This is a **feature list**, not a `/do` analysis graph.

## Versioning

Edit **only** `VERSION`. Everything else reads it.

Starting value: `0.0.1`.

| Change | Bump |
|--------|------|
| New feature | **minor** (`0.0.1` → `0.1.0`) |
| Fix or update of an existing feature | **patch** (`0.1.0` → `0.1.1`) |
| Release | **major** (`0.1.1` → `1.0.0`) |

## Install (conda only)

```bash
conda env create -f environment.yml
conda activate paragvae
```

## GitHub

Never `git push` unless the human explicitly asks. CI runs on GitHub after they push.

## Benchmarks

Do not add a new benchmark in this repository. Add it in MetaMetro (`metametro benchbuild`) and open a pull request there. Completion graphs are MetaMetro bench directories.

Do not hard-code a machine path. Prebuilt bundles use `VAEGBIN_DATA` and `VAEGBIN_GOLD_<NAME>`. If a bundle path is missing, stop. Gold is required only when `amber` is `true`; `auto` (the default) skips AMBER with one warning per graph when gold is unset (`run_arm` and, for the configured suite, `load_graphs`), and `false` disables it. If a graph has fewer than two usable evaluation labels (`>= 0`), `run_arm` warns once and returns `status=no_labels` with NaN ARI and F1 instead of writing 0.

Do not mock a benchmark graph, a taxonomy label, or a metric. The in-process MetaMetro bubble used by the test suite is a fixture, not a benchmark. A graph that exists only inside a test file stays there.

Do not copy an evaluation target into graph features or colours. The simulated taxon id is a score label.

## MetaMetro tensor

MetaMetro is a required dependency of this environment. The pinned tag is `PIN` in `scripts/ci_install_metametro.sh`, and CI installs it from there. Do not hard-code a path to its checkout in source, tests, configs, or examples.

Train and score on a MetaMetro coloured graph tensor (`Cgt`). Do not keep a second graph schema, and do not hand-build the CSR, features, or colours that MetaMetro already writes. If the input is still an assembly, CFA, or CDBG, call MetaMetro to produce the tensor, then train on that object.

`paragvae.metametro_path.ensure_metametro` fails when the package is missing or is older than that pin. A newer version is accepted. A hypothesis run records the installed package as `metametro_version` and that pin as `metametro_pin` in `benchmark/<hypothesis>/provenance.txt`. `research/run_registry.py` writes the same fields to `benchmark/registry/provenance.txt`. Both use `paragvae.run_record.installed_metametro_version` (the live `metametro.__version__`) and `installed_metametro_commit` (from package metadata or the checkout; `unknown` if it cannot be determined). `training_arrays` is the reader that turns one validated tensor into the arrays the GCN consumes.

## Colourings

Do not add a new colouring method in this repository. Add it in MetaMetro and open a pull request there.

Do not mock a colouring. The colouring arm (`cgt_colors`) uses only the `composition` namespace, loaded with `load_bench_cgt(..., namespaces=namespaces_for(("composition_kmeans",)))` when that namespace is present on the bench. It never asks `filter_colours` / `load_bench_*` for a namespace the graph lacks. `composition_colors` is a MetaMetro k-means helper used on graphs that have no bench (`mock_cgt` and VAEGbin bundles). A bench that lacks the composition layer is an error; it does not fall back to k-means.

Check that a selected colouring exists on the bench ToCUMG before training. If the colour matrix is empty and you expected a layer, stop.

`run_arm` records `coloring_source` on every row:

| Value | Meaning |
|---|---|
| `""` | The arm does not use colours |
| `composition_kmeans` | Stored MetaMetro composition layer on a bench |
| `kmeans_raw` | No bench; k-means colours from raw features. Warns once (`kmeans-fallback`) |

Use `load_bench_cgt(..., namespaces=...)` to keep the layers an arm needs, after checking those namespaces exist on the graph. Gold labels for a MetaMetro bench come from `load_bench_gold_cgt(bench_dir).node_colors` (row-wise argmax, or `-1` if the row is empty). Do not read `taxon` as a colour namespace.

## Evaluation taxids

On a simulated metagenome the scored id is the `tax_id` of the `sim` row in the accession table. That table is written before read generation.

Do not replace it with Kraken, Kaiju, CheckM, a k-means colour, or a mock label. Do not copy it into node features, edge features, edge weights, or node/edge colours of the graph passed to the GCN. A training loss that consumes that id is the same leak. The id is an evaluation vector only.

`paragvae.provenance_checks` enforces both checks. A benchmark that cannot name the pre-generation table stops.

## Citations

Add a `.bib` entry in `cite/` only for tools this package actually integrates. Do not invent papers.
