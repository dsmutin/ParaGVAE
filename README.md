# paragvae

[![version](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdsmutin%2FParaGVAE%2Fmain%2FVERSION&query=%24&label=version&color=blue)](VERSION)
[![required tests](https://img.shields.io/github/actions/workflow/status/dsmutin/ParaGVAE/required-tests.yml?branch=main&label=required%20tests)](https://github.com/dsmutin/ParaGVAE/actions/workflows/required-tests.yml)
[![full tests](https://img.shields.io/github/actions/workflow/status/dsmutin/ParaGVAE/full-tests.yml?branch=main&label=full%20tests)](https://github.com/dsmutin/ParaGVAE/actions/workflows/full-tests.yml)
[![warning](https://img.shields.io/badge/warning-in%20development-yellow)](https://shields.io/badges/static-badge)

Graph VAE binning on MetaMetro coloured graph tensors

**Warning: in development.** Interfaces may change. See `VERSION` (single source of truth).

## Install

Conda is the only supported install:

```bash
conda env create -f environment.yml
conda activate paragvae
```

`environment.yml` sets `PYTHONPATH=src`. Do not publish a pip-first install path.

## Usage

```bash
paragvae --version
paragvae
python examples/toy/run.py
python research/run_hypothesis.py feature_source
```

Hypothesis folders, external data paths, and the score definition are in [docs/reproduction.md](docs/reproduction.md). MetaMetro CFA colours and the CGT training tensor are in [docs/metametro.md](docs/metametro.md). The latest contig-F1 comparison is in [docs/findings.md](docs/findings.md). Charts are under `benchmark/`.

## Tests

```bash
pytest -m mandatory    # every commit
pytest                 # mandatory + optional (release / manual CI)
```

## License

MIT. See [CONTRIBUTING.md](CONTRIBUTING.md).
