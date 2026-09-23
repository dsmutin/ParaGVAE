# How to install

Conda is the only supported install. There is no pip-first path.

```bash
conda env create -f environment.yml
conda activate paragvae
```

The env name is `paragvae`. `environment.yml` sets `PYTHONPATH=src` and pins Python 3.12. Channels are conda-forge.

Check the scaffold:

```bash
paragvae --version
python examples/toy/run.py
pytest -m mandatory
```

`paragvae --version` prints the `VERSION` file. The toy script writes baseline JSON.

## Native trainer

The epoch loop is `cpp/gcn_train.cpp`. The first hypothesis run compiles it if the binary is missing or older than the source. A recorded run used the conda-forge `g++` named in `benchmark/<hypothesis>/provenance.txt`. The binary is not committed.

## Tests in CI

| Workflow | When | What |
|----------|------|------|
| `required-tests.yml` | push and pull request | `pytest -m mandatory` |
| `full-tests.yml` | release or manual | mandatory, optional, and `examples/toy/run.py` |

Both workflows create the conda env from `environment.yml`.
