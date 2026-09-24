# **paraGVAE**

*Graph VAE binning on MetaMetro coloured graph tensors*

paraGVAE trains a graph convolutional network on a MetaMetro coloured graph tensor and scores the bins. The command-line entry is still the scaffold: it prints the version or writes baseline JSON. Hypothesis runs are separate scripts.

**Status:** in development. The version is the `VERSION` file in the code repository.

```bash
conda env create -f environment.yml
conda activate paragvae
python examples/toy/run.py
python research/run_hypothesis.py feature_source
```

Source tree: [github.com/dsmutin/ParaGVAE](https://github.com/dsmutin/ParaGVAE).

---

## Where to look

### Setup

| Page | What is there |
|------|----------------|
| [How to install](How-to-install) | Conda env `paragvae`, Python 3.12, the C++ trainer compile. |
| [Contracts](Contracts) | What the scaffold still guarantees, and what the trainer implements. |
| [Testing](Testing) | Toy data, cached tensors, mandatory pytest, the full hypothesis run. |

### Data and training

| Page | What is there |
|------|----------------|
| [Data](Data) | CFA colours in the CGT `uint8` matrices, the five graphs, the schema gap. |
| [Training](Training) | Call path, early-stopped 2-layer GCN, cached intermediates, Altair charts. |

### Checks

| Page | What is there |
|------|----------------|
| [How we score](How-we-score) | Contig F1, ARI, and a separate AMBER `f1_score_seq` stage. VAMB is not rerun. |
| [Hypotheses](Hypotheses) | The six one-factor questions and their arms. |
| [Findings](Findings) | Means over seeds 0 and 1 after the scoring and split fixes. |

---

## Typical path

1. [How to install](How-to-install) → `conda env create -f environment.yml`
2. `python examples/toy/run.py` → baseline JSON (`status` is `baseline`)
3. Point `configs/datasets.yaml` at MetaMetro and the VAEGbin bundles: [Data](Data)
4. One hypothesis: `python research/run_hypothesis.py feature_source`
5. Read the score before the table: [How we score](How-we-score), then [Findings](Findings)

## Starting call graph

```
paragvae CLI → run_pipeline() → JSON {status, ok, input_path}

research/run_hypothesis.py
        → CGT (CSR; colours in uint8)
        → cpp/gcn_train (early-stopped 2-layer GCN)
        → k-means, or average-linkage when that arm says so
        → benchmark/<hypothesis>/results.csv
```
