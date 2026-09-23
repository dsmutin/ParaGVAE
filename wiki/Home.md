# paragvae wiki

Graph VAE binning on MetaMetro coloured graph tensors.

Version is the `VERSION` file in the code repository.

## Call path

```
VAEGbin bundle or MetaMetro fixture
        → CGT (CSR tensor; colours in uint8 matrices)
        → cpp/gcn_train (early-stopped 2-layer GCN)
        → k-means or agglomerative bins
        → benchmark/<hypothesis>/results.csv
```

The CLI `paragvae` still returns the scaffold JSON (`status=baseline`). Hypothesis runs are `python research/run_hypothesis.py <name>`.

## Pages

- [Contracts](Contracts)
- [Testing](Testing)
- [Findings](Findings)
