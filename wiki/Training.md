# Training

The long stage is C++. Python loads the tensor, writes a job directory, and reads the embedding back.

```
StudyGraph
    → normalized CSR (held-out edges removed from the operator)
    → cpp/gcn_train
    → embedding.f32
    → k-means or agglomerative clustering
    → benchmark/<hypothesis>/{results.csv,summary.csv,f1.html,f1.png}
```

## Budget

From `configs/datasets.yaml`: learning rate 0.05, at most 60 epochs, patience 8. One fifth of the undirected edges are held out. A stored reverse edge is the same edge, not a second sample. Those edges are removed from the propagation operator.

Illumina fits in the recorded grid hit the epoch cap, so those rows are a shared budget, not a fully stopped optimum. Each folder writes `benchmark/<hypothesis>/provenance.txt` (command, Python, numpy, scikit-learn, compiler, seeds, and the MetaMetro commit).

## What is kept

| Path | Keep it for |
|------|-------------|
| `intermediates/<dataset>/` | Features, labels, marker pairs, and the CGT directory |
| `benchmark/<hypothesis>/results.csv` | Every seed |
| `benchmark/<hypothesis>/summary.csv` | Mean and scatter |
| `benchmark/<hypothesis>/f1.html` and `f1.png` | Altair chart |
| `benchmark/<hypothesis>/jobs/` | Scratch for one fit. Not committed |

Re-run one question:

```bash
python research/run_hypothesis.py feature_source
```

All six:

```bash
python research/run_all.py
```

Names: `feature_source`, `loss`, `coloring`, `graph_type`, `multiscale`, `clustering`.
