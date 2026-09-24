# phage

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

MetaMetro `data/work/phage_x10`. The CGT node feature is only `out_degree`. Coverage is the coloured-CFA node column. Length and GC are computed from `nodes.fna`. Sample colours are the two sample columns and are not concatenated into the model. Label 0 is the unclassified or mixed class recorded in MetaMetro `docs/baseline-run.md`; the other labels are T1, T3, T4, T5, and T7. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 983. Classes: 6. Undirected edges, self-loops removed: 621. Isolated nodes: 506.
- Seed 0 contig F1 0.529, ARI 0.444, error rate 0.439, epochs 21.
- Error rate on nodes wrong for every seed: 0.357. Agreement of the error flag across seeds: 0.840.

### Degree

- 0 isolated: error rate 0.310 (506 nodes)
- 1 tip: error rate 0.571 (35 nodes)
- 2 path: error rate 0.540 (235 nodes)
- 3+ branch: error rate 0.618 (207 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.633 (90 nodes)
- mixed: error rate 0.804 (168 nodes)
- 1 all neighbours foreign: error rate 0.379 (219 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| n_sample_colors | not_in_model | -0.551 | 0.005 | 983 |
| class_size | diagnostic_uses_labels | -0.374 | 0.005 | 983 |
| boundary | diagnostic_uses_labels | -0.293 | 0.005 | 477 |
| gc | not_in_model | 0.280 | 0.005 | 983 |
| component_size | graph | 0.261 | 0.005 | 983 |
| coverage | not_in_model | 0.256 | 0.005 | 983 |
| degree | graph | 0.249 | 0.005 | 983 |
| clustering | graph | 0.042 | 0.219 | 983 |
| length | not_in_model | 0.019 | 0.537 | 983 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`

## Reading the correlations

The Spearman rows for sample colour, GC, coverage, and degree mostly track which genome a unitig belongs to. T4 (389 unitigs, mean degree 0.19) has error rate 0.023. T5 (177 unitigs, mean degree 0.23) has error rate 1. Degree does not separate those two. T5 and T7 are wrong on every unitig. Of the 339 unitigs seen in both samples, 260 are T4 and 76 are unclassified; those are the classes the clustering gets right. Among T1, T3, T5, and T7, a second sample colour is almost absent, so the colour correlation is not an independent cause.
