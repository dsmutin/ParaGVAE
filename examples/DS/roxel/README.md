# roxel

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

MetaMetro `data/work/roxel`, the sfnetworks Roxel street example. The node class is the road type. The node colour matrix is all zeros; street names are edge colours and are not the node target. Model inputs are longitude and latitude only. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1. The stored class names include a case split: unclassified and Unclassified. They are kept as separate classes.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 588. Classes: 9. Undirected edges, self-loops removed: 727. Isolated nodes: 107.
- Seed 0 contig F1 0.236, ARI 0.041, error rate 0.543, epochs 33.
- Error rate on nodes wrong for every seed: 0.488. Agreement of the error flag across seeds: 0.876.

### Degree

- 0 isolated: error rate 0.589 (107 nodes)
- 1 tip: error rate 0.581 (234 nodes)
- 2 path: error rate 0.477 (239 nodes)
- 3+ branch: error rate 0.750 (8 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.370 (262 nodes)
- mixed: error rate 0.623 (114 nodes)
- 1 all neighbours foreign: error rate 0.838 (105 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| class_size | diagnostic_uses_labels | -0.709 | 0.005 | 588 |
| boundary | diagnostic_uses_labels | 0.380 | 0.005 | 481 |
| component_size | graph | -0.138 | 0.005 | 588 |
| dist_from_mean_coordinate | model_input | 0.092 | 0.020 | 588 |
| degree | graph | -0.085 | 0.045 | 588 |
| clustering | graph | 0.062 | 0.209 | 588 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`
