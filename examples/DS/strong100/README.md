# strong100

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

Model inputs are the published 32-d VAE latent in `node_features.npy`, on the assembly CSR. Depth is `node_attributes_depth.npy` and is checked to be the last column of the raw matrix. The k-mer matrix is 136-d and is not a model input on this dataset. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 852. Classes: 81. Undirected edges, self-loops removed: 550. Isolated nodes: 456.
- Seed 0 contig F1 0.324, ARI 0.294, error rate 0.518, epochs 36.
- Error rate on nodes wrong for every seed: 0.393. Agreement of the error flag across seeds: 0.761.

### Degree

- 0 isolated: error rate 0.570 (456 nodes)
- 1 tip: error rate 0.491 (108 nodes)
- 2 path: error rate 0.514 (105 nodes)
- 3+ branch: error rate 0.404 (183 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.232 (95 nodes)
- mixed: error rate 0.407 (177 nodes)
- 1 all neighbours foreign: error rate 0.702 (124 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| boundary | diagnostic_uses_labels | 0.405 | 0.005 | 396 |
| class_size | diagnostic_uses_labels | -0.323 | 0.005 | 852 |
| kmer_distance_to_class_mean | diagnostic_uses_labels | 0.264 | 0.005 | 839 |
| component_size | graph | -0.124 | 0.005 | 852 |
| degree | graph | -0.121 | 0.005 | 852 |
| clustering | graph | -0.071 | 0.045 | 852 |
| depth | not_in_model | -0.015 | 0.632 | 852 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`
