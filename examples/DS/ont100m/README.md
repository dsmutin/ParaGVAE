# ont100m

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

This bundle has no separate 32-d VAE latent: `node_features.npy` equals k-mer (136-d) plus depth. The GCN therefore sees depth. The assembly graph is the CSR from the VAEGbin bundle. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 383. Classes: 10. Undirected edges, self-loops removed: 23. Isolated nodes: 353.
- Seed 0 contig F1 0.243, ARI 0.102, error rate 0.658, epochs 22.
- Error rate on nodes wrong for every seed: 0.486. Agreement of the error flag across seeds: 0.765.

### Degree

- 0 isolated: error rate 0.663 (353 nodes)
- 1 tip: error rate 0.524 (21 nodes)
- 2 path: error rate 0.833 (6 nodes)
- 3+ branch: error rate 0.667 (3 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.500 (16 nodes)
- mixed: error rate 0.750 (4 nodes)
- 1 all neighbours foreign: error rate 0.700 (10 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| class_size | diagnostic_uses_labels | -0.461 | 0.005 | 383 |
| boundary | diagnostic_uses_labels | 0.209 | 0.274 | 30 |
| kmer_distance_to_class_mean | diagnostic_uses_labels | 0.165 | 0.005 | 383 |
| depth | model_input | -0.051 | 0.279 | 383 |
| component_size | graph | -0.036 | 0.547 | 383 |
| degree | graph | -0.033 | 0.592 | 383 |
| clustering | graph | NA | NA | 383 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`
