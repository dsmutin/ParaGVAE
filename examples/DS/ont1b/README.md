# ont1b

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

This bundle has no separate 32-d VAE latent: `node_features.npy` equals k-mer (136-d) plus depth. The GCN therefore sees depth. The assembly graph is the CSR from the VAEGbin bundle. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 239. Classes: 10. Undirected edges, self-loops removed: 296. Isolated nodes: 11.
- Seed 0 contig F1 0.293, ARI 0.125, error rate 0.544, epochs 60.
- Error rate on nodes wrong for every seed: 0.410. Agreement of the error flag across seeds: 0.699.

### Degree

- 0 isolated: error rate 0.091 (11 nodes)
- 1 tip: error rate 0.675 (40 nodes)
- 2 path: error rate 0.473 (110 nodes)
- 3+ branch: error rate 0.641 (78 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.569 (102 nodes)
- mixed: error rate 0.539 (102 nodes)
- 1 all neighbours foreign: error rate 0.667 (24 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| class_size | diagnostic_uses_labels | -0.711 | 0.005 | 239 |
| kmer_distance_to_class_mean | diagnostic_uses_labels | 0.379 | 0.005 | 238 |
| depth | model_input | -0.170 | 0.010 | 239 |
| degree | graph | 0.125 | 0.075 | 239 |
| clustering | graph | 0.107 | 0.100 | 239 |
| boundary | diagnostic_uses_labels | 0.023 | 0.791 | 228 |
| component_size | graph | 0.010 | 0.881 | 239 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`
