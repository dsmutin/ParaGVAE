# illumina

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

This bundle has no separate 32-d VAE latent: `node_features.npy` equals k-mer (136-d) plus depth. The GCN therefore sees depth. The assembly graph is the CSR from the VAEGbin bundle. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 5140. Classes: 10. Undirected edges, self-loops removed: 6722. Isolated nodes: 1.
- Seed 0 contig F1 0.158, ARI 0.052, error rate 0.716, epochs 60.
- Error rate on nodes wrong for every seed: 0.611. Agreement of the error flag across seeds: 0.804.

### Degree

- 0 isolated: error rate 1.000 (1 nodes)
- 1 tip: error rate 0.836 (226 nodes)
- 2 path: error rate 0.691 (2664 nodes)
- 3+ branch: error rate 0.733 (2249 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.794 (2283 nodes)
- mixed: error rate 0.680 (2166 nodes)
- 1 all neighbours foreign: error rate 0.572 (690 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| class_size | diagnostic_uses_labels | -0.465 | 0.005 | 5140 |
| kmer_distance_to_class_mean | diagnostic_uses_labels | 0.195 | 0.005 | 5140 |
| boundary | diagnostic_uses_labels | -0.176 | 0.005 | 5139 |
| depth | model_input | -0.051 | 0.005 | 5140 |
| degree | graph | 0.020 | 0.119 | 5140 |
| clustering | graph | 0.018 | 0.209 | 5140 |
| component_size | graph | -0.009 | 1.000 | 5140 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`
