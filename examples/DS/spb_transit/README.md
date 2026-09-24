# spb_transit

## Question

Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?

## Protocol

MetaMetro `data/work/spb_ground_transit`. The node class is the sorted combination of `transport_type` colours (bus, tram, trolley, and the mixed labels). It is not a single route id. Model inputs are longitude, latitude, and the simulated passenger coverage from the CFA. Edge weight is that coverage on the hop. Route colours are multi-label and are scored afterwards as cluster cohesion, not as the k-means target. Standard edge loss, at most 60 epochs, patience 8, learning rate 0.05. Seeds 0 and 1.

A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.

## Result

- Nodes: 4570. Classes: 7. Undirected edges, self-loops removed: 7733. Isolated nodes: 26.
- Seed 0 contig F1 0.227, ARI 0.007, error rate 0.312, epochs 12.
- Error rate on nodes wrong for every seed: 0.312. Agreement of the error flag across seeds: 1.000.

### Degree

- 0 isolated: error rate 0.038 (26 nodes)
- 1 tip: error rate 0.264 (2044 nodes)
- 2 path: error rate 0.287 (1713 nodes)
- 3+ branch: error rate 0.498 (787 nodes)

### Label boundary

- 0 no foreign neighbour: error rate 0.131 (2892 nodes)
- mixed: error rate 0.623 (717 nodes)
- 1 all neighbours foreign: error rate 0.641 (935 nodes)

Boundary uses the ground-truth labels. It is not a feature the GCN sees.

## Associations

Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.

| Feature | Role | rho | permutation p | n |
|---|---|---:|---:|---:|
| class_size | diagnostic_uses_labels | -0.979 | 0.005 | 4570 |
| n_modes | not_in_model | 0.718 | 0.005 | 4570 |
| boundary | diagnostic_uses_labels | 0.517 | 0.005 | 4544 |
| dist_from_mean_coordinate | model_input | -0.406 | 0.005 | 4570 |
| n_routes | not_in_model | 0.167 | 0.005 | 4570 |
| degree | graph | 0.154 | 0.005 | 4570 |
| coverage | model_input | 0.141 | 0.005 | 4570 |
| component_size | graph | 0.017 | 0.463 | 4570 |
| clustering | graph | 0.014 | 0.373 | 4570 |

## Files

- `nodes.csv` — one row per node
- `associations.csv` — the table above
- `fits.csv` — F1, ARI, and epochs per seed
- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`

## Reading the correlations

Seed-0 error is exactly the non-bus stops: bus error rate is 0 on 3145 stops, and every other class, including tram, trolley, and every mixed label, has error rate 1. Stops with two or three transport modes (868 stops) are all errors. The boundary gap (0.131 with no foreign neighbour, 0.641 when every neighbour has another type) is the same split, because non-bus stops are the boundary of the bus majority.

## Routes

A route is a multi-label colour, so it is not the k-means target. Among 529 routes with at least 8 stops, mean cohesion is 0.582. A size-matched shuffle of the same clusters has mean cohesion 0.350. Span in `routes.csv` is the diagonal of the longitude/latitude box, in degrees.
