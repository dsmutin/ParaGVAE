# MetaMetro data structures

ParaGVAE does not invent a second graph schema. MetaMetro is a required installed package. The pinned tag is `PIN` in `scripts/ci_install_metametro.sh`, and CI installs it from there. Source code does not store a path to that checkout.

Training reads a coloured graph tensor (`Cgt`) and nothing assembled beside it. When a CFA, CDBG, or FASTG has to become a tensor, the call is a MetaMetro converter (`cdbg_to_cgt`, `cfa_to_cdbg`, `cgt_from_csr`, `load_cgt`).

| Field | Role |
|---|---|
| `indptr`, `indices` | CSR topology |
| `node_features`, `edge_features` | Model inputs. A width-0 edge matrix means unweighted edges. Weights are not invented. |
| `node_colors`, `edge_colors` | `uint8` colour mask |
| `node_color_weights`, `edge_color_weights` | Optional `float32` scores on those same colour columns. They are not features. |
| `node_labels`, `edge_labels` | Evaluation ids. Not features, colours, or the default loss. |

Schema version on the tensor is still `1.0`. Colour probabilities are the optional weight arrays, not a second mask.

`paragvae.graphs.cgt_from_csr` is a thin call to MetaMetro's external-CSR contract. VAEGbin bundles enter through that function. The hypothesis runner then trains on the resulting `Cgt`.

## Gold and geometric benches

From MetaMetro 0.23.0, evaluation namespaces are not colour namespaces. They are written under `ground_truth/` (`<namespace>_nodes.tsv`, `_edges.tsv`, `_values.tsv`) and are read with `load_bench_gold` / `load_bench_gold_cgt`. Asking `load_bench_cgt` for a gold namespace raises. The colouring arm never sees those columns.

`paragvae.ds.load.load_geometric` loads the Saint Petersburg and Roxel benches through `load_bench_states` (features, node classes, edge targets) and the gold helpers above. Saint Petersburg uses `node_namespace=transport_type` and `edge_namespaces=(transport_type, route)`. Roxel uses `node_namespace=type` and `edge_namespaces=(name,)`. Road type comes from `ground_truth/` `type`, not leftover CGT `node_labels`. Street names are the `name` gold (`edge_targets`), not edge colours. A bench that still has those namespaces only in `colors.tsv` (the pre-0.23.0 layout) fails with a clear error; there is no fallback. `n_sample_colors` counts only the `sample` namespace.

Pass `outdir` to point at a benchbuild directory; the default is MetaMetro's `default_outdir` for the named bench. Identity checks for a real default build of `spb_ground_transit` / `roxel` (prefixes published in MetaMetro v0.24.0 `docs/benchmarks.md`) run only when that build is present.

