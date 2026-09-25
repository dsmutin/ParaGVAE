# MetaMetro data structures

ParaGVAE does not invent a second graph schema. MetaMetro 0.16.0 is a required installed package. Source code does not store a path to that checkout.

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
