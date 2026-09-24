# Coloured-graph heads

`fit_coloured_graph` in `paragvae.heads` trains a numpy classifier on one totally coloured graph. The input is that graph. The output is a node head, an edge head, or both (`mode` is `node`, `edge`, or `both`).

## Inputs

Node features have shape `(N, F)`. Node colours have shape `(N, Cn)`, stored as `uint8` or float. `Cn` may be 0. Edges are a directed CSR (`indptr`, `indices`). Reverse arcs collapse to one undirected edge `i < j`, and self-loops are dropped. Optional edge features `(E_dir, Fe)` and edge colours `(E_dir, Ce)` follow that CSR order.

`edge_labels` use the undirected order returned as `edge_index`. A label matrix whose row count is the CSR arc count, and not the undirected count, is max-reduced onto those edges. Several 1s in one row are a valid multi-hot target.

## Two paths

Isolated nodes are common, and an isolated node gets no message from the convolution. Each node therefore has a residual MLP on the concatenation of z-scored node features, node colours as float, and degree. That path does not read neighbours. A two-layer graph convolution runs on the undirected normalised adjacency, without self-loops. The two paths are summed before the node head.

## Node head

The node head is a softmax over `n_node_classes`. The loss is cross-entropy on the train mask only. The weight of a class is `1/sqrt(train count)`.

Class imbalance, not degree, was the main failure mode: a majority class absorbed the others. The weights keep that class from owning the loss.

## Edge head

Each undirected edge is scored from the concatenation of the two endpoint states, the absolute difference of those states, the edge features, and the edge colours. A sigmoid per class returns probabilities `(E, C)`. This is multi-label, not a softmax: one edge may belong to several classes at once.

The loss is mean binary cross-entropy. A class with positive train rate `rate` weights its positive term by `(1 - rate) / rate`.

On some graphs the label boundary mattered. The absolute endpoint difference is the edge feature that exposes it. Depth was a weak feature, so there is no depth loss.

## Both heads

`mode="both"` trains one shared backbone. The node loss and the edge loss are added.

## Early stopping

A held-out fifth of the train mask is the stopping set. Those rows stay inside the train mask; they are not eval rows. Eval nodes and eval edges are never scored. When `edge_train_mask` is omitted and a node `train_mask` is set, an undirected edge is a training edge only if both endpoints are training nodes.

## Returns

`HeadFit.node_prob` is `(N, K)`, or `None` when `mode` is `edge`. `HeadFit.edge_prob` is `(E, C)`, or `None` when `mode` is `node`. `HeadFit.edge_index` is `(E, 2)` with `i < j`, in the same order as `edge_prob`.
