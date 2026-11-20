"""Re-export. The line-graph flip lives in ``paragvae.models``."""

from paragvae.models.flip import (
    EdgeGraph,
    edge_line_graph,
    infer_nodes_from_edges,
    train_edge_graph,
)
