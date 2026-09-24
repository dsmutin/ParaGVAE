# Where the GCN is wrong

Each folder trains the standard early-stopped GCN used in the benchmarks and marks a node wrong when its class is not the majority class of its cluster. Charts are Altair HTML and PNG.

| Example | Nodes | Isolated | Error rate | Stable error | Contig F1 | ARI |
|---|---:|---:|---:|---:|---:|---:|
| [strong100](strong100/README.md) | 852 | 456 | 0.518 | 0.393 | 0.324 | 0.294 |
| [ont100m](ont100m/README.md) | 383 | 353 | 0.658 | 0.486 | 0.243 | 0.102 |
| [ont1b](ont1b/README.md) | 239 | 11 | 0.544 | 0.410 | 0.293 | 0.125 |
| [illumina](illumina/README.md) | 5140 | 1 | 0.716 | 0.611 | 0.158 | 0.052 |
| [phage](phage/README.md) | 983 | 506 | 0.439 | 0.357 | 0.529 | 0.444 |
| [spb_transit](spb_transit/README.md) | 4570 | 26 | 0.312 | 0.312 | 0.227 | 0.007 |
| [roxel](roxel/README.md) | 588 | 107 | 0.543 | 0.488 | 0.236 | 0.041 |

Stable error counts nodes that are wrong for both seeds. The per-folder README has the degree bins, the label-boundary bins, and the Spearman table.

Assembly graphs are Strong100, ONT 100M, ONT 1B, and Illumina. MetaMetro examples are the T-phage graph, the Saint Petersburg ground-transit graph, and the Roxel street graph.

## What the errors line up with

Class size is the strongest Spearman row on every graph except the phage graph, where sample colour is stronger and is itself a stand-in for class. Small classes are absorbed into the majority cluster. That is a property of the label frequencies and of k-means, not of degree.

A label boundary (a neighbour with a different ground-truth class) raises the error rate on Strong100 (0.232 if every neighbour agrees, 0.702 if every neighbour disagrees; 396 nodes have a neighbour), on Roxel (0.370 vs 0.838), and on the Saint Petersburg stops (0.131 vs 0.641). On Illumina the boundary bins go the other way (0.794 vs 0.572), and inside the three largest genomes the rates sit between 0.46 and 0.57, so the boundary is not where that graph fails. ONT 100M has only 30 nodes with a neighbour, and the boundary correlation there does not beat a shuffle (permutation p 0.274).

Degree role (isolated, tip, path, branch) does not mark a shared failure mode. Strong100 is mostly isolated (456 of 852) and those contigs are wrong 0.570 of the time, about the same as tips and paths. ONT 100M is 353 isolated nodes out of 383. ONT 1B and Illumina are connected and still wrong on about half or more of every degree bin.

Depth is not the failure mode. On Strong100 it is not a model input and rho is -0.015 (permutation p 0.632). Where the GCN does see depth, rho is -0.051 on ONT 100M, -0.170 on ONT 1B, and -0.051 on Illumina. The Illumina p-value is small because there are 5140 nodes; the association is still tiny.

K-mer distance to the contig's own genome mean (a diagnostic that uses the labels) has rho 0.264, 0.165, 0.379, and 0.195 on Strong100, ONT 100M, ONT 1B, and Illumina. Composition outliers of a genome are somewhat harder. On Strong100 the model never sees that k-mer matrix.

The phage tensor gives the GCN only `out_degree`. T4 is almost entirely correct (error 0.023) and T5 and T7 are entirely wrong, even though T4 and T5 have nearly the same mean degree. Coverage, GC, and length correlations follow the genome, which the model cannot see. See `phage/README.md`.

Saint Petersburg k-means returns bus and nothing else: the error rate 0.312 is the fraction of non-bus stops. Route colours are not the training target. Across 529 routes with at least 8 stops, the fraction of stops that share one cluster averages 0.582, against 0.350 for a size-matched shuffle of the same clusters. Routes are tighter than chance and are still split. See `spb_transit/README.md`.

Roxel predicts road type from longitude and latitude. The node colour matrix is empty. Street names are edge colours and are not scored. The stored names `unclassified` and `Unclassified` are kept as two classes. Errors concentrate where a vertex meets a different road type.
