# Joint training

Numbers in this folder predate the contig-F1 matching fix and are not valid until the rerun.

Frozen 32-d VAE features versus a GCN on raw k-mer and depth.

Strong100 is the only bundle whose two matrices differ. Mean contig F1: joint 0.313, frozen 0.253, both still at or near the 60-epoch cap. Other datasets store the same array in both roles, so the arms match.

Numbers and the chart: `benchmark/joint_training/`.
