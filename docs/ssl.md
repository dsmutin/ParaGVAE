# Supervised SSL pipeline

The path is:

1. Species genomes in a non-empty `genome_dir`.
2. `samovar generate` writes an InSilicoSeq Snakemake script under `work/iss/.generate/generate.sh`.
3. That script simulates a metagenome (`n_samples=2`, `host_fraction=0`).
4. The per-sample FASTQ files are concatenated to `work/reads.fastq`.
5. MEGAHIT assembles those reads at one odd `k` (`k >= 15`), keeping intermediate contigs.
6. `kraken2` classifies `work/megahit/intermediate_contigs/k<k>.contigs.fa` with the caller-supplied database (`--threads 4`, `--output work/kraken.out`, `--report work/kraken.report`).
7. `megahit_toolkit contig2fastg` writes `work/k<k>.fastg`. MetaMetro `fastg_to_cfa` loads that graph as a CFA.
8. Kraken taxids become CFA colours (`namespace` `kraken2`, `value` the taxid) and `cfa_to_cdbg` / `cdbg_to_cgt` project them into CGT `node_colors` / `edge_colors` (`uint8`). Edges stay uncoloured. Training reads that tensor.
9. `fit_supervised_gcn` trains a two-layer GCN on an observed-node mask.

`samovar`, `megahit`, and `kraken2` are external binaries. They are not conda packages and they are not pinned in `environment.yml`. `require_tools` checks all three and raises `FileNotFoundError` naming every missing executable. `ssl_plan` only builds argument lists. `run_ssl` is what starts the processes, and only after `require_tools` succeeds.

`megahit_toolkit` is the FASTG converter shipped beside MEGAHIT. `run_ssl` raises `FileNotFoundError` if that executable is missing too.

## Kraken colours

`fastg_to_cfa` loads the assembly. Each Kraken2 `--output` sequence id must be a CFA `node_id`. A `C` line whose taxonomy field is `Species name (taxid 12345)` contributes that taxid. A `U` line, or a line with no taxid, leaves the node uncoloured. Unique taxids are sorted and mapped to colour ids `0..C-1`. `colour_cfa` writes that map onto nodes only. `cfa_to_cdbg` and `cdbg_to_cgt` copy the ids into the CGT colour matrices.

## Supervised head

`fit_supervised_gcn` does not call Samovar, MEGAHIT, or Kraken2.

The loss is class-weighted cross-entropy on the train mask only. The weight of class `c` is `1/sqrt(count)` among nodes on that mask. Early stopping holds out one fifth of the same mask (at least one node) and never scores nodes outside it. Labels outside the mask are not read. The result is a node embedding and class probabilities with shape `(N, n_classes)`.
