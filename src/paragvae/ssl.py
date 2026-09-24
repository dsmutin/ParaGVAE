"""Supervised SSL path: Samovar reads, MEGAHIT, Kraken2, then a masked GCN.

``samovar``, ``megahit``, and ``kraken2`` are external binaries. They are not
conda packages. :func:`require_tools` raises if any of them is missing.

MetaMetro builds the assembly graph (``fastg_to_cfa``) and can project CFA
colours into a CGT (``cfa_to_cdbg``, ``cdbg_to_cgt``). It has no function that
reads Kraken output into those colours. That gap is
``metametro.contracts.colouring.colour_by_kraken``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paragvae.metametro_path import ensure_metametro
from paragvae.train import normalized_adjacency

# Qualified name of the MetaMetro function that would paint a CFA from Kraken.
KRAKEN_COLOUR_FN = "metametro.contracts.colouring.colour_by_kraken"


@dataclass
class SupervisedFit:
    """Early-stopped supervised GCN on one observed-node mask."""

    embedding: np.ndarray
    probabilities: np.ndarray
    epochs_ran: int
    stopped_early: bool
    best_val_loss: float


def _executable_found(name: str) -> bool:
    if name == "":
        return False
    if shutil.which(name) is not None:
        return True
    return Path(name).is_file()


def require_tools(
    samovar: str = "samovar",
    megahit: str = "megahit",
    kraken2: str = "kraken2",
) -> None:
    """Raise ``FileNotFoundError`` naming every missing executable.

    All three are checked. A hit on the first name does not skip the rest,
    and a miss does not return success.
    """
    requested = (samovar, megahit, kraken2)
    missing = [name for name in requested if not _executable_found(name)]
    if missing:
        raise FileNotFoundError("missing executables: " + ", ".join(missing))


def _require_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 15 or k % 2 == 0:
        raise ValueError("k must be an odd integer >= 15")
    return k


def _require_genome_dir(genome_dir: str | Path) -> Path:
    path = Path(genome_dir)
    if not path.exists():
        raise FileNotFoundError(f"genome_dir not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"genome_dir is not a directory: {path}")
    if not any(path.iterdir()):
        raise ValueError(f"genome_dir is empty: {path}")
    return path


def _require_kraken_db(kraken_db: str | Path) -> Path:
    path = Path(kraken_db)
    if not path.exists():
        raise FileNotFoundError(f"kraken_db not found: {path}")
    return path


def _toolkit_for(megahit: str) -> str:
    """Use ``megahit_toolkit`` beside a MEGAHIT path when that file exists."""
    path = Path(megahit)
    if path.is_file():
        sibling = path.with_name("megahit_toolkit")
        if sibling.is_file():
            return str(sibling)
    return "megahit_toolkit"


def _stage(name: str, argv: list[str], stdout: str | None = None) -> dict:
    row = {"name": name, "argv": argv}
    if stdout is not None:
        row["stdout"] = stdout
    return row


def ssl_plan(
    genome_dir: str | Path,
    work_dir: str | Path,
    kraken_db: str | Path,
    *,
    k: int = 21,
    total_reads: int = 200,
    seed: int = 1,
    samovar: str = "samovar",
    megahit: str = "megahit",
    kraken2: str = "kraken2",
) -> dict:
    """Return staged argv for the SSL assembly path.

    Does not create ``work_dir`` and does not start processes. Samovar and
    MEGAHIT argument lists come from MetaMetro's command builders. Kraken2
    classifies MEGAHIT's intermediate contigs, the same FASTA that
    ``contig2fastg`` turns into the assembly graph.
    """
    k = _require_k(k)
    if isinstance(total_reads, bool) or not isinstance(total_reads, int) or total_reads < 1:
        raise ValueError("total_reads must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    genomes = _require_genome_dir(genome_dir)
    database = _require_kraken_db(kraken_db)
    work = Path(work_dir)
    iss_dir = work / "iss"
    reads = work / "reads.fastq"
    assembly_dir = work / "megahit"
    contigs = assembly_dir / "intermediate_contigs" / f"k{k}.contigs.fa"
    fastg = work / f"k{k}.fastg"
    kraken_out = work / "kraken.out"
    kraken_report = work / "kraken.report"
    ensure_metametro()
    from metametro.contracts.external import (
        contig2fastg_command,
        megahit_command,
        samovar_generate_command,
    )

    toolkit = _toolkit_for(megahit)
    stages = [
        _stage(
            "samovar_generate",
            samovar_generate_command(
                genomes,
                iss_dir,
                n_samples=2,
                total_reads=total_reads,
                host_fraction=0,
                seed=seed,
                samovar=samovar,
            ),
        ),
        _stage(
            "iss",
            ["bash", str(iss_dir / ".generate" / "generate.sh"), "--directory", str(iss_dir)],
        ),
        _stage(
            "megahit",
            megahit_command(
                reads,
                assembly_dir,
                k=k,
                threads=4,
                megahit=megahit,
                min_count=2,
            ),
        ),
        _stage(
            "kraken2",
            [
                kraken2,
                "--db",
                str(database),
                "--threads",
                "4",
                "--output",
                str(kraken_out),
                "--report",
                str(kraken_report),
                str(contigs),
            ],
        ),
        _stage(
            "contig2fastg",
            contig2fastg_command(contigs, k, toolkit=toolkit),
            stdout=str(fastg),
        ),
    ]
    return {
        "k": k,
        "seed": seed,
        "total_reads": total_reads,
        "genome_dir": str(genomes),
        "work_dir": str(work),
        "kraken_db": str(database),
        "iss_dir": str(iss_dir),
        "reads": str(reads),
        "assembly_dir": str(assembly_dir),
        "contigs": str(contigs),
        "fastg": str(fastg),
        "kraken_out": str(kraken_out),
        "kraken_report": str(kraken_report),
        "toolkit": toolkit,
        "stages": stages,
    }


def _concatenate_fastq(iss_dir: Path, destination: Path) -> None:
    """Write one FASTQ from Samovar's per-sample ``initial/*_full_R*.fastq`` files."""
    folder = iss_dir / "initial"
    paths = sorted(folder.glob("*_full_R*.fastq"))
    if not paths:
        raise FileNotFoundError(f"no InSilicoSeq FASTQ under {folder}")
    with destination.open("wb") as handle:
        for path in paths:
            data = path.read_bytes()
            if data == b"":
                raise ValueError(f"empty FASTQ: {path}")
            handle.write(data)
            if not data.endswith(b"\n"):
                handle.write(b"\n")
    if destination.stat().st_size == 0:
        raise ValueError(f"concatenated reads are empty: {destination}")


def _execute(stage: dict) -> None:
    from metametro.contracts.external import run_command, run_command_to_file

    argv = list(stage["argv"])
    if "stdout" in stage:
        run_command_to_file(argv, stage["stdout"])
    else:
        run_command(argv)


def _cgt_after_kraken(plan: dict) -> object:
    """Load the FASTG as a CFA, then colour it from Kraken if MetaMetro can.

    ``fastg_to_cfa`` builds the assembly graph. ``colour_by_reads`` paints
    sample depth. Kraken output needs :data:`KRAKEN_COLOUR_FN`. When that
    function exists, the coloured CFA continues through ``cfa_to_cdbg`` and
    ``cdbg_to_cgt``.
    """
    from metametro.contracts.assembly import fastg_to_cfa
    from metametro.contracts import colouring

    fastg = Path(plan["fastg"])
    if not fastg.is_file() or fastg.stat().st_size == 0:
        raise FileNotFoundError(f"FASTG not found: {fastg}")
    graph = fastg_to_cfa(fastg, k=int(plan["k"]), graph_id="ssl")
    painter = getattr(colouring, "colour_by_kraken", None)
    if painter is None:
        raise NotImplementedError(KRAKEN_COLOUR_FN)
    coloured = painter(graph, plan["kraken_out"], plan["kraken_report"])
    from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg
    from metametro.formats.cgt.validator import validate_cgt

    cdbg = cfa_to_cdbg(coloured)
    unitigs = sorted(cdbg.unitigs, key=lambda unitig: unitig.unitig_id)
    out_degree: dict[str, int] = {}
    for link in cdbg.links:
        out_degree[link.source] = out_degree.get(link.source, 0) + 1
    degree = np.asarray(
        [[out_degree.get(unitig.unitig_id, 0)] for unitig in unitigs],
        dtype=np.float32,
    )
    cgt = cdbg_to_cgt(
        cdbg,
        node_features=degree,
        node_feature_names=["out_degree"],
        edge_feature_names=[],
    )
    validate_cgt(cgt)
    return cgt


def run_ssl(
    genome_dir: str | Path,
    work_dir: str | Path,
    kraken_db: str | Path,
    *,
    k: int = 21,
    total_reads: int = 200,
    seed: int = 1,
    samovar: str = "samovar",
    megahit: str = "megahit",
    kraken2: str = "kraken2",
) -> object:
    """Run the planned stages, then project Kraken colours into a CGT.

    Returns a MetaMetro CGT when ``colour_by_kraken`` exists. When it does
    not, this raises ``NotImplementedError`` with :data:`KRAKEN_COLOUR_FN`
    after the external stages and ``fastg_to_cfa``.
    """
    require_tools(samovar=samovar, megahit=megahit, kraken2=kraken2)
    plan = ssl_plan(
        genome_dir,
        work_dir,
        kraken_db,
        k=k,
        total_reads=total_reads,
        seed=seed,
        samovar=samovar,
        megahit=megahit,
        kraken2=kraken2,
    )
    toolkit = str(plan["toolkit"])
    if not _executable_found(toolkit):
        raise FileNotFoundError(f"missing executables: {toolkit}")
    ensure_metametro()
    work = Path(plan["work_dir"])
    iss_dir = Path(plan["iss_dir"])
    work.mkdir(parents=True, exist_ok=True)
    iss_dir.mkdir(parents=True, exist_ok=True)
    (work / "ssl_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for stage in plan["stages"]:
        name = stage["name"]
        if name == "samovar_generate" and any(iss_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty {iss_dir}")
        if name == "megahit":
            assembly = Path(plan["assembly_dir"])
            if assembly.exists():
                raise FileExistsError(f"refusing to reuse {assembly}")
            _concatenate_fastq(iss_dir, Path(plan["reads"]))
        if name in {"kraken2", "contig2fastg"}:
            source = Path(stage["argv"][-1])
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"{name} input not found or empty: {source}")
        _execute(stage)
    return _cgt_after_kraken(plan)


def _zscore(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (values - center) / scale


def _init(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    scale = np.sqrt(2.0 / max(rows, 1))
    return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float64)


class _Adam:
    def __init__(self, lr: float) -> None:
        self.lr = lr
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, name: str, value: np.ndarray, grad: np.ndarray) -> None:
        if name not in self.m:
            self.m[name] = np.zeros_like(value)
            self.v[name] = np.zeros_like(value)
        self.m[name] = 0.9 * self.m[name] + 0.1 * grad
        self.v[name] = 0.999 * self.v[name] + 0.001 * (grad * grad)
        mhat = self.m[name] / (1 - 0.9**self.t)
        vhat = self.v[name] / (1 - 0.999**self.t)
        value -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def _class_weights(train_labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Weight of class ``c`` is ``1/sqrt(count)`` on the given training labels."""
    counts = np.bincount(train_labels, minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present].astype(np.float64))
    return weights


def _split_train_mask(train_mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one fifth of the train mask for early stopping.

    Both slices are subsets of ``train_mask``. Nodes outside the mask are
    not included.
    """
    index = np.flatnonzero(train_mask)
    if index.size < 2:
        raise ValueError("train_mask needs at least 2 nodes to hold out an early-stopping slice")
    rng = np.random.default_rng(seed)
    order = rng.permutation(index)
    n_val = max(1, int(index.size) // 5)
    if n_val >= index.size:
        n_val = index.size - 1
    return order[n_val:], order[:n_val]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _weighted_nll(
    probs: np.ndarray,
    index: np.ndarray,
    labels_at: np.ndarray,
    class_weight: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean weighted NLL on ``index`` and d(loss)/d(logits) for every node.

    Rows outside ``index`` get a zero gradient. ``labels_at`` is only the
    labels of ``index``.
    """
    grad = np.zeros_like(probs)
    count = int(index.size)
    if count == 0:
        return 0.0, grad
    chosen = probs[index]
    weight = class_weight[labels_at]
    picked = chosen[np.arange(count), labels_at]
    loss = float(np.mean(-weight * np.log(np.clip(picked, 1e-12, 1.0))))
    scale = weight / count
    grad[index] = chosen * scale[:, None]
    np.subtract.at(grad, (index, labels_at), scale)
    return loss, grad


def _leaky(pre: np.ndarray) -> np.ndarray:
    return np.where(pre > 0.0, pre, 0.01 * pre)


def fit_supervised_gcn(
    features: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    n_classes: int,
    max_epochs: int,
    patience: int,
    lr: float,
    seed: int,
) -> SupervisedFit:
    """Fit a two-layer GCN with class-weighted cross-entropy on ``train_mask``.

    The weight of class ``c`` is ``1/sqrt(count)`` among nodes on the train
    mask, so a majority class does not own the loss in proportion to its
    count. Early stopping uses a held-out slice of that same mask. Labels
    outside the mask are not read. The return is the node embedding and
    class probabilities of shape ``(N, n_classes)``.
    """
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs must be an integer >= 1")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if lr <= 0:
        raise ValueError("lr must be positive")
    raw_features = np.asarray(features, dtype=np.float64)
    if raw_features.ndim != 2:
        raise ValueError("features must be 2-d")
    if not np.isfinite(raw_features).all():
        raise ValueError("features must be finite")
    n = int(raw_features.shape[0])
    mask = np.asarray(train_mask, dtype=bool)
    label_array = np.asarray(labels)
    indptr = np.asarray(indptr)
    if mask.shape != (n,) or label_array.shape[0] != n:
        raise ValueError("labels and train_mask must have one entry per node")
    if indptr.shape != (n + 1,):
        raise ValueError(f"indptr length {indptr.shape[0]} does not match {n} nodes")
    fit_idx, val_idx = _split_train_mask(mask, seed)
    train_labels = np.asarray(label_array[np.flatnonzero(mask)], dtype=np.int64)
    fit_y = np.asarray(label_array[fit_idx], dtype=np.int64)
    val_y = np.asarray(label_array[val_idx], dtype=np.int64)
    if (
        np.any(train_labels < 0)
        or np.any(train_labels >= n_classes)
        or np.any(fit_y < 0)
        or np.any(fit_y >= n_classes)
        or np.any(val_y < 0)
        or np.any(val_y >= n_classes)
    ):
        raise ValueError("labels on the train mask must be in 0 .. n_classes-1")
    class_weight = _class_weights(train_labels, n_classes)
    encoded = raw_features
    if encoded.shape[1] == 0:
        encoded = np.ones((n, 1), dtype=np.float64)
    encoded = _zscore(encoded)
    adjacency = normalized_adjacency(indptr, indices, n).astype(np.float64)
    width = int(encoded.shape[1])
    hidden = 16
    latent = 8
    rng = np.random.default_rng(seed + 1)
    w0 = _init(rng, width, hidden)
    w1 = _init(rng, hidden, latent)
    w_cls = _init(rng, latent, n_classes)
    bias = np.zeros(n_classes, dtype=np.float64)
    opt = _Adam(lr=lr)
    best = (w0.copy(), w1.copy(), w_cls.copy(), bias.copy())
    best_val = np.inf
    stall = 0
    epochs = 0

    def forward(
        local_w0: np.ndarray,
        local_w1: np.ndarray,
        local_cls: np.ndarray,
        local_bias: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pre_h = np.asarray(adjacency @ (encoded @ local_w0), dtype=np.float64)
        hidden_state = _leaky(pre_h)
        latent_state = np.asarray(adjacency @ (hidden_state @ local_w1), dtype=np.float64)
        logits = latent_state @ local_cls + local_bias
        return pre_h, hidden_state, latent_state, logits

    for epoch in range(max_epochs):
        opt.t = epoch + 1
        epochs = epoch + 1
        pre_h, hidden_state, latent_state, logits = forward(w0, w1, w_cls, bias)
        probs = _softmax(logits)
        _train_loss, grad_logits = _weighted_nll(probs, fit_idx, fit_y, class_weight)
        grad_latent = grad_logits @ w_cls.T
        grad_cls = latent_state.T @ grad_logits
        grad_bias = np.sum(grad_logits, axis=0)
        grad_u = np.asarray(adjacency.T @ grad_latent, dtype=np.float64)
        grad_w1 = hidden_state.T @ grad_u
        grad_h = grad_u @ w1.T
        grad_pre = np.where(pre_h > 0.0, grad_h, 0.01 * grad_h)
        grad_xw = np.asarray(adjacency.T @ grad_pre, dtype=np.float64)
        grad_w0 = encoded.T @ grad_xw
        opt.step("w_cls", w_cls, grad_cls)
        opt.step("bias", bias, grad_bias)
        opt.step("w1", w1, grad_w1)
        opt.step("w0", w0, grad_w0)
        _pre, _hidden, _latent, val_logits = forward(w0, w1, w_cls, bias)
        val_loss, _val_grad = _weighted_nll(_softmax(val_logits), val_idx, val_y, class_weight)
        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best = (w0.copy(), w1.copy(), w_cls.copy(), bias.copy())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    w0, w1, w_cls, bias = best
    _pre, _hidden, embedding, logits = forward(w0, w1, w_cls, bias)
    return SupervisedFit(
        embedding=embedding.astype(np.float32),
        probabilities=_softmax(logits).astype(np.float32),
        epochs_ran=epochs,
        stopped_early=epochs < max_epochs,
        best_val_loss=float(best_val),
    )
