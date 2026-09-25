"""Require the installed MetaMetro package. No checkout path is stored here.

ParaGVAE trains on a MetaMetro coloured graph tensor (CGT). Building that
tensor from an assembly, a CFA, or a CDBG is MetaMetro's job. This module
checks the installed version and reads fields off a ``Cgt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED_METAMETRO = "0.15.0"


def ensure_metametro() -> str:
    """Import MetaMetro and return its version.

    The package must already be installed in this environment. A missing
    install or a different version is an error.
    """
    try:
        import metametro
    except ImportError as exc:
        raise ImportError(
            f"MetaMetro {REQUIRED_METAMETRO} is required. Install that package into this environment."
        ) from exc
    version = str(metametro.__version__)
    if version != REQUIRED_METAMETRO:
        raise ImportError(f"MetaMetro {version} is installed; paragvae requires {REQUIRED_METAMETRO}")
    return version


def metametro_root() -> Path:
    """Return the MetaMetro checkout that provided the installed package.

    An editable install keeps ``data/work`` beside the package. A wheel that
    does not ship that tree fails here instead of guessing a machine path.
    """
    ensure_metametro()
    import metametro

    root = Path(metametro.__file__).resolve().parents[2]
    if not (root / "VERSION").is_file():
        raise FileNotFoundError(
            "The installed MetaMetro package has no checkout VERSION beside it. "
            "Install it from the MetaMetro repository so data/work stays available."
        )
    if (root / "VERSION").read_text(encoding="utf-8").strip() != REQUIRED_METAMETRO:
        raise ImportError(f"MetaMetro checkout at {root} is not {REQUIRED_METAMETRO}")
    return root


def work_dir(name: str) -> Path:
    """Return ``data/work/<name>`` inside the installed MetaMetro checkout."""
    path = metametro_root() / "data" / "work" / name
    if not path.is_dir():
        raise FileNotFoundError(f"MetaMetro work directory is missing: {path}")
    return path


def training_arrays(cgt: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read the training view of one validated CGT.

    Features, CSR, and the first edge-feature column come from the tensor.
    A width-0 edge matrix means the graph is unweighted. Colours and labels
    are not returned: they are not the default GCN input.
    """
    ensure_metametro()
    from metametro.formats.cgt.validator import validate_cgt

    validate_cgt(cgt)
    features = np.asarray(cgt.node_features, dtype=np.float32)
    indptr = np.asarray(cgt.indptr)
    indices = np.asarray(cgt.indices)
    edges = np.asarray(cgt.edge_features, dtype=np.float32)
    if edges.size == 0 or (edges.ndim == 2 and edges.shape[1] == 0):
        weight = None
    elif edges.ndim == 1:
        weight = edges.reshape(-1)
    else:
        weight = np.asarray(edges[:, 0], dtype=np.float32)
    nnz = int(indptr[-1]) if len(indptr) else 0
    if weight is not None and weight.shape[0] != nnz:
        raise ValueError(f"edge feature length {weight.shape[0]} does not match {nnz} CSR entries")
    return features, indptr, indices, weight
