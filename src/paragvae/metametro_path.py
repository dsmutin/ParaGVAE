"""Locate the MetaMetro checkout without a pip install."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT = Path("/mnt/tank/scratch/dsmutin/tools/my/metametro/src")


def ensure_metametro(src: str | Path | None = None) -> Path:
    """Put MetaMetro's ``src`` on ``sys.path`` and return that path.

    Colour annotations stay in the CFA colour tables. Training reads the
    coloured graph tensor (CGT) only. The project brief calls the colouring
    step TCA; in MetaMetro schema 1.0 that step is the CFA colour dictionary
    projected into ``Cgt.node_colors`` and ``Cgt.edge_colors``.
    """
    chosen = Path(src or os.environ.get("METAMETRO_SRC", _DEFAULT))
    if not (chosen / "metametro" / "__init__.py").is_file():
        raise FileNotFoundError(f"MetaMetro package not found under {chosen}")
    text = str(chosen)
    if text not in sys.path:
        sys.path.insert(0, text)
    return chosen
