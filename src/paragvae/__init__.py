"""paragvae: Graph VAE binning on MetaMetro coloured graph tensors"""

from __future__ import annotations

from pathlib import Path


def _version() -> str:
    """Return the version from a source checkout, or from package metadata once installed."""
    checkout = Path(__file__).resolve().parents[2] / "VERSION"
    if checkout.is_file() and (checkout.parent / "pyproject.toml").is_file():
        return checkout.read_text(encoding="utf-8").strip()
    from importlib.metadata import version

    return version("paragvae")


__version__ = _version()
