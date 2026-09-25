"""Mandatory: learning rate is configured, and a run records its provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from run_hypothesis import write_provenance  # noqa: E402

pytestmark = pytest.mark.mandatory


def test_provenance_records_config(tmp_path: Path) -> None:
    """The sidecar names the learning rate and the MetaMetro checkout."""
    config = {
        "seeds": [0, 1],
        "max_epochs": 60,
        "patience": 8,
        "lr": 0.05,
        "vaegbin_data": "/data",
    }
    write_provenance(tmp_path, "loss", config)
    text = (tmp_path / "provenance.txt").read_text(encoding="utf-8")
    assert "lr: 0.05" in text
    assert "command: python research/run_hypothesis.py loss" in text
    assert "numpy:" in text
    assert "metametro_version: 0.15.0" in text
    assert "metametro_commit:" in text
