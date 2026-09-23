"""Mandatory: a chart failure fails the run."""

from __future__ import annotations

from pathlib import Path

import pytest

from paragvae import plots

pytestmark = pytest.mark.mandatory


def test_chart_error_is_recorded_and_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CSV can already be on disk, but the process must still exit with an error."""

    def fail(_rows: list[dict], _dest: Path) -> None:
        raise RuntimeError("chart broke")

    monkeypatch.setattr(plots, "write_charts", fail)
    with pytest.raises(RuntimeError, match="chart broke"):
        plots.save_charts([], tmp_path)
    assert "chart broke" in (tmp_path / "chart_error.txt").read_text(encoding="utf-8")
