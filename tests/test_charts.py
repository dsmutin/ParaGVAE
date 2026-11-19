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


def test_empty_amber_columns_do_not_break_charts(tmp_path: Path) -> None:
    """Empty amber_* values stay off the AMBER chart and do not fail the run."""
    rows = [
        {
            "dataset": "toy",
            "arm": "vae_latent",
            "f1": 0.5,
            "ari": 0.4,
            "epochs_ran": 2,
            "amber_f1": "",
            "amber_ap": "",
            "amber_ar": "",
        }
    ]
    plots.write_charts(rows, tmp_path)
    assert (tmp_path / "f1.html").is_file()
    assert not (tmp_path / "amber_f1.html").exists()

