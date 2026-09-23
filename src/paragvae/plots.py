"""Altair charts for one hypothesis benchmark."""

from __future__ import annotations

from pathlib import Path


def _summary(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["dataset"], row["arm"]), []).append(row)
    summary = []
    for (dataset, arm), group in buckets.items():
        f1 = [float(item["f1"]) for item in group]
        ari = [float(item["ari"]) for item in group]
        epochs = [float(item["epochs_ran"]) for item in group]
        mean = sum(f1) / len(f1)
        var = sum((value - mean) ** 2 for value in f1) / len(f1)
        summary.append(
            {
                "dataset": dataset,
                "arm": arm,
                "f1": mean,
                "f1_std": var**0.5,
                "ari": sum(ari) / len(ari),
                "epochs": sum(epochs) / len(epochs),
            }
        )
    return summary


def write_charts(rows: list[dict], dest: Path) -> None:
    """Write an HTML chart and a PNG when vl-convert is installed."""
    import altair as alt

    dest.mkdir(parents=True, exist_ok=True)
    summary_rows = _summary(rows)
    fields = list(summary_rows[0])
    with (dest / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = alt.Data(values=summary_rows)
    base = alt.Chart(summary).properties(width=120, height=160)
    bars = base.mark_bar().encode(
        x=alt.X("arm:N", title=None, sort=None),
        y=alt.Y("f1:Q", title="Contig F1"),
        color=alt.Color("dataset:N", title="Dataset"),
    )
    errors = base.mark_errorbar().encode(
        x=alt.X("arm:N", sort=None),
        y=alt.Y("f1:Q", title="Contig F1"),
        yError=alt.YError("f1_std:Q"),
    )
    chart = (
        (bars + errors)
        .facet(column=alt.Column("dataset:N", title=None))
        .resolve_scale(y="shared")
    )
    chart.save(str(dest / "f1.html"))
    try:
        chart.save(str(dest / "f1.png"), scale_factor=2)
    except Exception as error:  # noqa: BLE001 — PNG is optional if the converter is absent
        (dest / "f1_png_error.txt").write_text(str(error), encoding="utf-8")
