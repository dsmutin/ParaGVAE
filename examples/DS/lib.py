"""Shared error analysis for one early-stopped GCN.

A node is wrong when the majority ground-truth label of its k-means cluster
is not its own label. ``k`` is the number of labels. That is the same oracle
``k`` as the benchmarks. Genome and route labels are not model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from paragvae.native import train_gcn_native
from paragvae.score import contig_f1
from paragvae.study import _degree_and_clustering
from paragvae.train import undirected_edges

N_PERM = 200


def majority_error(labels: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """True where the node label is not the majority label of its cluster."""
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    error = np.zeros(len(labels), dtype=bool)
    for cluster in np.unique(clusters):
        member = clusters == cluster
        values, counts = np.unique(labels[member], return_counts=True)
        majority = values[np.argmax(counts)]
        error[member] = labels[member] != majority
    return error


def _self_check() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    clusters = np.array([0, 0, 1, 1, 1])
    if majority_error(labels, clusters).tolist() != [False, False, True, False, False]:
        raise RuntimeError("majority_error self-check failed")


def graph_metrics(indptr: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Degree, clustering, component size, and neighbour index lists. Self-loops are ignored."""
    structural = _degree_and_clustering(indptr, indices)
    n = len(indptr) - 1
    neighbors: list[list[int]] = []
    rows = []
    cols = []
    for node in range(n):
        start, stop = int(indptr[node]), int(indptr[node + 1])
        neigh = []
        for col in np.asarray(indices[start:stop]).tolist():
            other = int(col)
            if other == node or other in neigh:
                continue
            neigh.append(other)
            rows.append(node)
            cols.append(other)
        neighbors.append(neigh)
    if rows:
        undirected = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        undirected = undirected.maximum(undirected.T)
    else:
        undirected = sparse.csr_matrix((n, n))
    _n_components, component = connected_components(undirected, directed=False)
    sizes = np.bincount(component)
    return {
        "degree": structural[:, 0],
        "clustering": structural[:, 1],
        "component_size": sizes[component].astype(np.float64),
        "neighbors": neighbors,
    }


def boundary_fraction(labels: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """Fraction of neighbours with a different label. Isolated nodes are NaN.

    This uses the ground-truth labels. It is a diagnostic, not a model input.
    """
    out = np.full(len(labels), np.nan)
    for node, neigh in enumerate(neighbors):
        if not neigh:
            continue
        out[node] = float(np.mean(labels[neigh] != labels[node]))
    return out


def distance_to_class_mean(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Euclidean distance to the node's own class mean after column z-scoring."""
    matrix = np.asarray(features, dtype=np.float64)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (matrix - matrix.mean(axis=0)) / scale
    out = np.full(matrix.shape[0], np.nan)
    for label in np.unique(labels):
        member = labels == label
        if int(member.sum()) < 2:
            continue
        center = scaled[member].mean(axis=0)
        out[member] = np.linalg.norm(scaled[member] - center, axis=1)
    return out


def spearman_with_permutation(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    """Spearman rho of a feature against a 0/1 error flag, plus a permutation p.

    The p-value is ``(count + 1) / (n_perm + 1)`` where ``count`` is how often
    a shuffled error flag has an absolute rho at least as large.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    result = {"n": float(len(x)), "rho": float("nan"), "permutation_p": float("nan")}
    if len(x) < 8 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return result
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(N_PERM):
        shuffled = float(spearmanr(x, rng.permutation(y)).statistic)
        if abs(shuffled) >= abs(rho):
            count += 1
    result["rho"] = rho
    result["permutation_p"] = (count + 1) / (N_PERM + 1)
    return result


def _cluster(embedding: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_groups, random_state=seed, n_init=5)
    return model.fit_predict(np.asarray(embedding, dtype=np.float64))


def _save_chart(chart: alt.Chart, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(dest.with_suffix(".html")))
    try:
        chart.save(str(dest.with_suffix(".png")), scale_factor=2)
    except Exception as error:  # noqa: BLE001 — HTML remains if the PNG converter fails
        dest.with_suffix(".png_error.txt").write_text(str(error), encoding="utf-8")


def _degree_chart(frame: pd.DataFrame) -> alt.Chart:
    order = ["0 isolated", "1 tip", "2 path", "3+ branch"]
    grouped = (
        frame.groupby("degree_bin", as_index=False)
        .agg(error_rate=("error", "mean"), n=("error", "size"))
    )
    return (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("degree_bin:N", sort=order, title="Degree, self-loops removed"),
            y=alt.Y("error_rate:Q", title="Error rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["degree_bin", "n", alt.Tooltip("error_rate:Q", format=".3f")],
        )
        .properties(width=320, height=220, title="Errors by graph role")
    )


def _boundary_chart(frame: pd.DataFrame) -> alt.Chart:
    usable = frame.dropna(subset=["boundary_bin"])
    grouped = usable.groupby("boundary_bin", as_index=False).agg(error_rate=("error", "mean"), n=("error", "size"))
    order = ["0 no foreign neighbour", "mixed", "1 all neighbours foreign"]
    return (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("boundary_bin:N", sort=order, title="Ground-truth label boundary"),
            y=alt.Y("error_rate:Q", title="Error rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["boundary_bin", "n", alt.Tooltip("error_rate:Q", format=".3f")],
        )
        .properties(width=360, height=220, title="Errors at label boundaries")
    )


def _class_chart(frame: pd.DataFrame) -> alt.Chart:
    grouped = (
        frame.groupby(["label_name", "class_size"], as_index=False)
        .agg(error_rate=("error", "mean"), n=("error", "size"))
    )
    return (
        alt.Chart(grouped)
        .mark_circle(size=60)
        .encode(
            x=alt.X("class_size:Q", scale=alt.Scale(type="log"), title="Class size"),
            y=alt.Y("error_rate:Q", title="Error rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["label_name", "n", alt.Tooltip("error_rate:Q", format=".3f")],
        )
        .properties(width=360, height=240, title="Per-class error against class size")
    )


def _association_chart(table: pd.DataFrame) -> alt.Chart:
    shown = table.dropna(subset=["rho"]).copy()
    return (
        alt.Chart(shown)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("rho:Q", title="Spearman rho with node error"),
            color=alt.Color("role:N", title="Feature role"),
            tooltip=["feature", "role", alt.Tooltip("rho:Q", format=".3f"), alt.Tooltip("permutation_p:Q", format=".3f"), "n"],
        )
        .properties(width=360, height=28 * max(len(shown), 1) + 40, title="Association with node error")
    )


def _bin_boundary(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    if value == 0:
        return "0 no foreign neighbour"
    if value == 1:
        return "1 all neighbours foreign"
    return "mixed"


def _bin_degree(value: float) -> str:
    degree = int(value)
    if degree <= 0:
        return "0 isolated"
    if degree == 1:
        return "1 tip"
    if degree == 2:
        return "2 path"
    return "3+ branch"


def analyze(
    *,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    indptr: np.ndarray,
    indices: np.ndarray,
    edge_weight: np.ndarray | None,
    extras: dict[str, tuple[np.ndarray, str]],
    out: Path,
    protocol: str,
    epilogue: str = "",
    max_epochs: int,
    patience: int,
    lr: float,
    seeds: tuple[int, ...] = (0, 1),
) -> dict:
    """Train the standard GCN, score node errors, and write charts."""
    _self_check()
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError(f"{name}: features {features.shape} do not match {labels.shape[0]} labels")
    if not np.isfinite(features).all():
        raise ValueError(f"{name}: features contain a non-finite value")
    if len(label_names) != int(labels.max()) + 1:
        raise ValueError(f"{name}: {len(label_names)} names for labels up to {int(labels.max())}")
    n_groups = int(np.unique(labels).size)
    if n_groups < 2:
        raise ValueError(f"{name}: need at least two labels")
    metrics = graph_metrics(indptr, indices)
    boundary = boundary_fraction(labels, metrics["neighbors"])
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    scores = []
    cluster0 = None
    for seed in seeds:
        fit = train_gcn_native(
            features=features,
            indptr=indptr,
            indices=indices,
            different_pairs=None,
            edge_weight=edge_weight,
            loss="standard",
            max_epochs=max_epochs,
            patience=patience,
            latent=16,
            hidden=32,
            seed=seed,
            lr=lr,
            work=out / "jobs" / f"seed{seed}",
        )
        clusters = _cluster(fit.embedding, n_groups, seed)
        error = majority_error(labels, clusters)
        errors.append(error)
        if seed == seeds[0]:
            cluster0 = clusters
        scores.append(
            {
                "seed": seed,
                "f1": contig_f1(labels, clusters),
                "ari": float(adjusted_rand_score(labels, clusters)),
                "error_rate": float(error.mean()),
                "epochs_ran": fit.epochs_ran,
                "stopped_early": fit.stopped_early,
                "best_val_loss": fit.best_val_loss,
            }
        )
        np.save(out / f"embedding_seed{seed}.npy", fit.embedding)
    error0 = errors[0]
    stable = np.logical_and.reduce(errors) if len(errors) > 1 else error0
    agreement = float(np.mean(errors[0] == errors[1])) if len(errors) > 1 else 1.0
    class_size = np.array([int(np.sum(labels == label)) for label in labels], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "node": np.arange(len(labels)),
            "label": labels,
            "label_name": [label_names[int(label)] for label in labels],
            "degree": metrics["degree"],
            "clustering": metrics["clustering"],
            "component_size": metrics["component_size"],
            "boundary": boundary,
            "class_size": class_size,
            "error": error0.astype(int),
            "stable_error": stable.astype(int),
            "cluster": cluster0,
            "degree_bin": [_bin_degree(value) for value in metrics["degree"]],
            "boundary_bin": [_bin_boundary(value) for value in boundary],
        }
    )
    feature_table = {
        "degree": (metrics["degree"], "graph"),
        "clustering": (metrics["clustering"], "graph"),
        "component_size": (metrics["component_size"], "graph"),
        "boundary": (boundary, "diagnostic_uses_labels"),
        "class_size": (class_size, "diagnostic_uses_labels"),
    }
    feature_table.update(extras)
    associations = []
    for feature, (values, role) in feature_table.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(labels),):
            raise ValueError(f"{name}: feature {feature} has shape {values.shape}")
        frame[feature] = values
        stats = spearman_with_permutation(values, error0.astype(float), seed=0)
        associations.append({"feature": feature, "role": role, **stats})
    assoc = pd.DataFrame(associations)
    frame.to_csv(out / "nodes.csv", index=False)
    assoc.to_csv(out / "associations.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "fits.csv", index=False)
    _save_chart(_degree_chart(frame), out / "error_by_degree")
    _save_chart(_boundary_chart(frame), out / "error_by_boundary")
    _save_chart(_class_chart(frame), out / "error_by_class")
    _save_chart(_association_chart(assoc), out / "associations")
    summary = {
        "name": name,
        "n": int(len(labels)),
        "n_classes": n_groups,
        "n_edges_undirected": int(undirected_edges(indptr, indices)[0].shape[0]),
        "n_isolated": int(np.sum(metrics["degree"] == 0)),
        "error_rate": float(error0.mean()),
        "stable_error_rate": float(stable.mean()),
        "error_agreement": agreement,
        "f1_seed0": scores[0]["f1"],
        "ari_seed0": scores[0]["ari"],
        "epochs_seed0": scores[0]["epochs_ran"],
        "protocol": protocol,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_readme(out, summary, assoc, frame, protocol, epilogue)
    return summary


def _rate_line(frame: pd.DataFrame, column: str, value: str) -> str:
    part = frame[frame[column] == value]
    if part.empty:
        return f"- {value}: no nodes"
    return f"- {value}: error rate {part['error'].mean():.3f} ({len(part)} nodes)"


def _write_readme(
    out: Path,
    summary: dict,
    assoc: pd.DataFrame,
    frame: pd.DataFrame,
    protocol: str,
    epilogue: str,
) -> None:
    """Write the example README from the tables just saved."""
    lines = [
        f"# {summary['name']}",
        "",
        "## Question",
        "",
        "Which nodes does the standard early-stopped GCN put in a cluster whose majority label is a different class, and which graph features travel with that mistake?",
        "",
        "## Protocol",
        "",
        protocol,
        "",
        "A node is an error when its label is not the majority label of its k-means cluster. `k` is the number of classes. Seed 0 is the error flag used in the charts. `stable_error` is a node that is wrong for every seed.",
        "",
        "## Result",
        "",
        f"- Nodes: {summary['n']}. Classes: {summary['n_classes']}. Undirected edges, self-loops removed: {summary['n_edges_undirected']}. Isolated nodes: {summary['n_isolated']}.",
        f"- Seed 0 contig F1 {summary['f1_seed0']:.3f}, ARI {summary['ari_seed0']:.3f}, error rate {summary['error_rate']:.3f}, epochs {summary['epochs_seed0']}.",
        f"- Error rate on nodes wrong for every seed: {summary['stable_error_rate']:.3f}. Agreement of the error flag across seeds: {summary['error_agreement']:.3f}.",
        "",
        "### Degree",
        "",
    ]
    for label in ("0 isolated", "1 tip", "2 path", "3+ branch"):
        lines.append(_rate_line(frame, "degree_bin", label))
    lines.extend(["", "### Label boundary", ""])
    for label in ("0 no foreign neighbour", "mixed", "1 all neighbours foreign"):
        lines.append(_rate_line(frame, "boundary_bin", label))
    lines.extend(
        [
            "",
            "Boundary uses the ground-truth labels. It is not a feature the GCN sees.",
            "",
            "## Associations",
            "",
            "Spearman rho is between the feature and the seed-0 error flag. The permutation p shuffles that flag 200 times. A single p-value is not a discovery; the comparison across bins above is the check.",
            "",
            "| Feature | Role | rho | permutation p | n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in assoc.sort_values("rho", key=lambda series: series.abs(), ascending=False).itertuples(index=False):
        rho = "NA" if not np.isfinite(row.rho) else f"{row.rho:.3f}"
        p_value = "NA" if not np.isfinite(row.permutation_p) else f"{row.permutation_p:.3f}"
        lines.append(f"| {row.feature} | {row.role} | {rho} | {p_value} | {int(row.n)} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `nodes.csv` — one row per node",
            "- `associations.csv` — the table above",
            "- `fits.csv` — F1, ARI, and epochs per seed",
            "- `error_by_degree.html`, `error_by_boundary.html`, `error_by_class.html`, `associations.html`",
            "",
        ]
    )
    if epilogue:
        lines.extend(["", epilogue.strip(), ""])
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def route_cohesion(
    colors: np.ndarray,
    color_names: list[tuple[str, str]],
    clusters: np.ndarray,
    coordinates: np.ndarray,
    min_stops: int = 8,
) -> pd.DataFrame:
    """For each route colour, the fraction of its stops that share one cluster.

    ``color_names`` is ``(namespace, value)`` aligned with colour columns.
    A size-matched null shuffles cluster ids 50 times and averages cohesion.
    """
    if colors.shape[1] != len(color_names):
        raise ValueError("colour columns do not match colour names")
    rng = np.random.default_rng(0)
    rows = []
    for column, (namespace, value) in enumerate(color_names):
        if namespace != "route":
            continue
        member = np.flatnonzero(colors[:, column] > 0)
        if member.size < min_stops:
            continue
        assigned = clusters[member]
        cohesion = float(np.bincount(assigned).max() / member.size)
        nulls = []
        for _ in range(50):
            shuffled = rng.permutation(clusters)[member]
            nulls.append(float(np.bincount(shuffled).max() / member.size))
        coords = coordinates[member]
        span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        rows.append(
            {
                "route": value,
                "n_stops": int(member.size),
                "cohesion": cohesion,
                "null_cohesion": float(np.mean(nulls)),
                "span": span,
            }
        )
    return pd.DataFrame(rows)
