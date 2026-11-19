"""Load study graphs and the one-factor hypothesis grid."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import yaml

from paragvae.amber import load_gold
from paragvae.graphs import StudyGraph, load_metametro_bubble, load_vaegbin_bundle
from paragvae.metametro_path import ensure_metametro
from paragvae.study import Arm, amber_mode

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Read ``configs/datasets.yaml``."""
    chosen = path or (ROOT / "configs" / "datasets.yaml")
    return yaml.safe_load(chosen.read_text(encoding="utf-8"))


def _optional_path(value: object, env_name: str) -> Path | None:
    """Return a path from the config or from ``env_name``, or None if both are empty."""
    text = "" if value is None else str(value).strip()
    if not text:
        text = os.environ.get(env_name, "").strip()
    if not text:
        return None
    return Path(text)


def _configured_path(value: object, env_name: str) -> Path:
    """Return a path from the config or from ``env_name``. An empty value is an error."""
    path = _optional_path(value, env_name)
    if path is None:
        raise ValueError(f"set {env_name}; a machine path is not stored in this repository")
    return path


def _attach_gold(graph: StudyGraph, name: str, config: dict, mode: str, missing_gold: list[str]) -> None:
    """Load gold for ``name`` when the amber mode requires or allows it."""
    if mode == "false":
        return
    gold_paths = config.get("gold") or {}
    configured = gold_paths[name] if name in gold_paths else ""
    gold_path = _optional_path(configured, f"VAEGBIN_GOLD_{name.upper()}")
    if gold_path is None:
        missing_gold.append(name)
        if mode == "true":
            raise ValueError(
                f"{name} has no AMBER gold standard; set gold.{name} in "
                f"configs/datasets.yaml or VAEGBIN_GOLD_{name.upper()}, "
                "or set amber: auto or amber: false"
            )
        return
    if not gold_path.is_file():
        raise ValueError(f"{name} gold standard is missing: {gold_path}")
    graph.gold = load_gold(gold_path)
    missing = [sequence_id for sequence_id in graph.sequence_ids if sequence_id not in graph.gold]
    if missing:
        raise ValueError(f"{name}: {len(missing)} nodes are absent from {gold_path}")


def load_graphs(config: dict | None = None) -> list[StudyGraph]:
    """Bubble fixture plus the VAEGbin bundles named in the config."""
    config = config or load_config()
    ensure_metametro()
    graphs = [load_metametro_bubble()]
    mode = amber_mode(config.get("amber", "auto"))
    bundles = config.get("bundles") or {}
    if not bundles:
        return graphs
    root = _configured_path(config.get("vaegbin_data"), "VAEGBIN_DATA")
    missing_gold: list[str] = []
    for name, folder in bundles.items():
        graph = load_vaegbin_bundle(root / folder, name=name)
        _attach_gold(graph, name, config, mode, missing_gold)
        graphs.append(graph)
    if mode == "auto" and missing_gold:
        names = ", ".join(missing_gold)
        warnings.warn(
            "AMBER is disabled for graphs without gold "
            f"({names}); set VAEGBIN_GOLD_<NAME> or a gold path in "
            "configs/datasets.yaml. Contig F1 and ARI are still computed, "
            "and amber_* columns stay empty for those graphs.",
            UserWarning,
            stacklevel=2,
        )
    return graphs


def arms_for(hypothesis: str, max_epochs: int, patience: int, lr: float = 0.05) -> list[Arm]:
    """Return the arms that isolate one historical question."""
    common = {"max_epochs": max_epochs, "patience": patience, "lr": lr}
    if hypothesis == "feature_source":
        # Raw k-mer and depth versus the frozen VAE latent. This is not
        # end-to-end VAE+GCN training; only Strong100 has two different matrices.
        return [
            Arm("feature_source", "vae_latent", joint=False, **common),
            Arm("feature_source", "raw_features", joint=True, **common),
        ]
    if hypothesis == "loss":
        return [
            Arm("loss", name, loss=name, **common)
            for name in ("standard", "diff_c", "proxy", "contrastive")
        ]
    if hypothesis == "coloring":
        return [
            Arm("coloring", "uncoloured", colors=False, **common),
            Arm("coloring", "cgt_colors", colors=True, **common),
        ]
    if hypothesis == "graph_type":
        return [
            Arm("graph_type", "assembly", graph="assembly", **common),
            Arm("graph_type", "knn_vae", graph="knn_vae", **common),
            Arm("graph_type", "knn_kmer", graph="knn_kmer", **common),
        ]
    if hypothesis == "multiscale":
        return [
            Arm("multiscale", "latent_only", multiscale=False, **common),
            Arm("multiscale", "degree_clustering", multiscale=True, **common),
        ]
    if hypothesis == "clustering":
        return [
            Arm("clustering", "kmeans", clustering="kmeans", **common),
            Arm("clustering", "agglomerative", clustering="agglomerative", **common),
        ]
    raise ValueError(hypothesis)


HYPOTHESES = (
    "feature_source",
    "loss",
    "coloring",
    "graph_type",
    "multiscale",
    "clustering",
)
