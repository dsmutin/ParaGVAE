"""Load study graphs and the one-factor hypothesis grid."""

from __future__ import annotations

from pathlib import Path

import yaml

from paragvae.graphs import StudyGraph, load_metametro_bubble, load_vaegbin_bundle
from paragvae.metametro_path import ensure_metametro
from paragvae.study import Arm

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Read ``configs/datasets.yaml``."""
    chosen = path or (ROOT / "configs" / "datasets.yaml")
    return yaml.safe_load(chosen.read_text(encoding="utf-8"))


def load_graphs(config: dict | None = None) -> list[StudyGraph]:
    """Bubble fixture plus the VAEGbin bundles named in the config."""
    config = config or load_config()
    ensure_metametro(config["metametro_src"])
    graphs = [load_metametro_bubble()]
    root = Path(config["vaegbin_data"])
    for name, folder in config["bundles"].items():
        graphs.append(load_vaegbin_bundle(root / folder, name=name))
    return graphs


def arms_for(hypothesis: str, max_epochs: int, patience: int) -> list[Arm]:
    """Return the arms that isolate one historical question."""
    common = {"max_epochs": max_epochs, "patience": patience}
    if hypothesis == "joint_training":
        return [
            Arm("joint_training", "frozen_vae", joint=False, **common),
            Arm("joint_training", "joint_raw", joint=True, **common),
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
    "joint_training",
    "loss",
    "coloring",
    "graph_type",
    "multiscale",
    "clustering",
)
