"""Graph encoders from the May VAEGbin sweeps, trained with early stopping.

Architectures:

- ``gcn``: every layer is a normalized sparse multiply, leaky ReLU except the last.
- ``sage``: self-features plus a neighbour aggregate, then a dense map to the latent.
- ``gat``: one-head attention over neighbours, then a dense map.
- ``transformer``: one global attention block, a residual, one GCN block, then a dense map.
  The May config used four heads; this port uses one head so the graph still fits in memory.
- ``joint``: a linear VAE on the node features whose sample is the GCN input.
  The returned embedding concatenates the VAE mean and the GCN latent.

``n_layers`` is the number of message-passing blocks. The historical GCN arm is
``gcn`` with two layers. Losses match ``paragvae.train``: edge BCE, marker
weights, and InfoNCE. ``biological`` adds the Kraken auxiliary term and does
not read evaluation labels.
"""

from __future__ import annotations

import numpy as np

from paragvae.models.bioloss import BioTargets, biological_gradient
from paragvae.models.sota import SOTA, SotaModel
from paragvae.train import (
    TrainResult,
    _Adam,
    _bce_logits,
    _init,
    _zscore,
    draw_contrastive_negatives,
    infonce_loss,
    normalized_from_edges,
    split_edges,
    undirected_edges,
)

ARCHITECTURES = ("gcn", "sage", "gat", "transformer", "joint", *SOTA)
LOSSES = ("standard", "diff_c", "proxy", "contrastive", "biological")


def train_architecture(
    *,
    features: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    architecture: str = "gcn",
    n_layers: int = 2,
    loss: str = "standard",
    bio: BioTargets | None = None,
    different_pairs: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
    max_epochs: int = 30,
    patience: int = 5,
    latent: int = 16,
    hidden: int = 32,
    seed: int = 0,
    lr: float = 0.05,
    bio_alpha: float = 1.0,
    bio_beta: float = 1.0,
) -> TrainResult:
    """Fit one architecture and return the early-stopped embedding."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture}")
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss}")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if loss == "biological" and bio is None:
        raise ValueError("biological loss needs Kraken targets")
    features = _zscore(features)
    n, width = features.shape
    if n < 1:
        raise ValueError("architecture training needs at least one node")
    if width == 0:
        features = np.ones((n, 1), dtype=np.float32)
        width = 1
    pairs, weights = undirected_edges(indptr, indices, edge_weight)
    if edge_weight is None:
        weights = np.ones(pairs.shape[0], dtype=np.float32)
    train_ends, train_weights, val_ends, _val_weights = split_edges(pairs, weights, seed)
    adjacency = normalized_from_edges(train_ends, train_weights, n)
    hidden = int(min(hidden, max(4, width * 2)))
    latent = int(min(latent, hidden))
    rng = np.random.default_rng(seed)
    model = _Model(
        architecture=architecture,
        n_layers=n_layers,
        width=width,
        hidden=hidden,
        latent=latent,
        n=n,
        rng=rng,
        adjacency=adjacency,
        train_ends=train_ends,
    )
    opt = _Adam(lr=lr)
    marker = _markers(different_pairs, n, rng)
    marker_weight = {"diff_c": 1.0, "proxy": 2.0}.get(loss, 0.0)
    best_val = np.inf
    best_state = model.snapshot()
    stall = 0
    epochs = 0
    last_train = np.inf

    def negatives(count: int) -> np.ndarray:
        left = rng.integers(0, n, size=max(count, 1))
        right = rng.integers(0, n, size=max(count, 1))
        right[left == right] = (right[left == right] + 1) % n
        return np.stack([left, right], axis=1)

    val_negative = negatives(max(val_ends.shape[0] * 2, 1))
    val_contrastive = (
        draw_contrastive_negatives(val_ends, n, 8, rng) if loss == "contrastive" and val_ends.shape[0] else None
    )

    def objective(embedding: np.ndarray, positive: np.ndarray, negative: np.ndarray | None, contrastive) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(embedding)
        if loss == "contrastive":
            value, grad = infonce_loss(embedding, positive, contrastive)
        else:
            value, grad = _edge_bce(embedding, positive, negative if negative is not None else negatives(positive.shape[0] * 2), marker, marker_weight)
        extra = 0.0
        if loss == "biological":
            assert bio is not None
            extra, bio_grad = biological_gradient(embedding, bio, alpha=bio_alpha, beta=bio_beta, seed=seed)
            grad = grad + bio_grad
        return value + extra, grad

    for epoch in range(max_epochs):
        opt.t = epoch + 1
        epochs = epoch + 1
        embedding = model.forward(features, rng, deterministic=False)
        train_contrastive = draw_contrastive_negatives(train_ends, n, 8, rng) if loss == "contrastive" and train_ends.shape[0] else None
        last_train, grad_z = objective(embedding, train_ends, None, train_contrastive)
        model.backward(grad_z, opt)
        val_embedding = model.forward(features, rng, deterministic=True)
        val_value, _grad = objective(
            val_embedding,
            val_ends,
            val_negative,
            val_contrastive,
        )
        if val_value + 1e-5 < best_val:
            best_val = val_value
            best_state = model.snapshot()
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    model.restore(best_state)
    final = model.forward(features, rng, deterministic=True)
    return TrainResult(
        embedding=final.astype(np.float32),
        epochs_ran=epochs,
        stopped_early=epochs < max_epochs,
        best_val_loss=float(best_val),
        train_loss=float(last_train),
    )


def _markers(different_pairs: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
    marker = np.zeros((0, 2), dtype=np.int64) if different_pairs is None else np.asarray(different_pairs, dtype=np.int64)
    if marker.size:
        marker = marker[(marker[:, 0] >= 0) & (marker[:, 1] >= 0) & (marker[:, 0] < n) & (marker[:, 1] < n)]
        if marker.shape[0] > 4096:
            marker = marker[rng.choice(marker.shape[0], size=4096, replace=False)]
    return marker


def _edge_bce(
    latent_state: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    marker: np.ndarray,
    marker_weight: float,
) -> tuple[float, np.ndarray]:
    grad = np.zeros_like(latent_state)
    if positive.shape[0] == 0:
        return 0.0, grad
    pos_logit = np.sum(latent_state[positive[:, 0]] * latent_state[positive[:, 1]], axis=1)
    neg_logit = np.sum(latent_state[negative[:, 0]] * latent_state[negative[:, 1]], axis=1)
    pos_loss, pos_grad = _bce_logits(pos_logit, 1.0)
    neg_loss, neg_grad = _bce_logits(neg_logit, 0.0)
    for pairs, pair_grad in (
        (positive, 0.5 * pos_grad / max(positive.shape[0], 1)),
        (negative, 0.5 * neg_grad / max(negative.shape[0], 1)),
    ):
        np.add.at(grad, pairs[:, 0], pair_grad[:, None] * latent_state[pairs[:, 1]])
        np.add.at(grad, pairs[:, 1], pair_grad[:, None] * latent_state[pairs[:, 0]])
    total = 0.5 * (pos_loss + neg_loss)
    extra = 0.0
    if marker_weight and marker.shape[0]:
        delta = latent_state[marker[:, 0]] - latent_state[marker[:, 1]]
        energy = np.exp(-0.5 * np.sum(delta * delta, axis=1))
        extra = float(np.mean(energy))
        coeff = energy / max(marker.shape[0], 1)
        np.add.at(grad, marker[:, 0], marker_weight * coeff[:, None] * (latent_state[marker[:, 1]] - latent_state[marker[:, 0]]))
        np.add.at(grad, marker[:, 1], marker_weight * coeff[:, None] * (latent_state[marker[:, 0]] - latent_state[marker[:, 1]]))
    return total + marker_weight * extra, grad


def _leaky(pre: np.ndarray) -> np.ndarray:
    return np.where(pre > 0, pre, 0.01 * pre).astype(np.float32)


def _leaky_grad(pre: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return np.where(pre > 0, grad, 0.01 * grad).astype(np.float32)


class _Model:
    def __init__(self, *, architecture, n_layers, width, hidden, latent, n, rng, adjacency, train_ends) -> None:
        self.architecture = architecture
        self.n_layers = n_layers
        self.adjacency = adjacency
        self.neighbors = _neighbors(train_ends, n)
        self.cache: dict = {}
        self.weights: dict[str, np.ndarray] = {}
        if architecture == "joint":
            self.weights["w_mu"] = _init(rng, width, latent)
            self.weights["w_log"] = _init(rng, width, latent)
            self.weights["w_dec"] = _init(rng, latent, width)
            self.weights["g0"] = _init(rng, latent, hidden)
            self.weights["g1"] = _init(rng, hidden, latent)
        elif architecture == "sage":
            for layer in range(n_layers):
                incoming = width if layer == 0 else hidden
                self.weights[f"self{layer}"] = _init(rng, incoming, hidden)
                self.weights[f"neigh{layer}"] = _init(rng, incoming, hidden)
            self.weights["out"] = _init(rng, hidden, latent)
        elif architecture == "gat":
            for layer in range(n_layers):
                incoming = width if layer == 0 else hidden
                self.weights[f"w{layer}"] = _init(rng, incoming, hidden)
                self.weights[f"a_src{layer}"] = _init(rng, hidden, 1).ravel()
                self.weights[f"a_dst{layer}"] = _init(rng, hidden, 1).ravel()
            self.weights["out"] = _init(rng, hidden, latent)
        elif architecture == "transformer":
            self.weights["wq"] = _init(rng, width, hidden)
            self.weights["wk"] = _init(rng, width, hidden)
            self.weights["wv"] = _init(rng, width, hidden)
            self.weights["wr"] = _init(rng, width, hidden)
            self.weights["gcn"] = _init(rng, hidden, hidden)
            self.weights["out"] = _init(rng, hidden, latent)
        elif architecture in SOTA:
            self._sota = SotaModel(architecture, width, hidden, latent, adjacency, rng, _init)
            self.weights = self._sota.weights
        else:
            for layer in range(n_layers):
                incoming = width if layer == 0 else hidden
                outgoing = latent if layer == n_layers - 1 else hidden
                self.weights[f"w{layer}"] = _init(rng, incoming, outgoing)

    def snapshot(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.weights.items()}

    def restore(self, state: dict[str, np.ndarray]) -> None:
        self.weights = {name: value.copy() for name, value in state.items()}

    def forward(self, features: np.ndarray, rng: np.random.Generator, *, deterministic: bool) -> np.ndarray:
        if self.architecture == "joint":
            return self._joint(features, rng, deterministic)
        if self.architecture == "sage":
            return self._sage(features)
        if self.architecture == "gat":
            return self._gat(features)
        if self.architecture == "transformer":
            return self._transformer(features)
        if self.architecture in SOTA:
            return self._sota.forward(features)
        return self._gcn(features)

    def backward(self, grad_out: np.ndarray, opt: _Adam) -> None:
        if self.architecture == "joint":
            self._joint_backward(grad_out, opt)
        elif self.architecture == "sage":
            self._sage_backward(grad_out, opt)
        elif self.architecture == "gat":
            self._gat_backward(grad_out, opt)
        elif self.architecture == "transformer":
            self._transformer_backward(grad_out, opt)
        elif self.architecture in SOTA:
            self._sota.backward(grad_out, opt)
        else:
            self._gcn_backward(grad_out, opt)

    def _gcn(self, features: np.ndarray) -> np.ndarray:
        hidden_state = features
        layers = []
        for layer in range(self.n_layers):
            weight = self.weights[f"w{layer}"]
            pre = np.asarray(self.adjacency @ (hidden_state @ weight), dtype=np.float32)
            activated = pre if layer == self.n_layers - 1 else _leaky(pre)
            layers.append((hidden_state, pre, activated))
            hidden_state = activated
        self.cache = {"layers": layers}
        return hidden_state

    def _gcn_backward(self, grad_out: np.ndarray, opt: _Adam) -> None:
        grad = grad_out
        for layer in range(self.n_layers - 1, -1, -1):
            incoming, pre, _activated = self.cache["layers"][layer]
            weight = self.weights[f"w{layer}"]
            if layer != self.n_layers - 1:
                grad = _leaky_grad(pre, grad)
            grad_u = np.asarray(self.adjacency.T @ grad, dtype=np.float32)
            opt.step(f"w{layer}", weight, incoming.T @ grad_u)
            grad = grad_u @ weight.T

    def _sage(self, features: np.ndarray) -> np.ndarray:
        hidden_state = features
        layers = []
        for layer in range(self.n_layers):
            self_w = self.weights[f"self{layer}"]
            neigh_w = self.weights[f"neigh{layer}"]
            own = hidden_state @ self_w
            neigh = np.asarray(self.adjacency @ (hidden_state @ neigh_w), dtype=np.float32)
            pre = own + neigh
            activated = _leaky(pre)
            layers.append((hidden_state, pre, activated))
            hidden_state = activated
        self.cache = {"layers": layers, "hidden": hidden_state}
        return hidden_state @ self.weights["out"]

    def _sage_backward(self, grad_out: np.ndarray, opt: _Adam) -> None:
        hidden_state = self.cache["hidden"]
        opt.step("out", self.weights["out"], hidden_state.T @ grad_out)
        grad = grad_out @ self.weights["out"].T
        for layer in range(self.n_layers - 1, -1, -1):
            incoming, pre, _activated = self.cache["layers"][layer]
            grad = _leaky_grad(pre, grad)
            self_w = self.weights[f"self{layer}"]
            neigh_w = self.weights[f"neigh{layer}"]
            opt.step(f"self{layer}", self_w, incoming.T @ grad)
            neigh_in = np.asarray(self.adjacency.T @ grad, dtype=np.float32)
            opt.step(f"neigh{layer}", neigh_w, incoming.T @ neigh_in)
            grad = grad @ self_w.T + neigh_in @ neigh_w.T

    def _gat(self, features: np.ndarray) -> np.ndarray:
        hidden_state = features
        layers = []
        for layer in range(self.n_layers):
            projected = hidden_state @ self.weights[f"w{layer}"]
            activated, alpha = _gat_aggregate(projected, self.neighbors, self.weights[f"a_src{layer}"], self.weights[f"a_dst{layer}"])
            layers.append((hidden_state, projected, activated, alpha))
            hidden_state = activated
        self.cache = {"layers": layers, "hidden": hidden_state}
        return hidden_state @ self.weights["out"]

    def _gat_backward(self, grad_out: np.ndarray, opt: _Adam) -> None:
        hidden_state = self.cache["hidden"]
        opt.step("out", self.weights["out"], hidden_state.T @ grad_out)
        grad = grad_out @ self.weights["out"].T
        for layer in range(self.n_layers - 1, -1, -1):
            incoming, projected, activated, alpha = self.cache["layers"][layer]
            grad_h, grad_src, grad_dst = _gat_backward_layer(
                projected, activated, alpha, grad, self.neighbors, self.weights[f"a_src{layer}"], self.weights[f"a_dst{layer}"]
            )
            opt.step(f"a_src{layer}", self.weights[f"a_src{layer}"], grad_src)
            opt.step(f"a_dst{layer}", self.weights[f"a_dst{layer}"], grad_dst)
            weight = self.weights[f"w{layer}"]
            opt.step(f"w{layer}", weight, incoming.T @ grad_h)
            grad = grad_h @ weight.T

    def _transformer(self, features: np.ndarray) -> np.ndarray:
        query = features @ self.weights["wq"]
        key = features @ self.weights["wk"]
        value = features @ self.weights["wv"]
        scale = np.sqrt(query.shape[1])
        scores = (query @ key.T) / scale
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        attention = weights / weights.sum(axis=1, keepdims=True)
        attended = attention @ value
        residual = attended + features @ self.weights["wr"]
        pre = np.asarray(self.adjacency @ (residual @ self.weights["gcn"]), dtype=np.float32)
        hidden_state = _leaky(pre)
        self.cache = {
            "features": features,
            "query": query,
            "key": key,
            "value": value,
            "attention": attention,
            "attended": attended,
            "residual": residual,
            "pre": pre,
            "hidden": hidden_state,
        }
        return hidden_state @ self.weights["out"]

    def _transformer_backward(self, grad_out: np.ndarray, opt: _Adam) -> None:
        cache = self.cache
        opt.step("out", self.weights["out"], cache["hidden"].T @ grad_out)
        grad_h = _leaky_grad(cache["pre"], grad_out @ self.weights["out"].T)
        grad_u = np.asarray(self.adjacency.T @ grad_h, dtype=np.float32)
        opt.step("gcn", self.weights["gcn"], cache["residual"].T @ grad_u)
        grad_r = grad_u @ self.weights["gcn"].T
        opt.step("wr", self.weights["wr"], cache["features"].T @ grad_r)
        grad_attended = grad_r
        attention = cache["attention"]
        value = cache["value"]
        grad_value = attention.T @ grad_attended
        grad_attention = grad_attended @ value.T
        grad_scores = attention * (grad_attention - np.sum(attention * grad_attention, axis=1, keepdims=True))
        scale = np.sqrt(cache["query"].shape[1])
        grad_scores = grad_scores / scale
        grad_query = grad_scores @ cache["key"]
        grad_key = grad_scores.T @ cache["query"]
        features = cache["features"]
        opt.step("wq", self.weights["wq"], features.T @ grad_query)
        opt.step("wk", self.weights["wk"], features.T @ grad_key)
        opt.step("wv", self.weights["wv"], features.T @ grad_value)

    def _joint(self, features: np.ndarray, rng: np.random.Generator, deterministic: bool) -> np.ndarray:
        mu = features @ self.weights["w_mu"]
        logvar = np.clip(features @ self.weights["w_log"], -8.0, 8.0)
        if deterministic:
            sample = mu
        else:
            sample = mu + rng.normal(0.0, 1.0, size=mu.shape).astype(np.float32) * np.exp(0.5 * logvar)
        recon = sample @ self.weights["w_dec"]
        pre_h = np.asarray(self.adjacency @ (sample @ self.weights["g0"]), dtype=np.float32)
        hidden_state = _leaky(pre_h)
        gcn = np.asarray(self.adjacency @ (hidden_state @ self.weights["g1"]), dtype=np.float32)
        self.cache = {
            "features": features,
            "mu": mu,
            "logvar": logvar,
            "sample": sample,
            "recon": recon,
            "pre_h": pre_h,
            "hidden": hidden_state,
            "gcn": gcn,
        }
        return np.concatenate([mu, gcn], axis=1).astype(np.float32)

    def _joint_backward(self, grad_out: np.ndarray, opt: _Adam) -> None:
        cache = self.cache
        latent = self.weights["g1"].shape[1]
        grad_mu = grad_out[:, :latent]
        grad_gcn = grad_out[:, latent:]
        features = cache["features"]
        sample = cache["sample"]
        recon = cache["recon"]
        residual = (recon - features) / max(features.shape[0], 1)
        grad_sample = residual @ self.weights["w_dec"].T
        opt.step("w_dec", self.weights["w_dec"], sample.T @ residual)
        grad_u = np.asarray(self.adjacency.T @ grad_gcn, dtype=np.float32)
        opt.step("g1", self.weights["g1"], cache["hidden"].T @ grad_u)
        grad_h = _leaky_grad(cache["pre_h"], grad_u @ self.weights["g1"].T)
        grad_s = np.asarray(self.adjacency.T @ grad_h, dtype=np.float32)
        opt.step("g0", self.weights["g0"], sample.T @ grad_s)
        grad_sample = grad_sample + grad_s @ self.weights["g0"].T
        # Reparameterization: sample = mu + eps * exp(0.5 logvar). At eval eps is 0,
        # so training uses the stored sample and sends the full sample gradient to mu.
        grad_mu = grad_mu + grad_sample
        grad_log = grad_sample * (sample - cache["mu"])
        kl_mu = cache["mu"] / max(features.shape[0], 1)
        kl_log = 0.5 * (np.exp(cache["logvar"]) - 1.0) / max(features.shape[0], 1)
        grad_mu = grad_mu + 0.01 * kl_mu
        grad_log = grad_log + 0.01 * kl_log
        opt.step("w_mu", self.weights["w_mu"], features.T @ grad_mu)
        opt.step("w_log", self.weights["w_log"], features.T @ grad_log)


def _neighbors(pairs: np.ndarray, n: int) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(n)]
    for left, right in np.asarray(pairs, dtype=np.int64):
        buckets[int(left)].append(int(right))
        buckets[int(right)].append(int(left))
    return [np.asarray(bucket + [index], dtype=np.int64) for index, bucket in enumerate(buckets)]


def _gat_aggregate(projected, neighbors, a_src, a_dst):
    n, width = projected.shape
    out = np.zeros_like(projected)
    alpha = []
    src_score = projected @ a_src
    dst_score = projected @ a_dst
    for node in range(n):
        neigh = neighbors[node]
        logits = src_score[node] + dst_score[neigh]
        logits = np.where(logits > 0, logits, 0.01 * logits)
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        out[node] = weights @ projected[neigh]
        alpha.append(weights.astype(np.float32))
    return out.astype(np.float32), alpha


def _gat_backward_layer(projected, activated, alpha, grad_out, neighbors, a_src, a_dst):
    grad_h = np.zeros_like(projected)
    grad_src = np.zeros_like(a_src)
    grad_dst = np.zeros_like(a_dst)
    src_score = projected @ a_src
    dst_score = projected @ a_dst
    for node, neigh in enumerate(neighbors):
        weights = alpha[node]
        grad_h[neigh] += weights[:, None] * grad_out[node]
        message = projected[neigh] - activated[node]
        d_logit = weights * (message @ grad_out[node])
        raw = src_score[node] + dst_score[neigh]
        d_raw = np.where(raw > 0, d_logit, 0.01 * d_logit)
        grad_src += d_raw.sum() * projected[node]
        grad_dst += d_raw @ projected[neigh]
        grad_h[node] += a_src * d_raw.sum()
        grad_h[neigh] += a_dst * d_raw[:, None]
    return grad_h.astype(np.float32), grad_src.astype(np.float32), grad_dst.astype(np.float32)
