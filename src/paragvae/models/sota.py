"""Six encoders from the May 2026 architecture sweep.

Each one ends in a dense map to the latent and is trained by the same
early-stopped edge loss as the GCN. They read the normalized adjacency of
the coloured graph tensor. None of them reads evaluation labels.

- ``mixhop``: identity, one-hop, and two-hop neighbourhoods, concatenated.
- ``jknet``: every GCN layer is kept and concatenated.
- ``gps``: a local GCN block plus one global attention block.
- ``unet``: a two-level encoder with a skip into the decoder.
- ``han``: the adjacency and its square, mixed by a learned gate.
- ``diffpool``: a soft assignment into a few clusters, a coarse GCN, then an unpool.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

SOTA = ("mixhop", "jknet", "gps", "unet", "han", "diffpool")


class SotaModel:
    """One of the six encoders. ``weights`` is the dict the trainer snapshots."""

    def __init__(self, architecture: str, width: int, hidden: int, latent: int, adjacency, rng, init) -> None:
        if architecture not in SOTA:
            raise ValueError(f"unknown sota architecture {architecture}")
        self.architecture = architecture
        self.adjacency = adjacency.tocsr().astype(np.float32)
        self.hidden = hidden
        self.latent = latent
        self.weights: dict[str, np.ndarray] = {}
        self.cache: dict = {}
        two = self.adjacency @ self.adjacency
        self.hops = (sparse.eye(adjacency.shape[0], format="csr", dtype=np.float32), self.adjacency, two.tocsr())
        if architecture == "mixhop":
            for hop in range(3):
                self.weights[f"h{hop}"] = init(rng, width, hidden)
            self.weights["out"] = init(rng, hidden * 3, latent)
        elif architecture == "jknet":
            self.weights["w0"] = init(rng, width, hidden)
            self.weights["w1"] = init(rng, hidden, hidden)
            self.weights["out"] = init(rng, hidden * 2, latent)
        elif architecture == "gps":
            self.weights["local"] = init(rng, width, hidden)
            self.weights["wq"] = init(rng, width, hidden)
            self.weights["wk"] = init(rng, width, hidden)
            self.weights["wv"] = init(rng, width, hidden)
            self.weights["out"] = init(rng, hidden * 2, latent)
        elif architecture == "unet":
            self.weights["w1"] = init(rng, width, hidden)
            self.weights["w2"] = init(rng, hidden, hidden)
            self.weights["up"] = init(rng, hidden * 2, hidden)
            self.weights["out"] = init(rng, hidden, latent)
        elif architecture == "han":
            self.weights["w1"] = init(rng, width, hidden)
            self.weights["w2"] = init(rng, width, hidden)
            self.weights["gate"] = init(rng, hidden * 2, 2)
            self.weights["out"] = init(rng, hidden, latent)
        else:
            clusters = int(min(4, adjacency.shape[0]))
            self.clusters = clusters
            self.weights["assign"] = init(rng, width, clusters)
            self.weights["coarse"] = init(rng, width, hidden)
            self.weights["fine"] = init(rng, width, hidden)
            self.weights["out"] = init(rng, hidden * 2, latent)

    def forward(self, features: np.ndarray) -> np.ndarray:
        name = self.architecture
        if name == "mixhop":
            return self._mixhop(features)
        if name == "jknet":
            return self._jknet(features)
        if name == "gps":
            return self._gps(features)
        if name == "unet":
            return self._unet(features)
        if name == "han":
            return self._han(features)
        return self._diffpool(features)

    def backward(self, grad_out: np.ndarray, opt) -> None:
        name = self.architecture
        if name == "mixhop":
            self._mixhop_backward(grad_out, opt)
        elif name == "jknet":
            self._jknet_backward(grad_out, opt)
        elif name == "gps":
            self._gps_backward(grad_out, opt)
        elif name == "unet":
            self._unet_backward(grad_out, opt)
        elif name == "han":
            self._han_backward(grad_out, opt)
        else:
            self._diffpool_backward(grad_out, opt)

    def _mixhop(self, features: np.ndarray) -> np.ndarray:
        parts = []
        stored = []
        for hop, operator in enumerate(self.hops):
            pre = np.asarray(operator @ (features @ self.weights[f"h{hop}"]), dtype=np.float32)
            activated = np.where(pre > 0, pre, 0.01 * pre).astype(np.float32)
            parts.append(activated)
            stored.append(pre)
        hidden = np.concatenate(parts, axis=1)
        self.cache = {"features": features, "pre": stored, "hidden": hidden}
        return hidden @ self.weights["out"]

    def _mixhop_backward(self, grad_out: np.ndarray, opt) -> None:
        hidden = self.cache["hidden"]
        opt.step("out", self.weights["out"], hidden.T @ grad_out)
        grad_h = grad_out @ self.weights["out"].T
        width = self.weights["h0"].shape[1]
        features = self.cache["features"]
        for hop, operator in enumerate(self.hops):
            grad = grad_h[:, hop * width : (hop + 1) * width]
            pre = self.cache["pre"][hop]
            grad = np.where(pre > 0, grad, 0.01 * grad).astype(np.float32)
            grad_x = np.asarray(operator.T @ grad, dtype=np.float32)
            opt.step(f"h{hop}", self.weights[f"h{hop}"], features.T @ grad_x)

    def _jknet(self, features: np.ndarray) -> np.ndarray:
        pre0 = np.asarray(self.adjacency @ (features @ self.weights["w0"]), dtype=np.float32)
        h0 = np.where(pre0 > 0, pre0, 0.01 * pre0).astype(np.float32)
        pre1 = np.asarray(self.adjacency @ (h0 @ self.weights["w1"]), dtype=np.float32)
        h1 = np.where(pre1 > 0, pre1, 0.01 * pre1).astype(np.float32)
        hidden = np.concatenate([h0, h1], axis=1)
        self.cache = {"features": features, "pre0": pre0, "h0": h0, "pre1": pre1, "hidden": hidden}
        return hidden @ self.weights["out"]

    def _jknet_backward(self, grad_out: np.ndarray, opt) -> None:
        opt.step("out", self.weights["out"], self.cache["hidden"].T @ grad_out)
        grad = grad_out @ self.weights["out"].T
        width = self.weights["w0"].shape[1]
        grad0, grad1 = grad[:, :width], grad[:, width:]
        grad1 = np.where(self.cache["pre1"] > 0, grad1, 0.01 * grad1).astype(np.float32)
        back1 = np.asarray(self.adjacency.T @ grad1, dtype=np.float32)
        opt.step("w1", self.weights["w1"], self.cache["h0"].T @ back1)
        grad0 = grad0 + back1 @ self.weights["w1"].T
        grad0 = np.where(self.cache["pre0"] > 0, grad0, 0.01 * grad0).astype(np.float32)
        back0 = np.asarray(self.adjacency.T @ grad0, dtype=np.float32)
        opt.step("w0", self.weights["w0"], self.cache["features"].T @ back0)

    def _gps(self, features: np.ndarray) -> np.ndarray:
        local_pre = np.asarray(self.adjacency @ (features @ self.weights["local"]), dtype=np.float32)
        local = np.where(local_pre > 0, local_pre, 0.01 * local_pre).astype(np.float32)
        query = features @ self.weights["wq"]
        key = features @ self.weights["wk"]
        value = features @ self.weights["wv"]
        scale = np.sqrt(query.shape[1])
        scores = (query @ key.T) / scale
        scores = scores - scores.max(axis=1, keepdims=True)
        attention = np.exp(scores)
        attention = attention / attention.sum(axis=1, keepdims=True)
        attended = (attention @ value).astype(np.float32)
        hidden = np.concatenate([local, attended], axis=1)
        self.cache = {
            "features": features,
            "local_pre": local_pre,
            "query": query,
            "key": key,
            "value": value,
            "attention": attention,
            "hidden": hidden,
        }
        return hidden @ self.weights["out"]

    def _gps_backward(self, grad_out: np.ndarray, opt) -> None:
        cache = self.cache
        opt.step("out", self.weights["out"], cache["hidden"].T @ grad_out)
        grad = grad_out @ self.weights["out"].T
        width = self.weights["local"].shape[1]
        grad_local, grad_att = grad[:, :width], grad[:, width:]
        grad_local = np.where(cache["local_pre"] > 0, grad_local, 0.01 * grad_local).astype(np.float32)
        back = np.asarray(self.adjacency.T @ grad_local, dtype=np.float32)
        opt.step("local", self.weights["local"], cache["features"].T @ back)
        attention = cache["attention"]
        value = cache["value"]
        grad_value = attention.T @ grad_att
        grad_attention = grad_att @ value.T
        grad_scores = attention * (grad_attention - np.sum(attention * grad_attention, axis=1, keepdims=True))
        grad_scores = grad_scores / np.sqrt(cache["query"].shape[1])
        features = cache["features"]
        opt.step("wq", self.weights["wq"], features.T @ (grad_scores @ cache["key"]))
        opt.step("wk", self.weights["wk"], features.T @ (grad_scores.T @ cache["query"]))
        opt.step("wv", self.weights["wv"], features.T @ grad_value)

    def _unet(self, features: np.ndarray) -> np.ndarray:
        pre1 = np.asarray(self.adjacency @ (features @ self.weights["w1"]), dtype=np.float32)
        h1 = np.where(pre1 > 0, pre1, 0.01 * pre1).astype(np.float32)
        pre2 = np.asarray(self.adjacency @ (h1 @ self.weights["w2"]), dtype=np.float32)
        h2 = np.where(pre2 > 0, pre2, 0.01 * pre2).astype(np.float32)
        skip = np.concatenate([h2, h1], axis=1)
        pre_up = np.asarray(self.adjacency @ (skip @ self.weights["up"]), dtype=np.float32)
        hidden = np.where(pre_up > 0, pre_up, 0.01 * pre_up).astype(np.float32)
        self.cache = {"features": features, "pre1": pre1, "h1": h1, "pre2": pre2, "h2": h2, "skip": skip, "pre_up": pre_up}
        return hidden @ self.weights["out"]

    def _unet_backward(self, grad_out: np.ndarray, opt) -> None:
        cache = self.cache
        hidden = np.where(cache["pre_up"] > 0, cache["pre_up"], 0.01 * cache["pre_up"]).astype(np.float32)
        opt.step("out", self.weights["out"], hidden.T @ grad_out)
        grad = np.where(cache["pre_up"] > 0, grad_out @ self.weights["out"].T, 0.01 * (grad_out @ self.weights["out"].T))
        grad = grad.astype(np.float32)
        back = np.asarray(self.adjacency.T @ grad, dtype=np.float32)
        opt.step("up", self.weights["up"], cache["skip"].T @ back)
        grad_skip = back @ self.weights["up"].T
        width = cache["h2"].shape[1]
        grad2 = np.where(cache["pre2"] > 0, grad_skip[:, :width], 0.01 * grad_skip[:, :width]).astype(np.float32)
        grad1 = grad_skip[:, width:]
        back2 = np.asarray(self.adjacency.T @ grad2, dtype=np.float32)
        opt.step("w2", self.weights["w2"], cache["h1"].T @ back2)
        grad1 = grad1 + back2 @ self.weights["w2"].T
        grad1 = np.where(cache["pre1"] > 0, grad1, 0.01 * grad1).astype(np.float32)
        back1 = np.asarray(self.adjacency.T @ grad1, dtype=np.float32)
        opt.step("w1", self.weights["w1"], cache["features"].T @ back1)

    def _han(self, features: np.ndarray) -> np.ndarray:
        pre1 = np.asarray(self.adjacency @ (features @ self.weights["w1"]), dtype=np.float32)
        h1 = np.where(pre1 > 0, pre1, 0.01 * pre1).astype(np.float32)
        pre2 = np.asarray(self.hops[2] @ (features @ self.weights["w2"]), dtype=np.float32)
        h2 = np.where(pre2 > 0, pre2, 0.01 * pre2).astype(np.float32)
        stacked = np.concatenate([h1, h2], axis=1)
        gate_logit = stacked @ self.weights["gate"]
        gate_logit = gate_logit - gate_logit.max(axis=1, keepdims=True)
        gate = np.exp(gate_logit)
        gate = gate / gate.sum(axis=1, keepdims=True)
        hidden = (gate[:, :1] * h1 + gate[:, 1:] * h2).astype(np.float32)
        self.cache = {"features": features, "pre1": pre1, "h1": h1, "pre2": pre2, "h2": h2, "gate": gate, "hidden": hidden}
        return hidden @ self.weights["out"]

    def _han_backward(self, grad_out: np.ndarray, opt) -> None:
        cache = self.cache
        opt.step("out", self.weights["out"], cache["hidden"].T @ grad_out)
        grad_h = grad_out @ self.weights["out"].T
        gate = cache["gate"]
        # d(mix)/d(gate) and d(mix)/d(h)
        grad1 = gate[:, :1] * grad_h
        grad2 = gate[:, 1:] * grad_h
        mixed_diff = np.concatenate([cache["h1"], cache["h2"]], axis=1)
        # gate is softmax of stacked @ gate_w. Approximate the gate gradient
        # by the standard softmax Jacobian against (h * grad_h) summed.
        signal = np.concatenate([
            np.sum(cache["h1"] * grad_h, axis=1, keepdims=True),
            np.sum(cache["h2"] * grad_h, axis=1, keepdims=True),
        ], axis=1)
        grad_logit = gate * (signal - np.sum(gate * signal, axis=1, keepdims=True))
        stacked = np.concatenate([cache["h1"], cache["h2"]], axis=1)
        opt.step("gate", self.weights["gate"], stacked.T @ grad_logit)
        grad_stack = grad_logit @ self.weights["gate"].T
        width = cache["h1"].shape[1]
        grad1 = grad1 + grad_stack[:, :width]
        grad2 = grad2 + grad_stack[:, width:]
        grad1 = np.where(cache["pre1"] > 0, grad1, 0.01 * grad1).astype(np.float32)
        grad2 = np.where(cache["pre2"] > 0, grad2, 0.01 * grad2).astype(np.float32)
        back1 = np.asarray(self.adjacency.T @ grad1, dtype=np.float32)
        back2 = np.asarray(self.hops[2].T @ grad2, dtype=np.float32)
        opt.step("w1", self.weights["w1"], cache["features"].T @ back1)
        opt.step("w2", self.weights["w2"], cache["features"].T @ back2)
        del mixed_diff

    def _diffpool(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights["assign"]
        logits = logits - logits.max(axis=1, keepdims=True)
        assign = np.exp(logits)
        assign = assign / assign.sum(axis=1, keepdims=True)
        pooled = np.asarray(self.adjacency @ assign, dtype=np.float32)
        coarse_adj = assign.T @ pooled
        coarse_x = assign.T @ features
        pre = coarse_adj @ (coarse_x @ self.weights["coarse"])
        coarse = np.where(pre > 0, pre, 0.01 * pre).astype(np.float32)
        up = (assign @ coarse).astype(np.float32)
        fine_pre = np.asarray(self.adjacency @ (features @ self.weights["fine"]), dtype=np.float32)
        fine = np.where(fine_pre > 0, fine_pre, 0.01 * fine_pre).astype(np.float32)
        hidden = np.concatenate([fine, up], axis=1)
        self.cache = {
            "features": features,
            "assign": assign,
            "coarse_adj": coarse_adj,
            "coarse_x": coarse_x,
            "pre": pre,
            "coarse": coarse,
            "fine_pre": fine_pre,
            "hidden": hidden,
        }
        return hidden @ self.weights["out"]

    def _diffpool_backward(self, grad_out: np.ndarray, opt) -> None:
        cache = self.cache
        opt.step("out", self.weights["out"], cache["hidden"].T @ grad_out)
        grad = grad_out @ self.weights["out"].T
        width = self.weights["fine"].shape[1]
        grad_fine = np.where(cache["fine_pre"] > 0, grad[:, :width], 0.01 * grad[:, :width]).astype(np.float32)
        grad_up = grad[:, width:]
        back = np.asarray(self.adjacency.T @ grad_fine, dtype=np.float32)
        opt.step("fine", self.weights["fine"], cache["features"].T @ back)
        grad_coarse = cache["assign"].T @ grad_up
        grad_pre = np.where(cache["pre"] > 0, grad_coarse, 0.01 * grad_coarse).astype(np.float32)
        opt.step("coarse", self.weights["coarse"], cache["coarse_x"].T @ (cache["coarse_adj"].T @ grad_pre))
        # Assignment receives gradient through the unpool only.
        grad_assign = grad_up @ cache["coarse"].T
        assign = cache["assign"]
        grad_logit = assign * (grad_assign - np.sum(assign * grad_assign, axis=1, keepdims=True))
        opt.step("assign", self.weights["assign"], cache["features"].T @ grad_logit)
