"""Probabilistic prediction heads (design/models/probabilistic_heads.md).

A DistributionHead parameterizes an EXPLICIT distribution over the next latent from the spine context,
which is sampled and decoded. It is the swappable-policy analogue of collapse.py's CollapseStrategy, but --
unlike a collapse strategy -- it OWNS parameters (posterior/prior nets, embedding). It is therefore an
nn.Module held by the model as `self.dist_head`, so its params serialize under `dist_head.*` and the
missing-parameter tripwire in load_checkpoint keeps working (the resolution the design calls for).

Three concrete heads differ ONLY in (a) output params, (b) sampler, (c) loss term:
  CategoricalHead  DreamerV3 G x K one-hot groups, unimix floor, straight-through, balanced KL + free bits.
  GaussianHead     diagonal Gaussian; loss='kl' (stochastic latent, PlaNet/Dreamer) or 'nll' (deterministic
                   latent, Ward-2026: nll + latent_mse + kl-to-prior).
All are mutually exclusive with the flow head (one prediction mechanism per model).

Params are represented as plain dicts of tensors with the TIME axis at dim 1 (so loss_terms can slice
`{k: v[:, 1:]}` to align the posterior of the true-next frame with the prior that predicts it). The token
axis is the SECOND-to-last (n_state); the distribution's own axes (G,K for categorical; d for Gaussian) are
trailing."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _SpatialFusion(nn.Module):
    """Within-frame fusion: ONE bidirectional self-attention block over the n_state tokens of a single
    frame (no temporal context), so every modality token attends to every other -- the FUSED single-frame
    observation q(z_t | o_t) conditions on. Purely a function of o_t (per-frame, context-free), so the
    shared-encode + KV-cache paths are preserved. n_state==1 (proprio-only) -> attention over one token,
    a harmless near-identity."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        h = heads if d % heads == 0 else 1
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, feats: Tensor) -> Tensor:                    # (..., n_state, d) -> same
        lead = feats.shape[:-2]
        n, d = feats.shape[-2], feats.shape[-1]
        x = feats.reshape(-1, n, d)
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x.reshape(*lead, n, d)


class DistributionHead(nn.Module):
    """Policy object the model holds. Subclasses build their nets in `build(d, n_state)` (called by the model
    after super().__init__, so `d`/`n_state` are known) and implement posterior/prior/sample/losses."""

    needs_posterior: bool = False   # False -> encoder untouched (Gaussian-NLL / Dirac posterior)
    is_discrete: bool = False

    def build(self, d: int, n_state: int) -> None:
        raise NotImplementedError

    def posterior(self, feats: Tensor) -> dict:                    # per-token params from encoder features
        raise NotImplementedError

    def prior(self, cond: Tensor) -> dict:                         # per-token params from _cond(h)
        raise NotImplementedError

    def sample(self, params: dict, deterministic: bool = False) -> Tensor:   # -> bag (..., n_state, d)
        raise NotImplementedError

    def losses(self, post: dict | None, prior: dict, target: Tensor | None) -> tuple[dict, dict]:
        raise NotImplementedError

    def diagnostics(self, post: dict | None, prior: dict) -> dict:
        return {}

    def uncertainty(self, prior_params: dict) -> Tensor:          # Ward score (mean sigma / mean entropy)
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
class CategoricalHead(DistributionHead):
    """DreamerV3 categorical latent: G groups of K classes per token. unimix floor, straight-through
    one-hot, KL-balanced (dyn/rep) with a free-bits floor. The carried bag is embed(one-hot)."""

    needs_posterior = True
    is_discrete = True

    def __init__(self, groups: int = 16, classes: int = 16, unimix: float = 0.01, dyn_scale: float = 1.0,
                 rep_scale: float = 0.1, free_nats: float = 1.0, straight_through: bool = True):
        super().__init__()
        self.G, self.K = int(groups), int(classes)
        self.unimix = float(unimix)
        self.dyn_scale, self.rep_scale = float(dyn_scale), float(rep_scale)
        self.free_nats = float(free_nats)
        self.straight_through = bool(straight_through)

    def build(self, d: int, n_state: int) -> None:
        gk = self.G * self.K
        self.fusion = _SpatialFusion(d)
        self.post_net = nn.Linear(d, gk)                          # fused features -> posterior logits
        self.prior_net = nn.Linear(d, gk)                         # _cond(h)       -> prior logits
        self.embed = nn.Linear(gk, d)                             # one-hot sample -> d-dim carried token

    def _params(self, logits: Tensor) -> dict:
        logits = logits.reshape(*logits.shape[:-1], self.G, self.K)
        probs = (1.0 - self.unimix) * F.softmax(logits, dim=-1) + self.unimix / self.K   # unimix floor
        return {"logits": logits, "probs": probs}

    def posterior(self, feats: Tensor) -> dict:
        return self._params(self.post_net(self.fusion(feats)))

    def prior(self, cond: Tensor) -> dict:
        return self._params(self.prior_net(cond))

    def sample(self, params: dict, deterministic: bool = False) -> Tensor:
        probs = params["probs"]                                   # (..., G, K)
        if deterministic:
            idx = probs.argmax(dim=-1)
        else:
            idx = torch.multinomial(probs.reshape(-1, self.K), 1).reshape(probs.shape[:-1])
        onehot = F.one_hot(idx, self.K).to(probs.dtype)           # (..., G, K)
        if self.straight_through:
            onehot = onehot + probs - probs.detach()              # value == onehot; grad flows to probs
        return self.embed(onehot.reshape(*onehot.shape[:-2], self.G * self.K))

    @staticmethod
    def _kl(pa: Tensor, pb: Tensor) -> Tensor:
        """KL(a || b) of factorized categoricals: sum over classes then groups -> (...,). unimix keeps both
        strictly positive, so the logs are finite."""
        kl = (pa * (pa.clamp_min(1e-8).log() - pb.clamp_min(1e-8).log())).sum(dim=-1)   # over K -> (...,G)
        return kl.sum(dim=-1)                                     # over G -> (...,)

    def losses(self, post: dict, prior: dict, target: Tensor | None = None) -> tuple[dict, dict]:
        q, p = post["probs"], prior["probs"]
        dyn = self._kl(q.detach(), p).clamp_min(self.free_nats).mean()    # train prior toward posterior
        rep = self._kl(q, p.detach()).clamp_min(self.free_nats).mean()    # train posterior toward prior
        return ({"kl/dyn": dyn, "kl/rep": rep},
                {"kl/dyn": self.dyn_scale, "kl/rep": self.rep_scale})

    def diagnostics(self, post: dict | None, prior: dict) -> dict:
        out = {}
        p = prior["probs"]
        ent_p = -(p * p.clamp_min(1e-8).log()).sum(-1).mean()     # mean prior entropy (nats/group)
        out["dist/prior_entropy"] = ent_p
        if post is not None:
            q = post["probs"]
            out["dist/posterior_entropy"] = -(q * q.clamp_min(1e-8).log()).sum(-1).mean()
            out["dist/kl_floor_frac"] = (self._kl(q.detach(), prior["probs"]) < self.free_nats).float().mean()
        # code perplexity: how many classes the marginal actually uses (collapse detector)
        marg = p.reshape(-1, self.G, self.K).mean(0)
        out["dist/perplexity"] = (-(marg * marg.clamp_min(1e-8).log()).sum(-1)).exp().mean()
        return out

    def uncertainty(self, prior_params: dict) -> Tensor:
        p = prior_params["probs"]
        return -(p * p.clamp_min(1e-8).log()).sum(-1).mean(-1)    # mean prior entropy over groups (..., n_state)


# ---------------------------------------------------------------------------------------------------------
class GaussianHead(DistributionHead):
    """Diagonal Gaussian over the latent. loss='kl' -> stochastic latent + KL(post||prior) (PlaNet/Dreamer);
    loss='nll' -> deterministic latent, no posterior, prior trained by the likelihood of the true next latent
    plus Ward-2026 stabilizers (latent_mse + KL-to-N(0,I))."""

    def __init__(self, loss: str = "nll", min_std: float = 0.1, max_std: float | None = None,
                 dyn_scale: float = 1.0, rep_scale: float = 0.1, free_nats: float = 1.0,
                 kl_prior_scale: float = 0.05, latent_mse_scale: float = 2.0):
        super().__init__()
        if loss not in ("nll", "kl"):
            raise ValueError(f"GaussianHead loss must be 'nll' or 'kl', got {loss!r}")
        self.loss_kind = loss
        self.needs_posterior = loss == "kl"
        self.is_discrete = False
        self.min_std, self.max_std = float(min_std), (None if max_std is None else float(max_std))
        self.dyn_scale, self.rep_scale, self.free_nats = float(dyn_scale), float(rep_scale), float(free_nats)
        self.kl_prior_scale, self.latent_mse_scale = float(kl_prior_scale), float(latent_mse_scale)

    def build(self, d: int, n_state: int) -> None:
        self.prior_mean = nn.Linear(d, d)
        self.prior_std = nn.Linear(d, d)
        nn.init.zeros_(self.prior_std.weight); nn.init.zeros_(self.prior_std.bias)   # std starts constant
        if self.needs_posterior:
            self.fusion = _SpatialFusion(d)
            self.post_mean = nn.Linear(d, d)
            self.post_std = nn.Linear(d, d)
            nn.init.zeros_(self.post_std.weight); nn.init.zeros_(self.post_std.bias)

    def _std(self, raw: Tensor) -> Tensor:
        s = F.softplus(raw) + self.min_std
        return s if self.max_std is None else s.clamp_max(self.max_std)

    def posterior(self, feats: Tensor) -> dict:
        f = self.fusion(feats)
        return {"mean": self.post_mean(f), "std": self._std(self.post_std(f))}

    def prior(self, cond: Tensor) -> dict:
        return {"mean": self.prior_mean(cond), "std": self._std(self.prior_std(cond))}

    def sample(self, params: dict, deterministic: bool = False) -> Tensor:
        mean, std = params["mean"], params["std"]
        return mean if deterministic else mean + std * torch.randn_like(std)

    @staticmethod
    def _kl(ma, sa, mb, sb) -> Tensor:
        """KL(N(ma,sa^2) || N(mb,sb^2)) per dim, summed over the last (feature) axis -> (...,)."""
        va, vb = sa * sa, sb * sb
        kl = (sb.log() - sa.log()) + (va + (ma - mb) ** 2) / (2.0 * vb) - 0.5
        return kl.sum(dim=-1)

    def losses(self, post: dict | None, prior: dict, target: Tensor | None = None) -> tuple[dict, dict]:
        mp, sp = prior["mean"], prior["std"]
        if self.loss_kind == "kl":
            mq, sq = post["mean"], post["std"]
            dyn = self._kl(mq.detach(), sq.detach(), mp, sp).clamp_min(self.free_nats).mean()
            rep = self._kl(mq, sq, mp.detach(), sp.detach()).clamp_min(self.free_nats).mean()
            return {"kl/dyn": dyn, "kl/rep": rep}, {"kl/dyn": self.dyn_scale, "kl/rep": self.rep_scale}
        # nll mode (Ward-2026): target = sg(e_{t+1}) deterministic latent
        assert target is not None
        t = target.detach()
        nll = (0.5 * ((t - mp) / sp) ** 2 + sp.log() + 0.5 * math.log(2 * math.pi)).sum(dim=-1).mean()
        latent_mse = F.mse_loss(mp, t)
        kl_prior = (0.5 * (sp * sp + mp * mp - 1.0) - sp.log()).sum(dim=-1).mean()   # KL(N(mp,sp)||N(0,I))
        return ({"nll": nll, "latent_mse": latent_mse, "kl/prior": kl_prior},
                {"nll": 1.0, "latent_mse": self.latent_mse_scale, "kl/prior": self.kl_prior_scale})

    def diagnostics(self, post: dict | None, prior: dict) -> dict:
        out = {"dist/prior_sigma_mean": prior["std"].mean()}
        if post is not None:
            out["dist/posterior_sigma_mean"] = post["std"].mean()
        return out

    def uncertainty(self, prior_params: dict) -> Tensor:
        return prior_params["std"].mean(dim=-1)                   # mean sigma over dims (..., n_state)


def make_dist_head(model_cfg) -> DistributionHead:
    """Build the head from the model config block (model.dist_head + its hyperparameter sub-block)."""
    m = model_cfg
    get = (lambda k, v: m.get(k, v)) if hasattr(m, "get") else (lambda k, v: getattr(m, k, v))
    kind = str(get("dist_head", "categorical"))

    def sub(name):
        s = get(name, {}) or {}
        g = (lambda k, v: s.get(k, v)) if hasattr(s, "get") else (lambda k, v: getattr(s, k, v))
        return g

    if kind == "categorical":
        c = sub("categorical")
        return CategoricalHead(groups=int(c("groups", 16)), classes=int(c("classes", 16)),
                               unimix=float(c("unimix", 0.01)), dyn_scale=float(c("dyn_scale", 1.0)),
                               rep_scale=float(c("rep_scale", 0.1)), free_nats=float(c("free_nats", 1.0)),
                               straight_through=bool(c("straight_through", True)))
    if kind == "gaussian":
        g = sub("gaussian")
        mx = g("max_std", None)
        return GaussianHead(loss=str(g("loss", "nll")), min_std=float(g("min_std", 0.1)),
                            max_std=(None if mx is None else float(mx)), dyn_scale=float(g("dyn_scale", 1.0)),
                            rep_scale=float(g("rep_scale", 0.1)), free_nats=float(g("free_nats", 1.0)),
                            kl_prior_scale=float(g("kl_prior_scale", 0.05)),
                            latent_mse_scale=float(g("latent_mse_scale", 2.0)))
    if kind == "mvn":
        raise NotImplementedError(
            "dist_head='mvn' (low-rank / full multivariate-normal head) is DESIGNED "
            "(design/models/probabilistic_heads.md sec 4) but not yet implemented. Use 'categorical' or "
            "'gaussian'; MVN is the planned follow-up once the diagonal heads are validated.")
    raise ValueError(f"unknown model.dist_head: {kind!r} (options: categorical, gaussian)")
