"""Train-time shaping variations — an orthogonal shoot-out axis (design/models/variations.md).

Each variation is a self-contained, model-AGNOSTIC toggle that composes with any world model purely
through the `SequenceWorldModel` hook contract — it never touches DSAR/LSAR/RSSM internals. The seams:

  - noise_injection -> `transform_obs`: perturbs the obs INPUTS before the model's `encode_state`.
  - physical_loss   -> `loss` via `model.physical_state(pred)`: the physical 6-vector readout
                       (default `to_obs`; LSAR freezes its decoder; vision supplies a head or None).
  - contraction     -> `loss` via `model.one_step_states(...)`: the shared one-step state-map, with the
                       backbone routed through its differentiable eager attention path.

Adding a variation = one class here + one config key; no edits to `lit.py` or any model. This is what
keeps the set maintainable as the world model evolves (e.g. toward vision): as long as the hook
contract holds, this module is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..environments import torus as T


@dataclass
class VarContext:
    """Everything a variation might read, built once per step by the LightningModule. The STABLE
    interface between training and variations — extend this (not lit internals) for a new variation."""
    model: Any
    preds: Tensor          # predicted STATES (obs for DSAR, latent for LSAR), normalized
    future_obs: Tensor     # clean (un-noised) target obs, normalized
    obs_seq: Tensor        # clean full obs window (B, P+F, 6), normalized
    act_seq: Tensor        # actions (B, P+F-1, 2)
    norm: Any              # normalizer (denorm_obs for physical units)
    R: float
    r: float
    v_scale: float
    training: bool


class Variation:
    """Base toggle. A variation implements `transform_obs` (input augmentation) and/or `loss` (extra
    objective term). Both default to no-ops, so a subclass only overrides what it needs. Logs are
    returned as plain dicts; the suite namespaces them under `{name}/`."""
    name: str = "variation"

    def transform_obs(self, obs: Tensor, training: bool) -> tuple[Tensor, dict]:
        return obs, {}

    def loss(self, ctx: VarContext) -> tuple[Tensor | None, dict]:
        return None, {}


class NoiseInjection(Variation):
    """Add i.i.d. Gaussian noise to the normalized obs INPUTS (training only); targets stay clean, so
    the model learns off-manifold-input -> on-manifold-target (denoise/recover), attacking the
    compounding rollout drift. Injection sits before `encode_state` — the one point every model ingests
    obs — so it is modality-agnostic (multi-stream vision would dispatch a per-stream sigma here)."""
    name = "noise_injection"

    def __init__(self, std: float):
        self.std = float(std)

    def transform_obs(self, obs: Tensor, training: bool) -> tuple[Tensor, dict]:
        if not training or self.std <= 0.0:
            return obs, {}
        noise = torch.randn_like(obs) * self.std
        return obs + noise, {"sigma_desired": self.std, "sigma_measured": noise.std().detach()}


class PhysicalLoss(Variation):
    """Physics-informed penalty: the predicted state should lie ON the torus with a TANGENT velocity.
    L_phys = (d_off/r)^2 + (v_off/v_scale)^2, dimensionless. Operates on the physical 6-vector returned
    by `model.physical_state(pred)` (decoder frozen for LSAR), so the physics math (torus.py, the single
    geometry source) is fully decoupled from how each model produces that state."""
    name = "physical_loss"

    def __init__(self, weight: float):
        self.weight = float(weight)

    def loss(self, ctx: VarContext) -> tuple[Tensor | None, dict]:
        state = ctx.model.physical_state(ctx.preds)        # (..., 6) physical readout, or None
        if state is None:                                  # e.g. a vision model with no physical head
            return None, {"skipped": 1.0}
        obs_phys = ctx.norm.denorm_obs(state)              # to PHYSICAL units (affine -> grad preserved)
        p_hat = obs_phys[..., :3]
        d_off = T.signed_dist(p_hat, ctx.R, ctx.r) / ctx.r            # signed, smooth when squared
        th, ph = T.angles_from_point(p_hat, ctx.R)
        v_off = (obs_phys[..., 3:] * T.normal(th, ph)).sum(-1) / ctx.v_scale
        L = d_off.pow(2).mean() + v_off.pow(2).mean()
        logs = {"loss": L.detach(), "d_off": d_off.abs().mean().detach(), "v_off": v_off.abs().mean().detach()}
        return self.weight * L, logs


class Contraction(Variation):
    """Cap how much the one-step state-map amplifies a state error: hinge on the spectral norm
    sigma_max of J = d(next_state)/d(state). L = relu(sigma_max - target)^2 (one-sided: orbital sigma~1
    directions stay free; a dead latent's sigma=0 is NOT rewarded). Uses the shared
    `model.one_step_states` (so it's model-agnostic and, for latent models, modality-blind). sigma_max
    is estimated by power iteration, with Jacobian-vector products via the forward-over-reverse
    double-vjp trick (pure autograd double-backward — needs the eager sdpa(MATH) attention path, which
    FlexAttention can't double-back through). Runs in fp32 (autocast off) for a stable second-order."""
    name = "contraction"

    def __init__(self, weight: float, target: float, power_iters: int = 2, n_sample_steps: int = 4):
        self.weight = float(weight)
        self.target = float(target)
        self.power_iters = int(power_iters)
        self.n = int(n_sample_steps)

    @staticmethod
    def _jvp(f, x: Tensor, vec: Tensor, create_graph: bool) -> Tensor:
        """J @ vec via double-vjp: g(u)=J^T u, then d(vec . g)/du = J vec. Only reverse-mode (double-
        backward), so it works on the sdpa(MATH) path where forward-mode jvp is unsupported."""
        out = f(x)
        u = torch.zeros_like(out, requires_grad=True)
        g = torch.autograd.grad(out, x, grad_outputs=u, create_graph=True)[0]      # J^T u (function of u)
        return torch.autograd.grad(g, u, grad_outputs=vec, create_graph=create_graph)[0]

    @staticmethod
    def _vjp(f, x: Tensor, u: Tensor) -> Tensor:
        return torch.autograd.grad(f(x), x, grad_outputs=u)[0]                     # J^T u

    def _sigma_max(self, model, states: Tensor, act_w: Tensor) -> Tensor:
        prefix = states[:, :-1].detach()                            # fixed history (wrt: last_state)
        last = states[:, -1].detach().requires_grad_(True)          # the differentiated leaf

        def one_step(x: Tensor) -> Tensor:
            s_win = torch.cat([prefix, x[:, None]], dim=1)
            return model.one_step_states(s_win, act_w, attn_eager=True)

        v = torch.randn_like(last)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        for _ in range(self.power_iters):                           # find top right-singular vector (no graph)
            Jv = self._jvp(one_step, last, v, create_graph=False)
            w = self._vjp(one_step, last, Jv.detach())
            v = (w / (w.norm(dim=-1, keepdim=True) + 1e-12)).detach()
        Jv = self._jvp(one_step, last, v, create_graph=True)        # final Rayleigh step (grad -> params)
        return Jv.norm(dim=-1).mean()

    def loss(self, ctx: VarContext) -> tuple[Tensor | None, dict]:
        model, obs, act = ctx.model, ctx.obs_seq, ctx.act_seq
        Tlen = obs.shape[1]
        # a window of `win` states needs `win` aligned actions (token i consumes a_i, the last predicting
        # the next state), so the last state index s+win-1 must have an action: s+win-1 <= len(act)-1.
        win = min(model.window, Tlen - 1)
        max_start = Tlen - 1 - win
        if win < 1 or max_start < 0:
            return None, {}
        # fp32, autocast off: second-order AD through attention is unstable in bf16 (matches collapse diag).
        with torch.autocast(device_type=obs.device.type, enabled=False):
            n = min(self.n, max_start + 1)
            if max_start > 0:
                starts = torch.randint(0, max_start + 1, (n,), device=obs.device).tolist()
            else:
                starts = [0] * n
            sigmas = []
            for s in starts:
                states = model.encode_state(obs[:, s:s + win].float())   # (B, win, state)
                sigmas.append(self._sigma_max(model, states, act[:, s:s + win].float()))
            sigma_max = torch.stack(sigmas).mean()
        L = torch.relu(sigma_max - self.target).pow(2)
        return self.weight * L, {"loss": L.detach(), "sigma_max": sigma_max.detach()}


class VariationSuite:
    """Holds the enabled variations; the single seam the LightningModule talks to. Chains input
    transforms, sums loss terms, and namespaces every log under `{variation.name}/`."""

    def __init__(self, variations: list[Variation]):
        self.variations = variations

    def __bool__(self) -> bool:
        return bool(self.variations)

    def transform_obs(self, obs: Tensor, training: bool) -> tuple[Tensor, dict]:
        logs: dict = {}
        for v in self.variations:
            obs, l = v.transform_obs(obs, training)
            logs.update({f"{v.name}/{k}": val for k, val in l.items()})
        return obs, logs

    def losses(self, ctx: VarContext) -> tuple[Tensor | None, dict]:
        total: Tensor | None = None
        logs: dict = {}
        for v in self.variations:
            term, l = v.loss(ctx)
            if term is not None:
                total = term if total is None else total + term
            logs.update({f"{v.name}/{k}": val for k, val in l.items()})
        return total, logs


def make_variation_suite(cfg) -> VariationSuite:
    """Build the suite from a `variations` config node. A variation is enabled iff its knob is > 0, so
    the default (all 0) yields an empty suite -> zero overhead and a bit-identical baseline."""
    if cfg is None:
        return VariationSuite([])
    get = (lambda k, d=None: cfg.get(k, d)) if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    out: list[Variation] = []
    ni = get("noise_injection") or {}
    if float((ni.get("std", 0.0) if hasattr(ni, "get") else getattr(ni, "std", 0.0)) or 0.0) > 0.0:
        out.append(NoiseInjection(ni["std"]))
    pl = get("physical_loss") or {}
    if float((pl.get("weight", 0.0) if hasattr(pl, "get") else getattr(pl, "weight", 0.0)) or 0.0) > 0.0:
        out.append(PhysicalLoss(pl["weight"]))
    ct = get("contraction") or {}
    ctw = float((ct.get("weight", 0.0) if hasattr(ct, "get") else getattr(ct, "weight", 0.0)) or 0.0)
    if ctw > 0.0:
        cg = (lambda k, d: ct.get(k, d)) if hasattr(ct, "get") else (lambda k, d: getattr(ct, k, d))
        out.append(Contraction(ctw, cg("target", 1.02), cg("power_iters", 2), cg("n_sample_steps", 4)))
    return VariationSuite(out)
