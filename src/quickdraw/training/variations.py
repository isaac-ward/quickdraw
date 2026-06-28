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
    dt: float             # env timestep (physical seconds) — for the kinematic continuity term
    training: bool


class Variation:
    """Base toggle. A variation implements `transform_obs` (input augmentation) and/or `loss` (extra
    objective term). Both default to no-ops, so a subclass only overrides what it needs. `loss` returns
    (weighted_term_or_None, diag_dict): the weighted term is added to the objective AND logged under
    `{tag}/loss/<loss_name>` (so it sits with the other optimized loss components on both train and val);
    diag_dict holds sub-diagnostics, namespaced by the suite under `{name}/`."""
    name: str = "variation"
    loss_name: str | None = None   # key under {tag}/loss/ for this variation's additive term (None = none)

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
    """Physics-informed penalty on the predicted state via `model.physical_state(pred)` (decoder frozen
    for LSAR; physics math stays in torus.py). Three terms, each dimensionless:
      - ALGEBRAIC (weight): on-surface (d_off/r)^2 + tangent-velocity (v_off/v_scale)^2 — per-step;
      - CONTINUITY (continuity): the kinematic law v = dp/dt, as a central-difference residual over the
        rollout, ||v_hat_t - (p_{t+1}-p_{t-1})/(2 dt)||^2 / v_scale^2. This is the DIFFERENTIAL physics
        constraint linking the velocity channel to how the position actually moves across time (the
        algebraic terms are per-step and can't see it). Ambient coords are continuous through angle
        wraps, so the position difference is always well-defined."""
    name = "physical_loss"
    loss_name = "physical"

    def __init__(self, weight: float, continuity: float = 0.0):
        self.weight = float(weight)
        self.continuity = float(continuity)

    def loss(self, ctx: VarContext) -> tuple[Tensor | None, dict]:
        state = ctx.model.physical_state(ctx.preds)        # (B, H, 6) physical readout, or None
        if state is None:                                  # e.g. a vision model with no physical head
            return None, {"skipped": 1.0}
        obs_phys = ctx.norm.denorm_obs(state)              # to PHYSICAL units (affine -> grad preserved)
        p_hat = obs_phys[..., :3]
        d_off = T.signed_dist(p_hat, ctx.R, ctx.r) / ctx.r            # signed, smooth when squared
        th, ph = T.angles_from_point(p_hat, ctx.R)
        v_off = (obs_phys[..., 3:] * T.normal(th, ph)).sum(-1) / ctx.v_scale
        alg = d_off.pow(2).mean() + v_off.pow(2).mean()              # algebraic: on-surface + tangent
        total = self.weight * alg
        diag = {"d_off": d_off.abs().mean().detach(), "v_off": v_off.abs().mean().detach()}
        if self.continuity > 0.0 and obs_phys.shape[1] >= 3 and ctx.dt:   # kinematic continuity v = dp/dt
            sec = (p_hat[:, 2:] - p_hat[:, :-2]) / (2.0 * ctx.dt)         # central-diff velocity, interior t
            cont = ((obs_phys[:, 1:-1, 3:] - sec) / ctx.v_scale).pow(2).sum(-1).mean()
            total = total + self.continuity * cont
            diag["continuity"] = cont.detach()
        return total, diag


class Contraction(Variation):
    """Cap how much the one-step state-map amplifies a state error: hinge on the spectral norm
    sigma_max of J = d(next_state)/d(state). L = relu(sigma_max - target)^2 (one-sided: orbital sigma~1
    directions stay free; a dead latent's sigma=0 is NOT rewarded). Uses the shared
    `model.one_step_states` (so it's model-agnostic and, for latent models, modality-blind). sigma_max
    is estimated by power iteration, with Jacobian-vector products via the forward-over-reverse
    double-vjp trick (pure autograd double-backward — needs the eager sdpa(MATH) attention path, which
    FlexAttention can't double-back through). Runs in fp32 (autocast off) for a stable second-order."""
    name = "contraction"
    loss_name = "contraction"

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
        return self.weight * L, {"sigma_max": sigma_max.detach()}


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

    def losses(self, ctx: VarContext) -> tuple[Tensor | None, dict, dict]:
        """Returns (summed weighted term for the objective, {loss_name: term} for {tag}/loss/ logging,
        {name/diagkey: value} for diagnostics)."""
        total: Tensor | None = None
        comps: dict = {}
        diags: dict = {}
        for v in self.variations:
            term, diag = v.loss(ctx)
            if term is not None:
                total = term if total is None else total + term
                if v.loss_name:
                    comps[v.loss_name] = term.detach()
            diags.update({f"{v.name}/{k}": val for k, val in diag.items()})
        return total, comps, diags


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
    plg = (lambda k, d: pl.get(k, d)) if hasattr(pl, "get") else (lambda k, d: getattr(pl, k, d))
    plw, plc = float(plg("weight", 0.0) or 0.0), float(plg("continuity", 0.0) or 0.0)
    if plw > 0.0 or plc > 0.0:           # enable if EITHER the algebraic or the continuity term is active
        out.append(PhysicalLoss(plw, plc))
    ct = get("contraction") or {}
    ctw = float((ct.get("weight", 0.0) if hasattr(ct, "get") else getattr(ct, "weight", 0.0)) or 0.0)
    if ctw > 0.0:
        cg = (lambda k, d: ct.get(k, d)) if hasattr(ct, "get") else (lambda k, d: getattr(ct, k, d))
        out.append(Contraction(ctw, cg("target", 1.02), cg("power_iters", 2), cg("n_sample_steps", 4)))
    return VariationSuite(out)
