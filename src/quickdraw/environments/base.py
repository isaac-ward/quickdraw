"""The env-agnostic interface the whole pipeline talks to (design/gym_refactor.md).

`WorldEnv` is a MINIMAL BATCHED protocol: data-generation and internal rollouts are batched-torch for speed,
so envs expose batched `reset`/`step`. Torus implements it directly; arbitrary `gymnasium.Env`s are wrapped by
`GymBatchAdapter` (Phase 6). The pipeline (data-gen, control eval, eval-viz) is written against THIS protocol,
never against a concrete env — no env-specific branches.

Two render concerns, kept separate:
  - `render_obs`  -> the IMAGE MODALITY the model consumes (torus: FPV). Required for image world models.
  - `render_diagnostics(overlay, views)` -> OPTIONAL rich eval-video renderer. Declarative: the caller says WHAT
    to draw (a `SceneOverlay` of labelled paths/points/field, in world coords) and which camera `views`; the env
    draws its own geometry + those overlays. So one env method serves every eval (ood_horizon / control /
    language) — the env never knows which called it. If an env doesn't implement it, eval-viz falls back to the
    `render_obs` pred-vs-true filmstrip (see `wants_diagnostics`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from torch import Tensor


@dataclass
class SceneOverlay:
    """Declarative description of what a diagnostic render should draw, in WORLD coordinates. Env-agnostic:
    the eval routine fills this; the env turns roles into its own styling via `ROLE_STYLE`."""
    agents: dict[str, Tensor] = field(default_factory=dict)   # role -> (T, 3) world-space PATH  (e.g. true, pred)
    markers: dict[str, Tensor] = field(default_factory=dict)  # role -> (K, 3) world-space POINTS (e.g. goal)
    field_: Tensor | None = None                              # optional scalar field over the manifold (language)
    extras: dict = field(default_factory=dict)                # optional presentation hints from the eval routine
    #   (title, per-step action arrows, candidate fan, fork step, ...). An env MAY honor them (torus does, to
    #   keep its videos byte-identical to the pre-interface renders); any env can ignore them wholesale.


# Shared, env-agnostic role -> style map, so every env/eval draws the same semantics the same way. An env's
# `render_diagnostics` consults this to color/shape each overlay role; evals only ever refer to roles.
# Colors are matplotlib/pyvista NAMED colors — exactly the ones the shipped torus renderers always used,
# so drawing a role through this map is pixel-identical to the legacy direct viz calls.
ROLE_STYLE: dict[str, dict] = {
    "true":    {"color": "black",   "kind": "path"},   # ground-truth path
    "pred":    {"color": "dimgray", "kind": "path"},   # model-predicted    — grey
    "oracle":  {"color": "black",   "kind": "path"},   # control: true-dyn planner
    "learned": {"color": "dimgray", "kind": "path"},   # control: WM planner       — grey
    "goal":    {"color": "gold",    "kind": "ring"},   # target                — gold ring
    "concept": {"color": "crimson", "kind": "cross"},  # language concept pt   — red cross
}


@runtime_checkable
class WorldEnv(Protocol):
    """Batched environment. `reset`/`step` return observation vectors (B, obs_dim). Deterministic given a
    torch.Generator. `reward`/`render_obs` are required for control/vision; `render_diagnostics` is optional."""
    action_dim: int
    obs_dim: int

    def reset(self, generator=None) -> Tensor: ...            # -> (B, obs_dim)
    def step(self, action: Tensor) -> Tensor: ...             # action (B, action_dim) -> obs (B, obs_dim)
    def reward(self, obs: Tensor, goal: Tensor | None = None) -> Tensor: ...   # -> (B,) return for control eval
    def render_obs(self, obs: Tensor) -> Tensor: ...          # -> (B, H, W, 3) uint8 — the IMAGE MODALITY

    # Env-specific val ROLLOUT METRICS: {name: per-element Tensor} on denormalized obs. Envs without extra
    # geometry return `default_rollout_metrics` (generic); torus adds its manifold/tangent errors. An env MAY
    # also set `checkpoint_metric` (a key of this dict) naming the metric train.py monitors for best.ckpt
    # (default: pointwise_error).
    def rollout_metrics(self, pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]: ...

    # OPTIONAL — envs that can draw a rich scene implement this; others omit it (callers use `wants_diagnostics`).
    def render_diagnostics(self, overlay: SceneOverlay, views: list[str]) -> dict[str, np.ndarray]: ...


def default_rollout_metrics(pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]:
    """The generic, env-agnostic rollout metric — full-observation L2 error per step. Valid for ANY WorldEnv
    (recorded, pendulum, ...); geometry-aware envs override `rollout_metrics` to ADD their own errors."""
    return {"pointwise_error": (pred_obs - true_obs).norm(dim=-1)}


def wants_diagnostics(env: object) -> bool:
    """True if `env` provides a real `render_diagnostics` (not the missing/degenerate default) -> eval-viz can
    request the rich multi-view; otherwise it falls back to the `render_obs` pred-vs-true filmstrip."""
    fn = getattr(type(env), "render_diagnostics", None)
    return callable(fn) and getattr(fn, "_is_default", False) is False
