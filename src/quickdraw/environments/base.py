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


def continuity_residual(obs_phys, position_idx, velocity_idx, dt, v_scale: float = 1.0):
    """Reusable kinematic-continuity residual `v = dp/dt` (dimensionless), for any env's `physical_loss` hook.

    obs_phys: (..., T, D) rollout in PHYSICAL units. Compares the velocity channels to the CENTRAL difference
    of the position channels over the time axis (dim -2). Returns (..., T-2, len(position_idx)) — one residual
    per interior timestep per position dim; the physical_loss training variation Huber-penalizes it. Any env
    whose obs carries a position and its own time-derivative implements physical_loss in ~3 lines by calling
    this (no per-env physics to reimplement). See WorldEnv.physical_loss."""
    import torch  # local: base.py is imported very early; keep torch off the module import path
    p = obs_phys[..., list(position_idx)]                       # (...,T,k)
    v = obs_phys[..., list(velocity_idx)]                       # (...,T,k)
    dpdt = (p[..., 2:, :] - p[..., :-2, :]) / (2.0 * float(dt))  # central diff over time -> (...,T-2,k)
    return (v[..., 1:-1, :] - dpdt) / float(v_scale)

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
    """THE ENV CONTRACT — the single spec of what the pipeline may ask of an environment. Batched:
    `reset`/`step` return observation vectors (B, obs_dim); deterministic given a torch.Generator.

    REQUIRED (every env defines these; the shared pipeline calls them unconditionally):
      obs_dim / action_dim -- vector sizes                  -> unlocks model + normalizer construction.
      reset / step         -- batched simulation            -> unlocks data generation + live control rollouts.
                              (A dataset-only env, e.g. RecordedEnv, stubs them to raise — marked `not_provided`.)
      reward               -- per-step return for a goal    -> unlocks control scoring (MPPI candidate ranking).
      render_obs           -- the IMAGE MODALITY            -> unlocks image datagen + the pred-vs-true filmstrip.

    OPTIONAL (self-reported by `log_env_capabilities`; each has an env-agnostic fallback):
      rollout_metrics      -- extra val rollout errors      -> unlocks env-specific val+ood curves
                              (fallback: `default_rollout_metrics`, generic pointwise L2).
      checkpoint_metric    -- str key of rollout_metrics    -> which metric train.py monitors for best.ckpt
                              (default: 'pointwise_error').
      render_diagnostics   -- rich multi-view eval videos   -> unlocks rich eval-viz
                              (fallback: render_obs filmstrip; see `wants_diagnostics`).
      control_goals        -- world goal points             -> unlocks the goal-reaching control eval
                              (fallback: REWARD-ONLY control, maximize env.reward(obs, None)).
      fork                 -- batched state fork            -> unlocks the goal-race ORACLE's true-dynamics
                              planner (fallback: per-candidate deepcopy forks, slower).
      goal_point           -- obs -> goal-space world point -> how the control eval measures "at the goal"
                              (fallback: obs[..., :3], the torus convention).
      physical_loss        -- analytic physics residuals    -> unlocks the physical_loss training variation
                              (fallback: the variation skips itself).
      action_dist_split    -- obs -> (labels, low_name,     -> unlocks the by-state action-distribution
                              high_name) split by a             products (fallback: pooled-only panels).
                              MEANINGFUL state feature
      position_indices     -- obs dims that are world xyz   -> unlocks the flow/manifold WORLD-SPACE viz
                              (fallback: environments.position_idx config, else [0,1,2]).
      POLICIES             -- name -> factory(env, device)  -> unlocks env-shipped datagen behavior policies
                              (see environments/policies.make_policy; 'random' always available)."""
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

    # OPTIONAL — the env's control-GOAL source for the goal-reaching control eval: a list of
    # (name, goal_point (3,)) world points, or None. Envs WITHOUT goals (e.g. a gym Pendulum, where the task
    # is the env's OWN reward, not visiting points) omit it / return None -> the control eval runs
    # REWARD-ONLY (controller.mppi.run_control_reward_only maximizes env.reward(obs, None)).
    # batch/n_goals/generator let a future env sample per-episode goals; torus ignores them (its
    # per-episode goal subsetting lives in run_control, unchanged).
    def control_goals(self, batch: int, n_goals: int, generator=None, device=None): ...

    # OPTIONAL — batched STATE FORK for the goal-race oracle (controller.mppi._true_rollout_fn): return a
    # NEW env of batch B*k whose state is this env's current state repeat_interleaved k times (candidate k
    # of episode g at row g*k + k). Envs without it fall back to per-candidate deepcopy forks (the
    # reward-only oracle's mechanism) — correct but slower.
    def fork(self, k: int) -> "WorldEnv": ...

    # OPTIONAL — map obs -> the WORLD POINT compared against `control_goals` targets (settle/advance
    # distance in the control eval). Default (envs that omit it): obs[..., :3] — the torus convention,
    # where the first three obs channels ARE the ambient position. Pendulum maps obs -> rod tip.
    def goal_point(self, obs: Tensor) -> Tensor: ...

    # OPTIONAL — dimensionless analytic physics residuals {name: per-element Tensor} of a PHYSICAL-units obs
    # batch/rollout (torus: off-surface distance, normal velocity, kinematic continuity). Consumed by the
    # physical_loss training variation (training/variations.py), which Huber-penalizes EVERY key returned
    # (weights/warmup on top); the physics MATH lives here in the env. Envs without analytic physics omit it
    # -> the variation skips. To ADD physical_loss to a new env: return whatever residual dict you have; a
    # generic `v = dp/dt` term is one call to `continuity_residual(obs_phys, position_idx, velocity_idx, dt)`
    # (module fn above) -> `return {"continuity": continuity_residual(...)}` (see RecordedEnv).
    def physical_loss(self, obs_phys: Tensor) -> dict[str, Tensor]: ...

    # OPTIONAL — split obs rows by a MEANINGFUL state feature, for the by-state action-distribution eval
    # products (eval_action_distribution). obs: (..., obs_dim). Return (labels, low_name, high_name) where
    # `labels` is a bool array over obs rows (True -> low_name group, False -> high_name group), or None to
    # skip those products entirely (pooled-only panels). Envs without a meaningful split (e.g. RecordedEnv,
    # PendulumEnv) omit it.
    def action_dist_split(self, obs) -> tuple[np.ndarray, str, str] | None: ...

    # OPTIONAL — the obs dims that are ambient WORLD xyz, for the flow/manifold WORLD-SPACE viz (eval_flow
    # denoising_multistep/aggregate + the eval_ood_horizon paths). Return a length-3 list, or None. Torus &
    # pendulum return [0,1,2] (their first three obs channels ARE the ambient position). RecordedEnv omits it
    # (it can't know its dataset's layout) -> the dims come from the `environments.position_idx` CONFIG instead
    # (e.g. robocasa EEF = [7,8,9]). Resolution order (evaluation.routines._pos_idx): explicit config override
    # > this hook > [0,1,2] + a one-time warning. Unlocks the geometry-free 3D flow/manifold viz on any env.
    def position_indices(self) -> list[int] | None: ...


def default_rollout_metrics(pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]:
    """The generic, env-agnostic rollout metric — full-observation L2 error per step. Valid for ANY WorldEnv
    (recorded, pendulum, ...); geometry-aware envs override `rollout_metrics` to ADD their own errors."""
    return {"pointwise_error": (pred_obs - true_obs).norm(dim=-1)}


def wants_diagnostics(env: object) -> bool:
    """True if `env` provides a real `render_diagnostics` (not the missing/degenerate default) -> eval-viz can
    request the rich multi-view; otherwise it falls back to the `render_obs` pred-vs-true filmstrip."""
    return env_provides(env, "render_diagnostics")


def env_provides(env: object, member: str) -> bool:
    """True if `env` really implements `member`, rather than inheriting the `not_provided` stub/default.

    The same test `log_env_capabilities` uses for its per-hook checkmarks, exposed so a caller can ASK
    before committing to work the env cannot support. `wants_diagnostics` is the render_diagnostics
    special case of this; this is the general form.
    """
    fn = getattr(type(env), member, None)
    return callable(fn) and getattr(fn, "_is_default", False) is False


def not_provided(fn):
    """Decorator: mark a WorldEnv member an env defines only as a STUB or generic DEFAULT (raises, or returns
    the env-agnostic fallback) so `log_env_capabilities` reports it ✗. Same `_is_default` convention
    `wants_diagnostics` already checks. Reporting only — never changes behavior."""
    fn._is_default = True
    return fn


# The self-report rows of `log_env_capabilities`: (contract member, what ✓ unlocks, the ✗ fallback).
_CONTRACT_HOOKS = [
    ("reset",              "sim",               "no-sim"),
    ("step",               "sim",               "no-sim"),
    ("reward",             "control",           "no-control"),
    ("render_obs",         "datagen+filmstrip", "dataset-images-only"),
    ("rollout_metrics",    "val+ood",           "default(pointwise)"),
    ("render_diagnostics", "rich-viz",          "filmstrip"),
    ("control_goals",      "goal-control",      "reward-only"),
    ("physical_loss",      "physics-shaping",   "off"),
]


def log_env_capabilities(env, log_fn=print, name: str | None = None) -> str:
    """One-line ✓/✗ self-report of the WorldEnv contract (see the protocol docstring above, the source of
    truth): for each hook, ✓ + what it unlocks if the env provides it (present and not a `not_provided`
    stub/default), else ✗ + the fallback. Emitted at the START of training, data generation and the
    standalone evals — purely informational, never changes behavior."""
    def _has(member: str) -> bool:
        return env_provides(env, member)
    parts = [f"{m} {'✓ ' + ok if _has(m) else '✗ ' + miss}" for m, ok, miss in _CONTRACT_HOOKS]
    parts.append(f"checkpoint_metric={getattr(env, 'checkpoint_metric', 'pointwise_error')}")
    pol = list(getattr(env, "POLICIES", {}) or {})
    parts.append(f"policies={','.join(pol) if pol else '-'}")
    line = (f"[env-contract] {name or type(env).__name__} | obs_dim={env.obs_dim} action_dim={env.action_dim} | "
            + " | ".join(parts))
    log_fn(line)
    return line
