"""TorusEnv — the REFERENCE `WorldEnv` implementation (environments/base.py).

This file is deliberately ONLY the env contract: every method below is a member of the `WorldEnv`
protocol (or one of its documented optional hooks). Copy this file to build your own env. All the
torus-specific math (geometry, rollout errors, config dataclass, action samplers, color ground
truth) lives in `environments/torus_utils.py` — an env is the CONTRACT; utilities live elsewhere.

For a second, more minimal example see `environments/examples/pendulum.py` (reward-only, no
diagnostics). For wrapping an existing gymnasium.Env instead, see `environments/gym_adapter.py`.
"""

from __future__ import annotations

import numpy as _np
import torch
from torch import Tensor

from ..policies import SamplerPolicy
from ..torus_utils import (
    TWO_PI, BimodalActionSampler, OUActionSampler, TorusConfig, angles_from_point, control_goals,
    manifold_distance_error, normal, observation_vector, pointwise_error, signed_dist, split_obs,
    tangent_velocity_error,
)


class TorusEnv:
    """Batched torus dynamics. Holds state for `batch` parallel envs as tensors.

    Pure-tensor (no nn.Module needed); all ops run on `device`. Deterministic given a
    torch.Generator. The true state is always exactly on the torus.

    Implements the `WorldEnv` protocol (environments/base.py): `reset`/`step`/`reward`/`render_obs`.
    """

    action_dim = 2   # (a_theta, a_phi)
    obs_dim = 6      # [p; p_dot]

    # Env-shipped behavior policies (design/gym_refactor.md): name -> factory(env, device) -> Policy.
    # `make_policy` (environments/policies.py) resolves the generic 'random' itself and delegates these
    # here — pull Torus World and you get its policies. Wraps the samplers (torus_utils) unchanged (same math/RNG).
    POLICIES = {
        "ornstein_uhlenbeck": lambda env, device="cpu": SamplerPolicy(
            OUActionSampler(getattr(env, "batch", 1), float(env.cfg.a_max), device=device)),
        "bimodal": lambda env, device="cpu": SamplerPolicy(
            BimodalActionSampler(getattr(env, "batch", 1), float(env.cfg.a_max), device=device)),
    }

    def __init__(self, cfg: TorusConfig, batch: int, device="cpu", coloring: str = "hsv", fov: float = 100.0):
        self.cfg = cfg
        self.batch = batch
        self.device = torch.device(device)
        self.theta = torch.zeros(batch, device=self.device)
        self.phi = torch.zeros(batch, device=self.device)
        self.theta_dot = torch.zeros(batch, device=self.device)
        self.phi_dot = torch.zeros(batch, device=self.device)
        # render_obs appearance (NOT dynamics, so kept off TorusConfig — dataset_card.json stays unchanged):
        # texture coloring + camera fov, per split in data generation (hsv vs circles for ood_visual).
        self.coloring = str(coloring)
        self.fov = float(fov)

    # ---- WorldEnv contract: REQUIRED (reset / step / reward / render_obs) --------------------------
    def reset(self, generator: torch.Generator | None = None) -> Tensor:
        g = generator
        rand = lambda *s: torch.rand(*s, device=self.device, generator=g)
        self.theta = rand(self.batch) * TWO_PI
        self.phi = rand(self.batch) * TWO_PI
        # nonzero initial speed: fixed magnitude, random direction in (theta_dot, phi_dot)
        ang = rand(self.batch) * TWO_PI
        self.theta_dot = self.cfg.init_speed * torch.cos(ang)
        self.phi_dot = self.cfg.init_speed * torch.sin(ang)
        return self.observe()

    def step(self, action: Tensor) -> Tensor:
        """action: (batch, 2) = (a_theta, a_phi). Semi-implicit Euler. Returns observation."""
        a = action.clamp(-self.cfg.a_max, self.cfg.a_max)
        dt, gamma, m = self.cfg.dt, self.cfg.gamma, self.cfg.mass
        self.theta_dot = self.theta_dot + dt * (a[:, 0] / m - gamma * self.theta_dot)
        self.phi_dot = self.phi_dot + dt * (a[:, 1] / m - gamma * self.phi_dot)
        self.theta = (self.theta + dt * self.theta_dot) % TWO_PI
        self.phi = (self.phi + dt * self.phi_dot) % TWO_PI
        return self.observe()

    def observe(self) -> Tensor:
        return observation_vector(self.theta, self.phi, self.theta_dot, self.phi_dot, self.cfg.R, self.cfg.r)

    def reward(self, obs: Tensor, goal: Tensor | None = None, *,
               beta_vel: float = 0.0, r_settle: float = 0.5) -> Tensor:
        """Per-step control return (higher = better), mirroring controller.mppi._score's per-step term:
        negative ambient distance to `goal`, minus a near-goal velocity penalty gated inside `r_settle`
        (so the planner settles instead of orbiting). Defaults match conf/control/mppi.yaml (beta_vel=0
        -> pure negated goal distance). goal: (3,) or (B,3) world point; None -> zeros (no preference).
        obs (B,6) -> (B,)."""
        if goal is None:
            return torch.zeros(obs.shape[0], device=obs.device)
        p, v = split_obs(obs)
        d = (p - goal.to(obs)).norm(dim=-1)
        gate = (d < r_settle).float()
        return -d - beta_vel * gate * v.norm(dim=-1)

    def render_obs(self, obs: Tensor) -> Tensor:
        """The IMAGE MODALITY: egocentric FPV of each state, (B,6) -> (B,size,size,3) uint8 on `device`.
        Delegates to viz.fpv_frames with THIS env's coloring/fov (defaults hsv, fov=100 per
        conf/data/torus.yaml fpv_fov; size=viz.FPV_SIZE), so it is byte-identical to the data pipeline's
        renderer — including the sequential heading smoothing when `obs` is one trajectory over time."""
        from ...logging import viz   # lazy: keep this module import-light (viz pulls pyvista/matplotlib)
        frames = viz.fpv_frames(self.cfg.R, self.cfg.r, self.coloring, obs.detach().cpu().numpy(),
                                fov=self.fov, size=viz.FPV_SIZE)
        return torch.from_numpy(frames).to(self.device)

    # ---- WorldEnv contract: OPTIONAL hooks (each has an env-agnostic fallback; see base.py) --------
    # OPTIONAL WorldEnv goal source (base.py): the SAME 8 module-level control goals, so the control eval's
    # torus behavior is identical. batch/n_goals/generator are unused — run_control does its own per-episode
    # goal subsetting/ordering (unchanged); envs without goals return None -> reward-only control.
    def control_goals(self, batch: int = 0, n_goals: int = 0, generator=None, device=None):
        return control_goals(self.cfg.R, self.cfg.r, device=device)

    # train.py monitors this rollout metric for best.ckpt (WorldEnv default: pointwise_error).
    checkpoint_metric = "manifold_distance_error"

    def rollout_metrics(self, pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]:
        """WorldEnv val rollout metrics: the SAME three torus errors (same functions, same arguments —
        v_scale = init_speed) lit.py always logged, so torus training logs are byte-identical."""
        return {
            "manifold_distance_error": manifold_distance_error(pred_obs, self.cfg.R, self.cfg.r),
            "pointwise_error": pointwise_error(pred_obs, true_obs),
            "tangent_velocity_error": tangent_velocity_error(pred_obs, self.cfg.R, self.cfg.init_speed),
        }

    def physical_loss(self, obs_phys: Tensor) -> dict[str, Tensor]:
        """OPTIONAL WorldEnv physics hook (base.py): dimensionless analytic physics residuals of a
        PHYSICAL-units obs (..., 6) — the exact terms the physical_loss training variation penalizes
        (variations.py applies Huber/weights/warmup on top; the torus MATH lives here). Same functions and
        argument values as the variation always used (R/r from THIS env's config, v_scale = init_speed,
        dt = cfg.dt), so the variation's loss term is byte-identical to the pre-hook version:
          d_off      : signed off-surface distance / r                       (algebraic, per step)
          v_off      : velocity's normal component / init_speed              (algebraic, per step)
          continuity : (v - central-diff dp/dt) / init_speed, interior steps (kinematic; present only when
                       obs_phys is a rollout (B, T>=3, 6))."""
        p_hat, v_hat = split_obs(obs_phys)
        R, r, vs, dt = self.cfg.R, self.cfg.r, self.cfg.init_speed, self.cfg.dt
        d_off = signed_dist(p_hat, R, r) / r                          # signed, smooth when squared
        th, ph = angles_from_point(p_hat, R)
        out = {"d_off": d_off, "v_off": (v_hat * normal(th, ph)).sum(-1) / vs}
        if obs_phys.ndim >= 3 and obs_phys.shape[1] >= 3 and dt:      # kinematic continuity v = dp/dt
            sec = (p_hat[:, 2:] - p_hat[:, :-2]) / (2.0 * dt)         # central-diff velocity, interior t
            out["continuity"] = (v_hat[:, 1:-1] - sec) / vs
        return out

    def render_diagnostics(self, overlay, views) -> dict:
        """OPTIONAL rich diagnostic renderer (WorldEnv protocol; design/gym_refactor.md Phase 5): draw the
        overlay's labelled world-space paths/markers on the torus. ONE view is offered — "scene", the
        fig_torus_atlas composite (iso + 3 axial cameras in one frame) every eval video always used; other
        view names are ignored. A thin wrapper over the SAME viz renderers with the SAME arguments as the
        legacy direct calls, so the output is byte-identical. Two scene layouts, chosen by the presentation
        hints the eval routine put in `overlay.extras` (roles color via base.ROLE_STYLE either way):
          - `fork_step`: open-loop truth-vs-prediction compare (viz.traj_compare_frames). agents {true, pred}
            full paths sharing the first fork_step steps; extras avec (ambient applied action along the true
            path), n_frames/title/smooth_window/log.
          - `goal_seqs`: controller race (viz.control_compare_frames). One agent per overlay role; per-role
            extras goal_seqs/avecs; goal rings drawn iff markers['goal'] is present; extras
            fan_seq/reuse/n_frames/title/log.
        Returns {"scene": (T,H,W,3) uint8 frames}; {} when `views` requests nothing we can draw."""
        if "scene" not in views:
            return {}
        from ...logging import viz   # lazy: keep this module import-light (viz pulls pyvista/matplotlib)
        from ..base import ROLE_STYLE
        ex = overlay.extras
        R, r, coloring = self.cfg.R, self.cfg.r, ex.get("coloring", "hsv")
        paths = {role: _np.asarray(p) for role, p in overlay.agents.items()}
        if "fork_step" in ex:        # open-loop compare: true vs pred, forking at step fork_step
            frames = viz.traj_compare_frames(R, r, coloring, paths["true"], paths["pred"], ex["avec"],
                                             ex["fork_step"], n_frames=ex.get("n_frames", 120),
                                             title=ex.get("title", ""),
                                             smooth_window=ex.get("smooth_window", viz.ACTION_SMOOTH_WINDOW),
                                             log=ex.get("log"))
        elif "goal_seqs" in ex:      # controller race: one agent (+ its goal ring/action arrow) per role
            agents = [{"path": paths[role], "color": ROLE_STYLE[role]["color"],
                       "goal_seq": _np.asarray(ex["goal_seqs"][role]), "avec": _np.asarray(ex["avecs"][role])}
                      for role in overlay.agents]
            frames = viz.control_compare_frames(R, r, coloring, agents, n_frames=ex.get("n_frames", 10000),
                                                title=ex.get("title", ""), fan_seq=ex.get("fan_seq"),
                                                reuse=bool(ex.get("reuse", False)),
                                                show_goals=("goal" in overlay.markers), log=ex.get("log"))
        else:
            return {}
        return {"scene": frames}
