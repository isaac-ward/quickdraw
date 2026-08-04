"""PendulumEnv — the FULL `WorldEnv` example (environments/base.py): a batched torch pendulum.

The classic swing-up task (gym `Pendulum-v1` conventions: theta=0 is UP), written directly against the
`WorldEnv` protocol — a real batched-tensor env, not a gym wrapper (for that, see gym_adapter.py).
Copy this file to bring your own env; the torus (examples/torus.py) is the production reference.

Unlike the torus (whose math lives in torus_utils.py), EVERYTHING is in this one file so it reads
top-to-bottom as a teaching reference. It implements the four REQUIRED contract members AND every
OPTIONAL hook, so `log_env_capabilities` reports all ✓ and the whole pipeline exercises its rich paths
(datagen -> train -> ood_horizon -> control (goal AND reward-only) -> interpret/reward/language):

  REQUIRED  reset / step        : batched simulation, deterministic given a torch.Generator.
            reward              : goal=None -> the env's OWN swing-up return (reward-only control);
                                  goal=tip point -> -distance(tip(obs), goal) (goal-reaching control).
            render_obs          : a simple rod drawing, (B, obs_dim) -> (B, H, W, 3) uint8 (the image modality).
  OPTIONAL  rollout_metrics     : angle_error (wrapped |theta_pred - theta_true|) + the generic pointwise L2.
            checkpoint_metric   : "angle_error" — train.py selects best.ckpt on the pendulum-meaningful metric.
            control_goals       : 4 named TARGET tip points (upright/right/down/left) in the world plane.
            render_diagnostics  : draws the world plane — pivot, rod(s), overlay agent tip-paths + goal
                                  markers — one "scene" view (matplotlib, ROLE_STYLE-colored).
            physical_loss       : analytic residuals (off-unit-circle, energy drift beyond torque work,
                                  kinematic continuity) for the physical_loss training variation.
            POLICIES            : env-shipped datagen behavior policies — swingup (bang-bang energy
                                  pumping) and sinusoid (open-loop resonant driving)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..base import ROLE_STYLE, default_rollout_metrics

TWO_PI = 2.0 * math.pi


@dataclass
class PendulumConfig:
    dt: float = 0.05          # integration step (gym Pendulum-v1 default)
    max_torque: float = 2.0   # action clip: torque in [-max_torque, max_torque]
    g: float = 10.0           # gravity
    m: float = 1.0            # mass
    l: float = 1.0            # rod length


# ---------------------------------------------------------------------------------------------------
# Env-shipped datagen behavior policies (environments/policies.py duck type: reset(g) + sample(obs, g)).
# Registered in PendulumEnv.POLICIES below, so `data.action_sampler=swingup` (etc.) just works;
# the generic 'random' policy is always available on top (policies.make_policy).
# ---------------------------------------------------------------------------------------------------
class SwingUpPolicy:
    """Bang-bang energy-pumping swing-up: tau = max_torque * sign(-theta_dot * cos(theta)).

    The Astrom–Furuta bang-bang law is sign(phi_dot * cos(phi)) with phi measured from the DOWNWARD
    rest; our theta is measured from UPRIGHT (gym convention), so cos(phi) = -cos(theta) and the sign
    flips. Effect: torque power tau*theta_dot > 0 while the rod is below horizontal (pump energy in)
    and < 0 above it (brake near the top) -> the rod swings up from hanging and hovers around upright.
    Stateless and deterministic (draws nothing from the generator)."""

    def __init__(self, env, device="cpu"):
        self.a_max = float(env.a_max)

    def reset(self, generator: torch.Generator | None = None):
        pass

    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> Tensor:
        # obs = [cos(theta), sin(theta), theta_dot] -> sign(-theta_dot * cos(theta)), (B,) -> (B, 1)
        return (-self.a_max * torch.sign(obs[:, 2] * obs[:, 0])).unsqueeze(-1)


class SinusoidPolicy:
    """Open-loop sinusoidal driving: tau_t = max_torque * sin(omega * t * dt + phase_i), one random
    phase per parallel env (drawn in `reset` from the datagen generator — that draw is part of the
    dataset's RNG stream, like any policy draw). omega defaults near half the natural frequency
    sqrt(3g/2l) (~3.9 rad/s for the default config), which excites large, varied swings. Assumes it is
    stepped with the SAME batch it was built for (env.batch), as datagen always does."""

    def __init__(self, env, device="cpu", omega: float = 2.0):
        self.a_max, self.dt = float(env.a_max), float(env.cfg.dt)
        self.batch = int(getattr(env, "batch", 1))
        self.omega = float(omega)
        self.device = torch.device(device)
        self.t = 0
        self.phase = torch.zeros(self.batch, device=self.device)

    def reset(self, generator: torch.Generator | None = None):
        self.t = 0
        self.phase = torch.rand(self.batch, device=self.device, generator=generator) * TWO_PI

    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> Tensor:
        tau = self.a_max * torch.sin(self.omega * self.t * self.dt + self.phase.to(obs.device))
        self.t += 1
        return tau.unsqueeze(-1)                                # (B, 1)


class PendulumEnv:
    """Batched pendulum dynamics. Holds state for `batch` parallel envs as tensors.

    obs = [cos(theta), sin(theta), theta_dot] (gym-Pendulum convention, theta=0 = upright);
    action = (B, 1) torque. Implements the FULL `WorldEnv` protocol (environments/base.py):
    all four required members plus every optional hook."""

    action_dim = 1   # torque
    obs_dim = 3      # [cos(theta), sin(theta), theta_dot]

    # train.py monitors this rollout metric for best.ckpt (WorldEnv default: pointwise_error).
    checkpoint_metric = "angle_error"

    # Env-shipped behavior policies (name -> factory(env, device) -> Policy); resolved by
    # environments/policies.make_policy, so `data.action_sampler=swingup|sinusoid` works.
    POLICIES = {
        "swingup": lambda env, device="cpu": SwingUpPolicy(env, device),
        "sinusoid": lambda env, device="cpu": SinusoidPolicy(env, device),
    }

    def __init__(self, cfg: PendulumConfig, batch: int, device="cpu"):
        self.cfg = cfg
        self.batch = batch
        self.device = torch.device(device)
        self.a_max = float(cfg.max_torque)                      # action range, read by policies + MPPI
        self.theta = torch.zeros(batch, device=self.device)     # angle from upright, wrapped to [-pi, pi]
        self.theta_dot = torch.zeros(batch, device=self.device)
        self._torque = torch.zeros(batch, device=self.device)   # last applied torque (for the reward term)

    # ---- WorldEnv contract: REQUIRED (reset / step / reward / render_obs) --------------------------
    def reset(self, generator: torch.Generator | None = None) -> Tensor:
        """Random start: theta ~ U[-pi, pi], theta_dot ~ U[-1, 1] (gym Pendulum-v1 init)."""
        rand = lambda: torch.rand(self.batch, device=self.device, generator=generator)
        self.theta = (rand() * 2.0 - 1.0) * math.pi
        self.theta_dot = rand() * 2.0 - 1.0
        self._torque = torch.zeros(self.batch, device=self.device)
        return self.observe()

    def step(self, action: Tensor) -> Tensor:
        """action: (B, 1) torque, clipped to +-max_torque. Pendulum dynamics
        theta_dotdot = (3g / 2l) sin(theta) + 3 / (m l^2) * tau, semi-implicit Euler, wrap theta."""
        cfg = self.cfg
        tau = action[:, 0].clamp(-cfg.max_torque, cfg.max_torque)
        acc = (3.0 * cfg.g / (2.0 * cfg.l)) * torch.sin(self.theta) + 3.0 / (cfg.m * cfg.l**2) * tau
        self.theta_dot = self.theta_dot + cfg.dt * acc                              # velocity first, then
        self.theta = _wrap(self.theta + cfg.dt * self.theta_dot)                    # position (semi-implicit)
        self._torque = tau
        return self.observe()

    def observe(self) -> Tensor:
        return torch.stack((torch.cos(self.theta), torch.sin(self.theta), self.theta_dot), dim=-1)

    def reward(self, obs: Tensor, goal: Tensor | None = None) -> Tensor:
        """Per-step control return (higher = better), broadcast over leading dims (..., 3) -> (...).

        goal=None (REWARD-ONLY control): the gym Pendulum reward, unchanged —
        -(theta_wrapped^2 + 0.1 theta_dot^2 + 0.001 tau^2), max 0 when balanced upright at rest. The
        tau^2 term uses the LAST applied torque and so only applies to live (B, 3) calls — MPPI's
        imagined candidates get the state cost (its own beta_ctrl covers control cost there).

        goal=(..., 3) world TIP point (from `control_goals`, broadcastable like torus.reward):
        -||tip(obs) - goal|| — negative distance of the rod tip to the target tip, which is what makes
        the goal-reaching control race scoreable on this env."""
        if goal is not None:
            return -(self.tip_point(obs) - goal.to(obs)).norm(dim=-1)
        theta = torch.atan2(obs[..., 1], obs[..., 0])           # wrapped angle back out of [cos, sin]
        cost = theta**2 + 0.1 * obs[..., 2] ** 2
        if self._torque.shape == cost.shape:                    # live env call (not planner candidates)
            cost = cost + 0.001 * self._torque**2
        return -cost

    def render_obs(self, obs: Tensor, size: int = 96) -> Tensor:
        """The IMAGE MODALITY: draw the rod, (B, 3) -> (B, size, size, 3) uint8. Pure numpy (no plotting
        lib): per pixel, distance to the rod segment from the pivot (image center) to the tip; theta=0 UP."""
        o = obs.detach().cpu().numpy()
        span = 1.4 * self.cfg.l                                          # world half-extent shown
        ax = (np.arange(size) / (size - 1)) * 2.0 * span - span
        X, Y = ax[None, None, :], -ax[None, :, None]                     # world coords per pixel, y up
        tx = (o[:, 1] * self.cfg.l)[:, None, None]                       # rod tip = l * (sin, cos):
        ty = (o[:, 0] * self.cfg.l)[:, None, None]                       # theta measured from upright
        t = ((X * tx + Y * ty) / (self.cfg.l**2)).clip(0.0, 1.0)         # projection onto the segment
        d = np.hypot(X - t * tx, Y - t * ty)                             # (B, size, size) distance to rod
        img = np.full((o.shape[0], size, size, 3), 255, dtype=np.uint8)  # white background
        img[d < 0.10 * self.cfg.l] = (204, 77, 77)                       # the rod (gym's reddish bar)
        img[np.broadcast_to(np.hypot(X, Y) < 0.06 * self.cfg.l, d.shape)] = (0, 0, 0)   # pivot dot
        return torch.from_numpy(img).to(self.device)

    # ---- WorldEnv contract: OPTIONAL hooks (each has an env-agnostic fallback; see base.py) --------
    def tip_point(self, obs: Tensor) -> Tensor:
        """The rod TIP as an ambient world point in the pendulum's 2-D plane (z=0):
        tip = (l*sin(theta), l*cos(theta), 0) = (l*obs[1], l*obs[0], 0). Broadcasts (..., 3) -> (..., 3).
        This is the env's obs -> world-point map, shared by `reward(goal=...)` and `control_goals`."""
        l = self.cfg.l
        return torch.stack((l * obs[..., 1], l * obs[..., 0], torch.zeros_like(obs[..., 0])), dim=-1)

    # OPTIONAL WorldEnv hook (base.py): the control eval's obs -> goal-space point map (settle/advance
    # distance to `control_goals` targets). For the pendulum that map IS the tip point.
    goal_point = tip_point

    def fork(self, k: int) -> "PendulumEnv":
        """OPTIONAL WorldEnv state fork (base.py): a fresh env of batch B*k with this env's current state
        repeat_interleaved k times — the goal-race oracle's true-dynamics planner forks the live env with
        this (candidate k of episode g at row g*k + k), exactly like the torus."""
        sim = PendulumEnv(self.cfg, batch=self.batch * k, device=self.device)
        sim.theta = self.theta.repeat_interleave(k)
        sim.theta_dot = self.theta_dot.repeat_interleave(k)
        sim._torque = self._torque.repeat_interleave(k)
        return sim

    def rollout_metrics(self, pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]:
        """WorldEnv val rollout metrics, elementwise over leading dims ((N, H, 3) -> (N, H) curves,
        consumed by openloop/lit): the env-specific `angle_error` — the WRAPPED absolute angle gap
        |theta_pred - theta_true| recovered from the [cos, sin] channels (immune to the 2*pi seam,
        unlike raw obs L2) — PLUS the generic pointwise L2 from base.default_rollout_metrics."""
        th_p = torch.atan2(pred_obs[..., 1], pred_obs[..., 0])
        th_t = torch.atan2(true_obs[..., 1], true_obs[..., 0])
        return {"angle_error": _wrap(th_p - th_t).abs(), **default_rollout_metrics(pred_obs, true_obs)}

    def control_goals(self, batch: int = 0, n_goals: int = 0, generator=None, device=None):
        """OPTIONAL WorldEnv goal source (base.py): named TARGET angles for the goal-reaching control
        eval, each represented as the pendulum TIP position (l*sin(theta), l*cos(theta), 0) in the 2-D
        world plane — the same space `reward(goal=...)` and `render_diagnostics` markers live in.
        batch/n_goals/generator are unused (fixed goal set, like torus); the control eval does its own
        per-episode subsetting."""
        l = self.cfg.l
        pts = [("upright", (0.0, l)), ("right", (l, 0.0)), ("down", (0.0, -l)), ("left", (-l, 0.0))]
        return [(name, torch.tensor([x, y, 0.0], device=device)) for name, (x, y) in pts]

    def physical_loss(self, obs_phys: Tensor) -> dict[str, Tensor]:
        """OPTIONAL WorldEnv physics hook (base.py): dimensionless analytic residuals of a
        PHYSICAL-units obs (..., 3) batch or rollout (B, T, 3). KEY NAMES follow the consumer contract
        (training/variations.PhysicalLoss always reads d_off + v_off, and continuity when present);
        the semantics are this env's own:
          d_off      : ||[cos, sin]|| - 1 — how far the state falls off the UNIT CIRCLE, the pendulum's
                       state manifold (per step; the exact analog of the torus off-surface distance).
          v_off      : ENERGY-DRIFT residual per transition — relu(|dE| - max_torque*|dtheta|) / (m g l),
                       the part of the energy change NO admissible torque can explain (dE/dt = tau*theta_dot,
                       so |dE| <= max_torque*|dtheta| per step up to O(dt) integration error). Zeros when
                       obs_phys has no time axis (a per-step call has no dE).
          continuity : the kinematic law theta_dot = d theta/dt, as a wrapped central-difference residual
                       over interior steps, normalized by the natural frequency sqrt(3g/2l) (present only
                       when obs_phys is a rollout with T >= 3, like torus)."""
        cfg = self.cfg
        cos_t, sin_t, omega = obs_phys[..., 0], obs_phys[..., 1], obs_phys[..., 2]
        out = {"d_off": torch.sqrt(cos_t**2 + sin_t**2 + 1e-12) - 1.0}
        if obs_phys.ndim >= 3 and obs_phys.shape[1] >= 2:              # rollout (B, T, 3): time on dim 1
            theta = torch.atan2(sin_t, cos_t)
            dtheta = _wrap(theta[:, 1:] - theta[:, :-1])               # per-transition angle travel
            inertia = cfg.m * cfg.l**2 / 3.0                           # rod about its pivot (matches step's 3/ml^2)
            E = 0.5 * inertia * omega**2 + cfg.m * cfg.g * (cfg.l / 2.0) * cos_t   # kinetic + COM potential
            excess = (E[:, 1:] - E[:, :-1]).abs() - cfg.max_torque * dtheta.abs()  # |dE| beyond max torque work
            out["v_off"] = torch.relu(excess) / (cfg.m * cfg.g * cfg.l)
            if obs_phys.shape[1] >= 3:                                 # central diff needs interior steps
                omega0 = math.sqrt(3.0 * cfg.g / (2.0 * cfg.l))        # natural angular-frequency scale
                sec = _wrap(theta[:, 2:] - theta[:, :-2]) / (2.0 * cfg.dt)
                out["continuity"] = (omega[:, 1:-1] - sec) / omega0
        else:                                                          # per-step call: no transitions
            out["v_off"] = torch.zeros_like(out["d_off"])
        return out

    def render_diagnostics(self, overlay, views) -> dict:
        """OPTIONAL rich diagnostic renderer (WorldEnv protocol): draw the pendulum's 2-D world plane —
        the pivot, each overlay agent's CURRENT rod + its tip PATH so far (roles styled via
        base.ROLE_STYLE: true/oracle black, pred/learned grey), and goal MARKERS as gold rings. ONE view
        is offered — "scene"; other view names are ignored ({} when nothing to draw, so callers fall
        back to the render_obs filmstrip).

        overlay.agents paths arrive in either of the two (T, 3) forms the pipeline produces, detected by
        the z column (see `_tip_path`): raw obs rows [cos, sin, theta_dot] (the generic obs[..., :3]
        slice evals pass for this 3-dim obs) or world tip points (x, y, 0) (control_goals space).
        Honors overlay.extras title/n_frames/log; every other extra is ignored (allowed by the contract)."""
        if "scene" not in views or not (overlay.agents or overlay.markers):
            return {}
        from matplotlib.backends.backend_agg import FigureCanvasAgg    # lazy: keep this module import-light
        from matplotlib.figure import Figure
        agents = {role: self._tip_path(p) for role, p in overlay.agents.items()}     # role -> (T, 2) tips
        markers = {role: self._tip_path(p) for role, p in overlay.markers.items()}   # role -> (K, 2) points
        T = max((len(p) for p in agents.values()), default=1)
        ex = overlay.extras
        n_frames = int(ex.get("n_frames", T))
        tidx = np.unique(np.linspace(0, T - 1, max(1, min(T, n_frames))).round().astype(int))
        if callable(ex.get("log")):
            ex["log"](f"pendulum scene: {len(tidx)} frames, agents={list(agents)}, markers={list(markers)}")
        span = 1.4 * self.cfg.l
        fig = Figure(figsize=(3.2, 3.2), dpi=100)                      # (320, 320, 3) frames
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        frames = []
        for t in tidx:
            ax.clear()
            ax.set_xlim(-span, span), ax.set_ylim(-span, span)
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(str(ex.get("title", "")), fontsize=9)
            for role, pts in markers.items():                          # goal markers: gold rings
                color = ROLE_STYLE.get(role, {"color": "gold"})["color"]
                ax.scatter(pts[:, 0], pts[:, 1], s=140, facecolors="none", edgecolors=color, linewidths=2.0)
            for role, tips in agents.items():                          # each agent: path-so-far + current rod
                color = ROLE_STYLE.get(role, {"color": "dimgray"})["color"]
                k = min(t, len(tips) - 1)
                ax.plot(tips[: k + 1, 0], tips[: k + 1, 1], color=color, lw=1.0, alpha=0.6)
                ax.plot([0.0, tips[k, 0]], [0.0, tips[k, 1]], color=color, lw=3.0, label=role)
                ax.plot(tips[k, 0], tips[k, 1], "o", color=color, ms=5)
            ax.plot(0.0, 0.0, "o", color="black", ms=6)                # the pivot
            if agents:
                ax.legend(loc="upper right", fontsize=7, frameon=False)
            canvas.draw()
            frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
        return {"scene": np.stack(frames)}                             # (T, H, W, 3) uint8

    def _tip_path(self, pts) -> np.ndarray:
        """(T, 3) overlay path/points -> (T, 2) world tip coordinates. Accepts BOTH forms in play:
        world tip points (x, y, 0) — z exactly 0 (control_goals / goal_seqs) — and raw obs rows
        [cos, sin, theta_dot] — z is theta_dot, generically nonzero (the pipeline's obs[..., :3] slice).
        Degenerate corner: an obs path at exact rest (theta_dot == 0 throughout) reads as tip points
        and renders axis-mirrored — harmless for a diagnostic, and unreachable under normal dynamics."""
        p = np.asarray(pts.detach().cpu() if hasattr(pts, "detach") else pts, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0 or np.abs(p[:, 2]).max() < 1e-6:                # tip points already: (x, y)
            return p[:, :2]
        return self.cfg.l * np.stack([p[:, 1], p[:, 0]], axis=1)       # obs [cos, sin, ·] -> l*(sin, cos)


def _wrap(theta: Tensor) -> Tensor:
    """Wrap angle to [-pi, pi]."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi
