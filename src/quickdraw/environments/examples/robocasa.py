"""RoboCasaEnv — the `WorldEnv` example for a HEAVY THIRD-PARTY SIMULATOR (environments/base.py).

The other two examples are analytic torch envs (`torus.py`, `pendulum.py`) and `gym_adapter.py` is the
zero-code path for a `gymnasium.Env`. None of them shows the case this one does: a large external
simulator with its own asset tree, its own construction API, its own pinned dependencies, and a reset
measured in seconds rather than microseconds. Four things bit us here and NONE of them raises an error:

  1. THE OBSERVATION LAYOUT. See robocasa_utils.OBS_LAYOUT. `robot0_eef_pos` exists, is world-frame, and
     looks right; the dataset wants `robot0_base_to_eef_pos`. Only the value ranges give it away.
  2. ROBOSUITE MUST BE THE SOURCE CHECKOUT, not the PyPI wheel -- the wheel raises
     `ManipulationEnv.__init__() got an unexpected keyword argument 'load_model_on_init'`.
  3. ROBOCASA HARD-ASSERTS `numpy == 2.2.5` and `mujoco == 3.3.1` (exact allow-lists in its __init__).
  4. `render_obs` MUST MATCH THE TRAINING PIPELINE'S FILTER, not just its size. Frames were rendered at
     256 and AREA-downsampled to the training cache; rendering directly at 96 is a different filter and
     hands the model out-of-distribution input silently. Hence `data.dataset.resize_frames_area`, shared
     with the cache builder so the two cannot drift.

WHY B COPIES IN A LOOP. robosuite is single-instance, so a batched WorldEnv holds B of them and loops --
the same shape as `gym_adapter.GymBatchAdapter`. That overlap is ~10 lines of scaffolding and is
deliberately NOT factored into a shared base: the bodies diverge (explicit 16-dim concat vs
`gym.spaces.flatten`, named-camera sim rendering vs `env.render()`, robosuite's 4-tuple vs gym's 5-tuple),
and a base class both then fight is worse than ten duplicated lines. Revisit at a THIRD such adapter.

WHAT THIS ENV DELIBERATELY DOES NOT IMPLEMENT, and why (all are independent hooks with graceful
fallbacks -- see docs/byo.md's rung-3 table):

  * `fork` -> so NO ORACLE BASELINE, and control here is reward-only. This is a capacity fact, not an
    oversight: MPPI runs `n_episodes=8 x num_samples=128 = 1024` forked copies stepped `horizon=64`
    times per replan = 65,536 sim steps per replan, ~125 replans per episode. On the torus that fork is
    one tensor of shape (1024, obs_dim); here each copy is a full mujoco kitchen. Infeasible on memory
    and on time by orders of magnitude. Consequence to state whenever a control number is reported: a
    poor score cannot be attributed to the world model over the planner or the reward.
  * `control_goals` -> no goal race; MPPI maximises robosuite's own task reward instead.
  * `POLICIES` -> no env-generated data (we train on a 4 h recording, not on datagen).
  * `physical_loss` -> a kitchen has no clean analytic prior, unlike a pendulum's energy.
  * `checkpoint_metric` -> DELIBERATE. best.ckpt stays on `val/metric/<head>/visual`, because the
    project's objective is open-loop image quality; the proprio metrics below are logged as curves only.
"""

from __future__ import annotations

import os

# MUJOCO_GL must be chosen BEFORE mujoco initialises its GL context, and this container is headless.
# Set here rather than in the shell so `environments=robocasa` works from any entry point.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch
from torch import Tensor

from ..base import ROLE_STYLE, default_rollout_metrics
from ..robocasa_utils import OBS_DIM, obs_slices, pack_obs, quat_angle_error, scene_panels


class RoboCasaEnv:
    """B parallel robocasa kitchens as one batched `WorldEnv`.

    obs = the 16-dim vector of robocasa_utils.OBS_LAYOUT; action = (B, 12) for the PandaOmron composite
    controller. Implements the REQUIRED contract plus `position_indices`, `rollout_metrics` and
    `render_diagnostics`; see the module docstring for what is left out and why.
    """

    obs_dim = OBS_DIM        # 16
    action_dim = 12          # PandaOmron composite controller

    def __init__(self, cfg, batch: int, device="cpu"):
        self.cfg = cfg
        self.batch = int(batch)
        self.device = torch.device(device)
        self.a_max = float(getattr(cfg, "a_max", 1.0))          # action range, read by policies + MPPI
        self.tasks = list(getattr(cfg, "tasks", []) or [])
        self.task = getattr(cfg, "task", None)                  # None -> sample per reset
        self.cams = list(getattr(cfg, "cameras", ["robot0_agentview_left"]))
        self.render_size = int(getattr(cfg, "render_size", 256))   # what the SIM renders at
        self.out_hw = (int(getattr(cfg, "out_h", 96)), int(getattr(cfg, "out_w", 96)))  # what the MODEL sees
        assert self.tasks or self.task, "conf/environments/robocasa.yaml must set `task` or `tasks`"
        self._envs: list = []
        self._rew = torch.zeros(self.batch, device=self.device)
        self._raw: list[dict] = []                              # last obs dict per copy (for rendering)

    # ---- construction (lazy: a kitchen costs seconds, so only build when first reset) --------------
    def _make(self, task: str, seed: int):
        import robocasa            # noqa: F401  -- registers the 396 kitchen envs on import
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        robot = str(getattr(self.cfg, "robot", "PandaOmron"))
        return robosuite.make(
            env_name=task, robots=robot,
            controller_configs=load_composite_controller_config(robot=robot),
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=list(self.cams),
            camera_heights=self.render_size, camera_widths=self.render_size,
            layout_ids=int(getattr(self.cfg, "layout_ids", 4)),
            style_ids=int(getattr(self.cfg, "style_ids", 4)),
            control_freq=int(round(1.0 / float(self.cfg.dt))), seed=seed)

    # ---- WorldEnv contract: REQUIRED --------------------------------------------------------------
    def reset(self, generator: torch.Generator | None = None) -> Tensor:
        """Deterministic given `generator`: copy i is seeded `base + i` and, when `task` is None, draws
        its task from the SAME generator -- so the whole reset is one reproducible stream, matching
        GymBatchAdapter's contract."""
        base = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()) if generator else 0
        picks = ([self.task] * self.batch if self.task else
                 [self.tasks[int(i)] for i in torch.randint(0, len(self.tasks), (self.batch,),
                                                             generator=generator)])
        if not self._envs or [getattr(e, "_qd_task", None) for e in self._envs] != picks:
            for e in self._envs:                                # kitchens hold mujoco contexts; free them
                e.close()
            self._envs = []
            for i, t in enumerate(picks):
                e = self._make(t, base + i)
                e._qd_task = t
                self._envs.append(e)
        self._raw = [e.reset() for e in self._envs]
        self._rew = torch.zeros(self.batch, device=self.device)
        return self._observe()

    def step(self, action: Tensor) -> Tensor:
        """action (B, 12) -> obs (B, 16). A `done` copy is AUTO-RESET (standard vector semantics: the
        stored reward belongs to the step that ended, the returned obs is the fresh one)."""
        a = action.detach().cpu().numpy().astype(np.float64)
        rews = []
        for i, e in enumerate(self._envs):
            o, r, d, _ = e.step(a[i])
            if d:
                o = e.reset()
            self._raw[i] = o
            rews.append(float(r))
        self._rew = torch.tensor(rews, dtype=torch.float32, device=self.device)
        return self._observe()

    def _observe(self) -> Tensor:
        return torch.from_numpy(np.stack([pack_obs(o) for o in self._raw])).to(self.device)

    def reward(self, obs: Tensor | None = None, goal: Tensor | None = None) -> Tensor:
        """robosuite's native task reward from the LAST `step` (zeros before any step). `obs`/`goal` are
        accepted for protocol compatibility and IGNORED -- the reward is a function of full simulator
        state (object poses, fixture contacts) that the 16-dim obs does not carry, so it cannot be
        recomputed from an imagined observation. That is exactly why control here is reward-only and the
        MPPI candidate scoring falls back to its own state cost (see the module docstring)."""
        return self._rew

    def render_obs(self, obs: Tensor | None = None) -> Tensor:
        """THE image modality: (B, h, w, 3) uint8 per camera, concatenated along width when there are
        several, rendered at `render_size` and AREA-downsampled to `out_hw` -- see module note 4.

        `obs` is ignored: robosuite cannot render an arbitrary observation vector, only its own current
        state, so this returns the frames for the state reached by the last reset/step. Callers that
        need a frame for a specific state must step there."""
        from ...data.dataset import resize_frames_area           # THE shared filter, see note 4
        per_cam = []
        for cam in self.cams:
            raw = np.stack([o[f"{cam}_image"] for o in self._raw])          # (B, R, R, 3) uint8
            per_cam.append(resize_frames_area(raw, self.out_hw))
        img = per_cam[0] if len(per_cam) == 1 else np.concatenate(per_cam, axis=2)
        return torch.from_numpy(img).to(self.device)

    # ---- WorldEnv contract: OPTIONAL hooks --------------------------------------------------------
    def position_indices(self) -> list[int]:
        """Obs dims that ARE ambient world xyz: the mobile base position. Unlocks the proprio
        position-trajectory plots, which docs/byo.md notes require an EXPLICIT setting (the [0,1,2]
        fallback only warns and is not used for them). Here [0,1,2] happens to be correct on the
        merits, not by luck -- see robocasa_utils.OBS_LAYOUT."""
        return [0, 1, 2]

    def rollout_metrics(self, pred_obs: Tensor, true_obs: Tensor) -> dict[str, Tensor]:
        """Elementwise over leading dims ((N,H,16) -> (N,H) curves). Five env-meaningful errors instead
        of one whole-vector L2, because that L2 adds metres of base drive to radians of wrist rotation
        and to a gripper aperture in centimetres, and the sum is dominated by whichever happens to be
        largest. Quaternion errors are GEODESIC and sign-invariant (a model's prediction does not
        respect the dataset's canonical-w convention)."""
        s = obs_slices()
        p, t = pred_obs, true_obs
        pn, tn = p.detach().cpu().numpy(), t.detach().cpu().numpy()
        ang = lambda k: torch.from_numpy(quat_angle_error(pn[..., s[k]], tn[..., s[k]])).to(p.device).float()
        norm = lambda k: (p[..., s[k]] - t[..., s[k]]).norm(dim=-1)
        return {"base_pos_error": norm("robot0_base_pos"),
                "eef_pos_error": norm("robot0_base_to_eef_pos"),
                "base_quat_angle_error": ang("robot0_base_quat"),
                "eef_quat_angle_error": ang("robot0_base_to_eef_quat"),
                "gripper_qpos_error": norm("robot0_gripper_qpos"),
                **default_rollout_metrics(pred_obs, true_obs)}

    def render_diagnostics(self, overlay, views) -> dict:
        """Two plan/elevation panels of the kitchen's own extent with the overlay paths drawn on them
        (true/oracle black, pred/learned grey via base.ROLE_STYLE). Views offered: "floor" (world x-y,
        where BASE DRIVE ERROR shows -- the documented robocasa failure is the prediction holding its
        viewpoint while the base drives away) and "side" (x-z, arm height). Unknown view names are
        ignored and {} means "nothing to draw", so callers fall back to the render_obs filmstrip.
        Drawing lives in robocasa_utils.scene_panels; see there for why these are panels and not a
        camera render with projected paths."""
        ex = tuple(float(v) for v in getattr(self.cfg, "floor_extent", (-1.0, 5.0, -5.0, 1.0)))
        return scene_panels(overlay, views, ex, ROLE_STYLE,
                            title=(overlay.extras or {}).get("title") if hasattr(overlay, "extras") else None)
