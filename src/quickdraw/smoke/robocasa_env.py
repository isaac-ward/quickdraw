"""Does the live robocasa env stand in for the dataset it was built to match?

Line count is not the risk here; SILENT MISMATCH is. An env with the wrong obs keys, the wrong resize
filter or the wrong action width resets, steps and renders perfectly happily while emitting data the
trained model cannot interpret. These checks are the ones that would catch that.

    python -m quickdraw.smoke.robocasa_env
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from ..data.dataset import load_split_episodes, resize_frames_area
from ..environments.base import ROLE_STYLE, SceneOverlay, log_env_capabilities
from ..environments.registry import make_env
from ..environments.robocasa_utils import OBS_LAYOUT, obs_slices

DATA = "logs/recording_2026_09_01_05_30_23_robocasa_scene4_4h_3cam"
REPO = "robocasa_scene4_4h_3cam"
ok = bad = 0


def check(name, cond, extra=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}", flush=True)


def main() -> int:
    cfg = OmegaConf.load("conf/environments/robocasa.yaml")
    env = make_env(cfg.name, cfg, batch=2, device="cpu")
    check("registry builds it", type(env).__name__ == "RoboCasaEnv")
    check("dims match the dataset card", (env.obs_dim, env.action_dim) == (16, 12),
          f"obs {env.obs_dim} act {env.action_dim}")

    o = env.reset(torch.Generator().manual_seed(0))
    check("reset -> (B, 16)", tuple(o.shape) == (2, 16), str(tuple(o.shape)))
    o2 = env.step(torch.zeros(2, env.action_dim))
    check("step -> (B, 16)", tuple(o2.shape) == (2, 16), str(tuple(o2.shape)))
    check("reward -> (B,)", tuple(env.reward().shape) == (2,))

    # THE check that matters: every obs dim inside the range the DATASET actually contains.
    eps = load_split_episodes(DATA, "val", repo_id=REPO)
    D = np.concatenate([e for e, _ in eps])
    lo, hi = D.min(0), D.max(0)
    v = o2[0].numpy()
    pad = 0.15 * np.maximum(hi - lo, 1e-3)          # 15% slack: the env is one state, not the distribution
    inside = (v >= lo - pad) & (v <= hi + pad)
    check("every obs dim inside the dataset's range", bool(inside.all()),
          "" if inside.all() else f"outside at dims {list(np.where(~inside)[0])} -> "
          f"{[round(float(v[i]), 3) for i in np.where(~inside)[0]]} vs "
          f"{[(round(float(lo[i]),3), round(float(hi[i]),3)) for i in np.where(~inside)[0]]}")

    s = obs_slices()
    check("base_quat is unit norm, yaw-only (dims 3,4 == 0)",
          abs(np.linalg.norm(v[s['robot0_base_quat']]) - 1) < 1e-4 and abs(v[3]) < 1e-9 and abs(v[4]) < 1e-9)
    check("eef_quat is unit norm", abs(np.linalg.norm(v[s['robot0_base_to_eef_quat']]) - 1) < 1e-4)
    check("gripper dims are a mirror pair", abs(v[14] + v[15]) < 5e-3, f"{v[14]:+.4f} {v[15]:+.4f}")
    check("base z is pinned near the dataset's 0.70", abs(v[2] - 0.70) < 0.02, f"{v[2]:.4f}")

    # render_obs: right shape, and the SAME filter as the training cache.
    img = env.render_obs()
    exp_w = int(cfg.out_w) * len(cfg.cameras)
    check("render_obs -> (B, h, w*ncam, 3) uint8", tuple(img.shape) == (2, int(cfg.out_h), exp_w, 3)
          and img.dtype == torch.uint8, str(tuple(img.shape)))
    raw = np.stack([env._raw[0][f"{c}_image"] for c in cfg.cameras])
    check("render_obs uses the SHARED area filter (bit-identical to the cache builder)",
          bool((np.concatenate(list(resize_frames_area(raw, (int(cfg.out_h), int(cfg.out_w)))), axis=1)
                == img[0].numpy()).all()))

    # (N, H, 16) rollout tensors: N=2 episodes, H=3 steps. NOT o2[None].repeat(1,3,1) -- that stacks the
    # BATCH into the horizon axis and yields (1, 6, 16), which is how this check first mis-asserted (1, 3).
    P = o2.unsqueeze(1).expand(2, 3, 16).contiguous()
    T = o.unsqueeze(1).expand(2, 3, 16).contiguous()
    m = env.rollout_metrics(P, T)
    check("rollout_metrics: 5 env metrics + the generic fallback",
          {"base_pos_error", "eef_pos_error", "base_quat_angle_error", "eef_quat_angle_error",
           "gripper_qpos_error"} <= set(m), str(sorted(m)))
    check("rollout_metrics are elementwise (N,H) curves",
          all(tuple(x.shape) == (2, 3) for x in m.values()),
          str({k: tuple(v.shape) for k, v in m.items()}))
    check("identical obs -> zero error on every metric",
          max(float(x.abs().max()) for x in env.rollout_metrics(T, T).values()) < 1e-6)

    d = env.render_diagnostics(SceneOverlay(agents={"true": np.random.randn(12, 3) * 0.5,
                                                    "pred": np.random.randn(12, 3) * 0.5}),
                               ["floor", "side"])
    check("render_diagnostics -> floor + side panels", set(d) == {"floor", "side"}, str(sorted(d)))
    check("unknown view -> {} (caller falls back to the filmstrip)",
          env.render_diagnostics(SceneOverlay(agents={"true": np.zeros((3, 3))}), ["nope"]) == {})
    check("no fork -> the oracle is correctly unavailable", not hasattr(env, "fork"))

    print()
    log_env_capabilities(env, name="robocasa")
    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
