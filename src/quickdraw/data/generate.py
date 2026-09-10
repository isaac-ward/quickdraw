"""Dataset generation: roll out a WorldEnv -> LeRobotDataset splits (design/data.md, design/gym_refactor.md).

Splits: train, val, eval_ood_horizon, eval_ood_visual, eval_ood_geometric, eval_ood_dynamics.
Each split stores `observation_vector`, `action`, and `observation.images.fpv` (the egocentric video,
the lerobot-standard image observation). The FPV frames are rendered SEPARATELY at full parallelism
(see data_generation.py) and only INGESTED here, so the lerobot writing (one dataset per split) is
not the parallelism bottleneck.

NOTE: the lerobot writer calls are isolated in `write_lerobot_split` — that is the one place to
adjust if the installed lerobot API differs (it has drifted across versions).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import torch

from ..environments.examples.torus import TorusEnv
from ..environments.policies import make_policy
from ..environments.torus_utils import TorusConfig


def generate_episodes(env, n_traj: int, steps: int, seed: int, device="cpu",
                      action_sampler: str = "ornstein_uhlenbeck", policy=None):
    """Return obs (n_traj, steps, obs_dim) and act (n_traj, steps, action_dim) as float32 numpy arrays.

    Env-agnostic: `env` is any batched WorldEnv (batch == n_traj, on `device`) rolled with a behavior
    `policy` (environments/policies.py; default built by name from `action_sampler`). Legacy signature —
    `env` may be a TorusConfig, in which case the TorusEnv is built here. Both paths are byte-identical
    to the pre-refactor torus loop (proven by smoke/refactor_parity_datagen).

    All n_traj episodes are simulated in parallel as one batched env. action[:, t] is the action
    applied at step t (producing obs[:, t+1]); the final action is recorded but unused downstream.

    RNG CONTRACT — the data is a pure function of `seed` via ONE generator consumed in EXACTLY this
    order: env.reset(g), policy.reset(g), then per step policy.sample(obs, g) -> env.step(a). Adding,
    removing or reordering ANY draw changes every generated dataset.
    """
    if isinstance(env, TorusConfig):
        env = TorusEnv(env, batch=n_traj, device=device)
    if policy is None:
        policy = make_policy(action_sampler, env, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    obs0 = env.reset(g)
    policy.reset(g)
    obs_list, act_list = [obs0], []
    for _ in range(steps - 1):
        a = policy.sample(obs_list[-1], g)   # current obs -> state-dependent policies (bimodal)
        act_list.append(a)
        obs_list.append(env.step(a))
    act_list.append(policy.sample(obs_list[-1], g))  # pad last action so shapes match (unused)
    obs = torch.stack(obs_list, dim=1).cpu().numpy().astype(np.float32)
    act = torch.stack(act_list, dim=1).cpu().numpy().astype(np.float32)
    return obs, act


def compute_norm_stats(obs: np.ndarray, act: np.ndarray) -> dict:
    """Mean/std over the train split (flattened over traj & time)."""
    o = obs.reshape(-1, obs.shape[-1])
    a = act.reshape(-1, act.shape[-1])
    eps = 1e-6
    return {
        "observation_vector": {"mean": o.mean(0).tolist(), "std": (o.std(0) + eps).tolist()},
        "action": {"mean": a.mean(0).tolist(), "std": (a.std(0) + eps).tolist()},
    }


def _read_frames(path: str) -> np.ndarray:
    """Read an mp4 back to (T,H,W,3) uint8 (the pre-rendered per-episode FPV clip)."""
    import imageio.v2 as imageio

    rd = imageio.get_reader(path)
    frames = np.stack([f[..., :3] for f in rd])
    rd.close()
    return frames


def write_lerobot_split(root, repo_id: str, obs, act, fps: int,
                        fpv_dir=None, fpv_size=256, cam="fpv", task="torus"):
    """Write episodes to a LeRobotDataset on disk. ISOLATED lerobot API surface.

    MULTI-CAMERA. `cam` is a camera name OR a list of them, and `fpv_dir` correspondingly a directory
    OR a {cam: dir} mapping. Each episode's pre-rendered clip `<dir>/ep_<i>.mp4` is read back and
    stored as the lerobot-standard image observation `observation.images.<cam>` (dtype=video), aligned
    1:1 with the vector frames -- one such key PER CAMERA, which is exactly the layout
    `data/dataset.py` already reads (`<split>/videos/observation.images.<cam>/*/*.mp4`), so a
    multi-camera dataset needs no loader change at all: pick one with `data.cam`.

    `obs`/`act`: (n_traj, steps, dim) arrays OR lists of per-episode (T_i, dim) arrays (recorded
    episodes have variable length). `fpv_size`: int (square) or (H, W); all cameras share it.

    Passing a bare str + str (the torus/generate path) behaves exactly as before.

    `task`: one string for every episode (the historical behaviour), OR a per-episode sequence of
    len(obs) strings. WHY PER-EPISODE MATTERS: lerobot's `task` is the ONLY free-text field that
    survives packaging into `<split>/meta/tasks.parquet`, and a single dataset-wide constant throws
    away whatever distinguished one recording from another. Measured cost of getting this wrong: the
    `starling-2` eval split ships 49 episodes drawn from four separate OOD campaigns, every one of them
    labelled `'starling-2'`, so which episodes were the visual-shift ones and which the dynamics-shift
    ones is UNRECOVERABLE from the published dataset -- the information existed upstream and was
    flattened here. Per-episode tasks are how a consumer slices a dataset by condition.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cams = [cam] if isinstance(cam, str) else list(cam)
    dirs = {cams[0]: fpv_dir} if isinstance(fpv_dir, str) else dict(fpv_dir or {})
    video = bool(dirs)
    features = {
        "observation_vector": {"dtype": "float32", "shape": (obs[0].shape[-1],), "names": None},
        "action": {"dtype": "float32", "shape": (act[0].shape[-1],), "names": None},
    }
    if video:   # vector-only splits pass fpv_dir=None and no fpv_size -> no image feature
        h, w = (fpv_size, fpv_size) if isinstance(fpv_size, int) else tuple(fpv_size)
        for c in cams:
            features[f"observation.images.{c}"] = {"dtype": "video", "shape": (h, w, 3),
                                                   "names": ["height", "width", "channels"]}
    tasks = [str(task)] * len(obs) if isinstance(task, str) else [str(t) for t in task]
    assert len(tasks) == len(obs), f"task list has {len(tasks)} entries for {len(obs)} episodes"
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, features=features, use_videos=video)
    for i in range(len(obs)):
        # decode every camera's clip for THIS episode up front; they are frame-aligned by construction
        frames = {c: _read_frames(os.path.join(dirs[c], f"ep_{i:04d}.mp4")) for c in cams} if video else None
        for t in range(len(obs[i])):
            f = {"observation_vector": obs[i][t], "action": act[i][t], "task": tasks[i]}
            if video:
                for c in cams:
                    f[f"observation.images.{c}"] = frames[c][t]
            ds.add_frame(f)
        ds.save_episode()
    ds.finalize()
    return ds


def write_meta(root_dir: str, base_env: TorusConfig, splits, split_env: dict, coloring: dict,
               fps: int, stats: dict):
    """Write normalization_stats.json + dataset_card.json (the split_env is the resolved env per
    split, so eval scores each split on its OWN manifold)."""
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(root_dir, "dataset_card.json"), "w") as f:
        json.dump({"base_env": asdict(base_env), "split_env": split_env, "coloring": coloring,
                   # `split_rule` records HOW train/val were chosen -- "random, seed 0" or a
                   # processor's own rule. Without it a longest-first split is indistinguishable
                   # from a random one after the fact, and the two mean very different things
                   # when you read a val number.
                   "splits": {k: {"n_traj": int(v["n_traj"]), "steps": int(v["steps"]),
                                  "seed": int(v["seed"]),
                                  **({"split_rule": str(v["split_rule"])} if "split_rule" in v else {})}
                              for k, v in splits.items()}, "fps": fps}, f, indent=2)
