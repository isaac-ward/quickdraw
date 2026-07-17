"""Dataset generation: roll out TorusEnv -> LeRobotDataset splits (design/data.md).

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

from ..environments.torus import BimodalActionSampler, OUActionSampler, TorusConfig, TorusEnv


def _build_sampler(kind: str, n_traj: int, a_max: float, device):
    """Action process for data gen: 'ou' (unimodal OU) or 'bimodal' (two-basin action-magnitude)."""
    if kind == "bimodal":
        return BimodalActionSampler(n_traj, a_max, device=device)
    if kind == "ou":
        return OUActionSampler(n_traj, a_max, device=device)
    raise ValueError(f"unknown action_sampler {kind!r} (expected 'ou' or 'bimodal')")


def generate_episodes(env_cfg: TorusConfig, n_traj: int, steps: int, seed: int, device="cpu",
                      action_sampler: str = "ou"):
    """Return obs (n_traj, steps, 6) and act (n_traj, steps, 2) as float32 numpy arrays.

    All n_traj episodes are simulated in parallel as one batched env. action[:, t] is the action
    applied at step t (producing obs[:, t+1]); the final action is recorded but unused downstream.
    `action_sampler`: "ou" (unimodal OU, default) or "bimodal" (two-basin action-magnitude process).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    env = TorusEnv(env_cfg, batch=n_traj, device=device)
    sampler = _build_sampler(action_sampler, n_traj, env_cfg.a_max, device)
    env.reset(g)
    sampler.reset(g)
    obs_list, act_list = [env.observe()], []
    for _ in range(steps - 1):
        a = sampler.sample(g, state=obs_list[-1])   # state = current obs -> state-dependent samplers (bimodal)
        act_list.append(a)
        obs_list.append(env.step(a))
    act_list.append(sampler.sample(g, state=obs_list[-1]))  # pad last action so shapes match (unused)
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


def write_lerobot_split(root, repo_id: str, obs: np.ndarray, act: np.ndarray, fps: int,
                        fpv_dir: str | None = None, fpv_size: int = 256):
    """Write episodes to a LeRobotDataset on disk. ISOLATED lerobot API surface.

    When `fpv_dir` is given, each episode's pre-rendered clip `fpv_dir/ep_<i>.mp4` is read back and
    stored as the lerobot-standard image observation `observation.images.fpv` (dtype=video), aligned
    1:1 with the vector frames.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    video = fpv_dir is not None
    features = {
        "observation_vector": {"dtype": "float32", "shape": (obs.shape[-1],), "names": None},
        "action": {"dtype": "float32", "shape": (act.shape[-1],), "names": None},
    }
    if video:
        features["observation.images.fpv"] = {"dtype": "video", "shape": (fpv_size, fpv_size, 3),
                                              "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, features=features, use_videos=video)
    n_traj, steps, _ = obs.shape
    for i in range(n_traj):
        frames = _read_frames(os.path.join(fpv_dir, f"ep_{i:04d}.mp4")) if video else None
        for t in range(steps):
            f = {"observation_vector": obs[i, t], "action": act[i, t], "task": "torus"}
            if video:
                f["observation.images.fpv"] = frames[t]
            ds.add_frame(f)
        ds.save_episode()
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
                   "splits": {k: {"n_traj": int(v["n_traj"]), "steps": int(v["steps"]), "seed": int(v["seed"])}
                              for k, v in splits.items()}, "fps": fps}, f, indent=2)
