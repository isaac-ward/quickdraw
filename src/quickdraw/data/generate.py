"""Dataset generation: roll out TorusEnv -> LeRobotDataset splits (design/data.md).

Splits: train, val, eval_ind, eval_ood_visual, eval_ood_geometric, eval_ood_dynamics.
Low-dim only for the vector stage (observation_vector, action); images join at the image stage
through the same lerobot writer with no schema change.

NOTE: the lerobot writer calls are isolated in `write_lerobot_split` — that is the one place to
adjust if the installed lerobot API differs (it has drifted across versions).
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import torch

from ..environments.torus import OUActionSampler, TorusConfig, TorusEnv


def generate_episodes(env_cfg: TorusConfig, n_traj: int, steps: int, seed: int, device="cpu"):
    """Return obs (n_traj, steps, 6) and act (n_traj, steps, 2) as float32 numpy arrays.

    All n_traj episodes are simulated in parallel as one batched env. action[:, t] is the action
    applied at step t (producing obs[:, t+1]); the final action is recorded but unused downstream.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    env = TorusEnv(env_cfg, batch=n_traj, device=device)
    ou = OUActionSampler(n_traj, env_cfg.a_max, device=device)
    env.reset(g)
    ou.reset(g)
    obs_list, act_list = [env.observe()], []
    for _ in range(steps - 1):
        a = ou.sample(g)
        act_list.append(a)
        obs_list.append(env.step(a))
    act_list.append(ou.sample(g))  # pad last action so shapes match (unused)
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


def write_lerobot_split(root, repo_id: str, obs: np.ndarray, act: np.ndarray, fps: int):
    """Write episodes to a LeRobotDataset on disk. ISOLATED lerobot API surface."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation_vector": {"dtype": "float32", "shape": (obs.shape[-1],), "names": None},
        "action": {"dtype": "float32", "shape": (act.shape[-1],), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, features=features, use_videos=False)
    n_traj, steps, _ = obs.shape
    for i in range(n_traj):
        for t in range(steps):
            ds.add_frame({"observation_vector": obs[i, t], "action": act[i, t], "task": "torus"})
        ds.save_episode()
    return ds


def _gen_write_split(task: dict):
    """Worker: generate one split's episodes and write the lerobot split. Returns metadata + (for
    train) the normalization stats."""
    cfg = replace(TorusConfig(**task["base"]), **task["env"])
    obs, act = generate_episodes(cfg, task["n_traj"], task["steps"], task["seed"])
    write_lerobot_split(task["root_split"], f"torus/{task['name']}", obs, act, task["fps"])
    stats = compute_norm_stats(obs, act) if task["name"] == "train" else None
    return task["name"], asdict(cfg), task["coloring"], stats


def build_all(base_env: TorusConfig, root_dir: str, splits, fps: int = 30, device="cpu") -> dict:
    """Generate every split from the config `splits` mapping (in parallel across processes); return
    train-only norm stats. `splits[name]` has `n_traj, steps, seed, coloring, env`."""
    tasks = [{"name": name, "base": asdict(base_env), "env": dict(s.get("env", {}) or {}),
              "n_traj": int(s["n_traj"]), "steps": int(s["steps"]), "seed": int(s["seed"]),
              "coloring": s.get("coloring", "hsv"), "fps": fps,
              "root_split": os.path.join(root_dir, name)} for name, s in splits.items()]
    stats, split_env, coloring = None, {}, {}
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 4)) as ex:
        for nm, env_d, col, st in ex.map(_gen_write_split, tasks):
            split_env[nm] = env_d        # resolved env so eval scores each split on ITS OWN manifold
            coloring[nm] = col
            if st is not None:
                stats = st
    os.makedirs(root_dir, exist_ok=True)
    with open(os.path.join(root_dir, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(root_dir, "dataset_card.json"), "w") as f:
        json.dump({"base_env": asdict(base_env), "split_env": split_env, "coloring": coloring,
                   "splits": {k: {"n_traj": int(v["n_traj"]), "steps": int(v["steps"]), "seed": int(v["seed"])}
                              for k, v in splits.items()}, "fps": fps}, f, indent=2)
    return stats
