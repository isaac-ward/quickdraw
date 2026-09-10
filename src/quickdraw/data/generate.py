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


def compute_norm_stats(obs: np.ndarray, act: np.ndarray,
                       obs_identity: "list[int] | None" = None,
                       act_identity: "list[int] | None" = None) -> dict:
    """Mean/std over the train split (flattened over traj & time).

    `obs_identity` / `act_identity` list dims to LEAVE ALONE -- written as mean 0, std 1, so
    data/dataset.Normalizer passes them through unchanged. Use it for dims that are already
    bounded AND geometrically coupled to each other.

    WHY THIS EXISTS. A 6D rotation (the first two columns of a rotation matrix) is six numbers in
    [-1, 1] obeying |c0| = |c1| = 1 and c0 . c1 = 0. Z-scoring scales each of the six by a
    DIFFERENT factor, and on the block-stack corpus those factors are
    [3.5, 7.0, 324.4, 4.9, 2.5, 301.6] -- because yaw about the world vertical leaves the bottom
    row of the rotation matrix invariant, so two of the six barely move. Measured on real data,
    that turns exactly-orthonormal columns (norms 1.000, dot 0.0000) into norms spanning
    0.47-42.2 with dots up to 786.9. The very structure the 6D representation exists to expose is
    destroyed before the model sees it, and the two near-constant dims additionally have their
    sensor jitter amplified to 1.25 and 0.26 sigma.

    Per-dim z-scoring is right for dims that are independent and differ in unit or scale (mm vs
    radians). It is wrong for a group of dims that together encode one geometric object.
    """
    o = obs.reshape(-1, obs.shape[-1])
    a = act.reshape(-1, act.shape[-1])
    eps = 1e-6
    o_mean, o_std = o.mean(0), o.std(0) + eps
    a_mean, a_std = a.mean(0), a.std(0) + eps
    diag = {}
    for tag, x, arr_m, arr_s, idx in (("observation_vector", o, o_mean, o_std, obs_identity),
                                      ("action", a, a_mean, a_std, act_identity)):
        keep = set()
        for i in (idx or []):
            assert 0 <= int(i) < x.shape[-1], f"identity dim {i} out of range for {tag}"
            arr_m[int(i)], arr_s[int(i)] = 0.0, 1.0
            keep.add(int(i))
        diag[tag] = _concentration_warn(tag, x, arr_s, keep)
    return {
        "observation_vector": {"mean": o_mean.tolist(), "std": o_std.tolist()},
        "action": {"mean": a_mean.tolist(), "std": a_std.tolist()},
        # Advisory only; no consumer reads it. Kept so a dataset records what its own build knew.
        "concentration_diagnostic": diag,
    }


# std/range below this = the dim sits at one value with rare excursions, so z-scoring it mostly
# amplifies noise. Scale-free (units cancel) and offset-free, unlike 1/std, which is 1000x
# different for the same quantity in metres and millimetres. A Gaussian sits near 0.17; on the
# block-stack corpus the two structurally-pinned rotation dims scored 0.013 and 0.016 while the
# lowest healthy dim scored 0.072, so 0.05 separates them with margin on both sides.
CONCENTRATION_WARN = 0.05


def _concentration_warn(tag: str, x: np.ndarray, std: np.ndarray, exempt: set) -> dict:
    """Flag dims that are effectively constant, so nobody ships a 300x noise amplifier by accident.

    THIS ONLY WARNS. Whether a near-constant dim should be left unnormalized, dropped, or kept as
    is depends on what it MEANS, and only the processor knows that -- hence `obs_identity` rather
    than an automatic rule. What is not acceptable is the failure being silent: models/features.py
    documents robocasa action dims whose std floors near 1e-6, turning a real 0.5 deviation into
    5e5, and that was found by debugging a model rather than by building a dataset.
    """
    rng = x.max(0) - x.min(0)
    ratio = np.where(rng > 0, std / np.maximum(rng, 1e-12), 0.0)
    bad = [int(i) for i in np.argsort(ratio)
           if i not in exempt and ratio[i] < CONCENTRATION_WARN and rng[i] > 0]
    if bad:
        print(f"[norm] WARNING: {len(bad)} {tag} dim(s) are near-constant and will be z-scored "
              f"anyway (std/range < {CONCENTRATION_WARN}):", flush=True)
        for i in bad:
            print(f"[norm]   dim {i:3d}  std {std[i]:.6g}  range {rng[i]:.6g}  "
                  f"std/range {ratio[i]:.4f}  -> x{1 / std[i]:.0f} amplification", flush=True)
        print("[norm]   Mostly noise gets amplified. If these dims jointly encode one geometric "
              "object (a 6D rotation, a quaternion), pass them as obs_identity/act_identity.",
              flush=True)
    return {"std_over_range": [round(float(v), 6) for v in ratio],
            "flagged_dims": bad, "threshold": CONCENTRATION_WARN,
            "exempt_dims": sorted(exempt)}


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
