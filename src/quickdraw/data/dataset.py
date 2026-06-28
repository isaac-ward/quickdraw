"""Loading: windowed (train/val) and full-trajectory (eval) access + train-only normalization.

We store with lerobot (generate.py) but read episodes out and window them ourselves, so the
training loader does not depend on lerobot's `delta_timestamps` behavior. The lerobot read is
isolated in `load_split_episodes` (the one version-sensitive spot for loading).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Train-only mean/std, applied to every split (so OOD shift stays real)."""

    def __init__(self, stats: dict):
        self.o_mean = torch.tensor(stats["observation_vector"]["mean"])
        self.o_std = torch.tensor(stats["observation_vector"]["std"])
        self.a_mean = torch.tensor(stats["action"]["mean"])
        self.a_std = torch.tensor(stats["action"]["std"])

    @classmethod
    def from_file(cls, root: str) -> "Normalizer":
        with open(os.path.join(root, "normalization_stats.json")) as f:
            return cls(json.load(f))

    def norm_obs(self, o):
        return (o - self.o_mean.to(o)) / self.o_std.to(o)

    def denorm_obs(self, o):
        return o * self.o_std.to(o) + self.o_mean.to(o)

    def norm_act(self, a):
        return (a - self.a_mean.to(a)) / self.a_std.to(a)

    def denorm_act(self, a):
        return a * self.a_std.to(a) + self.a_mean.to(a)


def load_split_episodes(root: str, split: str):
    """Return list of (obs (T,6), act (T,2)) float32 arrays. ISOLATED lerobot read."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"torus/{split}", root=os.path.join(root, split))
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    obs_all = np.stack(hf["observation_vector"]).astype(np.float32)
    act_all = np.stack(hf["action"]).astype(np.float32)
    return [(obs_all[ep_idx == e], act_all[ep_idx == e]) for e in np.unique(ep_idx)]


class WindowDataset(Dataset):
    """Length-(P+F) windows. Returns normalized obs_seq (L,6) and act_seq (L,2)."""

    def __init__(self, episodes, P: int, F: int, normalizer: Normalizer):
        self.eps = episodes
        self.P, self.F, self.L = P, F, P + F
        self.norm = normalizer
        self.index = [
            (ei, s) for ei, (o, _) in enumerate(episodes) for s in range(0, len(o) - self.L + 1)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, s = self.index[i]
        o, a = self.eps[ei]
        obs = torch.from_numpy(o[s : s + self.L])
        act = torch.from_numpy(a[s : s + self.L])
        return {"obs_seq": self.norm.norm_obs(obs), "act_seq": self.norm.norm_act(act)}


def stack_windows(episodes, P: int, F: int, normalizer: Normalizer):
    """Pre-build ALL length-(P+F) windows into two normalized tensors (no per-item work later).
    Returns obs_windows (N,L,6), act_windows (N,L,2). For the vector stage N*L*8 floats is tiny."""
    L = P + F
    obs_w, act_w = [], []
    for o, a in episodes:
        if len(o) < L:
            continue
        obs_w.append(torch.from_numpy(o).unfold(0, L, 1).permute(0, 2, 1).contiguous())  # (n,L,6)
        act_w.append(torch.from_numpy(a).unfold(0, L, 1).permute(0, 2, 1).contiguous())  # (n,L,2)
    obs, act = torch.cat(obs_w), torch.cat(act_w)
    return normalizer.norm_obs(obs), normalizer.norm_act(act)


class GPUWindowLoader:
    """DataLoader-shaped iterable over windows held resident on `device`. Batches are produced by
    on-device index_select, so there is no host->device copy and no worker IPC in the train loop.
    Single-device only (no DistributedSampler); fine for one H100."""

    def __init__(self, obs_windows, act_windows, batch: int, shuffle: bool, device):
        self.obs = obs_windows.to(device)
        self.act = act_windows.to(device)
        self.N, self.batch, self.shuffle = self.obs.shape[0], batch, shuffle

    def __len__(self):
        return (self.N + self.batch - 1) // self.batch

    def __iter__(self):
        dev = self.obs.device
        idx = torch.randperm(self.N, device=dev) if self.shuffle else torch.arange(self.N, device=dev)
        for i in range(0, self.N, self.batch):
            j = idx[i : i + self.batch]
            yield {"obs_seq": self.obs.index_select(0, j), "act_seq": self.act.index_select(0, j)}


class TrajectoryDataset(Dataset):
    """Whole episodes for long-horizon eval. Returns normalized full obs/act sequences."""

    def __init__(self, episodes, normalizer: Normalizer):
        self.eps = episodes
        self.norm = normalizer

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        o, a = self.eps[i]
        return {
            "obs_seq": self.norm.norm_obs(torch.from_numpy(o)),
            "act_seq": self.norm.norm_act(torch.from_numpy(a)),
        }
