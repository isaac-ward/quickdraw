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


def load_split_episodes(root: str, split: str, repo_id: str = "torus"):
    """Return list of (obs (T,D), act (T,A)) float32 arrays. ISOLATED lerobot read. `repo_id` is the
    prefix the split was written with (<repo_id>/<split>; torus datasets = "torus")."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"{repo_id}/{split}", root=os.path.join(root, split))
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    obs_all = np.stack(hf["observation_vector"]).astype(np.float32)
    act_all = np.stack(hf["action"]).astype(np.float32)
    return [(obs_all[ep_idx == e], act_all[ep_idx == e]) for e in np.unique(ep_idx)]


def load_fpv_frames(root: str, split: str, size: int | tuple[int, int] | None = 128,
                    max_frames: int | None = None, cache: bool = True, cam: str = "fpv"):
    """All egocentric frames for a split (lerobot chunked video, camera `cam`), AREA-downsampled ONCE
    and cached to disk (npy next to the split). `size`: int -> size×size (torus default), (H, W) tuple,
    or None -> native resolution (no resize). Returns uint8 (N, H, W, 3). The downsample is the
    only per-frame work and it's cached, so repeat loads are instant (mmap)."""
    import glob as _glob

    import imageio.v2 as imageio
    hw = (size, size) if isinstance(size, int) else (tuple(size) if size is not None else None)
    tag = size if isinstance(size, int) else ("native" if hw is None else f"{hw[0]}x{hw[1]}")
    cache_path = os.path.join(root, split, f"{cam}_{tag}.npy")
    if cache and max_frames is None and os.path.exists(cache_path):
        return np.load(cache_path)
    vid = os.path.join(root, split, "videos", f"observation.images.{cam}")
    mp4s = sorted(_glob.glob(os.path.join(vid, "*", "*.mp4")))
    assert mp4s, f"no {cam} mp4s under {vid}"
    out, buf = [], []

    def _flush():
        if not buf:
            return
        if hw is None:                                                         # native: no resize
            out.append(np.stack(buf))
            buf.clear()
            return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float()       # (b,3,H,W)
        x = torch.nn.functional.interpolate(x, size=hw, mode="area")           # anti-aliased downsample
        out.append(x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy())
        buf.clear()

    def _have():
        return sum(len(o) for o in out) + len(buf)

    for p in mp4s:
        rd = imageio.get_reader(p)
        for fr in rd:
            buf.append(np.asarray(fr)[..., :3])
            if len(buf) >= 512:
                _flush()
            if max_frames is not None and _have() >= max_frames:
                break
        rd.close()
        if max_frames is not None and _have() >= max_frames:
            break
    _flush()
    frames = np.concatenate(out, 0)
    if max_frames is not None:
        frames = frames[:max_frames]
    elif cache:
        np.save(cache_path, frames)
    return frames


def load_split_episodes_mm(root: str, split: str, img_size: int | tuple[int, int] | None = 128,
                           cam: str = "fpv", repo_id: str = "torus"):
    """Like load_split_episodes but ALSO returns per-episode camera frames (area-downsampled to img_size,
    uint8), aligned 1:1 with obs steps. Returns list of (obs (T,D), act (T,A), img (T,H,W,3) uint8).
    The chunked video is read in dataset row order (== obs row order), then split by episode_index."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"{repo_id}/{split}", root=os.path.join(root, split))
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    obs_all = np.stack(hf["observation_vector"]).astype(np.float32)
    act_all = np.stack(hf["action"]).astype(np.float32)
    frames_all = load_fpv_frames(root, split, size=img_size, cam=cam)      # (N, H, W, 3), row order
    assert len(frames_all) == len(obs_all), f"{cam}/row count mismatch: {len(frames_all)} vs {len(obs_all)}"
    return [(obs_all[ep_idx == e], act_all[ep_idx == e], frames_all[ep_idx == e]) for e in np.unique(ep_idx)]


class MMWindowLoader:
    """The ONE GPU-resident window loader for every model. obs/act windows live on `device`; when an image
    modality is present (`image_head` set) its FRAME store (uint8, ~3 GB at 128²) is held resident too and
    gathered per batch by a pure GPU index (no host copy). PROPRIO-ONLY passes `image_head=None` -> no frames
    are loaded or gathered (episodes are (obs, act) pairs). Yields {obs_seq (B,L,6), act_seq (B,L,2)[,
    <image_head> (B,L,H,W,3) in [0,1]]}. Window order matches `stack_windows`, so all streams stay aligned."""

    def __init__(self, episodes, P: int, F: int, normalizer: Normalizer, batch: int, shuffle: bool, device,
                 image_head: str | None = None, stride: int = 1):
        L = P + F
        self.image_head = image_head
        obs_w, act_w = stack_windows([(e[0], e[1]) for e in episodes], P, F, normalizer, stride)
        self.obs, self.act = obs_w.to(device), act_w.to(device)
        self.frames = None
        if image_head is not None:   # concat all episode frames -> one GPU uint8 store + per-window GLOBAL frame idx
            frames, starts, off = [], [], 0
            for e in episodes:
                o, img = e[0], e[2]
                frames.append(torch.from_numpy(img))
                starts.extend(range(off, off + len(o) - L + 1, stride))   # stride matches stack_windows -> streams stay aligned
                off += len(img)
            self.frames = torch.cat(frames, 0).to(device)                    # (N_total,H,W,3) uint8, GPU-resident
            starts = torch.tensor(starts, device=device)
            self.win_idx = starts[:, None] + torch.arange(L, device=device)[None]
        self.batch, self.shuffle, self.device = batch, shuffle, device
        self.N = self.obs.shape[0]

    def __len__(self):
        return (self.N + self.batch - 1) // self.batch

    def __iter__(self):
        order = torch.randperm(self.N, device=self.device) if self.shuffle else torch.arange(self.N, device=self.device)
        for i in range(0, self.N, self.batch):
            j = order[i: i + self.batch]
            out = {"obs_seq": self.obs.index_select(0, j), "act_seq": self.act.index_select(0, j)}
            if self.frames is not None:
                out[self.image_head] = self.frames[self.win_idx.index_select(0, j)].float().div_(255.0)  # GPU gather
            yield out


def stack_windows(episodes, P: int, F: int, normalizer: Normalizer, stride: int = 1):
    """Pre-build the length-(P+F) windows (every `stride` starts) into two normalized tensors (no per-item
    work later). Returns obs_windows (N,L,6), act_windows (N,L,2). stride>1 drops near-duplicate overlapping
    windows (adjacent starts share L-1 steps) -> fewer batches/epoch, ~no coverage loss over many epochs."""
    L = P + F
    obs_w, act_w = [], []
    for o, a in episodes:
        if len(o) < L:
            continue
        obs_w.append(torch.from_numpy(o).unfold(0, L, stride).permute(0, 2, 1).contiguous())  # (n,L,6)
        act_w.append(torch.from_numpy(a).unfold(0, L, stride).permute(0, 2, 1).contiguous())  # (n,L,2)
    obs, act = torch.cat(obs_w), torch.cat(act_w)
    return normalizer.norm_obs(obs), normalizer.norm_act(act)


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
