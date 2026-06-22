"""Shared builders: cfg -> dataclasses, model, datasets, loaders. Used by all entrypoints."""

from __future__ import annotations

import os

from torch.utils.data import DataLoader

from ..data.dataset import Normalizer, TrajectoryDataset, WindowDataset, load_split_episodes
from ..environments.torus import TorusConfig
from ..models.base import BaseModelConfig, BaseWorldModel


def env_cfg(cfg) -> TorusConfig:
    e = cfg.environments
    return TorusConfig(R=e.R, r=e.r, dt=e.dt, gamma=e.gamma, a_max=e.a_max, init_speed=e.init_speed, mass=e.mass)


def build_model(cfg) -> BaseWorldModel:
    m = cfg.model
    return BaseWorldModel(BaseModelConfig(
        d=m.d, depth=m.depth, heads=m.heads, window=m.window,
        mlp_ratio=m.mlp_ratio, rope_theta=m.rope_theta,
    ))


def normalizer(cfg) -> Normalizer:
    return Normalizer.from_file(cfg.data.root)


def window_loaders(cfg, norm: Normalizer):
    P, F = cfg.data.P, cfg.data.F
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        ds = WindowDataset(load_split_episodes(cfg.data.root, split), P, F, norm)
        loaders[split] = DataLoader(
            ds, batch_size=cfg.data.batch, shuffle=shuffle, num_workers=cfg.data.workers,
            pin_memory=True, persistent_workers=cfg.data.workers > 0,
            prefetch_factor=4 if cfg.data.workers > 0 else None,
        )
    return loaders


def eval_episodes(cfg, norm: Normalizer, split: str):
    return TrajectoryDataset(load_split_episodes(cfg.data.root, split), norm)


def data_exists(cfg) -> bool:
    return bool(cfg.data.root) and os.path.exists(os.path.join(cfg.data.root, "normalization_stats.json"))


def load_checkpoint(model, path: str):
    """Load a Lightning checkpoint into a bare BaseWorldModel, stripping wrapper prefixes.

    `path` may be a .ckpt file or a train run dir (resolved to <dir>/checkpoints/best.ckpt).
    """
    import torch

    if path and os.path.isdir(path):
        path = os.path.join(path, "checkpoints", "best.ckpt")
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    clean = {}
    for k, v in sd.items():
        for pre in ("model._orig_mod.", "model."):
            if k.startswith(pre):
                k = k[len(pre):]
                break
        clean[k] = v
    model.load_state_dict(clean, strict=False)
    return model
