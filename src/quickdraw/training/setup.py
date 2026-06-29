"""Shared builders: cfg -> dataclasses, model, datasets, loaders. Used by all entrypoints."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from ..data.dataset import (
    GPUWindowLoader, Normalizer, TrajectoryDataset, WindowDataset, load_split_episodes, stack_windows,
)
from ..environments.torus import TorusConfig
from ..models.base import BaseModelConfig, BaseWorldModel


def env_cfg(cfg) -> TorusConfig:
    e = cfg.environments
    return TorusConfig(R=e.R, r=e.r, dt=e.dt, gamma=e.gamma, a_max=e.a_max, init_speed=e.init_speed, mass=e.mass)


def build_model(cfg):
    """Dispatch on cfg.model.name: data-space (DSAR) or latent-space (LSAR + a collapse mechanism)."""
    m = cfg.model
    name = str(m.get("name", "base"))
    if name in ("base", "dsar", "data_space_autoregressor"):
        return BaseWorldModel(BaseModelConfig(
            d=m.d, depth=m.depth, heads=m.heads, window=m.window,
            mlp_ratio=m.mlp_ratio, rope_theta=m.rope_theta,
        ))
    if name in ("lsar", "latent_space_autoregressor"):
        from ..models.collapse import Reconstruction, make_collapse
        from ..models.lsar import LatentSpaceAR, LSARConfig
        strat = make_collapse(cfg.collapse) if cfg.get("collapse", None) is not None else Reconstruction()
        return LatentSpaceAR(LSARConfig(
            d=m.d, dz=m.dz, depth=m.depth, heads=m.heads, window=m.window, mlp_ratio=m.mlp_ratio,
            rope_theta=m.rope_theta, dec_hidden=m.get("dec_hidden", 64),
            lambda_pred_obs=m.get("lambda_pred_obs", 1.0), lambda_reg=m.get("lambda_reg", 1.0),
            expander_hidden=m.get("expander_hidden", 256), expander_dim=m.get("expander_dim", 256)), strat)
    if name in ("diffusion",):
        from ..models.diffusion import Diffusion, DiffusionConfig
        df = m.get("diffusion", {}) or {}
        dfg = (lambda k, d: df.get(k, d)) if hasattr(df, "get") else (lambda k, d: getattr(df, k, d))
        # SUPPORTED parameterization/path: flow + linear only (rectified flow subsumes ddpm/ddim).
        param, path = str(dfg("parameterization", "flow")), str(dfg("path", "linear"))
        assert param == "flow", f"diffusion.parameterization={param!r} unsupported (only 'flow')"
        assert path == "linear", f"diffusion.path={path!r} unsupported (only 'linear')"
        # HARD ERROR: contraction + diffusion (design/models/diffusion.md) — the contraction penalty
        # differentiates the one-step state map, which for diffusion runs through the ODE sampler.
        cv = (cfg.get("variations") or {}).get("contraction", {}) or {}
        cw = float((cv.get("weight", 0.0) if hasattr(cv, "get") else getattr(cv, "weight", 0.0)) or 0.0)
        if cw > 0.0:
            raise ValueError("variations.contraction is mutually exclusive with the diffusion model "
                             "(it differentiates the one-step map, which runs through the ODE sampler). "
                             "Disable contraction (variations.contraction.weight=0) to train diffusion.")
        return Diffusion(DiffusionConfig(
            d=m.d, dz=m.dz, depth=m.depth, heads=m.heads, window=m.window, mlp_ratio=m.mlp_ratio,
            rope_theta=m.rope_theta, dec_hidden=m.get("dec_hidden", 64),
            lambda_pred_obs=m.get("lambda_pred_obs", 1.0), lambda_flow=m.get("lambda_flow", 1.0),
            lambda_consistency=m.get("lambda_consistency", 1.0),
            cond=str(dfg("cond", "concat")), shortcut=bool(dfg("shortcut", False)),
            sampling_steps=int(dfg("sampling_steps", 6)), predict=str(dfg("predict", "residual")),
            stochastic_eval=bool(dfg("stochastic_eval", False)),
            time_sampling=str(dfg("time_sampling", "uniform")), flow_hidden=int(dfg("flow_hidden", 0))))
    raise ValueError(f"unknown model.name: {name!r}")


def normalizer(cfg) -> Normalizer:
    return Normalizer.from_file(cfg.data.root)


def window_loaders(cfg, norm: Normalizer):
    P, F = cfg.data.P, cfg.data.F
    # vector stage: keep the whole windowed set resident on the GPU (tiny) -> no worker/copy overhead
    fast_gpu = cfg.data.get("fast_gpu", True) and torch.cuda.is_available()
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        eps = load_split_episodes(cfg.data.root, split)
        if fast_gpu:
            obs_w, act_w = stack_windows(eps, P, F, norm)
            loaders[split] = GPUWindowLoader(obs_w, act_w, cfg.data.batch, shuffle, torch.device("cuda"))
        else:
            ds = WindowDataset(eps, P, F, norm)
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
        if k.startswith("model."):
            k = k[len("model."):]
        k = k.replace("_orig_mod.", "")  # strip torch.compile wrapper anywhere (whole-model or submodule)
        clean[k] = v
    model.load_state_dict(clean, strict=False)
    return model
