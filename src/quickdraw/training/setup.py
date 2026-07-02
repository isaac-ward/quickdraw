"""Shared builders: cfg -> dataclasses, model, datasets, loaders. Used by all entrypoints."""

from __future__ import annotations

import os

import torch

from ..data.dataset import (
    MMWindowLoader, Normalizer, TrajectoryDataset,
    load_split_episodes, load_split_episodes_mm,
)
from ..environments.torus import TorusConfig


def env_cfg(cfg) -> TorusConfig:
    e = cfg.environments
    return TorusConfig(R=e.R, r=e.r, dt=e.dt, gamma=e.gamma, a_max=e.a_max, init_speed=e.init_speed, mass=e.mass)


def _modality_specs(cfg):
    """cfg.model.modalities -> list[ModalitySpec]. Defaults to a single proprio (6-vec) modality when none
    is configured, so proprio-only models are just the ONE spine with one modality (no separate vector path)."""
    from ..models.modalities import ModalitySpec
    ms = cfg.model.get("modalities", None)
    if not ms:
        return [ModalitySpec(name="proprio", kind="vector", dim=int(cfg.model.get("obs_dim", 6)))]
    return [ModalitySpec(**dict(e)) for e in ms]


def build_model(cfg):
    """Dispatch on cfg.model.name: data-space (DSAR) or latent-space (LSAR + a collapse mechanism) or
    diffusion; if cfg.model.modalities is set, build the MULTIMODAL variant (token-bag spine)."""
    m = cfg.model
    name = str(m.get("name", "base"))

    specs = _modality_specs(cfg)
    if specs is not None:
        from ..models.multimodal import MultiModalDiffusion, MultiModalDSAR, MultiModalLSAR
        common = dict(specs=specs, d=m.d, depth=m.depth, heads=m.heads, window=m.window,
                      mlp_ratio=m.mlp_ratio, rope_theta=m.rope_theta, action_dim=m.get("action_dim", 2))
        if name in ("mm_dsar", "dsar", "base"):
            return MultiModalDSAR(**common)
        if name in ("mm_lsar", "lsar"):
            from ..models.collapse import EMA, Reconstruction, make_collapse
            if cfg.get("collapse", None) is not None:          # conf/collapse group: naked/recon/ema/sigreg/vicreg
                strat = make_collapse(cfg.collapse)
            elif bool(m.get("ema", False)):                    # legacy mm_lsar_ema config -> EMA strategy
                strat = EMA(tau=float(m.get("ema_decay", 0.996)))
            else:
                strat = Reconstruction()
            return MultiModalLSAR(**common, lambda_pred_latent=m.get("lambda_pred_latent", 1.0),
                                  collapse=strat, lambda_reg=m.get("lambda_reg", 1.0))
        if name in ("mm_diffusion", "diffusion"):
            cv = (cfg.get("variations") or {}).get("contraction", {}) or {}
            cw = float((cv.get("weight", 0.0) if hasattr(cv, "get") else getattr(cv, "weight", 0.0)) or 0.0)
            if cw > 0.0:   # contraction differentiates the one-step map, which for diffusion runs through the ODE sampler
                raise ValueError("variations.contraction is mutually exclusive with the diffusion model "
                                 "(disable contraction, weight=0, to train diffusion).")
            d = m.get("diffusion", {})
            dfg = (lambda k, v: d.get(k, v)) if hasattr(d, "get") else (lambda k, v: getattr(d, k, v))
            return MultiModalDiffusion(**common, sampling_steps=int(dfg("sampling_steps", 6)),
                                       shortcut=bool(dfg("shortcut", False)), predict=str(dfg("predict", "residual")),
                                       stochastic_eval=bool(dfg("stochastic_eval", False)),
                                       time_sampling=str(dfg("time_sampling", "uniform")),
                                       flow_hidden=int(dfg("flow_hidden", 0)),
                                       lambda_flow=m.get("lambda_flow", 1.0),
                                       lambda_consistency=m.get("lambda_consistency", 1.0))
        raise ValueError(f"unknown model.name: {name!r}")


def normalizer(cfg) -> Normalizer:
    return Normalizer.from_file(cfg.data.root)


def window_loaders(cfg, norm: Normalizer):
    """The ONE GPU-resident loader for every model. Loads the FPV frame store only when an image modality
    is present; proprio-only just loads (obs, act) — no frames touched."""
    P, F = cfg.data.P, cfg.data.F
    specs = _modality_specs(cfg)
    img = next((s for s in specs if s.kind == "image"), None)   # image modality (if any) -> resident frame store
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        if img is not None:
            eps = load_split_episodes_mm(cfg.data.root, split, img_size=img.img_size)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev, image_head=img.name)
        else:                                                    # proprio-only: (obs, act) pairs, no FPV frames
            eps = load_split_episodes(cfg.data.root, split)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev)
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
