"""Shared loading + cube-mask helpers for the smear diagnostics (sampling_steps_sweep, rolled_zoom)."""
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

COLORS = ["red", "green", "blue", "purple"]


def rgb2hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn + 1e-8
    h = np.zeros_like(mx)
    m = (mx == r); h[m] = ((g - b) / d)[m] % 6
    m = (mx == g); h[m] = ((b - r) / d)[m] + 2
    m = (mx == b); h[m] = ((r - g) / d)[m] + 4
    return h * 60, d / (mx + 1e-8), mx


def any_cube_mask(rgb):
    """Canonical extractor threshold: saturated, bright, NON-table hue."""
    h, s, v = rgb2hsv(rgb)
    table = (h >= 18) & (h <= 55)
    return (s > 0.45) & (v > 0.2) & (~table)


def load(ckpt, dev="cuda", n_ep=4, H=64):
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    m = build_model(rcfg).to(dev).eval(); m = getattr(m, "_orig_mod", m)
    load_checkpoint(m, ckpt)
    norm = normalizer(rcfg); P = rcfg.data.P
    img_heads = [n for n, k in [(mm.get("name"), mm.get("kind")) for mm in rcfg.model.modalities] if k == "image"]
    eps = load_split_episodes_mm(resolve_data_root(rcfg), "val", dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=image_head_sizes(rcfg) or 128,
                                 cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                 repo_id=rcfg.data.get("repo_id", "torus"))[:n_ep]
    Hh = min(H, min(len(o) for o, _, _ in eps) - P - 1)
    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(dev)}
    for hd in img_heads:
        ctx[hd] = torch.stack([torch.from_numpy(fr[hd][:P]) for _, _, fr in eps]).float().div(255).to(dev)
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[:P + Hh - 1])) for _, a, _ in eps]).float().to(dev)
    gt = torch.stack([torch.from_numpy(fr["cam_scene"][P:P + Hh]) for _, _, fr in eps]).float().div(255).to(dev)
    return m, ctx, acts, gt, Hh


def rollout_decode(m, ctx, acts, Hh, stochastic, sampling_steps=None):
    m.stochastic_eval = stochastic
    if sampling_steps is not None:
        m.sampling_steps = int(sampling_steps)
    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    with torch.no_grad(), ac:
        rolled = m._rollout(ctx, acts, Hh, p_tf=0.0, true_future=None, detach_every=0, use_cache=False)
        return m.to_obs(rolled, heads=["cam_scene"], commit=True)["cam_scene"].float().cpu().numpy()
