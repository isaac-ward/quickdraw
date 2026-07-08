"""Language-steered control demo. `python -m quickdraw.eval_language checkpoint=<run> language.request=red`

Loads a trained reward head, steers MPPI to maximize R(latent, request), and logs the ep0 predicted-vs-actual
FPV video + the realized-reward curve so you can watch the agent steer toward the requested concept."""

from __future__ import annotations

import json
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from .controller.language_control import run_language_control
from .controller.mppi import MPPIConfig
from .controller.run import _plog
from .language.reward import LanguageReward
from .logging import viz
from .logging.writer import make_writer
from .training.setup import build_model, env_cfg, load_checkpoint, normalizer
from .utils.logging import make_run_dir


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # restore the trained model's arch from its saved config (any checkpoint loads), like the other eval entrypoints
    ck = cfg.get("checkpoint")
    run = os.path.dirname(os.path.dirname(ck)) if ck and str(ck).endswith(".ckpt") else ck
    cfgj = os.path.join(run, "logs", "config.json") if run else None
    if cfgj and os.path.exists(cfgj):
        OmegaConf.set_struct(cfg, False)
        cfg.model = OmegaConf.create(json.load(open(cfgj))).model
    model = build_model(cfg).to(device); load_checkpoint(model, cfg.checkpoint); model.eval()
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    core = getattr(model, "_orig_mod", model)
    reward = LanguageReward(cfg.language.head, device=device)
    request = str(cfg.language.request)
    assert request in reward.buckets, f"request {request!r} not in the head's vocab {reward.buckets}"

    img_size = next((m.ae.cfg.img_size for m in core.modalities.values() if hasattr(m, "ae")), 128)
    fpv = {"coloring": "hsv", "fov": float(cfg.data.fpv_fov), "size": img_size}
    fields = MPPIConfig.__dataclass_fields__
    mppi = MPPIConfig(**{k: v for k, v in OmegaConf.to_container(cfg.control, resolve=True).items() if k in fields})

    run_dir = make_run_dir("eval_language", cfg.experiment)
    writer = make_writer(run_dir, cfg, job_type="eval_language")
    _plog(writer, f"[eval_language:{request}] steering {mppi.n_episodes} eps, {mppi.num_samples} samples, "
                  f"H={mppi.horizon}, max_steps={mppi.max_steps}")
    res = run_language_control(model, norm, ecfg, reward, request, mppi, device=device, fpv=fpv,
                               log=lambda m: _plog(writer, f"[eval_language:{request}] {m}"))
    fps = round(1.0 / ecfg.dt)

    if "fpv_video" in res:  # pred (top) over actual (bottom), ep0, over the whole run
        pv, av = res["fpv_video"]["pred"], res["fpv_video"]["actual"]
        stacked = (np.concatenate([np.clip(pv, 0, 1), np.clip(av, 0, 1)], axis=1) * 255).astype(np.uint8)
        writer.video(f"eval_language/{request}/fpv_pred_top_actual_bottom", stacked, fps, 0)

    rc = res["reward_curve"]
    f = plt.figure(figsize=(9, 4)); ax = f.add_subplot(111)
    ax.plot(rc); ax.set_xlabel("control step"); ax.set_ylabel(f"reward R(state, '{request}')")
    ax.set_title(f"language steering: realized reward for '{request}' over the run (ep0)")
    ax.grid(alpha=0.3)
    writer.figure(f"eval_language/{request}/reward_curve", f, 0); plt.close(f)

    # ep0 path on the torus (where it went while steering)
    fp = viz.fig_torus_atlas(ecfg.R, ecfg.r, trajs=[{"xyz": res["path"], "color": "black", "start_sphere": True,
                             "end_sphere": True, "start_scale": 0.5}], coloring="hsv",
                             title=f"language steering '{request}' — ep0 path", torus_opacity=viz.TORUS_OPACITY)
    writer.figure(f"eval_language/{request}/path", fp, 0); plt.close(fp)

    writer.scalars({f"eval_language/{request}/reward_start": float(rc[0]),
                    f"eval_language/{request}/reward_end": float(rc[-1]),
                    f"eval_language/{request}/reward_delta": float(rc[-1] - rc[0])}, 0)
    writer.finalize()
    print(f"[eval_language] '{request}' reward {rc[0]:+.3f} -> {rc[-1]:+.3f} (Δ{rc[-1] - rc[0]:+.3f})  run_dir={run_dir}")


if __name__ == "__main__":
    main()
