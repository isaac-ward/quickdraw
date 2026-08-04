"""Collapse-axis smoke/validation: train each LSAR mechanism for a few hundred steps and track
participation_ratio / L_pred / obs-error. No Trainer/compile/wandb.

  uv run python -m quickdraw.smoke.collapse_validation [data_root] [--p_tf P] [--steps N]

Note: at p_tf=1 (one-step parallel) on high-framerate data the prediction task is near-trivial, so
collapse does not manifest; use --p_tf 0.5 (multi-step rollout) to actually stress the encoder.
"""
from __future__ import annotations

import argparse
import os

import torch
from hydra import compose, initialize_config_dir

from quickdraw.environments import torus_utils as T
from quickdraw.training.lit import LitWorldModel
from quickdraw.training.setup import build_model, env_cfg, normalizer, window_loaders

CONF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf"))
MECHS = ["naked", "reconstruction", "ema", "sigreg", "vicreg"]


def make_cfg(data_root, ov):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=[f"data.root={data_root}"] + ov)


def cycle(dl):
    while True:
        yield from dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", nargs="?", default="logs/data_generation_2026_06_24_06_57_53_regen")
    ap.add_argument("--p_tf", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log", type=int, default=60)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    base = make_cfg(args.data_root, [])
    norm = normalizer(base)
    loader = window_loaders(base, norm)["train"]
    e = env_cfg(base)
    P = base.data.P

    for mech in MECHS:
        cfg = make_cfg(args.data_root, ["model=latent_space_autoregressor", f"+collapse={mech}"])
        model = build_model(cfg).to(dev)
        lit = LitWorldModel(model, norm, e.R, e.r, e.init_speed, P, cfg.data.F,
                            args.p_tf, args.p_tf, 0, cfg.optim.lr, cfg.optim.weight_decay,
                            cfg.model.detach_every).to(dev)
        logged = {}
        lit.log = lambda k, v, **kw: logged.__setitem__(k, float(v))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        it = cycle(loader)
        print(f"--- {mech} (p_tf={args.p_tf}) ---", flush=True)
        for step in range(args.steps + 1):
            batch = next(it)
            opt.zero_grad()
            loss = lit._step(batch, "train")
            loss.backward()
            opt.step()
            model.on_optimizer_step()
            if step % args.log == 0:
                with torch.no_grad():
                    er = float(model.collapse_diagnostics(batch["obs_seq"])["rank/participation_ratio"])
                print(f"  step {step:4d}: PR={er:5.2f}  L_pred={logged.get('train/loss/pred_latent', 0):.4f}  "
                      f"mde={logged.get('train/manifold_distance_error', 0):.3f}  loss={float(loss):.3f}", flush=True)
    print("VALIDATION DONE", flush=True)


if __name__ == "__main__":
    main()
