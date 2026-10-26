"""Decode the observation head's ACTUAL sampling trajectory, one frame per Euler step.

The flow diagram needs a picture of a half-denoised sample. Synthesising one is an illustration; this
is the real thing -- `FlowField.sample(record_path=True)` hands back every point the Euler loop passes
through, and each of those points is a latent, so each one decodes to a frame.

    losses/flow_path.npz   image (steps+1, H, W, 3) uint8, tau (steps+1,), plus the true next frame

STARLING, NOT BLOCK-STACK. This needs a trained world model and only the drone has one in this repo, so
the frames in the flow diagram are the drone's. That is the whole reason the manipulator version of this
figure had to synthesise its corrupted frame.

TAU IS REPORTED THE PLOTS' WAY. The code integrates from tau=1 (noise) down to tau=0 (data); the slides
use the literature's convention, so what is stored is 1 - tau: 0 = noise, 1 = the finished sample.

    python src/paper_specific/figures/make_flow_path.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sorted(glob.glob("/app/logs/train_world_model_*_s2_sub4_concat"))[-1]
CKPT = f"{RUN}/checkpoints/ah_base_ep38.ckpt"
OUT = "/app/logs/paper_icra_2027/losses"
ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]
SEED = 0

_src = open(f"{HERE}/make_sequence_figs.py").read()
N = int(re.search(r"^N\s+= (\d+)", _src, re.M).group(1))
END = int(re.search(r"^END\s+= (\d+)", _src, re.M).group(1))
PANEL_EVERY = int(re.search(r"^PANEL_EVERY\s+= (\d+)", _src, re.M).group(1))
N_STATE = int(re.search(r"^N_STATE\s+= (\d+)", _src, re.M).group(1))

from quickdraw.data.dataset import set_action_aggregate, set_obs_keep, set_subsample   # noqa: E402
from quickdraw.training.setup import build_model, load_checkpoint, normalizer          # noqa: E402


def main() -> int:
    cfg = OmegaConf.create(json.load(open(f"{RUN}/logs/config.json")))
    OmegaConf.set_struct(cfg, False)
    S = int(cfg.data.get("subsample", 1) or 1)
    set_subsample(S)
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    set_obs_keep(cfg.data.get("obs_keep", None))
    P = int(cfg.data.P)

    idx = [END - (N - 1 - i) * S * PANEL_EVERY for i in range(N)]
    last = idx[N_STATE - 1]                              # the figure's last braced item
    ctx = [last - (P - 1 - j) * S for j in range(P)]     # P CONSECUTIVE strided frames ending there
    tgt = last + S                                       # the frame the head is predicting

    import imageio.v3 as iio3
    import pyarrow.parquet as pq
    tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
    ep = np.asarray(tab["episode_index"]); keep = np.flatnonzero(ep == ep[0])
    obs_all = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
    act_all = np.stack([np.asarray(x, np.float32) for x in tab["action"]])[keep]
    vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
    need = set(ctx) | {tgt}
    frames = {k: f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k in need}

    norm = normalizer(cfg)
    m = build_model(cfg)
    load_checkpoint(m, CKPT)
    m = m.eval()

    o = torch.from_numpy(np.stack([obs_all[t] for t in ctx])).float()
    obs = {"proprio": norm.norm_obs(o).unsqueeze(0),
           "image": torch.from_numpy(np.stack([frames[t] for t in ctx])).float().div(255.0).unsqueeze(0)}
    a = norm.norm_act(torch.from_numpy(np.stack([act_all[t: t + S].reshape(-1) for t in ctx])).float())
    a = a.unsqueeze(0)

    torch.manual_seed(SEED)
    with torch.no_grad():
        anchor = m.rel_anchor(obs) if m._rel_on() else None
        bag = m.encode_state(obs, anchor)                           # (1,P,n_state,d)
        # ONE step of the rollout, opened up: the same three calls readout() makes, with the sampler
        # asked to keep its trajectory instead of only its endpoint.
        h = m.backbone(m._to_input(bag, a))
        h_state = m._cond(h[:, -1], a[:, -1], m._proprio_slot(bag[:, -1]), m._z0(bag))
        _, path = m.flow.sample(h_state, steps=m.sampling_steps, deterministic=False,
                                record_path=True)
        prev = bag[:, -1]
        imgs = []
        for x in path:                                              # each point on the path IS a latent
            b = prev + x if m.predict_residual else x
            if m.latent_norm:
                b = torch.nn.functional.layer_norm(b, (b.shape[-1],))
            d = m.to_obs(b.unsqueeze(1), heads=["image"], anchor=anchor)["image"]
            imgs.append(d[0, 0].clamp(0, 1).numpy())

    tau = np.linspace(0.0, 1.0, len(path))
    true = frames[tgt].astype(np.uint8)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/flow_path.npz", image=(np.stack(imgs) * 255).astype(np.uint8),
             tau=tau.astype(np.float32), true=true, ctx_last=ctx[-1], target_frame=tgt,
             steps=m.sampling_steps)
    print(f"  context frames {ctx[0]}..{ctx[-1]}  ->  predicting frame {tgt}")
    print(f"  wrote {OUT}/flow_path.npz  {len(imgs)} decoded points along the trajectory")
    t = true.astype(np.float32) / 255.0
    for i, im in enumerate(imgs):
        mse = float(((im - t) ** 2).mean())
        print(f"    tau {tau[i]:.2f}   |decoded - true| = {np.abs(im - t).mean():.4f}   "
              f"{10 * np.log10(1.0 / max(mse, 1e-12)):5.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
