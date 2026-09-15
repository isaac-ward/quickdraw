"""Produce the ACTUAL model prediction the sequence figure shows as its output.

The figure's "predicted" frame and vector used to be the RECORDED item 10 -- ground truth wearing a hat.
This rolls the real world model instead and caches what it decodes, so the figure shows a prediction.

ALIGNMENT, which is not trivial. The cascade draws its items PANEL_EVERY model steps apart (so the camera
visibly moves between tiles), while the model's window is 8 CONSECUTIVE strided steps. So the context fed
here is NOT the eight drawn tiles: it is the eight consecutive strided frames ending at the last braced
tile, and the horizon is PANEL_EVERY steps, which is exactly the gap from that tile to the one the figure
labels as predicted. Same frame, real rollout.

    python logs/paper_icra_2027/make_prediction.py        # CPU; writes prediction.npz beside the figure
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import torch
from omegaconf import OmegaConf

RUN = sorted(glob.glob("/app/logs/train_world_model_*_s2_sub4_concat"))[-1]
CKPT = f"{RUN}/checkpoints/ah_base_ep38.ckpt"
OUT = "/app/logs/paper_icra_2027/prediction.npz"
ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]

# the figure's own constants, read from the figure script so the two cannot drift apart
_src = open("/app/logs/paper_icra_2027/make_sequence_figs.py").read()
N = int(re.search(r"^N\s+= (\d+)", _src, re.M).group(1))
END = int(re.search(r"^END\s+= (\d+)", _src, re.M).group(1))
PANEL_EVERY = int(re.search(r"^PANEL_EVERY\s+= (\d+)", _src, re.M).group(1))
PRED_AT = int(re.search(r"^PRED_AT\s+= (\d+)", _src, re.M).group(1))
N_STATE = int(re.search(r"^N_STATE\s+= (\d+)", _src, re.M).group(1))
PRED_STEPS = int(re.search(r"^PRED_STEPS\s+= (\d+)", _src, re.M).group(1))
BRACE_HI = N_STATE - 1   # last braced item: the context ends here

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
    last = idx[BRACE_HI]
    horizon = PRED_STEPS                                # one decoded frame per RED SLICE
    tgts = [last + (h + 1) * S for h in range(horizon)]  # the frames those slices predict
    ctx_raw = [last - (P - 1 - j) * S for j in range(P)]   # P CONSECUTIVE strided frames ending at `last`
    print(f"  context = {P} CONSECUTIVE strided frames {ctx_raw[0]}..{ctx_raw[-1]} | roll {horizon} steps "
          f"-> frames {tgts} (one per red slice; the last is drawn as item {PRED_AT})")

    import imageio.v3 as iio3
    import pyarrow.parquet as pq
    tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
    ep = np.asarray(tab["episode_index"]); keep = np.flatnonzero(ep == ep[0])
    obs_all = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
    act_all = np.stack([np.asarray(x, np.float32) for x in tab["action"]])[keep]
    vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
    need = max(max(tgts), ctx_raw[-1])
    frames = [f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k <= need]

    # actions are CONCAT: one strided step carries S raw commands, time-major -- rebuild exactly as the
    # loader does, for the context steps AND the rolled horizon
    def act_at(t0):                                      # the strided action whose window STARTS at t0
        return act_all[t0:t0 + S].reshape(-1)
    a_seq = np.stack([act_at(ctx_raw[j]) for j in range(P)]
                     + [act_at(last + (h + 1) * S) for h in range(horizon)])   # (P+horizon, S*4)

    norm = normalizer(cfg)
    model = build_model(cfg)
    load_checkpoint(model, CKPT)
    m = model.eval()

    o = torch.from_numpy(np.stack([obs_all[t] for t in ctx_raw])).float()
    ctx = {"proprio": norm.norm_obs(o).unsqueeze(0),
           "image": torch.from_numpy(np.stack([frames[t] for t in ctx_raw])).float().div(255.0).unsqueeze(0)}
    act = norm.norm_act(torch.from_numpy(a_seq).float()).unsqueeze(0)[:, : P - 1 + horizon]
    with torch.no_grad():
        out = m.imagine_eval(ctx, act, horizon, heads=["proprio", "image"])
    img = out["image"][0].clamp(0, 1).numpy()                         # (horizon,H,W,3) one per slice
    pro = norm.denorm_obs(out["proprio"][0].cpu()).numpy()             # (horizon,obs_dim)

    np.savez(OUT, image=(img * 255).astype(np.uint8), proprio=pro.astype(np.float32),
             target_frames=np.array(tgts), horizon=horizon, ctx_first=ctx_raw[0], ctx_last=ctx_raw[-1])
    print(f"  wrote {OUT}  image {img.shape}  proprio {pro.shape}")
    for h, t in enumerate(tgts):
        d = np.abs(img[h] - frames[t].astype(np.float32) / 255.0).mean()
        print(f"    step {h + 1} -> frame {t}: |pred - recorded| mean abs = {d:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
