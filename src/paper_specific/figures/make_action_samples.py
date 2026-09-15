"""Draw REAL samples from the trained action prior, for the distribution the figure shows at the action
head's output.

The head emits p(a_{t:t+K} | h) -- a distribution over a whole CHUNK, not over one command -- so the
honest picture of it is a set of draws overplotted: where they agree the prior is confident, where they
split it is not, and multimodality shows up as the fan separating rather than widening. A hand-drawn bell
curve would say "distribution" without saying anything about THIS one.

SAME MODEL, SAME CONTEXT AS THE REST OF THE FIGURE. The chunk-32 action head was trained from
ah_base_ep38.ckpt, which is the exact checkpoint make_prediction.py rolls for the predicted frames, on the
same dataset at the same stride -- so these samples and those frames describe one model. The context is
built the way make_prediction.py builds it (P CONSECUTIVE strided steps ending at the last braced tile),
then handed to controller/mppi._h_ctx, which is the same function eval_control uses, so the pooled context
is assembled under the alignment the head was TRAINED on: h[t-1] predicts a_t, and the freshest state has
no action to pair with because that action is the one being predicted.

    python logs/paper_icra_2027/make_action_samples.py     # writes action_samples.npz beside the figure
"""
from __future__ import annotations

import glob
import json
import re

import numpy as np
import torch
from omegaconf import OmegaConf

AH_RUN = "/app/logs/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full"
OUT = "/app/logs/paper_icra_2027/action_samples.npz"
ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]
N_DRAWS = 48
SEED = 0

_src = open("/app/logs/paper_icra_2027/make_sequence_figs.py").read()
N = int(re.search(r"^N\s+= (\d+)", _src, re.M).group(1))
END = int(re.search(r"^END\s+= (\d+)", _src, re.M).group(1))
PANEL_EVERY = int(re.search(r"^PANEL_EVERY\s+= (\d+)", _src, re.M).group(1))
N_STATE = int(re.search(r"^N_STATE\s+= (\d+)", _src, re.M).group(1))
BRACE_HI = N_STATE - 1

from quickdraw.controller.mppi import _h_ctx                                        # noqa: E402
from quickdraw.data.dataset import set_action_aggregate, set_obs_keep, set_subsample  # noqa: E402
from quickdraw.evaluation.steering import PriorProposal                             # noqa: E402
from quickdraw.training.setup import build_model, load_checkpoint, normalizer       # noqa: E402


def main() -> int:
    cfg = OmegaConf.create(json.load(open(f"{AH_RUN}/logs/config.json")))
    OmegaConf.set_struct(cfg, False)
    S = int(cfg.data.get("subsample", 1) or 1)
    set_subsample(S)
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    set_obs_keep(cfg.data.get("obs_keep", None))
    P = int(cfg.data.P)

    idx = [END - (N - 1 - i) * S * PANEL_EVERY for i in range(N)]
    last = idx[BRACE_HI]
    ctx_raw = [last - (P - 1 - j) * S for j in range(P)]

    import imageio.v3 as iio3
    import pyarrow.parquet as pq
    tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
    ep = np.asarray(tab["episode_index"]); keep = np.flatnonzero(ep == ep[0])
    obs_all = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
    act_all = np.stack([np.asarray(x, np.float32) for x in tab["action"]])[keep]
    vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
    frames = [f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k <= ctx_raw[-1]]

    def act_at(t0):                                      # one strided action = S raw commands, time-major
        return act_all[t0:t0 + S].reshape(-1)

    norm = normalizer(cfg)
    model = build_model(cfg)
    load_checkpoint(model, f"{AH_RUN}/checkpoints/last.ckpt")
    m = model.eval()
    core = getattr(m, "_orig_mod", m)
    assert getattr(core, "action_head_enabled", False), f"{AH_RUN} has no trained action head"
    K = int(core.action_head_chunk)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)

    obs_win = torch.from_numpy(np.stack([obs_all[t] for t in ctx_raw])).float().unsqueeze(0)
    fpv_win = torch.from_numpy(np.stack([frames[t] for t in ctx_raw])).float().div(255.0).unsqueeze(0)
    pa = torch.from_numpy(np.stack([act_at(t) for t in ctx_raw[:-1]])).float().unsqueeze(0)
    with torch.no_grad():
        h = _h_ctx(core, norm, obs_win, fpv_win, pa, img_head)         # (1, d_ctx)
        pr = PriorProposal(core, a_max=1e9, norm=norm)                 # norm -> RAW stick units out
        g = torch.Generator().manual_seed(SEED)
        A = pa.shape[-1]
        draws = pr.sample(torch.zeros(1, K, A), N_DRAWS, ctx=h, g=g)[0]   # (N, K, A) RAW
    # (N, K, S*4) -> (N, K*S, 4): unfold the concat back into real time, so a sample is the actual
    # sequence of stick commands, 4 axes at the capture rate, not one number per model step.
    smp = draws.reshape(N_DRAWS, K, S, A // S).reshape(N_DRAWS, K * S, A // S).numpy()
    # ...and what the pilot ACTUALLY did over the same span, where the episode still has it
    n_raw = min(K * S, len(act_all) - last)
    truth = act_all[last:last + n_raw]

    np.savez(OUT, samples=smp.astype(np.float32), truth=truth.astype(np.float32),
             chunk=K, subsample=S, start=last, ctx_first=ctx_raw[0], ctx_last=ctx_raw[-1], seed=SEED)
    print(f"  {AH_RUN.split('/')[-1]}  chunk {K} x {A}-wide actions, {N_DRAWS} draws")
    print(f"  context = {P} consecutive strided steps {ctx_raw[0]}..{ctx_raw[-1]}")
    print(f"  wrote {OUT}  samples {smp.shape} (raw stick units)  truth {truth.shape}")
    for j in range(smp.shape[-1]):
        print(f"    axis {j}: draws mean {smp[..., j].mean():+.3f} sd {smp[..., j].std():.3f}"
              f" | recorded mean {truth[:, j].mean():+.3f} sd {truth[:, j].std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
