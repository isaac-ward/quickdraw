"""Is the reward head informative in the region the PLANNER explores?

The head was fit on eval_interpret's latents: the model's committed next-state predictions from REAL val
contexts. The planner scores something else -- latents rolled 1..128 steps into a self-generated imagination
under actions the reward itself selected. If those two sets do not overlap, every steering number in the
record is a measurement of f_z outside its domain, and no amount of planning can fix it.

Three questions, in order of how badly a bad answer hurts:
  1. GEOMETRY   do the rolled latents live where the fitted ones do? (norm, and cosine to the fitted mean)
  2. SPREAD     does R vary across rolled latents at all, or is it pinned? A pinned score cannot rank, and
                MPPI consumes nothing but the ranking.
  3. DISCRIMINATION  on the FITTED latents the head separates requests (that is the AUC table). Does the
                same separation survive on rolled ones? Measured as the spread of R across the 10 request
                phrases for the same latent -- if a rolled latent scores every phrase alike, the words are
                dead on arrival no matter how good the planner is.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_reward_domain.py <reward_run> <interpret_run> <steer_run>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from quickdraw.train_reward_model import _embed_texts, _load_interpret

REQUESTS = ["strafe right", "strafe left", "rotate left", "climb", "descend", "fly toward the ladder",
            "the table", "fly forward toward the ladder", "face the glass wall", "do nothing"]


def main(reward_run: str, interpret_run: str, steer_run: str) -> int:
    d = torch.load(os.path.join(reward_run, "reward_head.pt"), map_location="cpu", weights_only=False)

    def mlp(sd, i, h, o):
        m = torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Dropout(0.0),
                                torch.nn.Linear(h, o))
        m.load_state_dict(sd); return m.eval()

    fz, ft = mlp(d["f_z"], d["latent_dim"], d["hidden"], d["embed_dim"]), mlp(d["f_t"], 384, d["hidden"], d["embed_dim"])
    fitted, _, _, _, _, _ = _load_interpret(interpret_run, list(d["factors"]))
    pf = sorted(glob.glob(os.path.join(steer_run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                       "latents.npy")))
    assert pf, ("no latents.npy in the steer run -- rerun eval_steer, which now dumps the scored latents "
                "alongside proprio.npy")
    per = [np.load(f) for f in pf]                      # each (H, D), in STEP ORDER
    rolled = np.concatenate(per, 0)
    stepi = np.concatenate([np.arange(len(x)) for x in per])
    print(f"  fitted (interpret) {fitted.shape}   rolled (planned) {rolled.shape}")

    Xf, Xr = torch.from_numpy(fitted).float(), torch.from_numpy(rolled).float()
    # 1. GEOMETRY, in the RAW latent space the head consumes
    mu = Xf.mean(0, keepdim=True)
    for nm, X in (("fitted", Xf), ("rolled", Xr)):
        cos = F.cosine_similarity(X, mu, dim=-1)
        print(f"  {nm:7s} |z| {X.norm(dim=-1).mean():7.2f} +/- {X.norm(dim=-1).std():5.2f} | "
              f"cos to fitted-mean {cos.mean():+.3f} +/- {cos.std():.3f}")
    with torch.no_grad():
        Zf, Zr = F.normalize(fz(Xf), dim=-1), F.normalize(fz(Xr), dim=-1)
        T = F.normalize(ft(_embed_texts(REQUESTS, d["text_model"], "cpu")), dim=-1)
        Rf, Rr = Zf @ T.T, Zr @ T.T                                    # (N, n_requests)
    # 2. SPREAD of R over latents, per request
    print(f"\n  {'request':32s} {'fitted R':>18s} {'rolled R':>18s}   {'std ratio':>9s}")
    for i, q in enumerate(REQUESTS):
        a, b = Rf[:, i], Rr[:, i]
        print(f"  {q:32s} {a.mean():+7.3f} +/-{a.std():5.3f} {b.mean():+7.3f} +/-{b.std():5.3f}   "
              f"{float(b.std() / max(1e-9, a.std())):>9.2f}")
    # 3. DISCRIMINATION: does one latent score the 10 phrases differently?
    fs = float(Rf.std(dim=1).mean())
    print(f"\n  spread of R ACROSS the {len(REQUESTS)} requests, for ONE latent (mean over latents):")
    print(f"    fitted (interpret, 15-step imaginations)  {fs:.4f}")
    print(f"    rolled (planned, all steps)               {float(Rr.std(dim=1).mean()):.4f}")
    print("    near 0 means the words stop mattering; near the fitted value means they still do.")
    # BY HORIZON: the head was fit on 15-step imaginations. Does discrimination survive past that?
    print(f"\n  by planned step (the fitted set only ever covered steps 1-15):")
    for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 64), (64, 128)):
        msk = (stepi >= lo) & (stepi < hi)
        if msk.sum() < 10:
            continue
        sub = Rr[torch.from_numpy(msk)]
        print(f"    steps {lo:>3d}-{hi:<3d} n={int(msk.sum()):>5d}  across-request spread {float(sub.std(dim=1).mean()):.4f}"
              f"  ({100 * float(sub.std(dim=1).mean()) / max(1e-9, fs):>3.0f}% of fitted)"
              f"  |z| {float(Xr[torch.from_numpy(msk)].norm(dim=-1).mean()):7.2f}")
    # the planner's own lever: within one request, how much can R vary over candidate latents?
    print(f"\n  within-request range over rolled latents (what MPPI can actually climb):")
    for i, q in enumerate(REQUESTS[:4]):
        b = Rr[:, i]
        print(f"    {q:24s} p5 {b.quantile(0.05):+.3f}  p50 {b.quantile(0.5):+.3f}  p95 {b.quantile(0.95):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3 + 1]))
