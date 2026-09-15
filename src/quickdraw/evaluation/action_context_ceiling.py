"""THE CEILING on the action prior: how much of the next K actions is LINEARLY readable from the
context vector it is given? Anything the head scores below this, it is leaving behind.

The action prior scores an energy skill of ~0.11 at lead 0 and ~0 at the far leads. Three explanations fit
that: the signal is not in the data, the signal is in the data but the flow is not extracting it, or the
signal is lost when the token bag is MEAN-POOLED into one vector. This separates the first from the other
two for the price of a matrix inverse.

A ridge regression from the frozen `h_ctx` straight to the recorded action chunk is a direct supervised
readout with no generative machinery in the way. Its R^2 is an UPPER BOUND on what any head can pull LINEARLY
out of this context vector -- so:

    probe R^2 ~ 0 at a lead   ->  nothing linear is there; a better objective cannot invent it
    probe R^2 >> the head's   ->  the signal is there and the head is missing it

`h_ctx` comes from the FROZEN world model, so the answer is a property of the context, not of any arm --
every action head trained on this checkpoint is reading the same vector.

Episodes are split 4 fit / 1 select / 2 test, so the ridge penalty is never chosen on the test episodes.
Reported alongside 1 - sqrt(1 - R^2), the energy skill an equally-informed sampler would score, so the
number is directly comparable to eval_action_distribution/lead_XX/energy_skill.

    CUDA_VISIBLE_DEVICES=0 python -m quickdraw.evaluation.action_context_ceiling <ckpt_or_run_dir> [K]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.data.dataset import (load_split_episodes_mm, set_action_aggregate,  # noqa: E402
                                    set_obs_keep, set_subsample)
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,  # noqa: E402
                                      load_checkpoint, normalizer, resolve_data_root)

FIT_SPLIT = os.environ.get("PROBE_FIT_SPLIT", "")   # "" -> time blocks within val; "train" -> the honest one
N_FIT_EP = int(os.environ.get("PROBE_FIT_EPS", "20"))
CAP = int(os.environ.get("PROBE_CAP", "600"))       # steps per fit episode (train episodes are ~1778)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def ridge(X, Y, lam):
    """Closed form with an intercept, solved on the centred problem."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, xm, ym


def r2(pred, Y):
    """Variance-weighted R^2 over all dims at once: 1 - SSE/SST, so dims with no variance cannot inflate it."""
    sse = float(((pred - Y) ** 2).sum())
    sst = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else 0.0


def main(path, K) -> int:
    run = os.path.dirname(os.path.dirname(path)) if path.endswith(".ckpt") else path
    ck = path if path.endswith(".ckpt") else os.path.join(run, "checkpoints", "last.ckpt")
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    OmegaConf.set_struct(cfg, False)
    set_subsample(int(cfg.data.get("subsample", 1) or 1))
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    set_obs_keep(cfg.data.get("obs_keep", None))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = normalizer(cfg)
    model = build_model(cfg)
    load_checkpoint(model, ck)
    m = model.to(dev).eval()
    mode = os.environ.get("PROBE_CONTEXT", "")
    if mode:
        m.action_head_context = mode          # pool_context reads this; the probe needs no head at all
        print(f"  context mode forced to {mode!r} -> width {m.context_dim()}")

    img_heads = [n for n, _ in m.layout if n != "proprio"]

    def contexts(split, n_ep=None, cap=None):
        """(H, A) for a split, ONE EPISODE AT A TIME -- the whole train split as one tensor is ~18 GB."""
        es = load_split_episodes_mm(resolve_data_root(cfg), split,
                                    img_size=image_head_sizes(cfg) or 128,
                                    cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                    repo_id=cfg.data.get("repo_id", "torus"))
        es = es[:n_ep] if n_ep else es
        Ls = min(len(o) for o, _, _ in es)
        Ls = min(Ls, cap) if cap else Ls
        hs, as_ = [], []
        for o, av, fr in es:
            c = {"proprio": norm.norm_obs(torch.from_numpy(o[:Ls])).float().unsqueeze(0).to(dev)}
            for h in img_heads:
                c[h] = torch.from_numpy(fr[h][:Ls]).float().div(255.0).unsqueeze(0).to(dev)
            av_ = norm.norm_act(torch.from_numpy(av[:Ls])).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                hs.append(m.action_context(c, av_)[0].cpu().numpy())
            as_.append(norm.denorm_act(av_[0, 1:].cpu()).numpy())
            del c, av_
            torch.cuda.empty_cache()
        return np.stack(hs), np.stack(as_)

    if FIT_SPLIT:
        H, A = contexts(FIT_SPLIT, n_ep=N_FIT_EP, cap=CAP)
        Hv, Av = contexts("val")
        print(f"  FIT on {H.shape[0]} {FIT_SPLIT} episodes x {H.shape[1]} steps, TEST on "
              f"{Hv.shape[0]} val episodes x {Hv.shape[1]} steps -- no flight appears in both")
    else:
        H, A = contexts("val")
        Hv, Av = None, None
    E, T, d = H.shape
    a_dim = A.shape[-1]
    K = min(K, T - 1)
    print(f"\n  {os.path.basename(run)}   context dim {d}   {E} val episodes x {T} steps   action_dim {a_dim}")
    print(f"  ridge probe h_ctx -> a[t+1+k], leads 0..{K - 1}, episodes split 4 fit / 1 select / 2 test\n")

    # SPLIT BY TIME INSIDE EACH EPISODE, not by episode. Different flights sit at different mean stick
    # positions, so an episode-level split makes the probe fail on the MEAN SHIFT alone -- every lambda
    # collapsed to maximum shrinkage and R^2 came out at -0.30 everywhere, which measures the shift and
    # nothing about predictability. Gaps between the blocks keep adjacent (highly correlated) frames apart.
    def blocks(n):
        f = slice(0, int(0.60 * n))
        s_ = slice(int(0.65 * n), int(0.75 * n))
        t_ = slice(int(0.80 * n), n)
        return f, s_, t_

    rows = []
    for k in range(K):
        Tk = T - k
        f, s_, t_ = blocks(Tk)
        if Hv is None:                                        # within-split time blocks
            Xs = {n_: H[:, sl].reshape(-1, d) for n_, sl in (("f", f), ("s", s_), ("t", t_))}
            Ys = {n_: A[:, k:k + Tk][:, sl].reshape(-1, a_dim) for n_, sl in (("f", f), ("s", s_), ("t", t_))}
        else:                                                 # fit on one split, test on ANOTHER
            Tv = Hv.shape[1] - k
            Xs = {"f": H[:, f].reshape(-1, d), "s": H[:, s_].reshape(-1, d),
                  "t": Hv[:, :Tv].reshape(-1, d)}
            Ys = {"f": A[:, k:k + Tk][:, f].reshape(-1, a_dim),
                  "s": A[:, k:k + Tk][:, s_].reshape(-1, a_dim),
                  "t": Av[:, k:k + Tv].reshape(-1, a_dim)}
        mu, sd = Xs["f"].mean(0), Xs["f"].std(0) + 1e-6        # standardise so lambda means the same thing
        Xs = {n_: (v - mu) / sd for n_, v in Xs.items()}
        best, best_lam = -9e9, LAMBDAS[0]
        for lam in LAMBDAS:
            W, xm, ym = ridge(Xs["f"], Ys["f"], lam)
            sc = r2((Xs["s"] - xm) @ W + ym, Ys["s"])
            if sc > best:
                best, best_lam = sc, lam
        W, xm, ym = ridge(Xs["f"], Ys["f"], best_lam)
        P = (Xs["t"] - xm) @ W + ym
        rt = r2(P, Ys["t"])
        per = []
        for i in range(min(4, a_dim)):
            v = float(Ys["t"][:, i].var())
            per.append(r2(P[:, i:i + 1], Ys["t"][:, i:i + 1]) if v > 1e-8 else float("nan"))
        rows.append((k, best_lam, rt, per))

    print("  lead |  ridge  | test R^2 | implied energy skill | per raw axis R^2 (sub-step 0)")
    print("  " + "-" * 92)
    for k, lam, rt, per in rows:
        sk = 1.0 - np.sqrt(max(0.0, 1.0 - rt)) if rt > 0 else rt / 2.0
        print(f"  {k:>4} | {lam:>7.4g} | {rt:>+8.4f} | {sk:>+20.4f} | "
              + "  ".join(f"a{i}:{v:+.3f}" for i, v in enumerate(per)))
    b = max(rows, key=lambda r: r[2])
    print(f"\n  BEST lead {b[0]} at R^2 {b[2]:+.4f} (implied skill {1 - np.sqrt(max(0, 1 - b[2])):+.3f})")
    far = [r for r in rows if r[0] >= 8]
    if far:
        print(f"  LEADS >= 8: mean R^2 {np.mean([r[2] for r in far]):+.4f}, max {max(r[2] for r in far):+.4f}")
    print("\n  READ IT AS: this is what a LINEAR readout of the frozen context achieves. The head's measured")
    print("  energy skill above this means it is finding nonlinear structure; well below means the signal is")
    print("  in the context and the head is leaving it there.\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m quickdraw.evaluation.action_context_ceiling <ckpt_or_run_dir> [K]")
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32))
