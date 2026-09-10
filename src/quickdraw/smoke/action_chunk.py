"""`data.action_aggregate` (five rules) and `action_head.chunk` (joint chunk prediction).

The defaults -- `sum` and `chunk: 1` -- must be BIT-IDENTICAL to every run before 2026-09-10, so most of
this file is that check rather than the new behaviour: the aggregation is compared against the literal old
expression, and the chunk alignment against the literal old two lines it replaced.

Why the knobs exist. `sum` is correct only for DELTA actions (robocasa EEF deltas compose additively over
the skipped frames); starling's `joy_axis_*` are absolute stick positions, so summing s of them scales the
std by ~s while normalization_stats.json is computed on the RAW actions -- measured 4.21 z-std at stride 4.
`chunk` predicts K consecutive post-subsample actions as ONE flattened vector, i.e. a joint over lead times
rather than K independent marginals.

    python -m quickdraw.smoke.action_chunk
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from ..data import dataset as D
from ..models.flow import FlowField

ok = bad = 0


def check(n, cond, extra=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{('  ' + extra) if extra else ''}", flush=True)


def _eps(n_ep=3, T=41, a=4, seed=0):
    r = np.random.default_rng(seed)
    return [(r.standard_normal((T, 6)).astype(np.float32), r.standard_normal((T, a)).astype(np.float32))
            for _ in range(n_ep)]


def _run(eps, s, mode):
    D._SUBSAMPLE, D._ACTION_AGGREGATE, D._SUBSAMPLE_USED = s, mode, False
    return D._subsample_episodes([tuple(e) for e in eps], "smoke/train")


def _old_sum(eps, s):
    """The literal pre-2026-09-10 body, for the bit-identicality comparison."""
    acts = np.concatenate([e[1] for e in eps], 0)
    hold = [d for d in range(acts.shape[1]) if len(np.unique(acts[:, d])) <= 2]
    out = []
    for ep in eps:
        o, aa_ = ep[0], ep[1]
        n = len(o) // s
        grp = aa_[:n * s].reshape(n, s, -1)
        aa = grp.sum(axis=1)
        if hold:
            aa[:, hold] = grp[:, -1, hold]
        out.append((o[:n * s:s], aa))
    return out


class _Head(torch.nn.Module):
    """Minimal stand-in exposing the real MultiModalFlow.action_pairs (an unbound method needs only the
    two attributes it reads), so the alignment is tested without building a whole world model."""

    def __init__(self, K):
        super().__init__()
        self.action_head_chunk = K

    action_pairs = None  # bound below


def main() -> int:
    from ..models.multimodal import MultiModalFlow
    _Head.action_pairs = MultiModalFlow.action_pairs
    eps = _eps()

    # ---- 1. BIT-IDENTICALITY of the default rule ----
    for s in (2, 3, 4, 5):
        got, want = _run(eps, s, "sum"), _old_sum(eps, s)
        same = (len(got) == len(want)
                and all(np.array_equal(g[0], w[0]) and np.array_equal(g[1], w[1]) for g, w in zip(got, want)))
        check(f"action_aggregate=sum at stride {s} is BIT-IDENTICAL to the old code", same)
    check("stride 1 short-circuits untouched (the episodes object itself)", _run(eps, 1, "sum") is not None
          and all(np.array_equal(a[1], b[1]) for a, b in zip(_run(eps, 1, "mean"), eps)))

    # ---- 2. the other four rules mean what they say ----
    s = 4
    A = np.concatenate([e[1] for e in eps], 0)
    for mode, fn in (("mean", lambda g: g.mean(1)), ("last", lambda g: g[:, -1]), ("first", lambda g: g[:, 0])):
        got = _run(eps, s, mode)
        want = [fn(e[1][:len(e[0]) // s * s].reshape(len(e[0]) // s, s, -1)) for e in eps]
        check(f"action_aggregate={mode} matches its definition",
              all(np.allclose(g[1], w) for g, w in zip(got, want)))
    got = _run(eps, s, "concat")
    check("concat widens action_dim by exactly s", got[0][1].shape[-1] == A.shape[-1] * s,
          f"{got[0][1].shape[-1]} vs {A.shape[-1] * s}")
    e0 = eps[0][1]
    check("concat is LOSSLESS and time-major (slot j = the j-th raw action of the window)",
          np.array_equal(got[0][1][0], e0[:s].reshape(-1)))
    check("mean == sum / s exactly (so it is the scale fix, not a different quantity)",
          np.allclose(_run(eps, s, "mean")[0][1], _run(eps, s, "sum")[0][1] / s))
    check("last/first do NOT alias the source array (a view would be mutable through it)",
          _run(eps, s, "last")[0][1].base is None and _run(eps, s, "first")[0][1].base is None)

    # ---- 3. the scale defect the knob exists to fix, measured ----
    corr = np.concatenate([np.cumsum(e[1], 0) for e in eps], 0)   # heavily autocorrelated, like a stick
    sd_raw = corr.std(0).mean()
    for s_ in (2, 4):
        n = len(corr) // s_
        sd_sum = corr[:n * s_].reshape(n, s_, -1).sum(1).std(0).mean()
        check(f"sum at stride {s_} inflates the action scale ~{s_}x on autocorrelated actions",
              sd_sum / sd_raw > s_ * 0.7, f"x{sd_sum / sd_raw:.2f}")

    # ---- 4. Normalizer.tile_act ----
    nz = D.Normalizer({"observation_vector": {"mean": [0.0], "std": [1.0]},
                       "action": {"mean": [1.0, 2.0], "std": [3.0, 4.0]}})
    nz.tile_act(3)
    check("tile_act repeats the stats time-major to match concat's layout",
          nz.a_mean.tolist() == [1.0, 2.0] * 3 and nz.a_std.tolist() == [3.0, 4.0] * 3)
    nz2 = D.Normalizer({"observation_vector": {"mean": [0.0], "std": [1.0]},
                        "action": {"mean": [1.0], "std": [2.0]}})
    check("tile_act(1) is a no-op", nz2.tile_act(1).a_mean.tolist() == [1.0])

    # ---- 5. action_pairs: K=1 reproduces the two lines it replaced ----
    B, L, d, a = 2, 12, 8, 4
    h_ctx, act = torch.randn(B, L - 1, d), torch.randn(B, L, a)
    for K in (1, 2, 3, 5):
        m = _Head(K)
        cond, tgt = m.action_pairs(h_ctx, act)
        check(f"K={K}: N rows = L-1-K", cond.shape[1] == L - 1 - K and tgt.shape[1] == L - 1 - K,
              f"{tuple(cond.shape)} {tuple(tgt.shape)}")
        check(f"K={K}: target width = K*action_dim", tgt.shape[-1] == K * a, str(tgt.shape[-1]))
        # row t must hold a[t+1..t+K], and cond must be h[t]
        for t in (0, cond.shape[1] - 1):
            want = act[:, t + 1:t + 1 + K].reshape(B, K * a)
            check(f"K={K}: row {t} target == a[{t+1}..{t+K}] flattened time-major",
                  torch.equal(tgt[:, t], want))
            check(f"K={K}: row {t} cond == h[{t}] (leak-free)", torch.equal(cond[:, t], h_ctx[:, t]))
    m1 = _Head(1)
    c1, t1 = m1.action_pairs(h_ctx, act)
    check("K=1 cond IS the old `h_ctx[:, :-1]`", torch.equal(c1, h_ctx[:, :-1]))
    check("K=1 target IS the old `act_seq[:, 1:L-1]`", torch.equal(t1, act[:, 1:L - 1]))
    check("target is detached (it is a target, not a path to the encoder)", not t1.requires_grad)

    # ---- 6. the short-window guard, and that it matches the old `L >= 3` at K=1 ----
    for K, L_, want in ((1, 3, True), (1, 2, False), (4, 6, True), (4, 5, False), (8, 10, True), (8, 9, False)):
        c, _ = _Head(K).action_pairs(torch.randn(B, L_ - 1, d), torch.randn(B, L_, a))
        check(f"K={K}, L={L_}: {'forms a chunk' if want else 'refuses (L < K+2)'}", (c is not None) == want)

    # ---- 7. the head really is wired K x wider, and samples a whole chunk ----
    for K in (1, 4):
        f = FlowField(a * K, h_dim=d, hidden=d, cond="concat", shortcut=False)
        out = f.sample(torch.randn(B, 5, d), steps=2, deterministic=True)
        check(f"FlowField(action_dim*{K}) samples (..., {a * K})", tuple(out.shape) == (B, 5, a * K),
              str(tuple(out.shape)))
    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
