"""`data/transforms.py` -- the invertibility and BIT-IDENTITY guarantees the PIT change rests on.

Two jobs, and the second is the important one. PIT is new, so its round trip, its normality and -- the whole
reason it exists -- its exact recovery of the ATOM are checked on real recorded actions when the checkout has
them, and on a synthetic atom+continuum mixture when it does not. But ZScore and Symlog REPLACE expressions
that are already in flight (Normalizer's inlined z-score, act_enc's symlog), so those are checked BITWISE
against the literal old expressions rather than merely numerically -- anything less and a refactor could
quietly move the inputs of every trained model.

    python -m quickdraw.smoke.transforms
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import torch

from ..data.transforms import PIT, Compose, Symlog, ZScore, from_state

ok = bad = 0


def check(n, cond, extra=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{('  ' + extra) if extra else ''}", flush=True)


def _actions():
    """Real recorded actions if this checkout has them, else a synthetic stand-in with the same pathology:
    a large atom at zero, a continuous part, and saturation at the stick stops."""
    fs = sorted(glob.glob("/app/scratch/recording_*_starling-2/train/data/**/*.parquet", recursive=True))
    if fs:
        import pyarrow.parquet as pq
        a = np.concatenate([np.stack([np.asarray(x, np.float32) for x in pq.read_table(f).to_pydict()["action"]])
                            for f in fs[:2]])
        return a, "recorded starling-2"
    r = np.random.default_rng(0)
    n = 40000
    a = r.normal(0, 0.3, (n, 4)).astype(np.float32)
    a[r.random((n, 4)) < 0.4] = 0.0
    return np.clip(a, -0.95, 0.95), "synthetic (no recording in this checkout)"


def main() -> int:
    g = torch.Generator().manual_seed(0)

    print("\nBIT-IDENTITY with the expressions these classes replace")
    mean, std = torch.randn(16, generator=g), torch.rand(16, generator=g) + 0.1
    x = torch.randn(7, 5, 16, generator=g)
    t = ZScore(mean, std)
    check("ZScore.apply  == (x - mean) / std        (Normalizer.norm_act)",
          torch.equal(t.apply(x), (x - mean.to(x)) / std.to(x)))
    check("ZScore.invert == x * std + mean          (Normalizer.denorm_act)",
          torch.equal(t.invert(x), x * std.to(x) + mean.to(x)))
    y = torch.randn(11, 13, generator=g) * 100.0
    s = Symlog()
    check("Symlog.apply  == sign(x) * log1p(|x|)    (features.symlog)",
          torch.equal(s.apply(y), torch.sign(y) * torch.log1p(y.abs())))
    check("Symlog.invert == sign(x) * expm1(|x|)    (features.symexp)",
          torch.equal(s.invert(y), torch.sign(y) * (y.abs().expm1())))

    print("\nRESHAPING, against Normalizer's own rules")
    m4, s4 = torch.randn(4, generator=g), torch.rand(4, generator=g) + 0.1
    z4 = ZScore(m4, s4)
    check("tile(4) repeats mean and std      (tile_act, concat)",
          torch.equal(z4.tile(4).mean, m4.repeat(4)) and torch.equal(z4.tile(4).std, s4.repeat(4)))
    check("tile(1) is a no-op                (tile_act, k<=1)", z4.tile(1) is z4)
    idx = torch.tensor([0, 2])
    check("subset selects dims               (subset_obs)", torch.equal(z4.subset(idx).mean, m4[idx]))

    print("\nPIT")
    a, src = _actions()
    xa = torch.from_numpy(np.asarray(a, np.float32))
    p = PIT.fit(a, n_knots=4096)
    z = p.apply(xa, generator=torch.Generator().manual_seed(0))
    back = p.invert(z)
    print(f"    source: {src}, {tuple(xa.shape)}")
    check("z is finite", bool(torch.isfinite(z).all()))
    check(f"z ~ N(0,1)   mean {float(z.mean()):+.4f}  std {float(z.std()):.4f}",
          abs(float(z.mean())) < 0.02 and abs(float(z.std()) - 1.0) < 0.02)
    for d in range(xa.shape[-1]):
        at = xa[:, d] == 0.0
        if float(at.float().mean()) < 0.05:
            continue
        # THE POINT: every draw inside the atom's slab inverts to EXACTLY the atom, never near it
        check(f"dim {d}: atom survives the round trip exactly  ({float(at.float().mean()) * 100:.1f}% of mass)",
              torch.equal(back[at, d], xa[at, d]))
        got, want = float((back[:, d] == 0.0).float().mean()), float(at.float().mean())
        check(f"dim {d}: slab carries the atom's mass  {got * 100:.2f}% vs {want * 100:.2f}%",
              abs(got - want) < 0.01)
    rng_ = float(xa.max() - xa.min())
    err = float((back - xa).abs().max())
    check(f"non-atom values return to knot resolution  max err {err:.2e} = {err / rng_ * 100:.3f}% of range",
          err / rng_ < 0.02)

    print("\nPIT reshaping and serialisation")
    p2 = PIT.fit(a, n_knots=1024)
    wide = torch.cat([xa[:512], xa[:512]], dim=-1)          # two concat slots of the same raw distribution
    D = xa.shape[-1]
    t2 = p2.tile(2)
    check("tile(2): slot 1 gets the SAME per-axis map as slot 0  (tile_act's argument, for PIT)",
          all(torch.equal(t2.knots[d], p2.knots[d]) and torch.equal(t2.knots[d + D], p2.knots[d])
              for d in range(D)))
    bw = t2.invert(t2.apply(wide, generator=torch.Generator().manual_seed(3)))
    check("tile(2): both slots round-trip to the same accuracy as one",
          float((bw[:, :D] - wide[:, :D]).abs().max()) < 1e-3
          and float((bw[:, D:] - wide[:, D:]).abs().max()) < 1e-3)
    r = from_state(p2.state_dict())
    check("PIT survives state_dict -> from_state bit-exactly", torch.equal(r.knots, p2.knots))

    print("\nNormalizer, refactored onto ZScore -- BITWISE against the expressions it used to inline")
    import glob as _g
    import json as _j
    from ..data.dataset import Normalizer
    roots = _g.glob("/app/scratch/recording_*_starling-2") + _g.glob("/app/logs/recording_*")
    if not roots:
        print("    SKIP: no dataset in this checkout to read normalization_stats.json from")
    else:
        st = _j.load(open(roots[0] + "/normalization_stats.json"))
        n = Normalizer(st)
        om, osd = torch.tensor(st["observation_vector"]["mean"]), torch.tensor(st["observation_vector"]["std"])
        am, asd = torch.tensor(st["action"]["mean"]), torch.tensor(st["action"]["std"])
        o = torch.randn(9, 7, om.numel(), generator=g)
        av = torch.randn(9, 7, am.numel(), generator=g)
        check("norm_obs   == (o - mean) / std", torch.equal(n.norm_obs(o), (o - om) / osd))
        check("denorm_obs == o * std + mean", torch.equal(n.denorm_obs(o), o * osd + om))
        check("norm_act   == (a - mean) / std", torch.equal(n.norm_act(av), (av - am) / asd))
        check("denorm_act == a * std + mean", torch.equal(n.denorm_act(av), av * asd + am))
        a4 = torch.randn(5, am.numel() * 4, generator=g)
        check("tile_act(4) then norm_act, against repeated stats",
              torch.equal(Normalizer(st).tile_act(4).norm_act(a4), (a4 - am.repeat(4)) / asd.repeat(4)))
        check("o_mean / a_std still readable as attributes",
              n.o_mean.numel() == om.numel() and n.a_std.numel() == asd.numel())

    print("\nact_enc's symlog, end to end against the formula it replaced")
    from ..models.multimodal import FourierMLP
    torch.manual_seed(0)
    enc = FourierMLP(6, 8, 16, n_freq=2, input_squash="symlog").eval()
    xi = torch.randn(4, 6, generator=g) * 50.0            # large, so the squash is doing real work
    with torch.no_grad():
        got = enc(xi)
        # the OLD path: features.symlog applied before both the raw copy and the fourier expansion
        from ..models.features import fourier_features
        xs = torch.sign(xi) * torch.log1p(xi.abs())
        want = enc.net(torch.cat([xs, fourier_features(xs, enc.freqs, squash=enc.squash)], dim=-1))
    check("FourierMLP(input_squash=symlog) is bitwise unchanged", torch.equal(got, want))

    print("\nTHE HEAD'S TARGET SPACE: action_pairs / sample_action under target_transform")
    from ..models.multimodal import MultiModalFlow

    class _Head(torch.nn.Module):
        """The real unbound methods on a minimal object -- the same stand-in trick smoke/action_chunk.py
        uses, so the target-space plumbing is checked without building a world model."""

        def __init__(self, K, mode, knots=None, flow=None):
            super().__init__()
            self.action_head_chunk, self.action_head_target_transform = K, mode
            self.action_head_sampling_steps, self.action_flow = 4, flow
            if knots is not None:
                self.register_buffer("action_pit_knots", knots)

        action_pairs = MultiModalFlow.action_pairs
        sample_action = MultiModalFlow.sample_action

    B, L, K = 64, 64, 3                                  # enough rows that "is z standard normal" is a real test
    # THE KNOTS ARE FITTED ON RAW ACTIONS BUT act_seq IS NORMALIZED, so the knots must be z-scored into the
    # same space -- setup._action_pit_knots does this. Feeding raw knots to a normalized target pinned 38.6%
    # of it on the clamp (measured, and it cost a training run), so the smoke feeds the NORMALIZED path.
    _m, _sd = torch.tensor(a.mean(0)), torch.tensor(a.std(0)) + 1e-6
    kn = ((PIT.fit(a, n_knots=512).knots.T - _m) / _sd).T
    An = (xa - _m) / _sd
    A = An[torch.randperm(An.shape[0], generator=g)[: B * L]].reshape(B, L, An.shape[-1])
    h = torch.randn(B, L - 1, 8, generator=g)
    c0, t0 = _Head(K, "none").action_pairs(h, A)
    check("target_transform=none leaves action_pairs untouched",
          torch.equal(t0, MultiModalFlow.action_pairs(_Head(K, "none"), h, A)[1]))
    c1, t1 = _Head(K, "pit", kn).action_pairs(h, A)
    check("target_transform=pit changes the target, keeps its shape",
          t1.shape == t0.shape and not torch.equal(t1, t0))
    check("pit target is standard-normal-ish  "
          f"mean {float(t1.mean()):+.3f} std {float(t1.std()):.3f}",
          abs(float(t1.mean())) < 0.15 and 0.85 < float(t1.std()) < 1.15)

    class _FakeFlow:                                     # returns z; sample_action must invert it
        def sample(self_, h_ctx, steps, deterministic=False, eps=None):
            return torch.randn(*h_ctx.shape[:-1], K * kn.shape[0], generator=g)

    out = _Head(K, "pit", kn, _FakeFlow()).sample_action(h)
    check("sample_action inverts the draw back into NORMALIZED action units",
          out.shape == (B, L - 1, K * kn.shape[0])
          and float(out.abs().max()) <= float(An.abs().max()) + 1e-6)
    atom_z = ((torch.zeros(An.shape[-1]) - _m) / _sd)     # where the atom sits after normalization
    atom_frac = float((out.reshape(-1, K, kn.shape[0]) == atom_z).float().mean())
    check(f"...and the inverted draws land ON the atom  ({atom_frac * 100:.1f}% exactly at rest)",
          atom_frac > 0.15)

    print("\nTHE THREE HARD ERRORS")
    try:
        _Head(K, "pit", kn, _FakeFlow()).sample_action(h, deterministic=True)
        check("deterministic draw under pit raises", False)
    except ValueError as e:
        check("deterministic draw under pit raises", "MEDIAN" in str(e))
    from ..training.setup import _action_pit_knots
    from omegaconf import OmegaConf as _OC
    try:
        _action_pit_knots(_OC.create({"data": {"action_aggregate": "sum"}}), "pit")
        check("action_aggregate=sum under pit raises", False)
    except ValueError as e:
        check("action_aggregate=sum under pit raises", "concat" in str(e))
    check("target_transform=off asks the dataset for nothing",
          _action_pit_knots(_OC.create({"data": {"action_aggregate": "sum"}}), "none") is None)
    print("\nCompose")
    c = Compose(ZScore(m4, s4), Symlog())
    xc = torch.randn(9, 4, generator=g)
    check("Compose inverts in reverse order", bool(torch.allclose(c.invert(c.apply(xc)), xc, atol=1e-5)))
    check("Compose round-trips through state_dict",
          torch.equal(from_state(c.state_dict()).apply(xc), c.apply(xc)))

    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
