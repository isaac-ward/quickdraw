"""Invertible per-dimension transforms, with fitted state, in ONE place.

Before this file the repo had two unrelated ways of reshaping a vector before a network saw it: the
z-score inside `Normalizer` (fitted, in the data package) and `symlog` inside `act_enc` (a bare formula, in
the models package). Both are the same kind of object -- a monotone per-dim map with an inverse -- and a
third one (PIT) has no home in either. So they all live here, behind one protocol.

    apply / invert       the map and its inverse
    fit                  TRAIN SPLIT ONLY, like every other statistic in this project
    state_dict / load    so a fitted map is stored next to mean/std and frozen for val/test

WHAT IS FITTED, AND WHAT ISN'T. ZScore and PIT carry data; Symlog is a pure formula and fits nothing. That
distinction is why Symlog needed no artifact and PIT does.

THE ONE THAT IS NEW: PIT, the probability integral transform.
    z = Phi^-1( F(a) )          F = the empirical CDF of the training data for that dim
Percentiles are uniform BY DEFINITION, so this flattens whatever shape the marginal had -- a spike at rest,
saturation at the stick stops, the quantisation grid -- with no assumption about WHERE the sharp parts are.
It reads them off the data. A rectified flow integrates a finite-Lipschitz velocity field, so its terminal
law is absolutely continuous: it CANNOT emit an atom, and ~38-58% of every recorded joystick axis is exactly
0.0. Under PIT that atom becomes a SLAB of z (F jumps there, and the jump is resolved by drawing uniformly
inside it), so the flow only has to get the slab's total mass right -- and every z in the slab inverts to
exactly 0.0, not to a smear around it. Measured on starling-2: round-trip error 0.0, z ~ N(0,1) to 3dp.

The transform is MARGINAL and UNCONDITIONAL: it makes sharp features of the POOLED distribution easy, and
does nothing about a conditional that is near-deterministic at a marginally rare value. That case stays
hard -- but it is finite-density hard, which a finite velocity field can do, unlike an atom.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor


@runtime_checkable
class Transform(Protocol):
    """A monotone per-dimension map with an exact inverse. `apply` and `invert` take and return tensors
    shaped (..., D); everything is per-dim, so leading dimensions are free."""

    def apply(self, x: Tensor) -> Tensor: ...
    def invert(self, x: Tensor) -> Tensor: ...
    def state_dict(self) -> dict: ...


class Symlog:
    """sign(x) * log(1 + |x|) -- monotone, invertible, near-identity for |x| < 1. Fits nothing.

    Moved here from models/features.py; the formula is unchanged and `act_enc` still applies it at the same
    point (on the conditioning path, after normalization). It is NOT part of the Normalizer's chain, because
    the head's regression target is built from the same normalized tensor the encoder conditions on, and
    folding symlog into that chain would silently move the target too.

    WHY IT EXISTS: (x - mean)/std centers and scales but does NOT bound. A 10-sigma value becomes 10.0, and
    robocasa action dims whose std floors near 1e-6 turn a real 0.5 deviation into 5e5 (measured); symlog
    compresses that to ~13.1 while PRESERVING ORDER, where a clamp would map all of that dim's variation
    onto one value. Needs no dataset statistics, unlike a true [0,1] rescale. GAIA-2 (arXiv:2503.20523)
    symlogs actions for this reason; DreamerV3 symlogs observations/rewards."""

    def apply(self, x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log1p(x.abs())

    def invert(self, x: Tensor) -> Tensor:
        return torch.sign(x) * (x.abs().expm1())

    def state_dict(self) -> dict:
        return {"kind": "symlog"}

    @classmethod
    def from_state(cls, _s: dict) -> "Symlog":
        return cls()


class ZScore:
    """(x - mean) / std. The transform every dataset has had since the beginning, now an object.

    Holds tensors rather than fitting from data here: mean/std are computed once when the dataset is built
    and stored in normalization_stats.json, and `Normalizer` hands them over. `subset` and `tile` exist
    because the loaders reshape the vector they apply to (an obs subset, or `subsample` raw actions laid
    end to end) and the stats have to follow."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean, self.std = mean, std

    def apply(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)

    def invert(self, x: Tensor) -> Tensor:
        return x * self.std.to(x) + self.mean.to(x)

    def subset(self, idx: Tensor) -> "ZScore":
        return ZScore(self.mean[idx], self.std[idx])

    def tile(self, k: int) -> "ZScore":
        return self if int(k) <= 1 else ZScore(self.mean.repeat(int(k)), self.std.repeat(int(k)))

    def state_dict(self) -> dict:
        return {"kind": "zscore", "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "ZScore":
        return cls(torch.tensor(s["mean"]), torch.tensor(s["std"]))


class PIT:
    """Probability integral transform: z = Phi^-1(F(x)), with F the EMPIRICAL CDF of the training data.

    Stored as `knots`, (D, K) per dim: the quantile function sampled at K percentiles from 0 to 1
    INCLUSIVE, so the first and last knots are the data's min and max and the whole recorded range is
    representable. Between knots the quantile function is treated as PIECEWISE LINEAR, which is what makes
    the round trip exact rather than "to the nearest stored value": nearest-knot lookup cost 18% of the
    range on the thin upper tail of one axis (13 samples above the last knot), because a thin tail is
    exactly where evenly spaced percentiles are furthest apart in VALUE.

    Linear interpolation does not weaken the atom guarantee, which is the property the whole transform
    exists for: an atom is a RUN of identical knots, and interpolating between two identical values returns
    that value exactly, in floating point, for any interpolation weight.

    TIES ARE THE POINT. Where the data has an atom, F jumps, and one x maps to an INTERVAL of percentiles.
    `apply` draws uniformly inside that interval, which is what turns the atom into a slab of positive
    width; `invert` maps every point of the slab back to the atom's exact value. This makes `apply`
    STOCHASTIC -- standard variational dequantisation, and the reason the flow sees the slab filled evenly
    rather than seeing one impossible point. Pass a generator to make it reproducible."""

    U_EPS = 1e-7                                             # keeps z finite at the ends of the range

    def __init__(self, knots: Tensor):
        self.knots = knots                                   # (D, K), sorted along K

    # ---- fitting ---------------------------------------------------------------------------------
    @classmethod
    def fit(cls, x, n_knots: int = 4096) -> "PIT":
        """x: (N, D) TRAIN-SPLIT samples. One quantile function per dim, sampled at K percentiles spanning
        [0, 1] inclusive. `inverted_cdf` keeps every knot an actually-recorded value."""
        a = np.asarray(x, dtype=np.float64)
        assert a.ndim == 2 and a.shape[0] > 1, f"PIT.fit wants (N, D) with N > 1, got {a.shape}"
        n = min(int(n_knots), a.shape[0])
        q = np.linspace(0.0, 1.0, n)
        k = np.stack([np.quantile(np.sort(a[:, d]), q, method="inverted_cdf") for d in range(a.shape[1])])
        return cls(torch.tensor(k, dtype=torch.float32))

    # ---- the map ---------------------------------------------------------------------------------
    def apply(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """x -> knot coordinate t -> percentile -> z. t is the position along the piecewise-linear quantile
        function; where x sits on a FLAT run (an atom) t is drawn uniformly along that run."""
        kn = self.knots.to(x.device, x.dtype)
        D, K = kn.shape
        assert x.shape[-1] == D, f"PIT fitted for {D} dims, got {x.shape[-1]}"
        flat = x.reshape(-1, D)
        u01 = torch.rand(flat.shape, dtype=flat.dtype, device=flat.device, generator=generator)
        t = torch.empty_like(flat)
        for d in range(D):                                   # searchsorted wants a 1-D sorted sequence
            v = flat[:, d].contiguous()
            lo = torch.searchsorted(kn[d], v, right=False)   # first knot >= v
            hi = torch.searchsorted(kn[d], v, right=True)    # first knot >  v
            on_knot = hi > lo                                # v IS a knot value, repeated (hi - lo) times
            t_run = lo.to(v.dtype) + u01[:, d] * (hi - 1 - lo).clamp(min=0).to(v.dtype)
            j = (lo - 1).clamp(0, K - 2)                     # v lies strictly between knots j and j+1
            gap = (kn[d][j + 1] - kn[d][j]).clamp(min=torch.finfo(v.dtype).tiny)
            t[:, d] = torch.where(on_knot, t_run, j.to(v.dtype) + (v - kn[d][j]) / gap)
        u = (t.clamp(0, K - 1) / (K - 1)).clamp(self.U_EPS, 1.0 - self.U_EPS)
        return _ndtri(u).reshape(x.shape)

    def invert(self, z: Tensor) -> Tensor:
        """z -> percentile -> t -> x, linearly between the two bracketing knots."""
        kn = self.knots.to(z.device, z.dtype)
        D, K = kn.shape
        assert z.shape[-1] == D, f"PIT fitted for {D} dims, got {z.shape[-1]}"
        flat = z.reshape(-1, D)
        t = (_ndtr(flat) * (K - 1)).clamp(0, K - 1)
        i0 = t.floor().long().clamp(0, K - 2)
        w = (t - i0.to(t.dtype)).unsqueeze(-1)
        knT = kn.T.contiguous()                              # (K, D): knT[i, d] = knots[d, i]
        a0 = torch.gather(knT, 0, i0)
        a1 = torch.gather(knT, 0, i0 + 1)
        return (a0 + w.squeeze(-1) * (a1 - a0)).reshape(z.shape)   # a0 == a1 on an atom -> exactly a0

    # ---- reshaping, mirroring ZScore ---------------------------------------------------------------
    def subset(self, idx: Tensor) -> "PIT":
        return PIT(self.knots[idx])

    def tile(self, k: int) -> "PIT":
        """`data.action_aggregate=concat` lays k raw actions end to end, and each slot holds the SAME raw
        distribution the knots were fitted on -- so repeating them is exactly right, for the same reason
        Normalizer.tile_act repeats mean/std."""
        return self if int(k) <= 1 else PIT(self.knots.repeat(int(k), 1))

    def state_dict(self) -> dict:
        return {"kind": "pit", "knots": self.knots.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "PIT":
        return cls(torch.tensor(s["knots"], dtype=torch.float32))


class Compose:
    """Transforms applied left to right; inverted right to left."""

    def __init__(self, *ts: Transform):
        self.ts = list(ts)

    def apply(self, x: Tensor) -> Tensor:
        for t in self.ts:
            x = t.apply(x)
        return x

    def invert(self, x: Tensor) -> Tensor:
        for t in reversed(self.ts):
            x = t.invert(x)
        return x

    def state_dict(self) -> dict:
        return {"kind": "compose", "ts": [t.state_dict() for t in self.ts]}

    @classmethod
    def from_state(cls, s: dict) -> "Compose":
        return cls(*[from_state(t) for t in s["ts"]])


_KINDS = {"symlog": Symlog, "zscore": ZScore, "pit": PIT, "compose": Compose}


def from_state(s: dict) -> Transform:
    kind = s["kind"]
    assert kind in _KINDS, f"unknown transform kind {kind!r}, expected one of {sorted(_KINDS)}"
    return _KINDS[kind].from_state(s)


# ---- the normal CDF and its inverse, in torch so the transform runs on whatever device it is handed ----
def _ndtr(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def _ndtri(u: Tensor) -> Tensor:
    return np.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
