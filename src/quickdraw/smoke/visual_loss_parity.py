"""VisualLoss must be BIT-IDENTICAL across the term-registry refactor.

`visual_lpips` is not merely a loss term -- it is the number 25+ historical runs are ranked on
(record §24's whole comparison table is LPIPS values). A refactor that shifts it by 1e-7 does not
error; it silently makes every past result incomparable to every future one. So this captures a
golden fingerprint of `forward()` and `temporal()` across the weight matrix, and asserts
`torch.equal` -- NOT `allclose` -- afterwards.

    python -m quickdraw.smoke.visual_loss_parity --write    # capture (run on UNCHANGED code)
    python -m quickdraw.smoke.visual_loss_parity            # verify

The golden lives next to this file and is committed. Regenerating it is the one thing that
invalidates the guarantee, so it must be a deliberate, separate, reviewed act.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from quickdraw.models.visual_loss import VisualLoss   # noqa: E402

GOLDEN = pathlib.Path(__file__).with_name("visual_loss_parity_golden.pt")

# (w_l2, w_l1, w_lpips) -- each alone, every pair, all three, and the all-zero degenerate case.
WEIGHTS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
           (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
           (1.0, 3.0, 1.0),            # the live block-stack recipe
           (0.0, 0.0, 0.0)]
FRAMES = [0, 8]                        # visual_frames: 0 = all, 8 = subsample (exercises randperm)
B, F, H, W, C = 2, 6, 32, 48, 3        # small but rank-5; H,W divisible by 16 for later DINO work


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(B, F, H, W, C, generator=g)
    t = torch.rand(B, F, H, W, C, generator=g)
    return p, t


def _fingerprint() -> dict:
    """Every (weights x frames x dtype) cell -> the two scalars. Seeded per cell, so subsampling
    and the LPIPS net's own state cannot leak across cells and make an ordering-dependent golden."""
    out = {}
    for w2, w1, wl in WEIGHTS:
        for fr in FRAMES:
            for autocast in (False, True):
                key = f"l2{w2}_l1{w1}_lp{wl}_fr{fr}_ac{int(autocast)}"
                torch.manual_seed(0)                       # _subsample draws from global RNG
                vl = VisualLoss(w_l2=w2, w_l1=w1, w_lpips=wl, lpips_net="vgg", frames=fr)
                p, t = _inputs(seed=1234)
                ctx = (torch.autocast("cpu", dtype=torch.bfloat16) if autocast
                       else torch.autocast("cpu", enabled=False))
                with ctx:
                    torch.manual_seed(0)
                    fwd5 = vl(p, t)                        # rank-5, the roundtrip-anchor shape
                    torch.manual_seed(0)
                    fwd4 = vl(p.reshape(-1, H, W, C), t.reshape(-1, H, W, C))   # rank-4, decode shape
                    torch.manual_seed(0)
                    tmp = vl.temporal(p, t, strides=(1,))
                out[key] = torch.stack([fwd5.float().detach(),
                                        fwd4.float().detach(),
                                        tmp.float().detach()])
    return out


def main() -> int:
    write = "--write" in sys.argv
    fp = _fingerprint()
    if write:
        torch.save(fp, GOLDEN)
        print(f"wrote {GOLDEN.name}: {len(fp)} cells x 3 scalars")
        for k in list(fp)[:3]:
            print(f"   {k:34} {[round(float(v), 6) for v in fp[k]]}")
        return 0
    if not GOLDEN.exists():
        print(f"FAIL no golden at {GOLDEN} -- run with --write on UNCHANGED code first")
        return 1
    ref = torch.load(GOLDEN)
    bad = []
    if set(ref) != set(fp):
        print(f"FAIL cell sets differ: missing {set(ref) - set(fp)}, extra {set(fp) - set(ref)}")
        return 1
    for k in sorted(ref):
        if not torch.equal(ref[k], fp[k]):
            bad.append((k, ref[k].tolist(), fp[k].tolist()))
    for k, a, b in bad:
        print(f"  MISMATCH {k}\n     golden {[round(x, 9) for x in a]}\n     now    {[round(x, 9) for x in b]}")
    print(f"\n{len(ref) - len(bad)}/{len(ref)} cells bit-identical"
          + ("" if bad else "  -- forward() and temporal() unchanged"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
