"""Smoke for LatentStraightness (training/variations.py):
  - a straight (constant-velocity) latent trajectory scores ~0 curvature,
  - a random-walk trajectory scores high,
  - the gradient reaches `preds`,
  - the suite gates it on weight>0 (off by default -> bit-identical baseline).
CPU, seeded. Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.straightness
"""

import torch
from omegaconf import OmegaConf

from ..training.variations import LatentStraightness, VarContext, make_variation_suite


def _ctx(z):
    return VarContext(model=None, preds=z, future_obs=None, obs_seq=None, act_seq=None, norm=None, training=True)


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def main():
    torch.manual_seed(0)
    B, Fh, N, d = 4, 12, 8, 16
    v = LatentStraightness(weight=1.0)

    # straight: z_t = base + vel*t  -> every step-vector == vel -> cos 1 -> curv ~0
    base = torch.randn(B, 1, N, d); vel = torch.randn(B, 1, N, d)
    t = torch.arange(Fh).float().view(1, Fh, 1, 1)
    straight = float(v.loss(_ctx(base + vel * t))[0])
    if straight > 1e-4:
        _fail(f"straight trajectory curv should be ~0, got {straight}")
    print(f"straight trajectory curv {straight:.2e}                 OK (~0)")

    # random walk: cumulative iid noise -> uncorrelated consecutive steps -> cos ~0 -> curv ~1
    rw = float(v.loss(_ctx(torch.randn(B, Fh, N, d).cumsum(dim=1)))[0])
    if rw < 0.5:
        _fail(f"random-walk curv should be high, got {rw}")
    print(f"random-walk trajectory curv {rw:.3f}             OK (high)")

    # gradient reaches preds
    z = torch.randn(B, Fh, N, d, requires_grad=True)
    v.loss(_ctx(z))[0].backward()
    if z.grad is None or float(z.grad.abs().sum()) == 0.0:
        _fail("no gradient reached preds")
    print("gradient flows to preds                          OK")

    # too-short trajectory -> no term (need >=3 frames), and non-latent (3D) preds -> skip
    if v.loss(_ctx(torch.randn(B, 2, N, d)))[0] is not None:
        _fail("2-frame trajectory should be skipped")
    if v.loss(_ctx(torch.randn(B, Fh, d)))[0] is not None:
        _fail("non-latent (DSAR obs) preds should be skipped")
    print("short / non-latent inputs skip cleanly           OK")

    # suite gates on weight>0 (off by default)
    off = [x.name for x in make_variation_suite(OmegaConf.create({"latent_straightness": {"weight": 0.0}})).variations]
    on = [x.name for x in make_variation_suite(OmegaConf.create({"latent_straightness": {"weight": 0.5}})).variations]
    if "latent_straightness" in off:
        _fail("weight=0 must NOT enable it")
    if "latent_straightness" not in on:
        _fail("weight>0 must enable it")
    print("suite gates on weight>0 (off by default)         OK")

    print("\nALL STRAIGHTNESS SMOKE PASSED")


if __name__ == "__main__":
    main()
