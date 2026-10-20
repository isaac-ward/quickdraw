"""Smoke for models/noise_transport (∫-noise, paper Eq 5 scatter transport). Claims checked:
  1. pixelize gives ~zero-mean, ~unit-variance, ~white noise.
  2. an integer-PIXEL flow advects EXACTLY: warp == roll(pixelize) in the interior.
  3. a fractional (sub-pixel) flow keeps output ~unit-variance and ~white -- the scatter/sqrt(count) transport
     does NOT dissipate variance the way bilinear grid_sample did (that version fell to std~0.90 here).
  4. zero flow == pixelize (recovers the constant decode_shared_noise case exactly).
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.noise_transport
"""
import torch
torch.set_num_threads(1)

from ..models.noise_transport import advect_and_pixelize, fine_noise, pixelize, warp_integral


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _white(x):                                    # max |lag-1 autocorrelation| over H and W (0 = white)
    x = x - x.mean()
    v = x.var().clamp_min(1e-8)
    ah = (x[..., 1:, :] * x[..., :-1, :]).mean() / v
    aw = (x[..., :, 1:] * x[..., :, :-1]).mean() / v
    return float(max(ah.abs(), aw.abs()))


def main():
    torch.manual_seed(0)
    B, C, H, W, up = 4, 3, 48, 64, 8
    fine = fine_noise(B, C, H, W, up)

    # 1. pixelize stats
    n0 = pixelize(fine, up)
    if n0.shape != (B, C, H, W):
        _fail(f"pixelize shape {tuple(n0.shape)} != {(B,C,H,W)}")
    print(f"pixelize: mean={n0.mean():+.3f} std={n0.std():.3f} whiteness={_white(n0):.3f}")
    if abs(float(n0.mean())) > 0.03 or abs(float(n0.std()) - 1.0) > 0.05 or _white(n0) > 0.1:
        _fail("pixelized field is not ~zero-mean unit-variance white noise")
    print("  ~N(0,1) and ~white   OK")

    # 2. integer-PIXEL flow == exact roll (scatter transport is correct). Content at p advects to p+(dx,dy),
    #    so G[p+dy, p+dx] = pixelize(p). The first (dy,dx) rows/cols of G are HOLES (nothing lands there) and
    #    the LAST row/col are clamp-piled, so compare the clean interior g[dy:H-1, dx:W-1] vs n0[:H-1-dy, :W-1-dx].
    dx, dy = 1, 2                                              # whole output pixels
    flow = torch.zeros(B, H, W, 2); flow[..., 0] = dx; flow[..., 1] = dy
    g = warp_integral(fine, flow, up)
    md = float((g[:, :, dy:H - 1, dx:W - 1] - n0[:, :, :H - 1 - dy, :W - 1 - dx]).abs().max())
    print(f"integer-pixel shift vs roll(pixelize): max|Δ| (clean interior) = {md:.2e}")
    if md > 1e-4:
        _fail("integer-pixel flow did not advect as an exact roll of the pixelized field")
    print("  exact advection   OK")

    # 3. fractional (sub-pixel) flow: variance and whiteness PRESERVED (no bilinear dissipation)
    flowf = torch.zeros(B, H, W, 2); flowf[..., 0] = 0.7; flowf[..., 1] = -0.4
    _, nf = advect_and_pixelize(fine, flowf, up)
    print(f"fractional flow: std={nf.std():.3f} whiteness={_white(nf):.3f}")
    if abs(float(nf.std()) - 1.0) > 0.06 or _white(nf) > 0.1:
        _fail("fractional-flow transport lost variance/whiteness (should be ~1.0 now, unlike bilinear)")
    print("  variance + whiteness preserved   OK")

    # 4. zero flow == pixelize (constant shared-noise case)
    _, nz = advect_and_pixelize(fine, None, up)
    if not torch.equal(nz, n0):
        _fail("zero/None flow did not recover the constant shared-noise field exactly")
    print("zero flow -> constant shared-noise recovered exactly   OK")

    print("\nALL NOISE-TRANSPORT SMOKE PASSED")


if __name__ == "__main__":
    main()
