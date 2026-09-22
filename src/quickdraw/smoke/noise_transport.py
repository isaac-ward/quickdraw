"""Smoke for models/noise_transport (∫-noise). The claim is "advect noise along the flow, still white":
  1. pixelize gives ~zero-mean, ~unit-variance, ~white (low lag-1 autocorr) noise.
  2. an INTEGER fine-grid shift advects EXACTLY (bit-exact roll) -- the transport is correct.
  3. a fractional (bilinear) flow keeps the output ~unit-variance and ~white (the approximation holds).
  4. zero flow is the identity -> recovers the constant decode_shared_noise case exactly.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.noise_transport
"""
import torch
torch.set_num_threads(1)

from ..models.noise_transport import advect_and_pixelize, fine_noise, pixelize, warp_fine


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
    B, C, H, W, up = 4, 3, 48, 64, 4
    fine = fine_noise(B, C, H, W, up)

    # 1. pixelize stats
    n0 = pixelize(fine, up)
    if n0.shape != (B, C, H, W):
        _fail(f"pixelize shape {tuple(n0.shape)} != {(B,C,H,W)}")
    print(f"pixelize: mean={n0.mean():+.3f} std={n0.std():.3f} whiteness={_white(n0):.3f}")
    if abs(float(n0.mean())) > 0.03 or abs(float(n0.std()) - 1.0) > 0.05 or _white(n0) > 0.1:
        _fail("pixelized field is not ~zero-mean unit-variance white noise")
    print("  ~N(0,1) and ~white   OK")

    # 2. integer fine-grid shift == exact roll (transport is correct)
    sx, sy = 2, 1                                              # fine-grid cells
    flow_px = torch.zeros(B, H, W, 2)
    flow_px[..., 0] = sx / up                                  # pixel-units; * up inside = sx fine cells
    flow_px[..., 1] = sy / up
    warped = warp_fine(fine, flow_px, up)
    exact = torch.roll(fine, shifts=(sy, sx), dims=(2, 3))
    inner = (slice(None), slice(None), slice(2, -2), slice(2, -2))   # ignore reflected border
    md = float((warped[inner] - exact[inner]).abs().max())
    print(f"integer shift vs exact roll: max|Δ| (interior) = {md:.2e}")
    if md > 1e-4:
        _fail("integer fine-grid shift did not advect exactly")
    print("  exact advection   OK")

    # 3. fractional (bilinear) flow: still ~unit-variance and ~white
    flow_frac = torch.zeros(B, H, W, 2)
    flow_frac[..., 0] = 0.7; flow_frac[..., 1] = -0.4          # sub-pixel, non-integer -> bilinear
    _, nf = advect_and_pixelize(fine, flow_frac, up)
    print(f"fractional flow: std={nf.std():.3f} whiteness={_white(nf):.3f}")
    if abs(float(nf.std()) - 1.0) > 0.12 or _white(nf) > 0.2:
        _fail("fractional-flow advection drifted too far from white unit-variance (raise `up`)")
    print("  approximation holds   OK")

    # 4. zero flow == identity (the constant decode_shared_noise case)
    warped0, n_z = advect_and_pixelize(fine, None, up)
    if not torch.equal(n_z, n0):
        _fail("zero/None flow did not recover the constant shared-noise field exactly")
    print("zero flow -> constant shared-noise recovered exactly   OK")

    print("\nALL NOISE-TRANSPORT SMOKE PASSED")


if __name__ == "__main__":
    main()
