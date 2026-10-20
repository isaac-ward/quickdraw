"""Smoke for the saliency-weighted decode loss (2026-09-28).
  1. OFF (saliency=none / weight 0) -> BIT-IDENTICAL to the unweighted VisualLoss.
  2. mask fires on cube-colored pixels, ~0 on table/background.
  3. targets cubes: a pred that errs ONLY on cube pixels costs MORE with weight>0; a pred that errs ONLY
     on background costs the SAME as unweighted (that region has weight 1).
  4. site gate: saliency applies at site="decode" only, NOT site="codec".
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.saliency_loss
"""
import torch

from ..models.visual_loss import VisualLoss


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _cube_frame(M=2, H=96, W=128):
    """A tan-table frame with a small saturated RED cube patch."""
    t = torch.zeros(M, H, W, 3)
    t[..., 0] = 0.55; t[..., 1] = 0.42; t[..., 2] = 0.22   # tan table (hue ~35, in the excluded band)
    t[:, 40:52, 60:72, :] = torch.tensor([0.9, 0.05, 0.05])  # red cube (saturated, hue ~0)
    return t


def main():
    torch.manual_seed(0)
    tgt = _cube_frame()

    # ---- 1. OFF == unweighted, bit-identical ----
    off = VisualLoss(w_l2=0.0, w_l1=1.0)                       # pure L1
    on = VisualLoss(w_l2=0.0, w_l1=1.0, saliency="hue", saliency_weight=8.0)
    pred = tgt + 0.1 * torch.randn_like(tgt)
    l_off = off(pred, tgt, site="decode")
    l_on_codec = on(pred, tgt, site="codec")                  # saliency gated OFF at codec site
    if not torch.allclose(l_off, l_on_codec):
        _fail(f"codec site must be unweighted: {l_off.item()} vs {l_on_codec.item()}")
    print(f"[1/4] OFF==codec-site bit-identical: {l_off.item():.6f}   OK")

    # ---- 2. mask fires on cube, ~0 on table ----
    w = on._saliency_map(tgt)                                  # (M,H,W,1), 1 + K*cube
    cube_region = w[:, 40:52, 60:72, 0]
    table_region = w[:, 0:20, 0:20, 0]
    if not (cube_region.mean() > 8.0 * 0.9 and table_region.mean() < 1.01):
        _fail(f"mask wrong: cube weight {cube_region.mean():.2f} (want ~9), table {table_region.mean():.2f} (want ~1)")
    print(f"[2/4] mask: cube weight {cube_region.mean():.2f}, table {table_region.mean():.2f}   OK")

    # ---- 3. targets cubes ----
    # (a) error ONLY on cube pixels -> weighted loss LARGER than unweighted
    p_cube = tgt.clone(); p_cube[:, 40:52, 60:72, :] += 0.3
    if not (on(p_cube, tgt, "decode") > 1.5 * off(p_cube, tgt, "decode")):
        _fail("cube-only error should cost MORE under saliency")
    # (b) error ONLY on background -> weighted == unweighted (weight 1 there)
    p_bg = tgt.clone(); p_bg[:, 0:20, 0:20, :] += 0.3
    if not torch.allclose(on(p_bg, tgt, "decode"), off(p_bg, tgt, "decode")):
        _fail("background-only error must be unchanged by saliency")
    print("[3/4] cube error costs more; background error unchanged   OK")

    # ---- 4. gradient flows, finite ----
    p = (tgt + 0.1 * torch.randn_like(tgt)).requires_grad_(True)
    on(p, tgt, "decode").backward()
    if not torch.isfinite(p.grad).all():
        _fail("non-finite grad")
    g_cube = p.grad[:, 40:52, 60:72, :].abs().mean()
    g_bg = p.grad[:, 0:20, 0:20, :].abs().mean()
    if not (g_cube > g_bg):
        _fail(f"cube grad {g_cube:.4g} should exceed background grad {g_bg:.4g}")
    print(f"[4/4] grad cube {g_cube:.4g} > background {g_bg:.4g}   OK")

    print("\nALL SALIENCY-LOSS SMOKE PASSED")


if __name__ == "__main__":
    main()
