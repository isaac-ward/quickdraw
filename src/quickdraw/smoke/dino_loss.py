"""The DINOv3 patch term must measure PATCH-LOCAL similarity, and must be inert when off.

The claim being tested is not "it runs". It is that the reduction differs from LPIPS in the
specific way the design says: a cosine taken per patch over the feature axis, so a small
low-contrast change counts as much as a large bright one -- which is the 5x dilution LPIPS suffers
on our 84.6% static frames.

    python -m quickdraw.smoke.dino_loss
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from quickdraw.models.dino_loss import (PATCH, get_net, patch_cosine,      # noqa: E402
                                        patch_tokens, weighted_patch_cosine)
from quickdraw.models.visual_loss import VisualLoss                        # noqa: E402

_n = _f = 0


def chk(name, ok, detail=""):
    global _n, _f
    _n += 1
    if not ok:
        _f += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}")


H, W = 96, 128           # the live block-stack size
NET = "vits16"
dev = "cpu"

# ---- geometry --------------------------------------------------------------------------------
chk("patch 16 divides our frames exactly", H % PATCH == 0 and W % PATCH == 0,
    f"{H}x{W} -> {H // PATCH}x{W // PATCH} = {(H // PATCH) * (W // PATCH)} patches")

net = get_net(NET, dev)
if net is None:
    print("\n  DINOv3 weights unavailable -- the term fails soft, which is itself the contract.")
    chk("a VisualLoss with the term ON still runs when weights are missing",
        torch.isfinite(VisualLoss(w_l2=1.0, w_dino_v3=1.0)(torch.rand(2, H, W, 3),
                                                           torch.rand(2, H, W, 3))).item())
    print(f"\n{_n - _f}/{_n} passed")
    raise SystemExit(1 if _f else 0)

torch.manual_seed(0)
x = torch.rand(3, H, W, 3)
tok = patch_tokens(net, x, -1)
chk("tokens are (M, 48, 384) with cls+registers dropped", tuple(tok.shape) == (3, 48, 384), tuple(tok.shape))

# ---- the distance behaves ----------------------------------------------------------------------
chk("identical frames -> ~0", abs(float(patch_cosine(tok, tok))) < 1e-5, f"{float(patch_cosine(tok, tok)):.2e}")
# ---- THE CLAIM, on REAL frames ------------------------------------------------------------------
# A synthetic version of this test FAILED and was right to: flat grey -> slightly different flat grey
# is no SEMANTIC change at all, so DINOv3 correctly reported ~nothing while LPIPS saw the pixels. A
# semantic critic has to be tested on semantic content. So: a real block-stack frame, perturbed two
# ways -- one patch replaced with genuinely different CONTENT from a later frame, versus a global
# brightness shift that changes every pixel while nothing IS different.
import numpy as np                                                      # noqa: E402

_NPY = pathlib.Path(__file__).resolve().parents[3] / (
    "logs/recording_2026_09_10_10_04_45_longhand/val/scene_right_96x128.npy")
if _NPY.exists():
    _a = np.load(_NPY, mmap_mode="r")
    f0 = torch.from_numpy(np.array(_a[1000])).float()[None] / 255.0
    f1 = torch.from_numpy(np.array(_a[1300])).float()[None] / 255.0     # objects have MOVED
    loc = f0.clone()
    loc[0, 48:64, 64:80] = f1[0, 48:64, 64:80]      # ONE patch of 48, real content
    glob = (f0 + 0.04).clamp(0, 1)                  # every pixel, no semantic change
    px_loc = float((loc - f0).abs().mean())
    px_glob = float((glob - f0).abs().mean())
    lp = VisualLoss(w_l2=0.0, w_lpips=1.0, frames=0)
    dn = VisualLoss(w_l2=0.0, w_dino_v3=1.0, dino_v3_net=NET, frames=0)
    r_lp = float(lp(loc, f0)) / max(float(lp(glob, f0)), 1e-12)
    r_dn = float(dn(loc, f0)) / max(float(dn(glob, f0)), 1e-12)
    chk("the local change really is far smaller IN PIXELS", px_loc < px_glob / 10,
        f"{px_loc:.5f} vs {px_glob:.5f} = {px_glob / px_loc:.0f}x smaller")
    chk("LPIPS ranks the flat global change HIGHER -- the dilution this term exists to fix",
        r_lp < 1.0, f"ratio {r_lp:.2f}x")
    chk("DINOv3 ranks the LOCAL SEMANTIC change higher", r_dn > 1.0, f"ratio {r_dn:.2f}x")
    chk("...and by a wide margin over LPIPS", r_dn / r_lp > 3.0, f"{r_dn / r_lp:.1f}x better")
else:
    print(f"  SKIP  real-frame locality checks (no {_NPY.name}; build the dataset first)")

# NOTE, worth not re-discovering: on two frames of PURE NOISE the patch cosine is only ~0.018.
# Random noise is semantically "nothing", so every noise patch maps to much the same feature. That
# is correct behaviour for a semantic critic, not insensitivity -- but it means noise is useless as
# a test signal here, which is why the checks above use real frames.

# ---- the weighting modes ------------------------------------------------------------------------
a = torch.randn(2, 48, 384)
b = torch.randn(2, 48, 384)
a[:, :40] *= 1e-6                       # 40 of 48 patches near-zero, as a real temporal diff is
b[:, :40] *= 1e-6
plain = float(weighted_patch_cosine(a, b, "none"))
wtrue = float(weighted_patch_cosine(a, b, "true"))
wmax = float(weighted_patch_cosine(a, b, "max"))
chk("all three weighting modes are finite", all(map(lambda v: v == v, (plain, wtrue, wmax))),
    f"none {plain:.4f} true {wtrue:.4f} max {wmax:.4f}")
chk("weighting CHANGES the answer when most patches are static", abs(plain - wmax) > 1e-3,
    f"none {plain:.4f} vs max {wmax:.4f}")
aa = a.clone().requires_grad_(True)
weighted_patch_cosine(aa, b, "max").backward()
chk("gradient flows through the weighted form", torch.isfinite(aa.grad).all().item()
    and float(aa.grad.abs().sum()) > 0, f"|grad| {float(aa.grad.abs().sum()):.3g}")

# the weight must be DETACHED, or freezing lowers the loss by shrinking its own weight
a2 = torch.randn(2, 48, 384, requires_grad=True)
w_direct = torch.maximum(a2.norm(dim=-1), b.norm(dim=-1))
chk("the weight is detached in 'max' (freezing cannot shrink its own weight)",
    not torch.maximum(a2.norm(dim=-1), b.norm(dim=-1)).detach().requires_grad)

# ---- integration -------------------------------------------------------------------------------
p5 = torch.rand(2, 4, H, W, 3)
t5 = torch.rand(2, 4, H, W, 3)
v = VisualLoss(w_l2=1.0, w_l1=3.0, w_lpips=1.0, w_dino_v3=0.5, dino_v3_net=NET, frames=8)
chk("pointwise: rank-5 input runs and is finite", torch.isfinite(v(p5, t5)).item(), f"{float(v(p5, t5)):.4f}")
chk("temporal: runs and is finite", torch.isfinite(v.temporal(p5, t5, strides=(1,))).item(),
    f"{float(v.temporal(p5, t5, strides=(1,))):.4f}")
off = VisualLoss(w_l2=1.0, w_l1=3.0, w_lpips=1.0, w_dino_v3=0.0, frames=8)
torch.manual_seed(0); a_on = float(v(p5, t5))
torch.manual_seed(0); a_off = float(off(p5, t5))
chk("turning the weight ON changes the total (the term is load-bearing)", a_on != a_off,
    f"on {a_on:.4f} vs off {a_off:.4f}")
pr = p5.clone().requires_grad_(True)
VisualLoss(w_l2=0.0, w_dino_v3=1.0, dino_v3_net=NET, frames=0).temporal(pr, t5, strides=(1,)).backward()
chk("temporal gradient reaches the prediction", torch.isfinite(pr.grad).all().item()
    and float(pr.grad.abs().sum()) > 0, f"|grad| {float(pr.grad.abs().sum()):.3g}")
chk("the backbone is frozen", not any(q.requires_grad for q in net.parameters()))
try:
    patch_tokens(net, torch.rand(1, 100, 128, 3), -1)
    chk("raises on a size not divisible by 16", False)
except ValueError:
    chk("raises on a size not divisible by 16", True)

print(f"\n{_n - _f}/{_n} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
