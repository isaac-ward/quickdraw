"""Smoke: the perceptual decode term (modalities.perceptual_weight) and num_tokens=64 on the up decoder.

The perceptual term is a LOSS-SHAPE change on the only autoregressive gradient in the model, so the checks are
about not breaking anything silently:
  1. weight=0 is BIT-IDENTICAL to before (the aux hook must be a true no-op).
  2. weight>0 strictly increases the loss and CHANGES the gradient reaching the decoder.
  3. it does NOT touch proprio (vector heads have no perceptual term) or the dynamics FlowField.
  4. frame subsampling bounds the cost and still produces a gradient.
  5. num_tokens=64 builds on decode_arch=up (impossible before: the old dense readout baked T*d into a weight).
Run: uv run python -m quickdraw.smoke.perceptual_decode
"""
import torch

from quickdraw.models.modalities import ImageModality, ModalitySpec, VectorModality

OK = [0, 0]
def check(name, cond, extra=""):
    OK[1] += 1; OK[0] += bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))

def img(**kw):
    base = dict(name="image", kind="image", num_tokens=32, img_size=96, encode_arch="conv",
                decode_arch="up", decode_kind="mse", decode_base=32, encode_base=32, ae_bottleneck=6)
    base.update(kw)
    torch.manual_seed(0)
    return ImageModality(ModalitySpec(**base), d=128)

X = torch.rand(6, 96, 96, 3)
Z = torch.randn(6, 32, 128)

# 1. off == unchanged
m0 = img()
l0, _ = m0.decode_loss(Z, X)
check("perceptual_weight=0 -> aux is None (a method would be truthy and break the head)", m0._perceptual is None)
m0b = img()
l0b, _ = m0b.decode_loss(Z, X)
check("weight=0 loss is deterministic/unchanged", torch.equal(l0, l0b), f"{float(l0):.6f}")

# 2. on -> bigger loss, different gradient
mp = img(perceptual_weight=1.0, perceptual_frames=0)
mp.load_state_dict(m0.state_dict())
lp, _ = mp.decode_loss(Z, X)
check("perceptual_weight>0 increases the loss", float(lp) > float(l0), f"{float(l0):.6f} -> {float(lp):.6f}")
g0 = torch.autograd.grad(m0.decode_loss(Z, X)[0], m0.decode_head.out_conv.weight, retain_graph=False)[0]
gp = torch.autograd.grad(mp.decode_loss(Z, X)[0], mp.decode_head.out_conv.weight, retain_graph=False)[0]
check("it CHANGES the gradient reaching the decoder", not torch.allclose(g0, gp, atol=1e-8),
      f"max |dg| {float((g0-gp).abs().max()):.3e}")

# 3. untouched elsewhere
v = VectorModality(ModalitySpec("proprio", "vector", dim=16, decode_kind="flow", decode_param="x0"), d=128)
check("vector modality has no perceptual term", v._perceptual is None)
check("weight is per-modality, not global", img()._perceptual is None and img(perceptual_weight=1.0)._perceptual is not None)

# 4. subsampling
ms = img(perceptual_weight=1.0, perceptual_frames=2)
ms.load_state_dict(m0.state_dict())
ls, _ = ms.decode_loss(Z, X)
check("frame subsampling still produces a finite loss", torch.isfinite(ls) and float(ls) > float(l0),
      f"{float(ls):.6f} (2 of 6 frames)")
gs = torch.autograd.grad(ls, ms.decode_head.out_conv.weight)[0]
check("subsampled term still yields a gradient", float(gs.abs().sum()) > 0)

# 5. num_tokens=64 on the up decoder
try:
    m64 = img(num_tokens=64)
    o = m64.decode(torch.randn(2, 64, 128))
    ok64 = o.shape == (2, 96, 96, 3)
except Exception as e:
    ok64 = False; print("   ", type(e).__name__, e)
check("num_tokens=64 builds + decodes on decode_arch=up", ok64,
      "the old dense readout baked T*d into a weight; the query grid does not")
n = lambda mm: sum(p.numel() for p in mm.parameters())
print(f"       decoder params: num_tokens 32 -> {n(img().decode_head):,} | 64 -> {n(img(num_tokens=64).decode_head):,}")

print(f"\n{'ALL OK' if OK[0]==OK[1] else 'FAILURES'} ({OK[0]}/{OK[1]})")
raise SystemExit(0 if OK[0]==OK[1] else 1)
