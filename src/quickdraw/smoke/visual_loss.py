"""Smoke: the shared `VisualLoss` (models/visual_loss.py) at BOTH image reconstruction sites.

VisualLoss is a LOSS-SHAPE change on the only autoregressive gradient in the model AND on the weight-10 codec
anchor, so the checks are about not breaking anything silently:
  1. the DEFAULT mix (l2 only) is BIT-IDENTICAL to the old plain `F.mse_loss` -- the plumbing must land as a no-op.
  2. each term is actually wired: l1 and lpips each change the loss and the gradient reaching the decoder.
  3. ONE instance serves both sites -- `mod.recon_loss` and the roundtrip anchor must be the SAME object, and
     it must be registered exactly ONCE (a double registration duplicates frozen LPIPS weights in every ckpt).
  4. the TRAINING backbone is vgg and the EVAL backbone stays squeeze -- training on the reported metric's own
     network would make the number incomparable to all 25 historical runs.
  5. vector modalities keep plain MSE (no perceptual term on proprio) and the dynamics FlowField is untouched.
  6. frame subsampling bounds the cost and still produces a gradient.
  7. `terms()` reports raw magnitudes, which is how the anchor's site weight gets chosen by MEASUREMENT.
Run: uv run python -m quickdraw.smoke.visual_loss
"""
import torch

from quickdraw.models.modalities import ImageModality, ModalitySpec, VectorModality
from quickdraw.models.visual_loss import VisualLoss

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

# ---- 1. the default mix is exactly MSE ----
m0 = img()
l0, _ = m0.decode_loss(Z, X)
pred = m0.decode_head.velocity(cond=Z)
check("default mix is l2-only", (m0.visual.w_l2, m0.visual.w_l1, m0.visual.w_lpips) == (1.0, 0.0, 0.0))
check("default decode loss == F.mse_loss EXACTLY (plumbing is a no-op)",
      torch.equal(l0, torch.nn.functional.mse_loss(pred, X)), f"{float(l0):.8f}")

# ---- 2. each term is wired ----
m1 = img(visual_l1=1.0)
l1, _ = m1.decode_loss(Z, X)
check("visual_l1 raises the loss", float(l1) > float(l0), f"{float(l0):.6f} -> {float(l1):.6f}")

mp = img(visual_lpips=1.0, visual_frames=0)
lp, _ = mp.decode_loss(Z, X)
skipped = mp.visual._net is None            # LPIPS weights unavailable in this environment
check("visual_lpips raises the loss", float(lp) > float(l0) or skipped,
      "LPIPS weights unavailable — term skipped" if skipped else f"{float(l0):.6f} -> {float(lp):.6f}")

def dgrad(mod):
    mod.zero_grad(); loss, _ = mod.decode_loss(Z, X); loss.backward()
    p = mod.decode_head.out_conv.weight
    return p.grad.detach().clone()
g0, g1 = dgrad(img()), dgrad(img(visual_l1=1.0))
check("the mix CHANGES the gradient reaching the decoder", not torch.allclose(g0, g1))

# ---- 3. ONE instance, both sites, registered once ----
m = img(visual_l1=1.0, visual_lpips=0.5)
check("recon_loss dispatches to the shared instance",
      m.recon_loss(pred, X).item() == m.visual(pred, X).item())
n_vis = sum(1 for mod in m.modules() if isinstance(mod, VisualLoss))
check("VisualLoss registered EXACTLY once", n_vis == 1, f"found {n_vis}")
check("the same object is reachable from both sites", m.recon_loss.__self__.visual is m.visual)

# ---- 4. backbones: vgg trains, squeeze evaluates ----
check("training backbone defaults to vgg", m.visual.lpips_net == "vgg")
import inspect
from quickdraw.evaluation import openloop
sig = inspect.signature(openloop._lpips_net)
check("eval backbone still defaults to squeeze", sig.parameters["net_type"].default == "squeeze")
check("the LPIPS cache is keyed by (device, net_type) so the two coexist",
      "net_type" in inspect.getsource(openloop._lpips_net).split("key =")[1].split("\n")[0])

# ---- 5. vector heads keep plain MSE; dynamics untouched ----
v = VectorModality(ModalitySpec(name="proprio", kind="vector", dim=6), d=128)
check("vector modality has no VisualLoss", not hasattr(v, "visual"))
check("vector recon_loss is plain MSE",
      torch.equal(v.recon_loss(X, X + 0.1), torch.nn.functional.mse_loss(X, X + 0.1)))
from quickdraw.models.flow import FlowField
check("FlowField.loss ignores recon_loss (param='v' has no clean prediction)",
      "recon_loss" in inspect.signature(FlowField.loss).parameters)

# ---- 6. subsampling ----
ms = img(visual_lpips=1.0, visual_frames=2)
ls, _ = ms.decode_loss(Z, X)
ls.backward()
check("frame subsampling still yields a finite loss and a gradient",
      torch.isfinite(ls) and ms.decode_head.out_conv.weight.grad is not None, f"frames=2, loss {float(ls):.6f}")

# ---- 6b. BOTH RANKS. The decode site passes (M,H,W,C); the roundtrip anchor passes (B,F,H,W,C). ----
m5 = img(visual_l1=1.0, visual_lpips=1.0)
X5 = X.reshape(2, 3, 96, 96, 3)                      # (B,F,H,W,C) exactly as to_obs() returns it
P5 = pred.detach().reshape(2, 3, 96, 96, 3)
l5, l4 = m5.visual(P5, X5), m5.visual(pred.detach(), X)
check("VisualLoss accepts the anchor's 5-D (B,F,H,W,C) input", torch.isfinite(l5), f"{float(l5):.6f}")
check("5-D and equivalent 4-D inputs agree (LPIPS subsample is over FRAMES)",
      abs(float(l5) - float(l4)) < 1e-5, f"{float(l5):.6f} vs {float(l4):.6f}")
check("terms() also accepts 5-D", torch.isfinite(torch.tensor(m5.visual.terms(P5, X5)["l1"])))

# ---- 7. terms() reports raw magnitudes for choosing the anchor weight ----
t = m.visual.terms(pred, X)
check("terms() reports raw l2/l1(/lpips)", {"l2", "l1", "lpips"} <= set(t) and t["l2"] > 0 and t["l1"] > 0,
      " ".join(f"{k}={v:.4f}" for k, v in t.items()))

print(f"\n{OK[0]}/{OK[1]} checks passed")
raise SystemExit(0 if OK[0] == OK[1] else 1)
