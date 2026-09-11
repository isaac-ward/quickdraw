"""`VisualLoss.temporal` — the first-order term (design/derivative_loss.md).

The third check is the load-bearing one: a CONSTANT-OFFSET sequence must score ~0. That is what proves the
term measures MOTION rather than POSITION, and it is the property the per-frame terms cannot provide.

    python -m quickdraw.smoke.derivative_loss
"""
import sys

import torch
import torch.nn.functional as F
from ..models.visual_loss import VisualLoss

ok = bad = 0
def chk(n, c, extra=""):
    global ok, bad; ok, bad = ok + bool(c), bad + (not c)
    print(f"  {'PASS' if c else 'FAIL'}  {n}{('  ' + extra) if extra else ''}")

torch.manual_seed(0)
B, Fr, H, W = 2, 6, 32, 32
g = torch.rand(B, Fr, H, W, 3)

pix = VisualLoss(w_l2=0.0, w_l1=3.0, w_lpips=0.0)
chk("identical sequences -> EXACTLY 0", float(pix.temporal(g, g)) == 0.0, f"{float(pix.temporal(g,g)):.3e}")

shifted = torch.cat([g[:, 1:], g[:, -1:]], 1)
chk("one-step-shifted -> > 0", float(pix.temporal(shifted, g)) > 0, f"{float(pix.temporal(shifted,g)):.5f}")

off = g + 0.137                                      # CONSTANT offset: wrong POSITION, correct MOTION
chk("CONSTANT OFFSET -> ~0  (measures MOTION, not POSITION)",
    float(pix.temporal(off, g)) < 1e-6, f"{float(pix.temporal(off,g)):.3e}")
chk("  ...while the per-frame term sees it clearly", float(pix(off.reshape(-1,H,W,3), g.reshape(-1,H,W,3))) > 0.3,
    f"{float(pix(off.reshape(-1,H,W,3), g.reshape(-1,H,W,3))):.4f}")

p = torch.rand(B, Fr, H, W, 3)
hand = 3.0 * F.l1_loss(p[:, 1:] - p[:, :-1], g[:, 1:] - g[:, :-1])
chk("strides=(1,) == the hand-written two-liner", torch.allclose(pix.temporal(p, g), hand, atol=0, rtol=0),
    f"{float(pix.temporal(p,g)):.8f} vs {float(hand):.8f}")

# episode seams: (B=2,F=4) must give 3 differences per episode, never 7 from flat-row differencing
q = torch.zeros(2, 4, 4, 4, 3); q[1] = 1.0                     # a HUGE jump between episodes
chk("episode seam is impossible (no spurious jump)", float(pix.temporal(q, q)) == 0.0,
    f"{float(pix.temporal(q,q)):.3e}")

feat = VisualLoss(w_l2=0.0, w_l1=0.0, w_lpips=1.0, frames=8)
torch.manual_seed(1); a = float(feat.temporal(g, g))
chk("FEATURE part: identical sequences -> ~0", a < 1e-8, f"{a:.3e}")
torch.manual_seed(1); b = float(feat.temporal(shifted, g))
chk("FEATURE part: shifted -> > 0", b > 0, f"{b:.6f}")
torch.manual_seed(1); c = float(feat.temporal(off, g))
chk("FEATURE part: constant offset -> smaller than shifted", c < b, f"offset {c:.6f} < shifted {b:.6f}")

# pred must DIFFER from target: with pred == target the L1 is exactly 0 and |x|'s subgradient at 0 is 0,
# so a vanishing gradient there is correct, not a bug. (This is how the check first mis-fired.)
pg = torch.rand(B, Fr, H, W, 3).requires_grad_(True)
VisualLoss(w_l2=0.0, w_l1=3.0, w_lpips=1.0, frames=8).temporal(pg, g).backward()
chk("gradient flows to pred", pg.grad is not None and float(pg.grad.abs().sum()) > 0,
    f"|grad| sum {float(pg.grad.abs().sum()):.2f}")

try:
    pix.temporal(g.reshape(-1, H, W, 3), g.reshape(-1, H, W, 3))
    chk("raises if the time axis was flattened away", False)
except AssertionError:
    chk("raises if the time axis was flattened away", True)

# ---- the WIRING: dispatch is polymorphic, defaults are inert, noised heads raise -------------------
import json

from hydra import initialize_config_dir, compose

from ..training.setup import build_model

def _losses(ov, seed=0):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        c = compose(config_name="config", overrides=ov)
    torch.manual_seed(seed); m = build_model(c).eval()
    torch.manual_seed(seed + 1)
    b, f = 2, 5
    im = [x for x in c.model.modalities if x.kind == "image"][0]
    h, w_ = (im.img_size, im.img_size) if isinstance(im.img_size, int) else tuple(im.img_size)
    o = {"proprio": torch.randn(b, f, c.model.modalities[0].dim)}
    for x in c.model.modalities:
        if x.kind == "image":
            o[x.name] = torch.rand(b, f, h, w_, 3)
    rec, wt = m.recon_losses(m.encode_state(o), o)
    return {k: (float(v), float(wt[k])) for k, v in sorted(rec.items())}

BASE = ["model=vl128_starling", "data=starling", "environments=recorded"]
d0 = _losses(BASE)
chk("default config: NO derivative term at all (feature is inert)",
    not any("derivative" in k for k in d0))
try:
    _losses(BASE + ["+model.modalities.0.derivative_weight=0.5"])
    chk("noised head + weight RAISES", False)
except ValueError as e:
    chk("noised head + weight RAISES, message names decode_kind and the 62%",
        "decode_kind='mse'" in str(e) and "62%" in str(e))
d1 = _losses(BASE + ["model.modalities.0.decode_kind=mse",
                     "+model.modalities.0.derivative_weight=0.5",
                     "+model.modalities.1.derivative_weight=0.5"])
chk("derivative/proprio AND derivative/image both appear, one code path",
    "derivative/proprio" in d1 and "derivative/image" in d1,
    f"proprio {d1.get('derivative/proprio',(0,))[0]:.4f} image {d1.get('derivative/image',(0,))[0]:.4f}")
chk("  weights travel with the losses",
    d1["derivative/proprio"][1] == 0.5 and d1["derivative/image"][1] == 0.5)
d2 = _losses(BASE + ["+model.modalities.1.derivative_weight=0.5"])
chk("the guard is PER-MODALITY (image on, proprio still noised, no raise)",
    "derivative/image" in d2 and "derivative/proprio" not in d2)

print(f"\n{ok} passed, {bad} failed")
sys.exit(1 if bad else 0)
