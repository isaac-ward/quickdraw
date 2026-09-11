"""`VisualLoss.temporal` — the first-order term (design/derivative_loss.md).

The third check is the load-bearing one: a CONSTANT-OFFSET sequence must score ~0. That is what proves the
term measures MOTION rather than POSITION, and it is the property the per-frame terms cannot provide.

    python -m quickdraw.smoke.derivative_loss
"""
import os
import pathlib
import sys

# The repo's conf/, resolved from THIS file rather than hardcoded. It was "/app/conf", the path
# inside the Docker image: outside the container hydra raises MissingConfigException, the 15
# config-dependent checks (byte-identicality, the eligibility guard, the training step) never run,
# and the script exits 1 having printed only PASS lines -- which reads as a pass.
_CONF = str(pathlib.Path(__file__).resolve().parents[3] / "conf")

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
    with initialize_config_dir(config_dir=_CONF, version_base=None):
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


# ---- FRAME ALIGNMENT: the silent-bug class ------------------------------------------------------
# If decode_loss's `reshape(-1, ...)` and `view(*lead, ...)` disagreed about layout, the term would
# difference the WRONG frames -- a different episode, or a shuffled time axis -- and every check above
# would still pass, because they all feed (B,F,...) directly. This is the one place the plumbing could be
# wrong with nothing noticing.
_B, _F, _H, _W, _C = 3, 5, 4, 4, 3
_t = torch.zeros(_B, _F, _H, _W, _C)
for _b in range(_B):
    for _i in range(_F):
        _t[_b, _i] = _b * 100 + _i                       # every frame a DISTINCT constant
_flat = _t.reshape(-1, _H, _W, _C)                       # what decode_loss hands the head
_back = _flat.view(_B, _F, _H, _W, _C)                   # what decode_loss hands derivative_loss
chk("flatten -> view is the identity", torch.equal(_back, _t))
chk("row b*F+t really IS frame [b, t]",
    all(torch.equal(_flat[b * _F + i], _t[b, i]) for b in range(_B) for i in range(_F)))
chk("temporal difference is exactly 1.0 everywhere (no seam, no shuffle)",
    bool(((_back[:, 1:] - _back[:, :-1]) == 1.0).all()))
_df = _flat[1:] - _flat[:-1]
chk("  (differencing FLAT rows WOULD fabricate seams -- confirming the hazard is real)",
    int((_df != 1.0).any(dim=(1, 2, 3)).sum()) == _B - 1)

# ---- A TRAINING STEP, minus Lightning and the GPU -----------------------------------------------
_opt_cfg = BASE + ["model.modalities.0.decode_kind=mse",      # eligibility, see the guard check above
                   "+model.modalities.0.derivative_weight=0.5",
                   "+model.modalities.1.derivative_weight=0.5"]
with initialize_config_dir(config_dir=_CONF, version_base=None):
    _c = compose(config_name="config", overrides=_opt_cfg)
_m = build_model(_c)
_opt = torch.optim.AdamW(_m.parameters(), lr=1e-4)
torch.manual_seed(0)
_o = {"proprio": torch.randn(2, 6, _c.model.modalities[0].dim),
      "image": torch.rand(2, 6, 112, 192, 3)}
_before = {k: v.detach().clone() for k, v in _m.named_parameters() if v.requires_grad}
_rec, _w = _m.recon_losses(_m.encode_state(_o), _o)
_loss = sum(_w[k] * _rec[k] for k in _rec)
chk("total loss with the term ON is finite", torch.isfinite(_loss).item(), f"{float(_loss):.4f}")
_loss.backward()
chk("gradients reach parameters and are all finite",
    all(torch.isfinite(v.grad).all() for v in _m.parameters() if v.grad is not None))
chk("  ...including the DECODER and the ENCODER",
    any("decode_head" in k and v.grad is not None and float(v.grad.abs().sum()) > 0
        for k, v in _m.named_parameters())
    and any(".ae" in k and v.grad is not None and float(v.grad.abs().sum()) > 0
            for k, v in _m.named_parameters()))
_opt.step()
chk("optimizer.step() moves parameters",
    sum(1 for k, v in _m.named_parameters()
        if v.requires_grad and not torch.equal(v.detach(), _before[k])) > 0)
with torch.autocast("cpu", dtype=torch.bfloat16):        # the real run is precision=bf16-mixed
    _r2, _w2 = _m.recon_losses(_m.encode_state(_o), _o)
    _l2 = sum(_w2[k] * _r2[k] for k in _r2)
chk("bf16 autocast: the term computes and stays finite",
    torch.isfinite(_l2).item() and torch.isfinite(_r2["derivative/image"]).item(),
    f"{float(_r2['derivative/image']):.4f}")
for _mod in _m.modalities.values():
    _mod.derivative_weight = 0.0
_r0, _w0 = _m.recon_losses(_m.encode_state(_o), _o)
chk("turning the weight off changes the total (the term is load-bearing)",
    abs(float(sum(_w0[k] * _r0[k] for k in _r0)) - float(_loss)) > 1e-6)

print(f"\n{ok} passed, {bad} failed")
sys.exit(1 if bad else 0)
