"""Smoke: the two DECODER CONDITIONING features on decode_arch=up.

  FEATURE 2  `decode_inject`         -- resample the bottleneck readout into every up level (spatial signal)
  FEATURE 3  `decode_xattn_max_res`  -- re-cross-attend the token bag at low-resolution up levels

Both exist because above the 6x6 bottleneck the ONLY latent signal reaching the conv trunk is `g`, one pooled
d-vector applied as a PER-CHANNEL FiLM. Both are zero-init on their output path, which is the property the
whole A/B design rests on: enabling either must be BIT-IDENTICAL at step 0, so any measured difference is
learned rather than an initialization shift.

Run: uv run python -m quickdraw.smoke.decoder_cond
"""
import torch

from quickdraw.models.decoders import LevelCrossAttn, TokenGridDecoder
from quickdraw.models.modalities import ImageModality, ModalitySpec

OK = [0, 0]
def check(name, cond, extra=""):
    OK[1] += 1; OK[0] += bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))

def img(**kw):
    base = dict(name="image", kind="image", num_tokens=32, img_size=96, encode_arch="conv",
                decode_arch="up", decode_kind="mse", decode_base=64, encode_base=32, ae_bottleneck=6)
    base.update(kw)
    torch.manual_seed(0)
    return ImageModality(ModalitySpec(**base), d=128)

def npar(m):
    return sum(p.numel() for p in m.decode_head.parameters())

X = torch.rand(4, 96, 96, 3)
Z = torch.randn(4, 32, 128)

base, inj, xa, both = img(), img(decode_inject=True), img(decode_xattn_max_res=24), \
    img(decode_inject=True, decode_xattn_max_res=24)

# ---- off by default ----
check("both features OFF by default", base.decode_head.inject is None and len(base.decode_head.xattn) == 0)

# ---- ZERO-INIT: identical output at step 0 ----
# Constructing extra modules CONSUMES RNG, so a variant built under the same seed gets different weights in
# every LATER layer. That is an artifact of the test, not of the feature. Copy the base weights in (strict=
# False leaves the new zero-init modules untouched) so this isolates the feature itself.
for m in (inj, xa, both):
    m.decode_head.load_state_dict(base.decode_head.state_dict(), strict=False)
with torch.no_grad():
    y0, yi, yx, yb = (m.decode_head.velocity(cond=Z) for m in (base, inj, xa, both))
check("decode_inject is BIT-IDENTICAL at init (zero-init conv)", torch.equal(y0, yi))
check("decode_xattn is BIT-IDENTICAL at init (zero-init out conv)", torch.equal(y0, yx))
check("both together are BIT-IDENTICAL at init", torch.equal(y0, yb))

# ---- but the parameters exist and are reachable by grad ----
check("decode_inject adds one conv per up level", len(inj.decode_head.inject) == len(inj.decode_head.ups),
      f"{len(inj.decode_head.inject)} convs / {len(inj.decode_head.ups)} levels")
check("decode_xattn_max_res=24 fires at exactly 2 levels (12, 24) on a 6x6 bottleneck",
      len(xa.decode_head.xattn) == 2, f"levels {sorted(xa.decode_head.xattn)}")
check("decode_xattn_max_res=0 fires nowhere", len(base.decode_head.xattn) == 0)
check("decode_xattn_max_res=96 fires at all 4 levels", len(img(decode_xattn_max_res=96).decode_head.xattn) == 4)

p0, pi, px = npar(base), npar(inj), npar(xa)
check("decode_inject param cost is ~116k at base 64 (d -> each block's INPUT width)",
      105_000 < pi - p0 < 130_000, f"+{pi - p0:,}")
check("decode_xattn(<=24) param cost is ~2x132k", 200_000 < px - p0 < 320_000, f"+{px - p0:,}")
print(f"       base {p0:,} | +inject {pi:,} | +xattn {px:,} | both {npar(both):,}")

# ---- gradients actually reach the new modules ----
for tag, m in (("inject", inj), ("xattn", xa)):
    m.zero_grad(); loss, _ = m.decode_loss(Z, X); loss.backward()
    mods = m.decode_head.inject if tag == "inject" else list(m.decode_head.xattn.values())
    gs = [p.grad for mm in mods for p in mm.parameters() if p.grad is not None]
    check(f"gradient reaches every {tag} module", len(gs) > 0 and all(torch.isfinite(g).all() for g in gs)
          and any(g.abs().sum() > 0 for g in gs), f"{len(gs)} grad tensors")

# ---- the features CHANGE the function once their weights are non-zero ----
with torch.no_grad():
    for cv in inj.decode_head.inject:
        cv.weight.normal_(0, 0.05)
    y_after = inj.decode_head.velocity(cond=Z)
check("decode_inject changes the output once trained away from zero", not torch.allclose(y0, y_after))

with torch.no_grad():
    for m_ in xa.decode_head.xattn.values():
        m_.to_ch.weight.normal_(0, 0.05)
    y_after_x = xa.decode_head.velocity(cond=Z)
check("decode_xattn changes the output once trained away from zero", not torch.allclose(y0, y_after_x))

# ---- the xattn module really reads the BAG (not just the feature map) ----
lc = LevelCrossAttn(64, 128, 8)
torch.nn.init.normal_(lc.to_ch.weight, 0, 0.1)
h = torch.randn(2, 64, 12, 12)
c1, c2 = torch.randn(2, 32, 128), torch.randn(2, 32, 128)
check("LevelCrossAttn output depends on the token bag", not torch.allclose(lc(h, c1), lc(h, c2)))
check("LevelCrossAttn preserves shape", lc(h, c1).shape == h.shape)

# ---- shapes still right end to end ----
check("output shape unchanged with both features on", yb.shape == (4, 96, 96, 3), str(tuple(yb.shape)))

print(f"\n{OK[0]}/{OK[1]} checks passed")
raise SystemExit(0 if OK[0] == OK[1] else 1)
