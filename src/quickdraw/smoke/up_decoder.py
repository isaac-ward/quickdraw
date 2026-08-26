"""Smoke: the UP-ONLY image decoder (decode_arch="up", models/decoders.py).

Checks the things that would silently invalidate a run: that the arch is actually BUILT (the old dispatch fell
through to ViT for unknown names), that the readout really carries the full latent (the whole point -- the
U-Net path was a rank-640 choke on 4096 floats), that `x` genuinely carries no information, that chunked
checkpointing stays exact, and that it can overfit a batch at all.
Run: uv run python -m quickdraw.smoke.up_decoder
"""
import torch
import torch.nn.functional as F

from quickdraw.models.decoders import TokenGridDecoder
from quickdraw.models.modalities import ImageModality, ModalitySpec

OK = [0, 0]
def check(name, cond, extra=""):
    OK[1] += 1; OK[0] += bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))

def spec(**kw):
    base = dict(name="image", kind="image", num_tokens=32, img_size=96, encode_arch="conv",
                decode_arch="up", decode_kind="mse", decode_base=32, encode_base=32, ae_bottleneck=6)
    base.update(kw)
    return ModalitySpec(**base)

# --- dispatch ---
m = ImageModality(spec(), d=128)
check("decode_arch='up' builds TokenGridDecoder", isinstance(m.decode_head, TokenGridDecoder),
      type(m.decode_head).__name__)
for bad in ("Up", "upp", "conv", ""):
    try:
        ImageModality(spec(decode_arch=bad), d=128); ok = False
    except ValueError:
        ok = True
    check(f"decode_arch={bad!r} RAISES (no silent ViT fallback)", ok)
try:
    ImageModality(spec(decode_kind="flow"), d=128); ok = False
except ValueError:
    ok = True
check("decode_arch='up' + decode_kind='flow' RAISES", ok)

# --- geometry + params ---
dec = m.decode_head
check("bottleneck grid is 6x6 at img 96 / ae_bottleneck 6", dec.bott_hw == (6, 6), str(dec.bott_hw))
n = lambda mod: sum(p.numel() for p in mod.parameters())
u = ImageModality(spec(decode_arch="unet"), d=128).decode_head
check("up-only decoder is much smaller than the U-Net", n(dec) < n(u),
      f"up {n(dec):,} vs unet {n(u):,} ({n(u)/n(dec):.2f}x)")
print(f"       readout {n(dec.readout):,} | gpool {n(dec.gpool):,} | mid {n(dec.mid):,} | ups {n(dec.ups):,}")
print(f"       readout bandwidth = {dec.bott_hw[0]*dec.bott_hw[1]}x128 = "
      f"{dec.bott_hw[0]*dec.bott_hw[1]*128} vs the U-Net's 512+128=640, on a 32x128={32*128}-float latent")

# --- shapes + the x-carries-nothing property ---
tok = torch.randn(3, 32, 128)
out = dec.velocity(None, None, tok, None)
check("velocity(cond) -> (M,96,96,3)", out.shape == (3, 96, 96, 3), str(tuple(out.shape)))
o2 = dec.velocity(torch.randn(3, 96, 96, 3), dec._temb(torch.ones(3, 1, 1, 1)), tok, None)
check("output is INDEPENDENT of x and temb", torch.equal(out, o2))
check("decode() through the Modality works", m.decode(tok).shape == (3, 96, 96, 3))

# --- the readout must NOT be rank-choked: perturb the latent, pixels must move ---
g = torch.Generator().manual_seed(0)
base_out = dec.velocity(None, None, tok, None)
moved = []
for t in range(32):                      # perturb ONE token at a time
    p = tok.clone(); p[:, t] += torch.randn(3, 128, generator=g) * 0.5
    moved.append(float((dec.velocity(None, None, p, None) - base_out).abs().mean()))
check("EVERY token individually changes the image (no null space)", min(moved) > 1e-6,
      f"min per-token pixel delta {min(moved):.2e}, max {max(moved):.2e}")

# --- chunked checkpointing must be exact ---
dch = TokenGridDecoder(m.ae.cfg, base=32, chunk=2)
dch.load_state_dict(dec.state_dict())
big = torch.randn(6, 32, 128, requires_grad=True)
a = dec.velocity(None, None, big, None)
b = dch._chunked_velocity(None, None, big, None) if False else dch.velocity(None, None, big, None)
check("chunk=2 matches chunk=0 in forward", torch.allclose(a, b, atol=1e-5), f"max {float((a-b).abs().max()):.2e}")

# --- can it learn? ---
torch.manual_seed(0)
d2 = TokenGridDecoder(m.ae.cfg, base=32)
tgt = torch.rand(2, 96, 96, 3)
z = torch.randn(2, 32, 128)
opt = torch.optim.Adam(d2.parameters(), lr=3e-3)
first = None
for i in range(120):
    loss = F.mse_loss(d2.velocity(None, None, z, None), tgt)
    if first is None: first = float(loss)
    opt.zero_grad(); loss.backward(); opt.step()
check("overfits a fixed (tokens -> image) pair", float(loss) < first * 0.2, f"{first:.4f} -> {float(loss):.4f}")

print(f"\n{'ALL OK' if OK[0]==OK[1] else 'FAILURES'} ({OK[0]}/{OK[1]})")
raise SystemExit(0 if OK[0]==OK[1] else 1)
