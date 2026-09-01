"""One-off: is the reconstruction COLOUR-SHIFTED, and does the loss mix explain it?

LPIPS scores VGG features, which are texture/edge driven and comparatively insensitive to a GLOBAL colour
shift -- a uniformly tinted image has near-identical deep features. The PIXEL term is what pins absolute
colour, and in the L1+LPIPS recipe the pixel term is only ~24% of the loss (measured: L1 0.0587 vs
LPIPS-vgg 0.1835). So the prediction is that a perceptually-trained codec drifts in colour more than an
MSE-trained one, visible as a per-channel SIGNED mean error rather than as a magnitude error.

Reports, per RGB channel, on real val frames through the real encode->decode path:
  bias      mean(pred - target)         a systematic tint; SIGN matters, this is the cast
  mae       mean|pred - target|         magnitude, for scale
  sat       mean per-pixel max-min      colour saturation of pred vs target (washed out / oversaturated)

Run: uv run python -m quickdraw._oneoff_colour_cast <ckpt> [n_frames]
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.training.setup import build_model, resolve_data_root

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RUN = os.path.dirname(os.path.dirname(CKPT))
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
dev = torch.device("cuda")

model = build_model(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
m = getattr(model, "_orig_mod", model)
name = next((n for n, _ in m.layout if n != "proprio"), None)
mod = m.modalities[name]
mix = (float(getattr(mod.visual, "w_l1", 0)), float(getattr(mod.visual, "w_l2", 0)),
       float(getattr(mod.visual, "w_lpips", 0))) if hasattr(mod, "visual") else ("?", "?", "?")

ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=mod.ae.cfg.img_size,
                            cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
rng = np.random.default_rng(0)
fr = []
for i in range(len(ds)):
    _, _, im = ds[i]
    fr.append(torch.from_numpy(im[rng.permutation(len(im))[:max(1, N // len(ds) + 1)]]).float().div(255.0))
X = torch.cat(fr)[:N].to(dev)

with torch.no_grad():
    rec = mod.decode_head.velocity(cond=mod.encode(X)).clamp(0, 1)

print(f"\n{os.path.basename(RUN)}   mix (l1,l2,lpips) = {mix}   n={X.shape[0]} val frames")
print(f"{'ch':>4s} {'bias (pred-tgt)':>16s} {'mae':>9s}")
for c, nm in enumerate("RGB"):
    b = float((rec[..., c] - X[..., c]).mean())
    a = float((rec[..., c] - X[..., c]).abs().mean())
    print(f"{nm:>4s} {b:+16.5f} {a:9.5f}")
sp = float((rec.max(-1).values - rec.min(-1).values).mean())
st = float((X.max(-1).values - X.min(-1).values).mean())
print(f"  saturation  pred {sp:.5f}  target {st:.5f}   ratio {sp/max(st,1e-9):.3f}"
      f"   ({'WASHED OUT' if sp < st*0.95 else 'oversaturated' if sp > st*1.05 else 'matched'})")
# a global tint is the SPREAD of the per-channel biases, not their magnitude
bs = [float((rec[..., c] - X[..., c]).mean()) for c in range(3)]
print(f"  channel-bias spread (max-min) = {max(bs)-min(bs):.5f}   <- this is the TINT")
