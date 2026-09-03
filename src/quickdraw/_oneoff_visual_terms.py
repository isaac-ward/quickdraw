"""One-off: the RAW magnitude of each VisualLoss term on a CONVERGED codec, to set the anchor's site weight.

The roundtrip anchor's `latent_loss_weight=10` was calibrated when its loss was `F.mse_loss`. Adopting the
literature mix (L1 1.0 + LPIPS 1.0) changes the anchor's MAGNITUDE, not just its shape: on [0,1] images MSE is
~0.01 while L1 is ~0.05 and LPIPS ~0.2, so reusing 10 silently scales the anchor by more than an order of
magnitude against an unchanged dynamics loss -- testing loss shape and a large reweighting at once, which is
exactly the confound this record keeps getting burned by.

Solve for the weight that PRESERVES the anchor's current contribution:
    w = 10 * mse / (w_l2*mse + w_l1*l1 + w_lpips*lpips_vgg)

Every number here is MEASURED on the real encode->decode path of a trained checkpoint. LPIPS-vgg in particular
is not logged anywhere (eval reports squeeze), and it is the term that decides the answer.

Run: uv run python -m quickdraw._oneoff_visual_terms <ckpt>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.evaluation.openloop import _lpips_net
from quickdraw.training.setup import build_model, normalizer, resolve_data_root

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/train_world_model_2026_08_27_17_06_28_dyn512/checkpoints/best.ckpt"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 256
RUN = os.path.dirname(os.path.dirname(CKPT))
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
dev = torch.device("cuda")

model = build_model(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
miss, unex = model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
m = getattr(model, "_orig_mod", model)
name = next((n for n, _ in m.layout if n != "proprio"), None)
mod = m.modalities[name]
ae_cfg = mod.ae.cfg
print(f"[load] {os.path.basename(RUN)} missing={len(miss)} unexpected={len(unex)} | "
      f"img_size={ae_cfg.img_size} bottleneck={getattr(ae_cfg,'bottleneck',8)} "
      f"num_tokens={ae_cfg.num_tokens} decode_arch={mod.decode_arch}", flush=True)

# frames come back as a DICT keyed by camera (data/dataset.py). Single-camera script:

# name the key once rather than indexing position 2 as if there were only ever one view.

_CAMK = str(cfg.data.get("cam", "fpv"))

ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=ae_cfg.img_size,
                            cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
rng = np.random.default_rng(0)
frames = []
for i in range(len(ds)):
    _, _, _fr = ds[i]
    im = _fr[_CAMK]
    idx = rng.permutation(len(im))[:max(1, N_FRAMES // max(len(ds), 1) + 1)]
    frames.append(torch.from_numpy(im[idx]).float().div(255.0))
X = torch.cat(frames)[:N_FRAMES].to(dev)
print(f"[data] {X.shape[0]} held-out val frames {tuple(X.shape[1:])}")

# ---- the REAL codec path: encode -> decode, exactly as the roundtrip anchor does ----
with torch.no_grad():
    tok = mod.encode(X) if hasattr(mod, "encode") else mod._encode(X)
    rec = mod.decode_head.velocity(cond=tok).clamp(0, 1)
mse, l1 = float(Fn.mse_loss(rec, X)), float(Fn.l1_loss(rec, X))
print(f"\nconverged codec, measured on the real encode->decode path:")
print(f"  L2 (mse)          {mse:.6f}   (PSNR {-10*np.log10(max(mse,1e-12)):.2f} dB)")
print(f"  L1                {l1:.6f}")
lp = {}
for net_type in ("squeeze", "vgg"):
    net = _lpips_net(dev, net_type=net_type)
    if net is None:
        print(f"  LPIPS {net_type}: UNAVAILABLE"); continue
    with torch.no_grad():
        vals = [float(net(rec[i:i+32].permute(0, 3, 1, 2), X[i:i+32].permute(0, 3, 1, 2)))
                for i in range(0, rec.shape[0], 32)]
    net.reset()
    lp[net_type] = float(np.mean(vals))
    tag = "   <- the REPORTED metric" if net_type == "squeeze" else "   <- the TRAINING loss"
    print(f"  LPIPS {net_type:8s}  {lp[net_type]:.6f}{tag}")

if "vgg" in lp:
    print(f"\n  vgg/squeeze = {lp['vgg']/max(lp.get('squeeze',1e-9),1e-9):.3f}")
    print("\nanchor site weight that PRESERVES the current contribution (10 * mse = "
          f"{10*mse:.4f}):")
    for wl2, wl1, wlp, label in ((0.0, 1.0, 1.0, "L1 + LPIPS      (IRIS/VQGAN/LDM)"),
                                 (0.0, 1.0, 0.5, "L1 + 0.5*LPIPS"),
                                 (1.0, 0.0, 1.0, "L2 + LPIPS      (ViTok stage 1)"),
                                 (1.0, 0.0, 0.5, "L2 + 0.5*LPIPS  (the arm as originally built)")):
        mix = wl2 * mse + wl1 * l1 + wlp * lp["vgg"]
        print(f"  {label:44s} mix={mix:.4f}  ->  latent_loss_weight = {10*mse/mix:.3f}")
