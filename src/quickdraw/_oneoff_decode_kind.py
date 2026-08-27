"""Price the mse -> flow decode swap BEFORE spending a pair on it. Two questions, one experiment:

  Q1 (LOSS SCALE)  flow-x0's loss averages over tau, including near-clean inputs the net can almost copy,
                   so its RAW value is systematically smaller than mse's (which is always the tau=1 case).
                   lit.py sums `w * raw` with flat weights, so swapping decode_kind silently REWEIGHTS the
                   only autoregressive gradient in the model -- the exact lever recon_frac=1.0 exists to
                   strengthen. We need the ratio to set model.modalities.1.weight instead of guessing.
  Q2 (SHARPNESS)   does a SAMPLED decode actually render sharper than the conditional mean? That is the whole
                   reason to want a generative decoder, and it has never been measured on this dataset.

Method: freeze the trained encoder from a checkpoint (so both heads see IDENTICAL real latents), then train
two fresh decode heads from the same seed on the same (latent -> frame) pairs for the same number of steps.
Compare raw losses, and compare committed-decode vs sampled-decode PSNR/LPIPS against the SAME frames.

Usage: uv run python -m quickdraw._oneoff_decode_kind <ckpt> <steps>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.evaluation.openloop import _lpips_net
from quickdraw.models.modalities import ImageModality, ModalitySpec
from quickdraw.training.setup import build_model, resolve_data_root

CKPT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
RUN = os.path.dirname(os.path.dirname(CKPT))
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
dev = torch.device("cuda")

model = build_model(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
m = getattr(model, "_orig_mod", model)
img_head = next((n for n, _ in m.layout if n != "proprio"), None)
mod = m.modalities[img_head]
spec_d = {e["name"]: dict(e) for e in OmegaConf.to_container(cfg.model.modalities)}[img_head]
img_size = mod.ae.cfg.img_size
ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                            cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
rng = np.random.default_rng(0)
frames = []
for _ in range(24):
    _, _, im = ds[int(rng.integers(len(ds)))]
    idx = rng.integers(0, len(im), size=8)
    frames.append(torch.from_numpy(im[idx]).float().div(255.0))
X = torch.cat(frames).to(dev)                                  # (N,H,W,3) real val frames
with torch.no_grad():                                          # FROZEN real latents -- identical for both heads
    Z = mod.encode(X).float()
print(f"[data] {X.shape[0]} frames {tuple(X.shape[1:])}  latents {tuple(Z.shape[1:])}", flush=True)

lp = _lpips_net(dev)
def quality(head_mod, stoch):
    head_mod.decode_stochastic = stoch
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 32):
            outs.append(head_mod.decode(Z[i:i + 32]).clamp(0, 1).float())
    P = torch.cat(outs)
    mse = (P - X).pow(2).mean(dim=(1, 2, 3))
    psnr = float((10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean())
    # _lpips_net builds the metric with normalize=True -> it wants [0,1], NOT [-1,1] (openloop.py:73)
    p, x = P.permute(0, 3, 1, 2).clamp(0, 1), X.permute(0, 3, 1, 2).clamp(0, 1)
    with torch.no_grad():
        lpv = float(torch.cat([lp(p[i:i + 32], x[i:i + 32]).flatten() for i in range(0, p.shape[0], 32)]).mean())
    return psnr, lpv

res = {}
for kind in ("mse", "flow"):
    sd_spec = dict(spec_d); sd_spec.update(decode_kind=kind, decode_param="x0", decode_steps=6,
                                           decode_stochastic=False)
    if not isinstance(sd_spec.get("img_size", 128), int):
        sd_spec["img_size"] = tuple(int(s) for s in sd_spec["img_size"])
    torch.manual_seed(0)
    h = ImageModality(ModalitySpec(**sd_spec), d=int(cfg.model.d)).to(dev)
    h.ae.load_state_dict(mod.ae.state_dict())                  # SAME frozen encoder
    for p_ in h.ae.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.Adam([p_ for p_ in h.decode_head.parameters()], lr=3e-4)
    hist = []
    every = max(1, STEPS // 5)
    for it in range(STEPS):
        i = torch.randint(0, Z.shape[0], (16,), device=dev)
        loss, _ = h.decode_loss(Z[i], X[i])
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.detach()))
        # TRAJECTORY, not just the endpoint: the flow head has the harder objective (denoise at EVERY tau,
        # not just tau=1), so it converges slower, and a single short-budget snapshot cannot tell "worse"
        # from "not there yet". Print the curve so the two can be distinguished.
        if (it + 1) % every == 0:
            q = quality(h, False)
            extra = quality(h, True) if kind == "flow" else None
            print(f"    [{kind:4} @{it+1:5}] raw {np.mean(hist[-50:]):.6f}  commit PSNR {q[0]:.2f} "
                  f"LPIPS {q[1]:.4f}" + (f"  | sampled PSNR {extra[0]:.2f} LPIPS {extra[1]:.4f}" if extra else ""),
                  flush=True)
            h.decode_stochastic = False
    raw = float(np.mean(hist[-50:]))
    det = quality(h, False)
    row = {"raw_loss": raw, "psnr_commit": det[0], "lpips_commit": det[1]}
    if kind == "flow":
        st = quality(h, True)
        row.update(psnr_sampled=st[0], lpips_sampled=st[1])
    res[kind] = row
    print(f"[{kind:4}] raw decode loss (last 50) {raw:.6f} | committed PSNR {det[0]:.2f} LPIPS {det[1]:.4f}"
          + (f" | SAMPLED PSNR {st[0]:.2f} LPIPS {st[1]:.4f}" if kind == "flow" else ""), flush=True)
    del h; torch.cuda.empty_cache()

r = res["mse"]["raw_loss"] / max(res["flow"]["raw_loss"], 1e-12)
print(f"\nQ1 LOSS SCALE  mse/flow raw ratio = {r:.3f}  ->  to keep the decode gradient at its current "
      f"strength, model.modalities.1.weight = {r:.2f} (from 1.0)")
print(f"Q2 SHARPNESS   sampled vs committed:  LPIPS {res['flow']['lpips_sampled']:.4f} vs "
      f"{res['flow']['lpips_commit']:.4f}   PSNR {res['flow']['psnr_sampled']:.2f} vs {res['flow']['psnr_commit']:.2f}")
print(f"               mse baseline:          LPIPS {res['mse']['lpips_commit']:.4f}   PSNR {res['mse']['psnr_commit']:.2f}")
