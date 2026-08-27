"""Is the trained encoder's latent EFFECTIVE RANK capped by the decoder's readout?

THE QUESTION. The mse decoder reads the 4,096-float token bag through only two routes:
`cond_to_spatial = Linear(4096 -> 512)` and `g = cond.mean(1)` (rank <= 128) -- a rank-<=640 cut. Both pixel
losses (the decode loss and the roundtrip anchor) backprop THROUGH that cut, so the encoder receives NO
reconstruction gradient in the ~3,456 orthogonal directions. If the encoder therefore never learned to use
them, its latent's effective rank is <= ~640 BY CONSTRUCTION -- and any experiment that swaps in a
wider-readout decoder on this FROZEN encoder is rigged: there is no extra information to read.

That invalidates _oneoff_decoder_ab.py as a verdict on decode_arch="up". This probe settles it with no
training. If effective rank >> 640 the encoder does carry information the U-Net cannot see, and the frozen
harness is at least measuring something real. If effective rank <= ~640 the harness is circular.

CONFOUND, stated up front: `latent_norm=layernorm` puts each TOKEN on a sphere of radius sqrt(d). That is a
per-token constraint (it removes 2 dof per token, mean and scale), NOT a global rank cap, so it cannot by
itself produce a rank far below 4096 -- it costs at most 2*num_tokens = 64 dimensions. The spectrum is
therefore still informative. Both the pre-LN and post-LN spectra are reported so the effect is visible.

Also reported: how much variance lies INSIDE the decoder's actual readout subspace (the rowspace of
cond_to_spatial, plus the token-mean directions). That is the direct measurement -- if the encoder adapted,
variance should be concentrated there far beyond what a random subspace of the same size would capture.

Usage: uv run python -m quickdraw._oneoff_latent_rank <ckpt> [n_frames]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.training.setup import build_model, resolve_data_root

CKPT = sys.argv[1]
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
RUN = os.path.dirname(os.path.dirname(CKPT))
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
m = getattr(model, "_orig_mod", model)
img_head = next((n for n, _ in m.layout if n != "proprio"), None)
mod = m.modalities[img_head]
ae = mod.ae.cfg
T, d = ae.num_tokens, ae.d
D = T * d
print(f"[load] {os.path.basename(RUN)} | img {ae.img_size} bott {getattr(ae,'bottleneck',8)} "
      f"tokens {T} d {d} -> latent {D} floats | decode_arch={mod.decode_arch} latent_norm={m.latent_norm}",
      flush=True)

ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=ae.img_size,
                            cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
rng = np.random.default_rng(0)
fr = []
per = max(1, NF // max(1, len(ds)))
for i in range(len(ds)):
    _, _, im = ds[i]
    fr.append(torch.from_numpy(im[rng.permutation(len(im))[:per]]).float().div(255.0))
X = torch.cat(fr)[:NF].to(dev)
with torch.no_grad():
    Zraw = torch.cat([mod.encode(X[i:i + 64]).float() for i in range(0, X.shape[0], 64)])
Zln = Fn.layer_norm(Zraw, (d,))
print(f"[data] {X.shape[0]} val frames -> latents {tuple(Zraw.shape)}", flush=True)

def spectrum(Z, tag):
    F_ = Z.reshape(Z.shape[0], -1).double()
    F_ = F_ - F_.mean(0, keepdim=True)
    s = torch.linalg.svdvals(F_)
    ev = (s ** 2) / (s ** 2).sum()
    csum = torch.cumsum(ev, 0)
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())        # participation ratio
    k90 = int((csum < 0.90).sum()) + 1
    k99 = int((csum < 0.99).sum()) + 1
    print(f"  {tag:10} participation-ratio {pr:8.1f} | comps for 90% var {k90:5d} | for 99% {k99:5d} "
          f"| of {min(F_.shape)} available")
    return ev

print(f"\n=== EFFECTIVE RANK of the {D}-float latent (n={X.shape[0]} frames) ===")
ev_raw = spectrum(Zraw, "pre-LN")
ev_ln = spectrum(Zln, "post-LN")
print("  NOTE: per-token LayerNorm removes at most 2*num_tokens = "
      f"{2*T} of {D} dims, so it cannot explain a rank far below {D}.")

# ---- how much variance lies in the decoder's ACTUAL readout subspace? ----
head = mod.decode_head
W = None
for nm, p in head.named_parameters():
    if nm.endswith("cond_to_spatial.weight"):
        W = p.detach().double()
print(f"\n=== variance captured by the decoder's REAL readout subspace ===")
if W is None:
    print("  cond_to_spatial not found (this head does not use a dense flatten) — skipping")
else:
    Fm = Zln.reshape(Zln.shape[0], -1).double()
    Fm = Fm - Fm.mean(0, keepdim=True)
    tot = float((Fm ** 2).sum())
    # rowspace of cond_to_spatial (<=512 dims) + the token-mean directions (d dims, one per feature channel)
    mean_dirs = torch.zeros(d, D, dtype=torch.float64, device=W.device)
    for j in range(d):
        mean_dirs[j, j::d] = 1.0 / T                      # g = cond.mean(1): averages token t's channel j
    B = torch.cat([W, mean_dirs], 0)                      # (<=512+d, D)
    Q = torch.linalg.qr(B.T)[0]                           # orthonormal basis of the readout subspace
    k = Q.shape[1]
    cap = float(((Fm @ Q) ** 2).sum()) / tot
    gen = torch.Generator(device=Q.device).manual_seed(0)
    R = torch.linalg.qr(torch.randn(D, k, generator=gen, dtype=torch.float64, device=Q.device))[0]
    cap_rand = float(((Fm @ R) ** 2).sum()) / tot
    print(f"  readout subspace dim {k} of {D} ({100*k/D:.1f}%)")
    print(f"  variance captured by the REAL readout subspace : {100*cap:6.2f}%")
    print(f"  variance captured by a RANDOM subspace, same k : {100*cap_rand:6.2f}%   <- the null")
    print(f"  concentration factor (real / random)           : {cap/max(cap_rand,1e-12):6.2f}x")
    print("\nREAD: if the encoder ADAPTED to the rank-640 readout, the real subspace should capture far more")
    print("than the random one, and effective rank should sit near the readout dim. If real ~= random and")
    print("effective rank >> 640, the encoder spreads information the decoder CANNOT see -- the frozen-encoder")
    print("A/B is then measuring something real, and the widened readout has genuine headroom to exploit.")
