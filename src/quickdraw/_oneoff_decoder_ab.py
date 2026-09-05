"""DECODER A/B on a frozen encoder: does the query-grid readout beat the U-Net, and is it readout RANK or
trunk CAPACITY that matters?

WHY IT EXISTS. `decode_arch="up"` (models/decoders.py) fixes two measured pathologies of the mse decoder: a
down path that convolves zeros, and a rank-640 choke on a 4,096-float latent. But the single piece of prior
evidence that a wider decoder moves the OBJECTIVE -- OL LPIPS@+128 improving 0.333 -> 0.310 at matched settings
-- came from `decode_base` 32 -> 64, which widened the ENTIRE conv trunk (2.3x params), not the readout alone.
The up-only decoder widens the readout (640 -> 4,608) while SHRINKING the trunk ~2.9x. If that earlier win was
trunk capacity, this design loses it. Three arms separate the two axes:

    up@32      readout 4608, trunk small      <- the new design
    unet@32    readout  640, trunk small      <- the current default, the control
    unet@64    readout 1152, trunk 2.3x       <- the arm that produced 0.333 -> 0.310

TWO TESTS, because the floor and the objective are different questions and the record repeatedly shows floor
levers failing to transfer (bott16 won the floor and lost every rollout metric).

  TEST 1 -- FLOOR, on HELD-OUT frames. Freeze the trained encoder so all three heads see IDENTICAL real
  latents, train each from scratch with the same seed/data/steps, and score reconstruction on frames never
  trained on. The held-out split matters: an earlier version of this probe trained 6000 steps on 192 frames and
  reached LPIPS 0.05 against a real floor of ~0.24, i.e. it measured memorisation, not reconstruction.

  TEST 2 -- THE OBJECTIVE-RELEVANT ONE. Roll the model open-loop, take the PREDICTED (drifted) latents at each
  horizon, and decode them with each head. The dynamics is held constant, so this isolates what the decoder
  does with a WRONG latent -- which is the crux: the rank-640 choke currently acts as an accidental DRIFT
  FILTER (the decoder is blind to 84% of the latent, so it ignores most of the rollout's error). Widening the
  readout could therefore make the decoder faithfully render a wrong latent and WORSEN @+128 while improving
  the floor. That is exactly the bott16 pattern, and this test is the cheapest way to see it before a pair is
  spent. Heads are trained ONLY on true latents, so decoding predicted ones is genuinely out-of-distribution
  for all three equally.

Usage: uv run python -m quickdraw._oneoff_decoder_ab <ckpt> [steps] [arm,arm,...]
       arms: up@32 up@64 unet@32 unet@64   (default: up@32,unet@32,unet@64)
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.evaluation.openloop import _lpips_net
from quickdraw.models.decoders import TokenGridDecoder
from quickdraw.models.flow import ImageUNetFlowHead
from quickdraw.training.setup import build_model, normalizer, resolve_data_root

CKPT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
RUN = os.path.dirname(os.path.dirname(CKPT))
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
dev = torch.device("cuda")
HZ = [1, 8, 16, 32, 64, 128]

model = build_model(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
miss, unex = model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
m = getattr(model, "_orig_mod", model)
img_head = next((n for n, _ in m.layout if n != "proprio"), None)
mod = m.modalities[img_head]
ae_cfg = mod.ae.cfg
print(f"[load] {os.path.basename(RUN)} missing={len(miss)} unexpected={len(unex)} | img_size={ae_cfg.img_size} "
      f"bottleneck={getattr(ae_cfg,'bottleneck',8)} d={ae_cfg.d} num_tokens={ae_cfg.num_tokens} "
      f"predict={'residual' if m.predict_residual else 'absolute'}", flush=True)

norm = normalizer(cfg)
# frames come back as a DICT keyed by camera (data/dataset.py). Single-camera script:
# name the key once rather than indexing position 2 as if there were only ever one view.
_CAMK = str(cfg.data.get("cam", "fpv"))
ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=ae_cfg.img_size,
                           cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
P = int(cfg.data.P)
rng = np.random.default_rng(0)

# ---- frames for head training, with a HELD-OUT split (see TEST 1 note) ----
tr_f, te_f = [], []
for i in range(len(ds)):
    _, _, _fr = ds[i]
    im = _fr[_CAMK]
    idx = rng.permutation(len(im))[:40]
    cut = int(0.75 * len(idx))
    (tr_f if i % 4 else te_f).append(torch.from_numpy(im[idx[:cut] if i % 4 else idx]).float().div(255.0))
Xtr = torch.cat(tr_f).to(dev)
Xte = torch.cat(te_f).to(dev)
print(f"[data] {len(ds)} val episodes -> {Xtr.shape[0]} train frames / {Xte.shape[0]} HELD-OUT frames", flush=True)
# The heads must be TRAINED on latents in the SAME space the rollout produces, or TEST 2 compares apples to
# oranges. `encode_state` concatenates the per-modality encodes and then applies a non-affine per-token
# LayerNorm over the whole bag (multimodal.py `_ln`), which `mod.encode` alone does not. LayerNorm here is
# per TOKEN over the feature axis and the concat does not mix tokens, so slicing commutes with it:
#     _ln(encode_state(obs))[..., img_slice, :] == _ln(mod.encode(img))
# so applying _ln to the modality encode reproduces the bag's image slice exactly.
_ln = (lambda t: Fn.layer_norm(t, (t.shape[-1],))) if m.latent_norm else (lambda t: t)
with torch.no_grad():
    Ztr = torch.cat([_ln(mod.encode(Xtr[i:i + 64]).float()) for i in range(0, Xtr.shape[0], 64)])
    Zte = torch.cat([_ln(mod.encode(Xte[i:i + 64]).float()) for i in range(0, Xte.shape[0], 64)])
print(f"[latent] latent_norm={m.latent_norm} -> per-token LN {'applied' if m.latent_norm else 'off'}; "
      f"train latents {tuple(Ztr.shape)}", flush=True)

# ---- open-loop rollout ONCE, shared by all heads: predicted (drifted) latents + their GT frames ----
picks = []
while len(picks) < 6:
    i = int(rng.integers(len(ds)))
    o, _, _ = ds[i]
    if len(o) > P + max(HZ) + 2:
        picks.append((i, int(rng.integers(P, len(o) - max(HZ) - 1))))
roll_z, roll_gt = {h: [] for h in HZ}, {h: [] for h in HZ}
with torch.no_grad():
    for ei, t0 in picks:
        o, a, _fr = ds[ei]
        im = _fr[_CAMK]
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(dev),
               img_head: torch.from_numpy(im).float().div(255.0)[None].to(dev)}
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.encode_state(obs).float()
        hist = z[:, :t0 + 1]
        for h in range(1, max(HZ) + 1):
            t = t0 + h - 1
            w = min(m.window, hist.shape[1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = m.backbone(m._to_input(hist[:, -w:], act[:, t - w + 1:t + 1]))[:, -1]
                nb = m.predict_next(m._cond(hb, act[:, t]), hist[:, -1]).float()
            hist = torch.cat([hist, nb[:, None]], dim=1)
            if h in HZ:
                roll_z[h].append(nb[0])                                    # (n_state,d) PREDICTED bag
                roll_gt[h].append(torch.from_numpy(im[t0 + h]).float().div(255.0).to(dev))
# predict_next returns the FULL bag (every modality's tokens concatenated); the decode head takes only this
# modality's slice. Compute the offset from the layout -- there is no helper for it.
off, n_img = 0, None
for _nm, _nt in m.layout:
    if _nm == img_head:
        n_img = _nt
        break
    off += _nt
assert n_img is not None
for h in HZ:
    roll_z[h] = torch.stack(roll_z[h])[:, off:off + n_img]      # (n_chains, n_img_tokens, d)
    roll_gt[h] = torch.stack(roll_gt[h])
assert roll_z[HZ[0]].shape[1:] == Ztr.shape[1:], (roll_z[HZ[0]].shape, Ztr.shape)
print(f"[rollout] {len(picks)} chains, horizons {HZ}; image tokens = bag[..., {off}:{off + n_img}, :] "
      f"-> {tuple(roll_z[HZ[0]].shape)}", flush=True)

lp = _lpips_net(dev)
# Drive every arm through the head's OWN no_noise machinery (TransportHead.loss / .sample) rather than calling
# velocity directly: the U-Net needs a real (all-zero) `x` while TokenGridDecoder ignores it, and loss()/sample()
# are exactly what training and eval use -- `loss` builds x0=zeros + temb(ones), `sample(steps=1,
# deterministic=True)` returns velocity(zeros, temb(ones), cond, None). Uniform across arms, and faithful.
def score(head, X, Z, bs=32):
    outs = []
    with torch.no_grad():
        for i in range(0, Z.shape[0], bs):
            outs.append(head.sample(Z[i:i + bs], steps=1, deterministic=True).clamp(0, 1).float())
    Pm = torch.cat(outs)
    mse = (Pm - X).pow(2).mean(dim=(1, 2, 3))
    psnr = float((10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean())
    p, x = Pm.permute(0, 3, 1, 2).clamp(0, 1), X.permute(0, 3, 1, 2).clamp(0, 1)
    with torch.no_grad():
        l = float(torch.cat([lp(p[i:i + 32], x[i:i + 32]).flatten() for i in range(0, p.shape[0], 32)]).mean())
    return psnr, l

# Arms are selectable so the MATCHED-TRUNK-WIDTH follow-up can be run without touching the defaults: the
# first pass (up@32 vs unet@32 vs unet@64) confounds the design change with a 2.9x smaller trunk, so
# `up@64 unet@64` isolates readout-and-no-down-path at IDENTICAL base width.
_MAKE = {"up@32":   lambda: TokenGridDecoder(ae_cfg, base=32),
         "up@64":   lambda: TokenGridDecoder(ae_cfg, base=64),
         "unet@32": lambda: ImageUNetFlowHead(ae_cfg, base=32, param="x0", no_noise=True),
         "unet@64": lambda: ImageUNetFlowHead(ae_cfg, base=64, param="x0", no_noise=True)}
_names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["up@32", "unet@32", "unet@64"]
ARMS = [(k, _MAKE[k]) for k in _names]
res = {}
for name, make in ARMS:
    torch.manual_seed(0)
    head = make().to(dev)
    npar = sum(p.numel() for p in head.parameters())
    opt = torch.optim.Adam(head.parameters(), lr=3e-4)
    g = torch.Generator(device=dev).manual_seed(1)
    for it in range(STEPS):
        i = torch.randint(0, Ztr.shape[0], (32,), device=dev, generator=g)
        loss, _ = head.loss(Ztr[i], Xtr[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % max(1, STEPS // 4) == 0:
            ps, l = score(head, Xte, Zte)
            print(f"    [{name:8} @{it+1:5}] held-out PSNR {ps:.2f} LPIPS {l:.4f}", flush=True)
    head.eval()
    ps_te, l_te = score(head, Xte, Zte)
    # TEST 2: decode the PREDICTED (drifted) latents. Heads never saw a predicted latent in training.
    ol = {}
    for h in HZ:
        ol[h] = score(head, roll_gt[h], roll_z[h], bs=16)
    res[name] = {"params": npar, "floor": (ps_te, l_te), "ol": ol}
    del head, opt
    torch.cuda.empty_cache()

print("\n=== TEST 1 — FLOOR on HELD-OUT frames (true latents) ===")
print(f"{'arm':10} {'params':>11} {'PSNR':>7} {'LPIPS':>8}")
for n2, r in res.items():
    print(f"{n2:10} {r['params']:11,} {r['floor'][0]:7.2f} {r['floor'][1]:8.4f}")
print("\n=== TEST 2 — decoding the SAME open-loop PREDICTED latents (dynamics held constant) ===")
print("LPIPS (lower better) by horizon")
print(f"{'arm':10} " + "".join(f"+{h:<8}" for h in HZ))
for n2, r in res.items():
    print(f"{n2:10} " + "".join(f"{r['ol'][h][1]:<9.4f}" for h in HZ))
print("PSNR by horizon")
for n2, r in res.items():
    print(f"{n2:10} " + "".join(f"{r['ol'][h][0]:<9.2f}" for h in HZ))
print("\nREAD: TEST 2 at the deepest horizons is the objective-relevant number. If up@32 beats unet@32 there,")
print("the readout widening survives contact with a DRIFTED latent and the design is worth a pair. If up@32")
print("wins TEST 1 but loses TEST 2, the rank-640 choke was acting as a drift filter and widening it trades")
print("floor for objective -- the bott16 pattern. If unet@64 beats up@32 on BOTH, the earlier 0.333->0.310 win")
print("was trunk CAPACITY, not readout rank, and this design is the wrong lever.")
