"""JACOBIAN PROBE -- is the collapse an exploded ROLLOUT JACOBIAN, or a loss-SCALE problem?

WHY. Two independent changes (record 18's dynamics_follows_p_tf, and predict=absolute) produced the same
failure: grad/norm/flow explodes, then inf on backbone+encoders+act_enc, while the decoders -- which sit
OUTSIDE the recurrent path -- stay finite. The proposed fix (detach_every) only helps if the cause is a
COMPOUNDING Jacobian along the rollout. The competing explanation is pure SCALE: predict=absolute regresses
the full LayerNormed bag instead of a small residual, so every dynamics gradient is bigger everywhere, which
a lower LR or lambda_flow addresses and detach_every does not. The two are distinguishable by SHAPE:

  TEST 1  ||dL/d bag_h|| vs rollout depth h.  EXPONENTIAL growth toward the context = compounding Jacobian.
                                              FLAT-but-large = scale. (Read the ratio, not the magnitude.)
  TEST 2  per-step operator norm ||d bag_{h+1} / d bag_h||, power-iterated. This is the actual multiplier.
                                              >1 compounds, <1 contracts. Residual carries an identity skip
                                              (I + J) so it should sit near 1; absolute is J alone.

Both tests are run on WHATEVER checkpoints exist. NOTE the absolute run has no pre-collapse checkpoint --
val ran every 4 epochs so its first save was e3, AFTER the blow-up -- so its weights are already broken and
a huge reading there is not evidence of anything. The load-bearing comparison is TEST 2 at matched epochs,
because the per-step multiplier is a property of the MAP, and the residual control is healthy at every epoch.

Usage: uv run python -m quickdraw._oneoff_jacobian_probe <ckpt> [<ckpt> ...] out.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.training.setup import build_model, normalizer, resolve_data_root

*CKPTS, OUT = sys.argv[1:]
DEPTH = 32           # rollout steps to differentiate through (>= detach_every=32, so nothing is truncated)
N_SAMP = 4           # (episode, t0) samples averaged
N_ITER = 12          # power iterations for the operator norm
dev = torch.device("cuda")


def load(ck):
    run = os.path.dirname(os.path.dirname(ck))
    cfg = OmegaConf.load(os.path.join(run, "checkpoints", "config.resolved.yaml"))
    model = build_model(cfg).to(dev).eval()
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    miss, unex = model.load_state_dict(sd, strict=False)
    m = getattr(model, "_orig_mod", model)
    tag = f"{os.path.basename(run).split('_')[-1]}/{os.path.basename(ck).split('-')[0]}"
    print(f"[load] {tag}  predict={'absolute' if not m.predict_residual else 'residual'} "
          f"detach_every={cfg.model.get('detach_every')} missing={len(miss)} unexpected={len(unex)}", flush=True)
    return cfg, m, tag


def step(m, hist, act, t):
    """ONE rollout advance, differentiable. Mirrors the training rollout and the filmstrip chain."""
    w = min(m.window, hist.shape[1])
    hb = m.backbone(m._to_input(hist[:, -w:], act[:, t - w + 1:t + 1]))[:, -1]
    return m.predict_next(m._cond(hb, act[:, t]), hist[:, -1])


def probe(cfg, m, tag):
    norm = normalizer(cfg)
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)
    img_size = next((md.ae.cfg.img_size for md in m.modalities.values() if hasattr(md, "ae")), 128)
    ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    P = int(cfg.data.P)
    rng = np.random.default_rng(0)
    picks = []
    while len(picks) < N_SAMP:
        i = int(rng.integers(len(ds)))
        o, _, _ = ds[i]
        if len(o) > P + DEPTH + 2:
            picks.append((i, int(rng.integers(P, len(o) - DEPTH - 1))))
    gnorm = {h: [] for h in range(DEPTH)}
    opnorm = {h: [] for h in range(DEPTH)}
    for (ei, t0) in picks:
        o, a, im = ds[ei]
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(dev),
               img_head: torch.from_numpy(im).float().div(255.0)[None].to(dev)}
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(dev)
        with torch.no_grad():
            z = m.encode_state(obs).float()

        # ---- TEST 1: ||dL/d bag_h|| vs depth, ONE backward through the whole chain ----
        hist = z[:, :t0 + 1]
        bags = []
        for k in range(DEPTH):
            nb = step(m, hist, act, t0 + k)
            nb.retain_grad()
            bags.append(nb)
            hist = torch.cat([hist, nb[:, None]], dim=1)
        # the loss the rollout actually carries: dynamics regression of the final bag onto its true latent
        L = F.mse_loss(bags[-1], z[:, t0 + DEPTH].detach())
        L.backward()
        for k, b in enumerate(bags):
            gnorm[k].append(float(b.grad.norm()))

        # ---- TEST 2: per-step operator norm ||d bag_{h+1}/d bag_h||, power iteration ----
        with torch.no_grad():
            hist = z[:, :t0 + 1]
            states = [hist]
            for k in range(DEPTH):
                nb = step(m, hist, act, t0 + k)
                hist = torch.cat([hist, nb[:, None]], dim=1)
                states.append(hist)
        for k in range(DEPTH):
            h_in = states[k].detach().clone().requires_grad_(True)
            out = step(m, h_in, act, t0 + k)
            v = torch.randn_like(out)
            v = v / v.norm()
            s = 0.0
            for _ in range(N_ITER):                       # ||J|| via J^T J power iteration (VJP only)
                g = torch.autograd.grad(out, h_in, v, retain_graph=True)[0][:, -1]
                s = float(g.norm())
                if s < 1e-12:
                    break
                v = torch.zeros_like(out)
                v[:] = (g / s)
            opnorm[k].append(s)
    return {"grad_vs_depth": {h: float(np.mean(v)) for h, v in gnorm.items()},
            "per_step_opnorm": {h: float(np.mean(v)) for h, v in opnorm.items()}}


res = {}
for ck in CKPTS:
    cfg, m, tag = load(ck)
    res[tag] = probe(cfg, m, tag)
    g = res[tag]["grad_vs_depth"]
    op = res[tag]["per_step_opnorm"]
    # h=0 is the DEEPEST-backprop point (nearest the context); h=DEPTH-1 is the loss itself.
    ratio = g[0] / max(g[DEPTH - 1], 1e-30)
    print(f"\n=== {tag} ===")
    print(f"  ||dL/dbag|| at loss (h={DEPTH-1}): {g[DEPTH-1]:.4g}   at context (h=0): {g[0]:.4g}"
          f"   AMPLIFICATION over {DEPTH} steps: {ratio:.4g}   per-step geo-mean: {ratio ** (1.0/(DEPTH-1)):.4f}")
    print("  grad vs depth: " + " ".join(f"h{h}:{g[h]:.3g}" for h in (0, 4, 8, 16, 24, DEPTH - 1)))
    vals = [op[h] for h in range(DEPTH)]
    print(f"  per-step ||J||: mean {np.mean(vals):.4f}  max {np.max(vals):.4f}  min {np.min(vals):.4f}"
          f"   (residual carries I+J -> expect ~1; >1 compounds)")
    print("  ||J|| vs depth: " + " ".join(f"h{h}:{op[h]:.3f}" for h in (0, 4, 8, 16, 24, DEPTH - 1)))
    del m
    torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
