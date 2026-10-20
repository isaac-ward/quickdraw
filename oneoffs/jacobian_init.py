"""Companion to _oneoff_jacobian_probe: the missing CONTROL.

The absolute run has no pre-collapse checkpoint (val every 4 epochs -> first save at e3, AFTER the blow-up),
so its per-step ||J|| there cannot separate CAUSE from CONSEQUENCE. This measures the same quantity on
FRESHLY INITIALISED models -- same seed, same architecture, only `predict` differs -- where no collapse has
happened and any difference is therefore STRUCTURAL: residual's step map is I + J, absolute's is J alone.

Usage: uv run python -m quickdraw._oneoff_jacobian_init <a_config.resolved.yaml>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.training.setup import build_model

DEPTH, N_ITER, SEED = 32, 12, 0
dev = torch.device("cuda")
cfg = OmegaConf.load(sys.argv[1])
OmegaConf.set_struct(cfg, False)

def step(m, hist, act, t):
    w = min(m.window, hist.shape[1])
    hb = m.backbone(m._to_input(hist[:, -w:], act[:, t - w + 1:t + 1]))[:, -1]
    return m.predict_next(m._cond(hb, act[:, t]), hist[:, -1])

for mode in ("residual", "absolute"):
    cfg.model.diffusion.predict = mode
    torch.manual_seed(SEED)
    m = getattr(build_model(cfg).to(dev).eval(), "_orig_mod", None) or build_model(cfg).to(dev).eval()
    torch.manual_seed(SEED + 1)
    P = int(cfg.data.P)
    T = P + DEPTH + 2
    bag = torch.randn(1, P, m.n_state, m.d, device=dev)
    bag = torch.nn.functional.layer_norm(bag, (m.d,)) if m.latent_norm else bag
    act = torch.randn(1, T, int(cfg.model.action_dim), device=dev)
    with torch.no_grad():
        hist, states = bag, [bag]
        for k in range(DEPTH):
            hist = torch.cat([hist, step(m, hist, act, P + k - 1)[:, None]], dim=1)
            states.append(hist)
    ops = []
    for k in range(DEPTH):
        h_in = states[k].detach().clone().requires_grad_(True)
        out = step(m, h_in, act, P + k - 1)
        v = torch.randn_like(out); v = v / v.norm()
        s = 0.0
        for _ in range(N_ITER):
            g = torch.autograd.grad(out, h_in, v, retain_graph=True)[0][:, -1]
            s = float(g.norm())
            if s < 1e-12:
                break
            v = (g / s).clone()
        ops.append(s)
    a = np.array(ops)
    print(f"{mode:9} per-step ||J||: mean {a.mean():.4f}  max {a.max():.4f}  min {a.min():.4f}  "
          f"p90 {np.percentile(a,90):.4f}   32-step product (geo) {a.prod() ** (1/DEPTH):.4f}")
    print(f"          vs depth: " + " ".join(f"h{h}:{a[h]:.3f}" for h in (0, 4, 8, 16, 24, DEPTH - 1)))
    del m; torch.cuda.empty_cache()
