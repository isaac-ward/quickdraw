"""Diagnostic (b): is the rolled cube SMEARED (a blend) or CRISP-but-DISPLACED? Filmstrip of the cube region:
GT | rolled-DETERMINISTIC | rolled-STOCHASTIC, at several horizons, 4x zoom. A blend => washed/ghosted cube;
a displacement => a sharp cube in the wrong spot. Out: logs/oracle_cube/plots/rolled_zoom.png
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/rolled_zoom.py <ckpt>
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from _cube_common import load, rollout_decode, any_cube_mask


def crop(fr, msk):
    """Zoom to the GT cube bbox (padded) so we can SEE sharpness vs blur."""
    ys, xs = np.where(msk)
    if len(ys) == 0:
        return fr
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = 14; H, W = fr.shape[:2]
    y0, y1 = max(0, y0 - pad), min(H, y1 + pad); x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
    c = fr[y0:y1, x0:x1]
    return np.kron(c, np.ones((4, 4, 1)))  # 4x nearest-neighbor zoom


def main():
    ck = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, ctx, acts, gt, Hh = load(ck, dev)
    g = gt.cpu().numpy()
    det = rollout_decode(m, ctx, acts, Hh, stochastic=False)
    sto = rollout_decode(m, ctx, acts, Hh, stochastic=True)
    b = 0                                                    # one episode; the first
    ts = [4, 16, 28, 40, 52, Hh - 1]
    fig, ax = plt.subplots(len(ts), 3, figsize=(9, 2.3 * len(ts)))
    for i, t in enumerate(ts):
        msk = any_cube_mask(g[b, t])
        for j, (name, im) in enumerate([("GT", g), ("rolled DET", det), ("rolled STOCH", sto)]):
            ax[i, j].imshow(np.clip(crop(im[b, t], msk), 0, 1)); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(name)
            if j == 0: ax[i, j].set_ylabel(f"t={t}")
    fig.suptitle(f"rolled cube zoom (smear vs displacement) — {os.path.basename(os.path.dirname(os.path.dirname(ck)))}")
    fig.tight_layout(); os.makedirs("logs/oracle_cube/plots", exist_ok=True)
    fig.savefig("logs/oracle_cube/plots/rolled_zoom.png", dpi=110); plt.close(fig)
    print("[rolled_zoom] wrote logs/oracle_cube/plots/rolled_zoom.png", flush=True)


if __name__ == "__main__":
    main()
