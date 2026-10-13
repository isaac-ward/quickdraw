"""Diagnostic (a): does bumping the flow sampling_steps reduce the smear? Tests the under-resolution
hypothesis (6-step ODE lands between modes -> munge). Reports cube survival + cube-only L1 over horizon
for sampling_steps in {6,16,32} (stochastic) and deterministic. Out: logs/oracle_cube/plots/sampling_sweep.png
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/sampling_steps_sweep.py <ckpt>
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from _cube_common import load, rollout_decode, any_cube_mask


def curves(g, r, Hh):
    surv, l1 = [], []
    for t in range(Hh):
        sv, e = [], []
        for b in range(g.shape[0]):
            msk = any_cube_mask(g[b, t])
            if msk.sum() < 8: continue
            sv.append((any_cube_mask(r[b, t]) & msk).sum() / max(msk.sum(), 1))
            e.append(np.abs(r[b, t][msk] - g[b, t][msk]).mean())
        surv.append(np.mean(sv) if sv else np.nan); l1.append(np.mean(e) if e else np.nan)
    return np.array(surv), np.array(l1)


def main():
    ck = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, ctx, acts, gt, Hh = load(ck, dev)
    g = gt.cpu().numpy(); hs = list(range(Hh))
    configs = [("det", False, 6), ("stoch s6", True, 6), ("stoch s16", True, 16), ("stoch s32", True, 32)]
    res = {}
    for name, stoch, steps in configs:
        r = rollout_decode(m, ctx, acts, Hh, stochastic=stoch, sampling_steps=steps)
        res[name] = curves(g, r, Hh)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, (sv, l1) in res.items():
        ax[0].plot(hs, sv, label=name); ax[1].plot(hs, l1, label=name)
    ax[0].set_title("cube SURVIVAL vs sampling_steps"); ax[0].set_ylim(0, 1); ax[0].set_xlabel("horizon"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_title("cube-only L1 vs sampling_steps"); ax[1].set_xlabel("horizon"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"sampling_steps sweep — {os.path.basename(os.path.dirname(os.path.dirname(ck)))}")
    fig.tight_layout(); os.makedirs("logs/oracle_cube/plots", exist_ok=True)
    fig.savefig("logs/oracle_cube/plots/sampling_sweep.png", dpi=110); plt.close(fig)
    print("[sampling_sweep] wrote logs/oracle_cube/plots/sampling_sweep.png", flush=True)
    print("  cfg        | surv h16 h32 h48 h63 | L1 h16 h32 h48 h63", flush=True)
    for name, (sv, l1) in res.items():
        qs = [Hh // 4, Hh // 2, 3 * Hh // 4, Hh - 1]
        print(f"  {name:10s} | surv {' '.join(f'{sv[q]:.2f}' for q in qs)} | L1 {' '.join(f'{l1[q]:.3f}' for q in qs)}", flush=True)


if __name__ == "__main__":
    main()
