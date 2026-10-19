"""Per-frame error over the first 30 s of the Cosmos rollouts, via the repo's own image_curves so the
numbers are the same arithmetic eval reports. Source is the raw npz (native 144x192, float32)."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/app/src")
from quickdraw.evaluation.openloop import image_curves

HZ = 15.0
ROOT = "/app/logs/eval_ood_horizon_2026_09_22_19_16_{t}_cosmos_val_{a}/logs/epoch_0000/eval_ood_horizon"
ARMS = {"scene only": ("39", "pos"), "scene + negative": ("38", "posneg")}
rows = {}
for arm, (t, a) in ARMS.items():
    for mode, sec in (("open_loop", 30), ("closed_loop_16_steps", 17)):
        P, G = [], []
        for ep in (0, 1):
            z = np.load(f"{ROOT.format(t=t, a=a)}/{mode}/image/raw_filmstrip_frames_{ep}.npz")
            n = min(int(sec * HZ), len(z["pred"]))
            P.append(z["pred"][:n]); G.append(z["gt"][:n])
        pred = torch.from_numpy(np.stack(P)).float().cuda().clamp(0, 1)
        true = torch.from_numpy(np.stack(G)).float().cuda()
        c = image_curves(pred, true)
        rows[(arm, mode)] = (n, {k: float(np.mean(v)) for k, v in c.items()})
        del pred, true; torch.cuda.empty_cache()

hdr = f"{'arm':18s} {'mode':12s} {'frames':>7s} {'sec':>5s} {'L1':>8s} {'L2 (MSE)':>9s} {'PSNR dB':>8s} {'LPIPS':>7s} {'SSIM':>7s}"
print(hdr); print("-" * len(hdr))
for (arm, mode), (n, m) in rows.items():
    print(f"{arm:18s} {mode.replace('_steps',''):12s} {n:7d} {n/HZ:5.1f} "
          f"{m['l1']:8.4f} {m['mse']:9.4f} {m['psnr']:8.2f} {m['lpips']:7.4f} {m['ssim']:7.4f}")
