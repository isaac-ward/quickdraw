"""Falsifiable check on record §13: is torus-world's per-step motion ABOVE its own codec floor?

§13 claims the reason robocasa shows no arm motion is that the per-step image change (0.0389 RMSE) is
0.61x the frozen TAESD's reconstruction error (0.0637) -- signal below the noise floor, so predicting zero
is correct. The user reports torus-world learns video prediction from very little data. If torus's per-step
delta is ALSO below its codec floor and it still learns motion, §13 IS WRONG.

SELF-VALIDATION: this script first reproduces TAESD's known 23.92 dB on robocasa. If that number does not
come back, the TAESD scaling convention here is wrong and the torus number must not be trusted.
"""
import glob
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderTiny
from huggingface_hub import snapshot_download

from quickdraw.data.dataset import load_fpv_frames, load_split_episodes_mm

dev = "cuda" if torch.cuda.is_available() else "cpu"
ae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(dev).eval()


@torch.no_grad()
def codec_rmse(x, n=384):
    """x: (N,H,W,3) float in [0,1]. TAESD takes [0,1] in, .sample out (the repo maps pm1 on decode)."""
    errs = []
    for i in range(0, min(n, len(x)), 32):
        b = x[i:i + 32].permute(0, 3, 1, 2).to(dev)
        y = ae.decode(ae.encode(b).latents).sample
        errs.append(((y.clamp(0, 1) - b) ** 2).mean().item())
    m = float(np.mean(errs))
    return m ** 0.5, -10 * np.log10(m)


def deltas(eps, codec, strides, n_eps=40):
    print(f"{'stride':>7} {'delta RMSE':>11} {'x codec floor':>14} {'% pix > floor':>14}")
    print("-" * 50)
    for s in strides:
        ds, fr = [], []
        for e in eps[:n_eps]:
            im = e[2].astype(np.float32) / 255.0
            if len(im) <= s:
                continue
            d = im[s:] - im[:-s]
            ds.append(float(np.sqrt((d ** 2).mean())))
            fr.append(float((np.abs(d) > codec).mean()))
        print(f"{s:>7} {np.mean(ds):>11.4f} {np.mean(ds)/codec:>13.2f}x {100*np.mean(fr):>13.2f}%", flush=True)


RC = "/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71eb0b7d9ecbf1a7ada843da026d4bc0785"
rc = load_fpv_frames(RC, "val", size=128, cam="robot0_agentview_left", max_frames=384, cache=False)
r, db = codec_rmse(torch.from_numpy(rc).float().div(255.0))
print(f"\n[VALIDATION] TAESD on robocasa val @128: RMSE {r:.4f} = {db:.2f} dB  (record says 23.92 dB)")
print(f"[VALIDATION] {'PASS - convention correct' if abs(db-23.92) < 1.5 else 'FAIL - do NOT trust the torus number below'}\n")

root = snapshot_download(repo_id="isaac-ronald-ward/torus-world", repo_type="dataset")
for split in ("val", "train"):
    p = os.path.join(root, split, "meta", "info.json")
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"[torus] {split}: " + str({k: d.get(k) for k in ("fps", "total_frames", "total_episodes")}))
vids = glob.glob(os.path.join(root, "val", "videos", "observation.images.*"))
cam = os.path.basename(vids[0]).split("observation.images.")[-1] if vids else "fpv"
print(f"[torus] cam = {cam}")

tf = load_fpv_frames(root, "val", size=128, cam=cam, max_frames=384, cache=False)
tr, tdb = codec_rmse(torch.from_numpy(tf).float().div(255.0))
print(f"[torus] TAESD recon: RMSE {tr:.4f} = {tdb:.2f} dB   <-- torus's OWN codec floor\n")

eps = load_split_episodes_mm(root, "val", img_size=128, cam=cam, repo_id="torus")
print(f"[torus] {len(eps)} val eps, first T={len(eps[0][0])}")
print("\n=== TORUS: per-step motion vs its own codec floor ===")
deltas(eps, tr, (1, 2, 4, 8))
print("\n=== ROBOCASA reference (record §13) ===")
print("stride 1 (20 Hz): 0.0389 RMSE = 0.61x floor,  3.18% of pixels")
print("stride 5  (4 Hz): 0.0863 RMSE = 1.35x floor,  8.08% of pixels")
print("\nVERDICT: §13 survives if torus stride 1 is comfortably ABOVE 1.0x its own floor.")
