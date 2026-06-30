"""Greenlight smoke for the ViT image autoencoder (models/vision.py). Loads real FPV frames (downsampled
to 128² once + cached), fits the AE briefly under plain MSE, reports train/val recon MSE + PSNR, and writes
a recon grid (TOP = reconstruction, BOTTOM = ground truth) to logs/viz_preview/ for visual review.

  uv run python -m quickdraw.smoke.vision_ae <data_root> [steps]
"""
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quickdraw.data.dataset import load_fpv_frames
from quickdraw.models.vision import ImageAutoencoder, VisionAEConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/app/logs/viz_preview/vision_ae_recon.png"


def psnr(mse):
    return float("inf") if mse <= 0 else -10.0 * np.log10(mse)


def main():
    root = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    bsz, n_frames = 128, 16000
    print(f"[vision_ae] loading FPV frames (<= {n_frames}, 128x128) from {root} ...")
    frames = load_fpv_frames(root, "train", size=128, max_frames=n_frames, cache=False)
    print(f"[vision_ae] {len(frames)} frames {frames.shape} {frames.dtype}")
    x = torch.from_numpy(frames).float().div_(255.0)                 # (N,128,128,3) in [0,1]
    n_val = 1024
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(x), generator=g)
    val = x[perm[:n_val]].to(DEV)
    train = x[perm[n_val:]]

    m = ImageAutoencoder(VisionAEConfig()).to(DEV)
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    print(f"[vision_ae] AE {n_params/1e6:.1f}M params | training {steps} steps, batch {bsz}")
    m.train()
    gg = torch.Generator().manual_seed(1)
    for it in range(steps):
        idx = torch.randint(0, len(train), (bsz,), generator=gg)
        img = train[idx].to(DEV)
        recon, _ = m(img)
        loss = torch.nn.functional.mse_loss(recon, img)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0 or it == steps - 1:
            print(f"[vision_ae]   step {it:5d}  train_mse {loss.item():.5f}  psnr {psnr(loss.item()):.2f}")

    m.eval()
    with torch.no_grad():
        vr, _ = m(val)
        vmse = torch.nn.functional.mse_loss(vr, val).item()
    print(f"[vision_ae] VAL recon_mse {vmse:.5f}  psnr {psnr(vmse):.2f} dB")

    # recon grid: 8 val frames, TOP = reconstruction, BOTTOM = ground truth
    with torch.no_grad():
        sel = val[:8]
        rec = m(sel)[0].clamp(0, 1)
    sel, rec = sel.cpu().numpy(), rec.cpu().numpy()
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.2))
    for j in range(8):
        axes[0, j].imshow(rec[j]); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].imshow(sel[j]); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        if j == 0:
            axes[0, j].set_ylabel("recon", fontsize=10); axes[1, j].set_ylabel("true", fontsize=10)
    fig.suptitle(f"ViT image AE — val PSNR {psnr(vmse):.2f} dB ({steps} steps, num_tokens=8, 128²)", fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.02, wspace=0.04, hspace=0.04)
    fig.savefig(OUT, dpi=110); plt.close(fig)
    print(f"[vision_ae] wrote {OUT}")
    print("[vision_ae] OK" if vmse < 0.02 else "[vision_ae] DONE (review recon; vmse high)")


if __name__ == "__main__":
    main()
