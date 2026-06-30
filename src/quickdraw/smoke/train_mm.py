"""End-to-end multimodal LSAR training smoke on REAL FPV data — the P3 capstone. Trains MultiModalLSAR
(proprio + image) on val episodes for a few hundred steps, then on a HELD-OUT episode does an
autoregressive rollout and writes artifacts to logs/viz_preview/:
  - mm_lsar_filmstrip.png : 8 horizon steps, predicted FPV (top) vs ground-truth (bottom)
  - mm_lsar_rollout.mp4    : synced rollout video (pred top, black through context; GT bottom)
Reports per-head recon + pred_latent over training and rollout proprio error + image PSNR.

  uv run python -m quickdraw.smoke.train_mm <data_root> [steps]
"""
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from quickdraw.data.dataset import MMWindowLoader, Normalizer, load_split_episodes_mm
from quickdraw.logging import viz
from quickdraw.models.modalities import ModalitySpec
from quickdraw.models.multimodal import MultiModalLSAR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/app/logs/viz_preview"


def psnr(mse):
    return float("inf") if mse <= 0 else -10.0 * np.log10(max(mse, 1e-12))


def main():
    root = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    P, Fh, B, d = 8, 24, 16, 256
    print(f"[train_mm] loading VAL episodes (obs+act+FPV128) from {root} ...")
    eps = load_split_episodes_mm(root, "val", img_size=128)
    norm = Normalizer.from_file(root)
    train_eps, hold = eps[:-4], eps[-4]                         # hold out 1 episode for the rollout viz
    loader = MMWindowLoader(train_eps, P, Fh, norm, batch=B, shuffle=True, device=DEV)
    specs = [ModalitySpec("proprio", "vector", dim=6, weight=1.0),
             ModalitySpec("image", "image", num_tokens=8, weight=1.0)]
    m = MultiModalLSAR(specs, d=d, depth=4, heads=4, window=P + Fh, mlp_ratio=4.0,
                       rope_theta=10000.0, action_dim=2).to(DEV)
    wts = {mm.name: mm.weight for mm in m.modalities.values()}
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    print(f"[train_mm] MultiModalLSAR {sum(p.numel() for p in m.parameters())/1e6:.1f}M params | "
          f"{len(loader)} batches/epoch | training {steps} steps")

    m.train()
    it = 0
    while it < steps:
        for batch in loader:
            obs = {"proprio": batch["obs_seq"], "image": batch["image"]}
            pf = m({k: v[:, :-1] for k, v in obs.items()}, batch["act_seq"][:, :-1])
            preds = pf[:, P - 1:]
            future = {k: v[:, P:] for k, v in obs.items()}
            dec = m.to_obs(preds)
            rl = {k: F.mse_loss(dec[k], future[k]) for k in future}
            raw, w = m.loss_terms(preds, future, obs, 1.0, batch["act_seq"])
            loss = sum(wts[k] * rl[k] for k in rl) + sum(w[k] * raw[k] for k in raw)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 200 == 0 or it == steps - 1:
                print(f"[train_mm]   step {it:5d}  proprio {rl['proprio'].item():.4f}  "
                      f"image {rl['image'].item():.4f} (psnr {psnr(rl['image'].item()):.1f})  "
                      f"pred_latent {raw['pred_latent'].item():.4f}")
            it += 1
            if it >= steps:
                break

    # ---- held-out autoregressive rollout ----
    m.eval()
    o, a, img = hold
    H = min(Fh, len(o) - P - 1)
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[:P])).float()[None].to(DEV),
           "image": torch.from_numpy(img[:P]).float().div(255.0)[None].to(DEV)}
    actions = torch.from_numpy(a[:P + H - 1]).float()[None].to(DEV)
    with torch.no_grad():
        out = m.imagine_eval(ctx, actions, H)
    pred_img = out["image"][0].clamp(0, 1).cpu().numpy()                     # (H,128,128,3)
    true_img = (img[P:P + H].astype(np.float32) / 255.0)                     # (H,128,128,3)
    pred_pro = norm.denorm_obs(out["proprio"][0].cpu()).numpy()              # (H,6)
    true_pro = o[P:P + H]
    img_mse = float(np.mean((pred_img - true_img) ** 2))
    pro_err = float(np.mean(np.linalg.norm(pred_pro[:, :3] - true_pro[:, :3], axis=-1)))
    print(f"[train_mm] HELD-OUT rollout H={H}: image PSNR {psnr(img_mse):.2f} dB | proprio pos err {pro_err:.4f}")

    # artifacts (reuse the viz utils)
    f = viz.fig_image_filmstrip(pred_img, true_img, n_cols=8,
                                title=f"MM-LSAR rollout — pred (top) vs GT (bottom), H={H}, img PSNR {psnr(img_mse):.1f}dB")
    f.savefig(f"{OUT}/mm_lsar_filmstrip.png", dpi=110)
    full_true = (img[: P + H].astype(np.float32) / 255.0)                    # context + future GT
    vid = viz.image_rollout_video(full_true, pred_img, context_len=P)
    imageio.mimwrite(f"{OUT}/mm_lsar_rollout.mp4", list((vid).astype(np.uint8)), fps=10, macro_block_size=2, quality=8)
    print(f"[train_mm] wrote {OUT}/mm_lsar_filmstrip.png and mm_lsar_rollout.mp4")
    print("[train_mm] OK" if img_mse < 0.02 else "[train_mm] DONE (review artifacts)")


if __name__ == "__main__":
    main()
