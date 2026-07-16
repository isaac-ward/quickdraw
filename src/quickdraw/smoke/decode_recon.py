"""Guard for the generative decode heads: a decode head must be able to RECONSTRUCT (not just have a
dropping loss). Overfit one fixed (cond -> target) and assert the deterministic decode matches + stays
in-range. This is the check that would have caught the flow-velocity image-decode collapse (param=v gave
mse ~0.16 + out-of-[0,1]; param=x0 gives ~0). Run: uv run python -m quickdraw.smoke.decode_recon"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.flow import FlowField, ImageFlowHead, ImageUNetFlowHead
from ..models.vision import VisionAEConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _overfit(head, cond, tgt, steps=800):
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        lf, lc = head.loss(cond, tgt)
        (lf + (lc if lc is not None else 0)).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return head.sample(cond, steps=1, deterministic=True)


def main():
    torch.manual_seed(0)
    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond)); print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")

    cfg = VisionAEConfig(img_size=32, patch=8, d=128, heads=8, num_tokens=8, enc_depth=4, dec_depth=4)

    # IMAGE: x0 must reconstruct precisely + in-range; v is expected to be poor (documents the failure mode).
    ci, ti = torch.randn(4, 8, 128, device=DEV), torch.rand(4, 32, 32, 3, device=DEV)
    r_x0 = _overfit(ImageFlowHead(cfg, depth=4, param="x0").to(DEV), ci, ti)
    mse_x0 = float(F.mse_loss(r_x0, ti)); rng = (float(r_x0.min()), float(r_x0.max()))
    check("image x0 (ViT) overfit reconstructs", mse_x0 < 0.02, f"mse {mse_x0:.4f}")
    check("image x0 (ViT) stays ~in-range [0,1]", -0.2 < rng[0] and rng[1] < 1.2, f"range [{rng[0]:.2f},{rng[1]:.2f}]")

    # IMAGE (U-Net flow head): the conv alternative must ALSO reconstruct (x0, 1-step deterministic).
    r_u = _overfit(ImageUNetFlowHead(cfg, param="x0").to(DEV), ci, ti)
    mse_u = float(F.mse_loss(r_u, ti)); rngu = (float(r_u.min()), float(r_u.max()))
    check("image x0 (U-Net) overfit reconstructs", mse_u < 0.02, f"mse {mse_u:.4f}")
    check("image x0 (U-Net) stays ~in-range [0,1]", -0.2 < rngu[0] and rngu[1] < 1.2, f"range [{rngu[0]:.2f},{rngu[1]:.2f}]")

    # PROPRIO: both should reconstruct (low-dim); x0 near-exact.
    cp, tp = torch.randn(8, 128, device=DEV), torch.randn(8, 6, device=DEV)
    r_p = _overfit(FlowField(dz=6, h_dim=128, hidden=64, param="x0").to(DEV), cp, tp)
    mse_p = float(F.mse_loss(r_p, tp))
    check("proprio x0 overfit reconstructs", mse_p < 0.02, f"mse {mse_p:.4f}")

    print(f"\n{'ALL OK' if all(ok) else 'FAILURES'} ({sum(ok)}/{len(ok)})")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
