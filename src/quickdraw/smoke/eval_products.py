"""Smoke: the eval PRODUCT path (image_curves -> products.log_error_curves -> viz.fig_error_vs_step).

Exists because a 30-epoch run lost EVERY eval to `TypeError: fig_error_vs_step() got an unexpected keyword
argument 'split_bottom'` -- products.py and viz.py disagreed on a signature. Nothing exercised that seam, so a
one-line change in one file silently disabled all eval products. Run this before launching.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from quickdraw.evaluation.openloop import emit_horizon_readouts, image_curves
from quickdraw.evaluation.products import log_error_curves

OK = [0, 0]


def check(name, cond, extra=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))


class _W:                                    # writer stub: records tags, swallows media
    def __init__(self): self.tags = []
    def figure(self, tag, fig, step): self.tags.append(tag)
    def scalar(self, tag, v, step): self.tags.append(tag)
    def scalars(self, d, step): self.tags.extend(d)


def main():
    torch.manual_seed(0)
    pred = torch.rand(3, 12, 32, 32, 3)
    true = torch.rand(3, 12, 32, 32, 3)
    c = image_curves(pred, true)
    check("image_curves returns psnr/ssim/mse/l1", {"psnr", "ssim", "mse", "l1"} <= set(c), str(sorted(c)))
    check("image_curves includes lpips (fail-soft: absent only if weights unavailable)", "lpips" in c)
    for k, v in c.items():
        check(f"  {k} is finite, len=H", np.isfinite(v).all() and len(v) == 12)

    # the seam that broke: products -> viz, WITH the third panel, at BOTH yscales
    w = _W()
    log_error_curves(w, "eval_ood_horizon/open_loop", c, 1, head="image",
                     split_top={"psnr"}, split_bottom={"lpips"}, colors={"psnr": "red", "lpips": "purple"})
    check("products->viz 3-panel path (linear AND log)", sum("error_vs_step_avg" in t for t in w.tags) == 2,
          f"{sum('error_vs_step_avg' in t for t in w.tags)} figures")
    check("per-metric _mean scalars emitted", all(any(f"{k}_mean" in t for t in w.tags) for k in c))

    # degradation: no lpips (2 panels) and proprio (1 panel) must still work
    w2 = _W()
    log_error_curves(w2, "r", {k: v for k, v in c.items() if k != "lpips"}, 1, head="image",
                     split_top={"psnr"}, split_bottom={"lpips"})
    check("products->viz without lpips", sum("error_vs_step_avg" in t for t in w2.tags) == 2)
    w3 = _W()
    log_error_curves(w3, "r", {"obs_error": np.linspace(1, 0.1, 12)}, 1, head="proprio")
    check("products->viz proprio (no split)", sum("error_vs_step_avg" in t for t in w3.tags) == 2)

    w4 = _W()
    emit_horizon_readouts(w4, "eval_ood_horizon/open_loop", "image", c, 12, 1)
    check("emit_horizon_readouts covers every metric incl. lpips",
          all(any(f"/{k}/@+" in t for t in w4.tags) for k in c), f"{len(w4.tags)} scalars")

    # recon_losses on a BESPOKE trunk (no pretrained AE -> roundtrip_losses has no heads). This is the exact
    # config that broke with "not enough values to unpack": the early return was a bare dict, not a tuple.
    from quickdraw.models.modalities import ModalitySpec
    from quickdraw.models.multimodal import MultiModalFlow
    for tag, spec in (("bespoke", dict(pretrained=False, encode_arch="conv", decode_arch="unet")),
                      ("pretrained", dict(pretrained=True, pretrained_name="madebyollin/taesd",
                                          pretrained_init=True, freeze=True))):
        sp = [ModalitySpec(name="proprio", kind="vector", dim=16, num_tokens=1),
              ModalitySpec(name="image", kind="image", num_tokens=8, img_size=64, patch=16, channels=3, **spec)]
        torch.manual_seed(0)
        m = MultiModalFlow(sp, d=128, depth=2, heads=8, window=8, mlp_ratio=2.0, rope_theta=1e4,
                           action_dim=12, latent_norm="layernorm")
        obs = {"proprio": torch.randn(2, 4, 16), "image": torch.rand(2, 4, 64, 64, 3)}
        bag = m.encode_state(obs)
        try:
            losses, weights = m.recon_losses(bag, obs)
            ok = set(losses) == set(weights) and any(k.startswith("decode/") for k in losses)
        except Exception as e:
            ok, losses = False, {"error": e}
        check(f"recon_losses returns (losses, weights) on a {tag} trunk", ok, str(sorted(losses)))

    print(f"{'ALL OK' if OK[0] == OK[1] else 'SOME FAILED'} ({OK[0]}/{OK[1]})")
    return 0 if OK[0] == OK[1] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
