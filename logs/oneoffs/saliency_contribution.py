"""How big is the saliency term's actual gradient signal? Load a checkpoint, roll out, and compute the scene
decode loss WITH vs WITHOUT saliency on the SAME predictions, plus the raw cube-pixel L1 over horizon.
Answers: does weighting the cube pixels 8x actually move the decode loss, or is it a rounding error?
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/saliency_contribution.py <ckpt>
"""
import os, sys
import numpy as np, torch
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.models.visual_loss import VisualLoss
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

N_EP, H = 4, 64


def main():
    ckpt = sys.argv[1]
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(rcfg).to(dev).eval(); m = getattr(m, "_orig_mod", m)
    load_checkpoint(m, ckpt)
    norm = normalizer(rcfg); P = rcfg.data.P
    img_heads = [n for n, k in [(mm.get("name"), mm.get("kind")) for mm in rcfg.model.modalities] if k == "image"]
    eps = load_split_episodes_mm(resolve_data_root(rcfg), "val", dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=image_head_sizes(rcfg) or 128,
                                 cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                 repo_id=rcfg.data.get("repo_id", "torus"))[:N_EP]
    Hh = min(H, min(len(o) for o, _, _ in eps) - P - 1)
    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(dev)}
    for hd in img_heads:
        ctx[hd] = torch.stack([torch.from_numpy(fr[hd][:P]) for _, _, fr in eps]).float().div(255).to(dev)
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[:P + Hh - 1])) for _, a, _ in eps]).float().to(dev)
    gt = torch.stack([torch.from_numpy(fr["cam_scene"][P:P + Hh]) for _, _, fr in eps]).float().div(255).to(dev)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        rolled = m._rollout(ctx, acts, Hh, p_tf=0.0, true_future=None, detach_every=0, use_cache=False)
        pred = m.to_obs(rolled, heads=["cam_scene"], commit=True)["cam_scene"].float()   # (B,Hh,H,W,3)

    # two VisualLosses: exactly the recipe's mix, saliency ON (as trained) vs OFF
    def vl(sal):
        return VisualLoss(w_l2=0.0, w_l1=3.0, w_lpips=1.0, saliency=("hue" if sal else "none"),
                          saliency_weight=8.0, saliency_dilate=2).to(dev)
    on, off = vl(True), vl(False)
    with torch.no_grad():
        d_on = float(on(pred, gt, site="decode"))
        d_off = float(off(pred, gt, site="decode"))
    # raw cube-pixel vs whole-frame L1 at horizons, and cube pixel fraction
    B_, T_, Hp, Wp, _ = gt.shape
    wmap = on._saliency_map(gt.reshape(-1, Hp, Wp, 3))[..., 0].reshape(B_, T_, Hp, Wp).cpu().numpy()  # 1+8*mask
    mask = wmap > 1.5
    err = (pred - gt).abs().mean(-1).cpu().numpy()             # (B,Hh,H,W)
    print(f"scene decode loss (recipe mix 3*L1+LPIPS):  saliency ON {d_on:.4f}   OFF {d_off:.4f}   "
          f"delta {d_on-d_off:+.4f}  ({100*(d_on-d_off)/d_off:+.1f}%)")
    print(f"cube pixel fraction (dilated mask): {mask.mean()*100:.1f}%")
    print("  horizon | whole-frame L1 | CUBE-pixel L1 | cube/frame ratio")
    for q in [Hh // 4, Hh // 2, 3 * Hh // 4, Hh - 1]:
        fr = err[:, q].mean()
        cu = err[:, q][mask[:, q]].mean() if mask[:, q].any() else float("nan")
        print(f"   h{q:2d}   |   {fr:.4f}      |   {cu:.4f}    |  {cu/fr:.2f}x")


if __name__ == "__main__":
    main()
