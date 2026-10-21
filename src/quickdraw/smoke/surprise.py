"""Smoke for the general OOD surprise tools (evaluation/surprise.py).
  1. image_maps: an OOD blob (obs deviates from the ensemble there) -> surprise_patch high IN the blob, low out.
  2. auc_vs_mask: that surprise vs the blob mask -> AUC ~ 1.
  3. proprio_zscore: a single perturbed obs dim -> its z-score dominates.
  4. model path: surprise_image / surprise_proprio run on a tiny model, right shapes, finite, sigma>0 (stochastic).
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.surprise
"""
import numpy as np, torch
torch.set_num_threads(1)
from ..evaluation.surprise import image_maps, proprio_zscore, auc_vs_mask, surprise_image, surprise_proprio


def _fail(m): print("FAIL:", m); raise SystemExit(1)


def main():
    torch.manual_seed(0)
    H = W = 32
    # ---- 1&2. image: ensemble ~ gray with small spread; obs = gray EXCEPT an OOD blob ----
    mu0 = 0.5
    samples = mu0 + 0.02 * torch.randn(16, H, W, 3)
    obs = torch.full((H, W, 3), mu0)
    mask = torch.zeros(H, W, dtype=torch.bool); mask[8:16, 8:16] = True   # the OOD region
    obs[mask] = 0.95                                                      # strongly off the ensemble there
    maps = image_maps(samples, obs, patch=8)
    sp = maps["surprise_patch"]
    if not (sp[mask].mean() > 5 * sp[~mask].mean()):
        _fail(f"surprise_patch not localized: blob {float(sp[mask].mean()):.2f} vs bg {float(sp[~mask].mean()):.2f}")
    a = auc_vs_mask(sp.numpy(), mask.numpy())
    if not (a > 0.95):
        _fail(f"AUC {a:.3f} not ~1 for a clean OOD blob")
    print(f"[1&2/4] image surprise localizes (blob {float(sp[mask].mean()):.1f} vs bg {float(sp[~mask].mean()):.2f}); AUC {a:.3f}   OK")

    # ---- 3. proprio z-score: one perturbed dim dominates ----
    D = 17
    ps = torch.randn(16, D)
    pobs = ps.mean(0).clone(); pobs[5] += 6.0                            # dim 5 way off the ensemble mean
    z = proprio_zscore(ps, pobs)
    if int(z.argmax()) != 5:
        _fail(f"proprio z-score peak at dim {int(z.argmax())}, expected 5")
    print(f"[3/4] proprio z-score peaks on the perturbed dim (z[5]={float(z[5]):.1f})   OK")

    # ---- 4. model path: shapes + stochastic spread ----
    from hydra import compose, initialize_config_dir
    from ..training.setup import build_model
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        # target_transform=none: mm_flow (main) defaults the action prior's target to `pit`, whose
        # knots need a packed dataset this smoke does not use. longhand's flow config + main's pit
        # default first met in the 2026-10-20 merge. Same pin as smoke/xstep_ahead.
        cfg = compose(config_name="config",
                      overrides=["model=vl128_blockstack_flow",
                                 "model.action_head.target_transform=none"])
    m = build_model(cfg).eval(); m.stochastic_eval = True

    class _IdNorm:                                   # identity norm (no data root needed for the smoke)
        norm_obs = staticmethod(lambda x: x.float())
        norm_act = staticmethod(lambda x: x.float())
        denorm_obs = staticmethod(lambda x: x.float())
    norm = _IdNorm(); P = cfg.data.P
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    head = img_heads[0]
    sh = m.modalities[head].ae.cfg.img_size; sh = sh if isinstance(sh, (tuple, list)) else (sh, sh)
    L = P + 4
    o = np.random.randn(L, m.modalities["proprio"].dim).astype(np.float32)
    a = np.random.randn(L, 5).astype(np.float32)
    fr = {h: (np.random.rand(L, sh[0], sh[1], 3) * 255).astype(np.uint8) for h in img_heads}   # ALL heads for ctx
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev)
    mp, mu = surprise_image(m, norm, o, a, fr, head, P, t=P, n=8, patch=8, device=dev)
    if mp["surprise_patch"].shape != (sh[0], sh[1]):
        _fail(f"image surprise shape {tuple(mp['surprise_patch'].shape)} != {sh}")
    if not float(mp["std"].mean()) > 0:
        _fail("ensemble sigma==0 (stochastic_eval not giving draws)")
    zz = surprise_proprio(m, norm, o, a, fr, P, t=P, n=8, device=dev)
    if zz.shape != (m.modalities["proprio"].dim,):
        _fail(f"proprio surprise shape {tuple(zz.shape)} != (obs_dim,)")
    print(f"[4/4] model path: image map {tuple(mp['surprise_patch'].shape)} sigma>0; proprio z {tuple(zz.shape)}   OK")

    print("\nALL SURPRISE SMOKE PASSED")


if __name__ == "__main__":
    main()
