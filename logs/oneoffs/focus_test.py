"""Pick a surprise-focus map that isolates the OOD purple cube from general (arm) movement. One purple frame,
compute the ensemble once, derive candidate maps, render side-by-side with the purple-mask contour.
  a) surprise (z)           = |obs-mu|/(sigma+eps)            (current; fires on arm + purple)
  b) confident-error        = |obs-mu| gated where sigma LOW  (arm = model-uncertain -> suppressed)
  c) local-contrast         = surprise / blur(surprise)       (compact purple pops over diffuse arm; per-frame self-baseline, no episode mismatch)
  d) absdiff                = |obs-mu|
Out: logs/ood/purple_surprise/focus_test.png
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/focus_test.py <ckpt>
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.evaluation.surprise import _ensemble
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, N = "cam_scene", 48


def purple_mask(f):
    f = f.astype(np.float32) / 255.0; r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); d = mx - f.min(-1) + 1e-8; h = np.zeros_like(mx)
    m = mx == r; h[m] = ((g - b) / d)[m] % 6
    m = (mx == g) & ~(mx == r); h[m] = ((b - r) / d)[m] + 2
    m = (mx == b) & ~(mx == r) & ~(mx == g); h[m] = ((r - g) / d)[m] + 4
    h = (h * 60) % 360; s = d / (mx + 1e-8)
    return (h > 260) & (h < 325) & (s > 0.35) & (mx > 0.25)


def blur(x, s): return cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)


def main():
    ck = sys.argv[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck); m.stochastic_eval = True
    norm = normalizer(rc); P = rc.data.P
    eps = load_split_episodes_mm(resolve_data_root(rc), "eval_purple_play", dcfg=DataConfig.from_cfg(rc),
                                 img_size=image_head_sizes(rc) or 128, cam=image_head_cams(rc) or "fpv",
                                 repo_id=rc.data.get("repo_id", "torus"))[:1]
    o, a, fr = eps[0]
    t = len(o) // 2                                            # a mid-episode frame
    samp = _ensemble(m, norm, o, a, fr, HEAD, P, t, N, dev)    # (N,H,W,3)
    obs = torch.from_numpy(fr[HEAD][t]).float().div(255.0).to(dev)
    mu, sd = samp.mean(0), samp.std(0)
    diff = (obs - mu).abs().mean(-1).cpu().numpy()
    sdm = sd.mean(-1).cpu().numpy()
    sur = (diff / (sdm + 1e-3))
    sur_p = F.avg_pool2d(torch.from_numpy(sur)[None, None], 8, 1, 4)[0, 0].numpy()[:sur.shape[0], :sur.shape[1]]
    conf = diff * (sdm < np.median(sdm))                       # confident-error: low-sigma regions only
    conf_p = F.avg_pool2d(torch.from_numpy(conf)[None, None], 8, 1, 4)[0, 0].numpy()[:sur.shape[0], :sur.shape[1]]
    local = sur_p / (blur(sur_p, 12) + 1e-3)                   # local contrast (per-frame self-baseline)
    pm = purple_mask(fr[HEAD][t])

    cands = [("GT", fr[HEAD][t].astype(np.float32) / 255.0, None), ("pred mu", mu.cpu().numpy(), None),
             ("a) surprise z", sur_p, "inferno"), ("b) confident-err", conf_p, "inferno"),
             ("c) local-contrast", local, "inferno"), ("d) absdiff", diff, "inferno")]
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.2))
    ys, xs = np.where(pm)
    for k, (ttl, img, cmap) in enumerate(cands):
        if cmap:
            ax[k].imshow(img, cmap=cmap, vmax=np.percentile(img, 99))
            if len(ys):
                ax[k].contour(pm, levels=[0.5], colors="cyan", linewidths=0.6)   # purple region outline
        else:
            ax[k].imshow(np.clip(img, 0, 1))
        ax[k].set_title(ttl, fontsize=9); ax[k].axis("off")
        # AUC of this map vs purple mask
        if cmap:
            fv = img.ravel(); mk = pm.ravel()
            if mk.sum() >= 8:
                lo = np.sort(fv[~mk]); r = (np.searchsorted(lo, fv[mk], "left") + np.searchsorted(lo, fv[mk], "right")) / 2
                _au=float(r.mean()/(~mk).sum()); ax[k].set_xlabel(f"AUC {_au:.3f}", fontsize=9); print(f"  {ttl:18s} AUC {_au:.3f}", flush=True)
    fig.suptitle("focus candidates vs purple mask (cyan outline) — eval_purple_play")
    fig.tight_layout(); os.makedirs("logs/ood/purple_surprise", exist_ok=True)
    fig.savefig("logs/ood/purple_surprise/focus_test.png", dpi=120); plt.close(fig)
    print("[focus_test] wrote focus_test.png", flush=True)


if __name__ == "__main__":
    main()
