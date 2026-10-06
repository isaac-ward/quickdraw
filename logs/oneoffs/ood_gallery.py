"""OOD surprise GALLERY: 10 in-distribution (val) + 10 OOD (purple-cube) clips, each 30s @30Hz, rendered in TWO
layouts -> 40 mp4s in logs/ood/gallery/:
  *_2panel.mp4 : Ground truth | Uncertainty
  *_4panel.mp4 : Ground truth | Prediction | Prediction error | Uncertainty
"Uncertainty" = CHROMA error (Lab a,b colour error; isolates the OOD object from the moving arm). "Prediction
error" = |obs - pred|. Prediction is the model's deterministic one-step (=10-frame-ahead) forecast from the TRUE
past (sigma ~0.001 here, so deterministic == ensemble mean, and much faster). 30Hz via the phase trick (predict
every original frame T from phase T%s step T//s).
Also emits traj_separation.png: TRAJECTORY-level detection -- aggregate each clip's per-frame chroma-p99.5 score by
max / p95 / mean -> one number per clip -> IND vs OOD (far more separation than per-frame, since OOD clips only
sometimes show the cube). Clears logs/ood/gallery/ at start.
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/ood_gallery.py <ckpt>
"""
import os, sys, json, glob, dataclasses
import numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm, _subsample_episodes
from quickdraw.evaluation.surprise import _ensemble, chroma_err
from quickdraw.logging.viz import save_mp4, save_gif, label_panels
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, WIN, SCALE = "cam_scene", 900, 4            # 900 orig frames = 30s @30Hz
N_IND_PER_EP = 5                                  # val has 2 long eps -> 5 windows each = 10 IND
OOD_PLAN = [("eval_purple_play", 5), ("eval_purple_stack", 5)]   # 10 OOD
TITLES4 = ["Ground truth", "Prediction", "Prediction error", "Uncertainty"]
TITLES2 = ["Ground truth", "Uncertainty"]
OUT = "logs/ood/gallery"


def purple_mask(f):
    f = f.astype(np.float32) / 255.0; r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); d = mx - f.min(-1) + 1e-8; h = np.zeros_like(mx)
    m = mx == r; h[m] = ((g - b) / d)[m] % 6
    m = (mx == g) & ~(mx == r); h[m] = ((b - r) / d)[m] + 2
    m = (mx == b) & ~(mx == r) & ~(mx == g); h[m] = ((r - g) / d)[m] + 4
    h = (h * 60) % 360; s = d / (mx + 1e-8)
    return (h > 260) & (h < 325) & (s > 0.35) & (mx > 0.25)


def best_ood_window(rawfr, lo):
    """[T0,T1) WIN-long window with the most purple motion (full clip if shorter than WIN)."""
    L = len(rawfr)
    if L - lo <= WIN:
        return lo, L
    cen = []
    for t in range(lo, L):
        ys, xs = np.where(purple_mask(rawfr[t])); cen.append((ys.mean(), xs.mean()) if len(ys) else None)
    disp = np.zeros(L); prev = None
    for k, c in enumerate(cen):
        if c is not None and prev is not None: disp[lo + k] = np.hypot(c[0] - prev[0], c[1] - prev[1])
        if c is not None: prev = c
    cs = np.cumsum(disp)
    best = max(range(lo, L - WIN), key=lambda a: cs[a + WIN] - cs[a])
    return best, best + WIN


def heat(a, hi): return (cm.inferno(np.clip(a / max(hi, 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)


def stats(pos, neg):
    lo = np.sort(neg); r = (np.searchsorted(lo, pos, "left") + np.searchsorted(lo, pos, "right")) / 2
    auc = float(r.mean() / neg.size)
    thr = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 400)
    acc = max(((pos >= t).sum() + (neg < t).sum()) / (pos.size + neg.size) for t in thr)
    bal = max(0.5 * ((pos >= t).mean() + (neg < t).mean()) for t in thr)
    return auc, float(acc), float(bal)


def main():
    ck = sys.argv[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    for f in glob.glob(os.path.join(OUT, "*")):
        try: os.remove(f)
        except OSError: pass
    os.makedirs(OUT, exist_ok=True)
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck)
    m.stochastic_eval = False                          # deterministic one-step (sigma~0.001 -> == ensemble mean, faster)
    norm = normalizer(rc); P = rc.data.P; s = int(rc.data.subsample); lo = P * s
    fps = 1.0 / float(env_cfg(rc).dt)
    base = DataConfig.from_cfg(rc)
    raw_dcfg = dataclasses.replace(base, subsample=1, subsample_all_phases=False)
    ph_dcfg = dataclasses.replace(base, subsample=s, subsample_all_phases=True)

    def load(split, k):
        return load_split_episodes_mm(resolve_data_root(rc), split, dcfg=raw_dcfg,
                                      img_size=image_head_sizes(rc) or 128,
                                      cam=image_head_cams(rc) or rc.data.get("cam", "fpv"),
                                      repo_id=rc.data.get("repo_id", "torus"))[:k]

    # assemble the 20 windows: (name, phase_eps, T0, T1, is_ood)
    windows = []
    vi = 0
    for ep in load("val", 2):
        phe = _subsample_episodes([ep], "clip/train", dcfg=ph_dcfg)
        L = len(ep[0])
        for st in np.linspace(lo, max(lo, L - WIN), N_IND_PER_EP).astype(int):
            windows.append((f"ind_{vi:02d}", phe, int(st), int(st) + WIN, False)); vi += 1
    oi = 0
    for split, k in OOD_PLAN:
        for ep in load(split, k):
            phe = _subsample_episodes([ep], "clip/train", dcfg=ph_dcfg)
            T0, T1 = best_ood_window(ep[2][HEAD], lo)
            windows.append((f"ood_{oi:02d}", phe, T0, T1, True)); oi += 1
    print(f"gallery: {vi} IND + {oi} OOD windows, {WIN/fps:.0f}s @ {fps:.0f}Hz, x{SCALE}", flush=True)

    traj = {}                                          # name -> {"ood":bool, "scores":[per-frame chroma p99.5]}
    for name, phe, T0, T1, ood in windows:
        f2, f4, sc = [], [], []
        for T in range(T0, T1):
            ph, i = T % s, T // s
            pe = phe[ph]
            if i < P or i >= len(pe[0]):
                continue
            samp = _ensemble(m, norm, pe[0], pe[1], pe[2], HEAD, P, i, 1, dev)     # n=1 deterministic
            pred = samp[0]
            gt = pe[2][HEAD][i].astype(np.uint8); pred_u8 = (pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            diff = (torch.from_numpy(gt).float().div(255).to(dev) - pred).abs().mean(-1).cpu().numpy()
            chro = chroma_err(gt, pred_u8)
            sc.append(float(np.percentile(chro, 99.5)))
            ch = heat(chro, 60.0)
            f2.append(label_panels([gt, ch], TITLES2, scale=SCALE))
            f4.append(label_panels([gt, pred_u8, heat(diff, np.percentile(diff, 99)), ch], TITLES4, scale=SCALE))
        if f2:
            save_mp4(os.path.join(OUT, f"{name}_2panel.mp4"), np.stack(f2), fps=fps)
            save_mp4(os.path.join(OUT, f"{name}_4panel.mp4"), np.stack(f4), fps=fps)
            # 2-panel GIF for Google Slides: downscale 0.5 + 10 fps -> well under the 50 MB cap (aim small)
            gstep = max(1, round(fps / 10))
            gf = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA) for f in f2[::gstep]]
            gpath = os.path.join(OUT, f"{name}_2panel.gif"); save_gif(gpath, gf, fps / gstep)
            traj[name] = {"ood": ood, "scores": sc}
            print(f"[{name}] {len(f2)}f ood={ood} | Uncertainty score max {max(sc):.1f} p95 {np.percentile(sc,95):.1f} mean {np.mean(sc):.1f} | gif {os.path.getsize(gpath)/1e6:.1f}MB", flush=True)

    # TRAJECTORY-level separation: aggregate each clip's per-frame scores -> one value/clip -> IND vs OOD
    aggs = {"max": np.max, "p95": lambda a: np.percentile(a, 95), "mean": np.mean}
    ind = [v for v in traj.values() if not v["ood"]]; ood_ = [v for v in traj.values() if v["ood"]]
    res = {}
    fig, axs = plt.subplots(1, len(aggs), figsize=(4.6 * len(aggs), 4.6))
    for ax, (an, af) in zip(axs, aggs.items()):
        iv = np.array([af(v["scores"]) for v in ind]); ov = np.array([af(v["scores"]) for v in ood_])
        auc, acc, bal = stats(ov, iv)
        res[an] = {"auc": auc, "acc": acc, "bal": bal}
        ax.boxplot([iv, ov], tick_labels=["IND", "OOD"], showfliers=False)
        ax.scatter(np.ones(iv.size), iv, c="#000000", s=22, zorder=3)
        ax.scatter(np.full(ov.size, 2), ov, c="#7e2fb0", s=22, zorder=3)
        ax.set_title(f"per-clip {an}\nAUC {auc:.2f}  acc {acc:.2f}  bal {bal:.2f}", fontsize=10)
        ax.set_ylabel(f"Uncertainty score (per-clip {an})"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"TRAJECTORY-level OOD detection ({len(ind)} IND vs {len(ood_)} OOD clips) — Uncertainty score aggregated per clip", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "traj_separation.png"), dpi=120); plt.close(fig)
    json.dump({"trajectory_detection": res,
               "clips": {k: {"ood": v["ood"], "max": float(np.max(v["scores"])),
                             "p95": float(np.percentile(v["scores"], 95)), "mean": float(np.mean(v["scores"]))}
                         for k, v in traj.items()}},
              open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
