"""Purple-cube OOD surprise videos at TRUE 30 Hz (no downsampling) via evaluation/surprise.py.
block-stack is 30 Hz native with subsample=10 (model step = 10 frames). To render every original frame: load raw
(subsample=1), build all 10 phase-subsampled views via the loader's own _subsample_episodes (exact action
aggregation), and for each original frame T run the model's 1-step (=10-frame-ahead) prediction from phase T%10,
step T//10. Interleaving the phases in T-order = smooth 30 Hz with the model's native prediction stride.
4 panels (upscaled, crisp centered titles via viz.label_panels): Ground truth | Prediction (ensemble mean) |
|observation - prediction| (all error) | Chroma error (OOD colour, arm removed). The chroma panel (Lab a,b, no
luminance) isolates the OOD colour from the moving arm at AUC(purple>arm) 0.998 -- sigma/z CANNOT (arm sigma ==
purple sigma). No purple outline (unfair). Detection: per-frame pixel (absdiff/chroma x mean/p99.5), ensemble-
VARIANCE (pixel + latent std), and LATENT-surprise metrics -> AUC + acc + balanced acc. Clears OUT at start.
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/purple_surprise.py <ckpt>
"""
import os, sys, json, glob, dataclasses, math
import numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm, _subsample_episodes
from quickdraw.evaluation.surprise import _ensemble, auc_vs_mask, encode_true_latent
from quickdraw.logging.viz import save_mp4, label_panels
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, N_ENS, SCALE = "cam_scene", 16, 4
WIN_VAL, WIN_OOD = 900, 300                       # original frames @30Hz: val 30s, purple 10s (keep punchy)
PLAN = [("val", 2), ("eval_purple_play", 3), ("eval_purple_stack", 1)]
TITLES = ["Ground truth", "Prediction (ensemble mean)", "|observation - prediction|  (all error)", "Chroma error  (OOD colour, arm removed)"]
METRICS = ["absdiff_mean", "absdiff_p99.5", "chroma_mean", "chroma_p99.5", "pixel_std_mean", "latent_absdiff", "latent_std"]
OUT = "logs/ood/purple_surprise"


def purple_mask(f):
    f = f.astype(np.float32) / 255.0; r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); d = mx - f.min(-1) + 1e-8; h = np.zeros_like(mx)
    m = mx == r; h[m] = ((g - b) / d)[m] % 6
    m = (mx == g) & ~(mx == r); h[m] = ((b - r) / d)[m] + 2
    m = (mx == b) & ~(mx == r) & ~(mx == g); h[m] = ((r - g) / d)[m] + 4
    h = (h * 60) % 360; s = d / (mx + 1e-8)
    return (h > 260) & (h < 325) & (s > 0.35) & (mx > 0.25)


def centroid(mk):
    ys, xs = np.where(mk); return np.array([ys.mean(), xs.mean()]) if len(ys) else None


def best_window(rawfr, s, P, ood):
    """[T0,T1) in ORIGINAL frames; needs T0 >= P*s (context) and T1 <= len. OOD: max purple motion."""
    win = WIN_OOD if ood else WIN_VAL
    L = len(rawfr); lo = P * s
    if L - lo <= win:
        return lo, L
    if not ood:
        t0 = (L - win) // 2; return t0, t0 + win
    disp = np.zeros(L); cp = centroid(purple_mask(rawfr[lo]))
    for t in range(lo + 1, L):
        c = centroid(purple_mask(rawfr[t]))
        if c is not None and cp is not None: disp[t] = np.linalg.norm(c - cp)
        if c is not None: cp = c
    cs = np.cumsum(disp)
    best = max(range(lo, L - win), key=lambda a: cs[a + win] - cs[a])
    return best, best + win


def heat(a, hi): return (cm.inferno(np.clip(a / max(hi, 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)


def chroma_err(obs_u8, mu_u8):
    """Prediction error in COLOR only (Lab a,b, luminance discarded). Measured: isolates the OOD purple cube from
    the moving arm at AUC(purple>arm) 0.998 -- the arm error is luminance/position (same grey arm, shifted edges),
    the purple error is chroma (model paints purple as red/orange). No purple knowledge baked in -- any wrong hue."""
    lo = cv2.cvtColor(obs_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lm = cv2.cvtColor(mu_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.linalg.norm(lo[..., 1:] - lm[..., 1:], axis=-1)


def main():
    ck = sys.argv[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    for f in glob.glob(os.path.join(OUT, "*")):          # CLEAR the folder on restart
        try: os.remove(f)
        except OSError: pass
    os.makedirs(OUT, exist_ok=True)
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck); m.stochastic_eval = True
    norm = normalizer(rc); P = rc.data.P; s = int(rc.data.subsample)
    fps = 1.0 / float(env_cfg(rc).dt)                    # TRUE native rate (30 Hz) -- every original frame
    base = DataConfig.from_cfg(rc)
    raw_dcfg = dataclasses.replace(base, subsample=1, subsample_all_phases=False)
    ph_dcfg = dataclasses.replace(base, subsample=s, subsample_all_phases=True)
    print(f"30Hz render: native {fps:.1f}Hz, subsample {s}, val {WIN_VAL/fps:.0f}s / purple {WIN_OOD/fps:.0f}s, upscale x{SCALE}", flush=True)

    scores = {met: {} for met in METRICS}; results = {}
    for split, k in PLAN:
        ood = split != "val"
        raw = load_split_episodes_mm(resolve_data_root(rc), split, dcfg=raw_dcfg,
                                     img_size=image_head_sizes(rc) or 128,
                                     cam=image_head_cams(rc) or rc.data.get("cam", "fpv"),
                                     repo_id=rc.data.get("repo_id", "torus"))[:k]
        for ei, ep in enumerate(raw):
            phase_eps = _subsample_episodes([ep], "clip/train", dcfg=ph_dcfg)    # s phases, in order 0..s-1
            rawfr = ep[2][HEAD]
            T0, T1 = best_window(rawfr, s, P, ood)
            moved = 0.0; cp = centroid(purple_mask(rawfr[T0]))
            for T in range(T0 + 1, T1):
                c = centroid(purple_mask(rawfr[T]))
                if c is not None and cp is not None: moved += np.linalg.norm(c - cp)
                if c is not None: cp = c
            tag = f"{split}_ep{ei}" + ("_MOVING" if ood and moved > 20 else "")
            frames, aucs = [], []
            for T in range(T0, T1):
                ph, i = T % s, T // s
                pe = phase_eps[ph]
                if i < P or i >= len(pe[0]):
                    continue
                samp, pred_bag = _ensemble(m, norm, pe[0], pe[1], pe[2], HEAD, P, i, N_ENS, dev, return_bag=True)
                z_true = encode_true_latent(m, norm, pe[0], pe[2], i, dev)        # (n_tok, d) true latent at step i
                obs = torch.from_numpy(pe[2][HEAD][i]).float().div(255.0).to(dev)
                mu = samp.mean(0)
                gt = pe[2][HEAD][i].astype(np.uint8); mu_u8 = (mu.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                diff = (obs - mu).abs().mean(-1).cpu().numpy()                    # all error (motion + color)
                chro = chroma_err(gt, mu_u8)                                      # color error (isolates OOD, arm gone)
                pix_std = float(samp.std(0).mean())                               # ensemble VARIANCE of draws (pixels)
                lat_absdiff = float((z_true - pred_bag.mean(0)).norm(dim=-1).mean())   # latent surprise ||z_true - mean pred||
                lat_std = float(pred_bag.std(0).norm(dim=-1).mean())             # ensemble VARIANCE of draws (latent)
                for met, v in (("absdiff_mean", diff.mean()), ("absdiff_p99.5", np.percentile(diff, 99.5)),
                               ("chroma_mean", chro.mean()), ("chroma_p99.5", np.percentile(chro, 99.5)),
                               ("pixel_std_mean", pix_std), ("latent_absdiff", lat_absdiff), ("latent_std", lat_std)):
                    scores[met].setdefault(split, []).append(float(v))
                pm = purple_mask(gt)
                if ood and pm.sum() >= 8: aucs.append(auc_vs_mask(chro, pm))      # localization now on the chroma map
                frames.append(label_panels([gt, mu_u8, heat(diff, np.percentile(diff, 99)), heat(chro, 60.0)], TITLES, scale=SCALE))
            if frames:
                save_mp4(os.path.join(OUT, f"v_{tag}.mp4"), np.stack(frames), fps=fps)
                if ood: results[tag] = {"chroma_localization_auc": float(np.nanmean(aucs)) if aucs else float("nan"), "purple_moved_px": float(moved)}
                print(f"[{tag}] {len(frames)}f @30Hz {frames[0].shape[1]}x{frames[0].shape[0]} | purple moved {moved:.0f}px" + (f" | chroma loc-AUC {np.nanmean(aucs):.3f}" if ood and aucs else ""), flush=True)

    # detection: for EACH metric, AUC + best-threshold acc + BALANCED acc (val=negatives, purple=positives)
    def stats(pos, neg):
        lo = np.sort(neg); r = (np.searchsorted(lo, pos, "left") + np.searchsorted(lo, pos, "right")) / 2
        auc = float(r.mean() / neg.size)
        thr = np.linspace(min(neg.min(), pos.min()), max(neg.max(), pos.max()), 300)
        acc = max(((pos >= t).sum() + (neg < t).sum()) / (pos.size + neg.size) for t in thr)              # best plain acc
        bal = max(0.5 * ((pos >= t).mean() + (neg < t).mean()) for t in thr)                              # best balanced acc
        return {"detection_auc": auc, "best_acc": float(acc), "best_balanced_acc": float(bal)}

    oods = [s2 for s2, _ in PLAN if s2 != "val"]
    dets = {}
    ncol = 3; nrow = math.ceil(len(METRICS) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axs = axs.ravel()
    for ax, met in zip(axs, METRICS):
        val = np.array(scores[met]["val"])
        ax.boxplot([val] + [np.array(scores[met][s2]) for s2 in oods],
                   tick_labels=["val(ID)"] + [s2.replace("eval_purple_", "purple-") for s2 in oods], showfliers=False)
        dets[met] = {s2: stats(np.array(scores[met][s2]), val) for s2 in oods}
        ax.set_ylabel(met); ax.grid(axis="y", alpha=0.3)
        ax.set_title("\n".join(f"{s2.replace('eval_purple_','')}: AUC {dets[met][s2]['detection_auc']:.2f}  acc {dets[met][s2]['best_acc']:.2f}  bal {dets[met][s2]['best_balanced_acc']:.2f}" for s2 in oods), fontsize=9)
    for ax in axs[len(METRICS):]:
        ax.axis("off")
    fig.suptitle("OOD detection (val vs purple) by per-frame score metric — pixel / variance / latent", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "id_vs_ood.png"), dpi=120); plt.close(fig)
    results["detection"] = dets
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
