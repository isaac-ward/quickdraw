"""OOD visual-anomaly detection via per-head one-step re-grounded VoE (Violation of Expectation).

The world model is used as an anomaly detector. VoE = one-step re-grounded prediction error: at each step t the
model predicts obs[t] from the true context obs[t-P:t] (+ actions), and we score how wrong it is. Nominal (the
base-ISS training scene) -> low; OOD (an extra vehicle berthed in the FPV background) -> high. Computed PER HEAD:
the IMAGE head is the detector; PROPRIO is a NEGATIVE CONTROL (divergence check: proprio identical nom-vs-OOD).

PER-EPISODE ISS-VISIBLE GATING: most of the approach is empty space/Earth (nothing to predict). A model-free
"structure" detector (fraction of dark + strongly-edged pixels -> fires on the metallic station, not on bright
clouds or black space) marks, per frame, whether the ISS is in view. We evaluate the reduction ONLY over each
episode's visible slice, so each timeseries line starts/ends where the ISS enters/leaves (not 0..end).

MULTIPLE STATISTICS: per-episode scalar (what conformal calibrates against) is reported for max / mean / median /
p90 of the gated per-step errors. The conformal THRESHOLD per statistic is the 90th percentile of the calibration
episodes' scalars (~10% FPR). Max should barely move under gating (empty frames were already low-error); mean/
median are where gating should pay off if the vehicle adds a consistent small error across the visible window.

Design (see wizard/records/owm.md): nominal = dock-success-100ep (base ISS) restricted to 4 matched ports;
OOD = dock-attempts-by-port cygnus @ {harmony_fwd_pma2, rassvet_nadir, zvezda_aft}, dragon @ harmony_zenith_cbm.
Calibrate on ~34 matched nominals, test on 17 nominal + 30 balanced OOD (15 cygnus, 15 dragon). Numeric only.

Outputs (under --out, use a timestamped dir): results.json, scores.npz, per head timeseries_<head>.png
(gated per-step VoE over approach time; black=nominal / orange=cygnus / red=dragon) and scores_<head>.png
(one score strip per statistic, each with its threshold + TPR/FPR/AUC).

Run in the app container, e.g.:
  TS=$(date +%Y%m%d_%H%M); docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run python -m \
    quickdraw.scripts.ood_voe --run <run> --ckpt last --out logs/ood/voe_$TS
"""
import argparse, json, os
import numpy as np
import torch
from omegaconf import OmegaConf

NOMINAL_ROOT = "logs/recorded_hf/owm-iss-numerical-dock-success-100ep"
ATTEMPTS_ROOT = "logs/recorded_hf/owm-iss-numerical-dock-attempts-by-port-20ep"
OOD_SOURCES = [("cygnus", "harmony_fwd_pma2"), ("cygnus", "rassvet_nadir"), ("cygnus", "zvezda_aft"),
               ("dragon", "harmony_zenith_cbm")]
N_OOD_PER_SHIP, N_NOM_TEST, FPR_Q, RAW_DT = 15, 17, 90, 0.05
STATS = {"max": np.max, "min": np.min, "mean": np.mean, "median": np.median, "p90": lambda a: np.percentile(a, 90)}
NOM_C, SHIP_COLOR = "black", {"cygnus": "#e8820c", "dragon": "#d62728"}
SEED = 0


def load_model(run_dir, ckpt_arg, device):
    cfg = OmegaConf.create(json.load(open(os.path.join(run_dir, "logs", "config.json"))))
    from quickdraw.training.setup import build_model, load_checkpoint, normalizer, env_cfg
    from quickdraw.data.dataset import set_action_aggregate, set_subsample, set_obs_keep
    from quickdraw.environments.registry import make_env
    set_subsample(int(cfg.data.get("subsample", 1) or 1))
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))   # beside the stride: same one-shot rule
    set_obs_keep(cfg.data.get("obs_keep", None))
    model = build_model(cfg).to(device)
    ckpt = run_dir if ckpt_arg in (None, "best") else (
        os.path.join(run_dir, "checkpoints", "last.ckpt") if ckpt_arg == "last" else ckpt_arg)
    load_checkpoint(model, ckpt)
    model.eval()
    m = getattr(model, "_orig_mod", model)
    return cfg, m, normalizer(cfg), make_env(cfg.environments.get("name", "recorded"), cfg.environments, 1, "cpu")


def frame_structure(im):
    """Model-free ISS-visible score for a frame (H,W,3 uint8): fraction of dark + strongly-edged pixels.
    Fires on dark metallic station structure; NOT on bright clouds/Earth or edgeless black space."""
    g = im.astype(np.float32).mean(-1) / 255.0
    grad = np.abs(g[:-1, 1:] - g[:-1, :-1]) + np.abs(g[1:, :-1] - g[:-1, :-1])   # (H-1,W-1) edge magnitude
    dark = g[:-1, :-1] < 0.55                                                    # exclude bright cloud/Earth
    return float(((grad > 0.06) & dark).mean())


def head_slices(m):
    """Token-range per head in the latent bag, from m.layout = [(name, n_tokens), ...]."""
    sl, c = {}, 0
    for name, n in m.layout:
        sl[name] = (c, c + n)
        c += n
    return sl


def spatial_pool(em, pool, frac):
    """Pool a per-pixel error/variance map (B,H,W) to (B,). mean buries a small localized anomaly; max / top-k%
    keep it (the standard localized-anomaly pooling)."""
    if pool == "mean":
        return em.mean(dim=(-2, -1))
    if pool == "max":
        return em.amax(dim=(-2, -1))
    flat = em.flatten(1)                                          # topk
    k = max(1, int(flat.shape[1] * frac))
    return flat.topk(k, dim=1).values.mean(dim=1)


@torch.no_grad()
def encode_frames(m, norm, batch_o, batch_im, img_heads, slices, device, dev_type):
    """Encode a batch of single frames -> {head: (B, n_tokens*d) flattened latent vector} (anchor=None -> absolute,
    consistent between reference and test)."""
    ac = torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda"))
    actual = {"proprio": norm.norm_obs(torch.from_numpy(batch_o)[:, None].float()).to(device)}
    for h in img_heads:
        actual[h] = torch.from_numpy(batch_im)[:, None].float().div(255.0).to(device)
    with ac:
        bag = m.encode_state(actual, None)                                    # (B,1,n_state,d)
    out = {}
    for h in ["proprio"] + img_heads:
        aa, bb = slices[h]
        out[h] = bag[:, 0, aa:bb, :].reshape(len(batch_o), -1).float()        # (B, n_tok*d)
    return out


@torch.no_grad()
def build_reference(m, norm, eps, P, img_heads, slices, device, dev_type, max_refs=4000):
    """Nominal latent-manifold reference cloud {head: (N,D)} for --metric manifold (kNN OOD)."""
    heads = ["proprio"] + img_heads
    acc = {h: [] for h in heads}
    per_ep = max(1, max_refs // max(1, len(eps)))
    for o, a, im in eps:
        idx = np.linspace(P, len(o) - 1, per_ep).astype(int)
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]
            v = encode_frames(m, norm, np.stack([o[t] for t in b]), np.stack([im[t] for t in b]),
                              img_heads, slices, device, dev_type)
            for h in heads:
                acc[h].append(v[h])
    return {h: torch.cat(acc[h], 0) for h in heads}


@torch.no_grad()
def episode_voe(m, norm, ep, P, img_heads, pos, device, dev_type, dt_eff, metric, K, slices, pool, topk_frac,
                ref=None, knn=10, target_pts=400, bs=32):
    """Per sampled step, one of four OOD signals (returns {head:(n,)err}, (n,)times_s, (n,)vis). Image error maps
    are spatially pooled by `pool` (mean/max/topk) so a small localized novelty survives; proprio is position-L2:
      pixel    : one-step re-grounded PREDICTION error (dynamics)   image=pooled pixel-MSE map, proprio=pos-L2
      ae       : encode->decode RECONSTRUCTION error (no dynamics)  image=pooled pixel-MSE map, proprio=pos-L2
      latent   : ||predicted-next bag - encode(actual)|| per head's token slice (deterministic; pool n/a)
      variance : across-K spread of K stochastic predictions        image=pooled var map, proprio=pos std over K."""
    o, a, im = ep
    T = len(o)
    stride = max(1, (T - P) // target_pts)
    ts = list(range(P, T, stride))
    heads = ["proprio"] + img_heads
    errs = {h: [] for h in heads}
    step_bs = bs if metric != "variance" else max(1, bs // K)
    ac = lambda: torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda"))
    for i in range(0, len(ts), step_bs):
        batch = ts[i:i + step_bs]
        ctx = {"proprio": norm.norm_obs(torch.stack([torch.from_numpy(o[t - P:t]) for t in batch]).float()).to(device)}
        for h in img_heads:
            ctx[h] = torch.stack([torch.from_numpy(im[t - P:t]) for t in batch]).float().div(255.0).to(device)
        acts = norm.norm_act(torch.stack([torch.from_numpy(a[t - P:t]) for t in batch]).float()).to(device)

        if metric in ("pixel", "ae"):
            true = torch.stack([torch.from_numpy(im[t]) for t in batch]).float().div(255.0).to(device)  # (B,H,W,3)
            with ac():
                if metric == "pixel":
                    out = m.imagine_eval(ctx, acts, 1, heads=heads)                       # dynamics prediction
                else:
                    anchor = m.rel_anchor(ctx) if m._rel_on() else None                   # encode->decode actual (no dynamics)
                    actual = {"proprio": norm.norm_obs(torch.from_numpy(np.stack([o[t] for t in batch]))[:, None].float()).to(device)}
                    for h in img_heads:
                        actual[h] = true[:, None]
                    out = {k: v for k, v in m.to_obs(m.encode_state(actual, anchor), heads=heads, anchor=anchor).items()}
            p_hat = norm.denorm_obs(out["proprio"])[:, 0].float().cpu().numpy()
            p_true = np.stack([o[t] for t in batch])
            errs["proprio"].extend(np.linalg.norm((p_hat - p_true)[:, pos], axis=1).tolist())
            for h in img_heads:
                em = ((out[h][:, 0].clamp(0, 1).float() - true) ** 2).sum(dim=-1)          # (B,H,W) error map
                errs[h].extend(spatial_pool(em, pool, topk_frac).cpu().numpy().tolist())

        elif metric == "latent":                                   # deterministic bag prediction vs encoded actual
            anchor = m.rel_anchor(ctx) if m._rel_on() else None
            actual = {"proprio": norm.norm_obs(torch.from_numpy(np.stack([o[t] for t in batch]))[:, None].float()).to(device)}
            for h in img_heads:
                actual[h] = torch.from_numpy(np.stack([im[t] for t in batch]))[:, None].float().div(255.0).to(device)
            with ac():
                bag_pred = m._rollout(ctx, acts, 1, 0.0, None, 0, anchor=anchor)          # (B,1,n_state,d)
                bag_true = m.encode_state(actual, anchor)                                 # (B,1,n_state,d)
            diff = ((bag_pred.float() - bag_true.float()) ** 2)[:, 0]                     # (B,n_state,d)
            for h in heads:
                aa, bb = slices[h]
                errs[h].extend(diff[:, aa:bb, :].mean(dim=(-1, -2)).cpu().numpy().tolist())

        elif metric == "variance":                                 # spread over K stochastic samples
            ctxK = {k: v.repeat_interleave(K, dim=0) for k, v in ctx.items()}
            actsK = acts.repeat_interleave(K, dim=0)
            with ac():
                out = m.imagine_eval(ctxK, actsK, 1, heads=heads)
            B = len(batch)
            ph = norm.denorm_obs(out["proprio"])[:, 0].float().reshape(B, K, -1)[..., pos]    # (B,K,3)
            errs["proprio"].extend(np.sqrt(ph.var(dim=1).mean(dim=-1).cpu().numpy()).tolist())  # position std over K (m)
            for h in img_heads:
                pk = out[h][:, 0].clamp(0, 1).float().reshape(B, K, *out[h].shape[2:])         # (B,K,H,W,3)
                vm = pk.var(dim=1).sum(dim=-1)                                                  # (B,H,W) variance map
                errs[h].extend(spatial_pool(vm, pool, topk_frac).cpu().numpy().tolist())

        elif metric == "manifold":                                 # kNN distance of frame latent to nominal cloud
            v = encode_frames(m, norm, np.stack([o[t] for t in batch]), np.stack([im[t] for t in batch]),
                              img_heads, slices, device, dev_type)
            for h in heads:
                d = torch.cdist(v[h], ref[h])                                                   # (B, N_ref)
                errs[h].extend(d.topk(min(knn, d.shape[1]), dim=1, largest=False).values.mean(dim=1).cpu().numpy().tolist())
    vis = np.array([frame_structure(im[t]) for t in ts])
    return {h: np.array(v) for h, v in errs.items()}, np.array(ts) * dt_eff, vis


def score_episodes(m, norm, eps, P, img_heads, pos, device, dev_type, dt_eff, vis_thresh, metric, K, slices,
                   pool, topk_frac, ref, knn, target_pts, tag):
    """Gate each episode to its ISS-visible slice (vis > vis_thresh; fallback to all if <3 visible), then store
    per-stat per-head scalars + the gated per-step curve. Returns (list of {stat:{head:scalar}}, curves list)."""
    heads = ["proprio"] + img_heads
    scores, curves = [], []
    for k, ep in enumerate(eps):
        v, times, vis = episode_voe(m, norm, ep, P, img_heads, pos, device, dev_type, dt_eff, metric, K, slices,
                                    pool, topk_frac, ref, knn, target_pts)
        gate = vis > vis_thresh
        if gate.sum() < 3:
            gate = np.ones_like(gate, dtype=bool)
        sc = {s: {h: float(fn(v[h][gate])) for h in heads} for s, fn in STATS.items()}
        scores.append(sc)
        frac = (times[gate].min() / times.max(), times[gate].max() / times.max()) if times.max() > 0 else (0, 1)
        curves.append({"times": times[gate], **{h: v[h][gate] for h in heads}, "gate_frac": frac, **tag})
        print(f"    ep {k+1}/{len(eps)}: gate=[{frac[0]:.2f},{frac[1]:.2f}] "
              + ", ".join(f"{h}:max={sc['max'][h]:.3g}/mean={sc['mean'][h]:.3g}" for h in heads), flush=True)
    return scores, curves


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(neg) * len(pos))


def plot_timeseries(head, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in curves:                                        # OOD drawn transparent so black nominal shows through
        nom = c["kind"] == "nominal"
        ax.plot(c["times"], c[head], color=(NOM_C if nom else SHIP_COLOR.get(c["ship"], "#d62728")),
                lw=0.7, alpha=(0.6 if nom else 0.3), zorder=(2 if nom else 1))
    if head == "image":
        ax.set_yscale("log")
    ax.set_xlabel("approach time (s)  [each line gated to its ISS-visible slice]")
    ax.set_ylabel(f"{head} one-step VoE" + ("  (pixel MSE)" if head == "image" else "  (position L2, m)"))
    ax.set_title(f"{head} VoE over the ISS-visible approach window")
    ax.legend(handles=[Line2D([], [], color=NOM_C, label="nominal"),
                       Line2D([], [], color=SHIP_COLOR["cygnus"], label="cygnus (OOD)"),
                       Line2D([], [], color=SHIP_COLOR["dragon"], label="dragon (OOD)")], loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"timeseries_{head}.png"), dpi=90)
    plt.close(fig)


def plot_scores(head, scores_by_kind, res_head, out):
    """One score strip per statistic (nominal / cygnus / dragon), each with its threshold + TPR/FPR/AUC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(STATS), figsize=(4.5 * len(STATS), 5.5), sharex=True)
    rng = np.random.default_rng(0)
    for ax, stat in zip(axes, STATS):
        cats = [("nominal", NOM_C, scores_by_kind["nominal"]), ("cygnus", SHIP_COLOR["cygnus"], scores_by_kind["cygnus"]),
                ("dragon", SHIP_COLOR["dragon"], scores_by_kind["dragon"])]
        for x, (name, col, byship) in enumerate(cats):
            ys = [d[stat][head] for d in byship]
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(ys)), ys, s=22, color=col, alpha=0.8, edgecolor="none")
        r = res_head[stat]
        ax.axhline(r["threshold"], color="#1f77b4", ls="--", lw=1.6)
        if head == "image":
            ax.set_yscale("log")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["nom", "cyg", "drg"])
        ax.set_title(f"{stat}\nTPR={r['tpr']:.2f}@FPR={r['fpr']:.2f} AUC={r['auc_test']:.2f}", fontsize=10)
    axes[0].set_ylabel(f"{head} per-episode score")
    fig.suptitle(f"{head} VoE per-episode score by statistic (threshold = 90th-pct of calib nominal)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"scores_{head}.png"), dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--out", default="logs/ood/voe")
    ap.add_argument("--vis-thresh", type=float, default=0.02, help="ISS-visible structure gate (0/neg => no gating)")
    ap.add_argument("--metric", default="pixel", choices=["pixel", "ae", "latent", "variance", "manifold"])
    ap.add_argument("--knn", type=int, default=10, help="k nearest neighbours for --metric manifold")
    ap.add_argument("--pool", default="topk", choices=["mean", "max", "topk"], help="spatial pooling of image error map")
    ap.add_argument("--topk-frac", type=float, default=0.01, help="top-k fraction of pixels for --pool topk")
    ap.add_argument("--k", type=int, default=8, help="samples for --metric variance")
    ap.add_argument("--target-pts", type=int, default=400, help="approx sampled steps per episode")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    from quickdraw.data.dataset import load_split_episodes_mm
    from quickdraw.evaluation.routines import _pos_idx
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_type = "cuda" if device == "cuda" else "cpu"
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cfg, m, norm, env = load_model(args.run, args.ckpt, device)
    P = int(cfg.data.P)
    dt_eff = RAW_DT * int(cfg.data.get("subsample", 1) or 1)
    img_size = next((md.ae.cfg.img_size for md in m.modalities.values() if hasattr(md, "ae")), 128)
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    heads = ["proprio"] + img_heads
    pos, pos_explicit = _pos_idx(cfg, env=env)
    pos = list(pos) if pos_explicit else [0, 1, 2]
    slices = head_slices(m)
    m.stochastic_eval = (args.metric == "variance")   # instance attr (not a file edit): det for pixel/latent, stochastic for variance
    print(f"[ood_voe] device={device} P={P} dt_eff={dt_eff}s metric={args.metric} pool={args.pool}(frac={args.topk_frac}) "
          f"K={args.k} stochastic_eval={m.stochastic_eval} vis_thresh={args.vis_thresh} heads={heads} pos={pos} slices={slices}", flush=True)

    def load(root):
        # frames come back as a DICT keyed by camera (data/dataset.py). This script is single-camera, so
        # flatten to the (obs, act, frames) triples its helpers expect rather than threading a key through.
        eps = load_split_episodes_mm(root, "rollout", img_size=img_size, cam="fpv", repo_id="owm")
        return [(o, a, fr["fpv"]) for o, a, fr in eps]

    ood, dock_ref = {}, {}
    for ship, port in OOD_SOURCES:
        eps = load(os.path.join(ATTEMPTS_ROOT, ship, port))
        if args.smoke:
            eps = eps[:args.smoke]
        ood[(ship, port)] = eps
        dock_ref[port] = np.mean([e[0][-1][pos] for e in eps], axis=0)
        print(f"[ood_voe] OOD {ship}/{port}: {len(eps)} eps", flush=True)

    nom_all = load(NOMINAL_ROOT)
    nom_matched, nom_labels = [], []
    for e in nom_all:
        p, d = min(((p, np.linalg.norm(e[0][-1][pos] - r)) for p, r in dock_ref.items()), key=lambda x: x[1])
        if d < 5.0:
            nom_matched.append(e)
            nom_labels.append(p)
    from collections import Counter
    print(f"[ood_voe] matched nominal: {len(nom_matched)} {dict(Counter(nom_labels))}", flush=True)
    if args.smoke:
        nom_matched, nom_labels = nom_matched[:args.smoke], nom_labels[:args.smoke]

    ref = None
    if args.metric == "manifold":                              # HELD-OUT reference (no leakage): build the nominal
        half = max(1, len(nom_matched) // 2)                   # manifold from one half, score the OTHER half + OOD
        ref_eps, nom_matched, nom_labels = nom_matched[:half], nom_matched[half:], nom_labels[half:]
        ref = build_reference(m, norm, ref_eps, P, img_heads, slices, device, dev_type)
        print(f"[ood_voe] manifold: reference from {len(ref_eps)} nominal eps ("
              + ", ".join(f"{h}={tuple(ref[h].shape)}" for h in ref)
              + f"); scoring {len(nom_matched)} HELD-OUT nominal + OOD (no leakage)", flush=True)

    print("[ood_voe] scoring nominal ...", flush=True)
    nom_sc, curves = score_episodes(m, norm, nom_matched, P, img_heads, pos, device, dev_type, dt_eff, args.vis_thresh,
                                    args.metric, args.k, slices, args.pool, args.topk_frac, ref, args.knn, args.target_pts, {"kind": "nominal"})
    ood_sc, ood_ships = [], []
    for (ship, port), eps in ood.items():
        print(f"[ood_voe] scoring OOD {ship}/{port} ...", flush=True)
        s, cs = score_episodes(m, norm, eps, P, img_heads, pos, device, dev_type, dt_eff, args.vis_thresh,
                               args.metric, args.k, slices, args.pool, args.topk_frac, ref, args.knn, args.target_pts,
                               {"kind": "ood", "ship": ship, "port": port})
        ood_sc.extend(s)
        ood_ships.extend([ship] * len(eps))
        curves.extend(cs)
    ood_ships = np.array(ood_ships)

    # calibration/test split (nominal) + balanced OOD test
    idx = rng.permutation(len(nom_matched))
    n_test = max(1, len(idx) // 2) if args.smoke else min(N_NOM_TEST, len(idx) // 2)
    test_i, calib_i = idx[:n_test], idx[n_test:]
    ood_test_i = []
    for ship in ("cygnus", "dragon"):
        ii = np.where(ood_ships == ship)[0]
        ood_test_i.extend(ii if args.smoke else ii[:N_OOD_PER_SHIP])
    ood_test_i = np.array(ood_test_i)

    def arr(scorelist, stat, head, sel=None):
        L = [scorelist[i][stat][head] for i in (sel if sel is not None else range(len(scorelist)))]
        return np.array(L)

    result = {"run": args.run, "ckpt": args.ckpt, "metric": args.metric, "pool": args.pool, "topk_frac": args.topk_frac,
              "K": args.k, "vis_thresh": args.vis_thresh, "P": P, "n_calib": len(calib_i), "n_nom_test": len(test_i),
              "n_ood_test": len(ood_test_i), "heads": {}}
    for h in heads:
        result["heads"][h] = {}
        for stat in STATS:
            thr = float(np.percentile(arr(nom_sc, stat, h, calib_i), FPR_Q))
            nom_t, ood_t = arr(nom_sc, stat, h, test_i), arr(ood_sc, stat, h, ood_test_i)
            per_ship = {s: float((arr(ood_sc, stat, h, ood_test_i)[ood_ships[ood_test_i] == s] > thr).mean())
                        for s in ("cygnus", "dragon")}
            result["heads"][h][stat] = {
                "threshold": thr, "fpr": float((nom_t > thr).mean()), "tpr": float((ood_t > thr).mean()),
                "auc_test": auc(nom_t, ood_t), "auc_all": auc(arr(nom_sc, stat, h), arr(ood_sc, stat, h)),
                "tpr_per_ship": per_ship}
        print(f"[ood_voe] {h}: " + " | ".join(
            f"{s} AUC={result['heads'][h][s]['auc_test']:.2f} TPR={result['heads'][h][s]['tpr']:.2f}" for s in STATS), flush=True)
        by_kind = {"nominal": nom_sc, "cygnus": [ood_sc[i] for i in np.where(ood_ships == "cygnus")[0]],
                   "dragon": [ood_sc[i] for i in np.where(ood_ships == "dragon")[0]]}
        plot_timeseries(h, curves, args.out)
        plot_scores(h, by_kind, result["heads"][h], args.out)

    save = {"ood_ships": ood_ships}
    for h in heads:
        for s in STATS:
            save[f"nom_{h}_{s}"] = np.array([d[s][h] for d in nom_sc])
            save[f"ood_{h}_{s}"] = np.array([d[s][h] for d in ood_sc])
    np.savez(os.path.join(args.out, "scores.npz"), **save)
    json.dump(result, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("[ood_voe] wrote", os.path.join(args.out, "results.json"), "+ plots", flush=True)


if __name__ == "__main__":
    main()
