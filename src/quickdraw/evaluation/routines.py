"""Eval routines — one definition each, used both as in-training subscriptions (on a cadence) and
as standalone post-hoc steps. REGISTRY maps name -> routine.

A routine: (cfg, model, norm, ecfg, writer, device, step) -> summary dict. It logs every scalar and
plot through `writer` (one call -> local + wandb identically). Routines are read-only (no grad).
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ..controller.run import _plog, run_and_log_control
from ..environments.registry import make_env
from ..logging import viz
from ..training.setup import eval_episodes, resolve_data_root
import torch

from .openloop import emit_horizon_readouts, eval_batched, image_curves, proprio_curves
from .products import emit_openloop


def _is_mm(model):
    return hasattr(getattr(model, "_orig_mod", model), "layout")


_POS_IDX_WARNED = [False]


def _pos_idx(cfg, env=None):
    """The obs dims that are ambient world xyz, for the flow/manifold world-space viz. LAYERED resolution:
    explicit `environments.position_idx` CONFIG override > the env's optional `position_indices()` HOOK
    (torus/pendulum declare [0,1,2] in code; RecordedEnv omits it) > [0,1,2] with a ONE-TIME warning. Config
    wins so a recorded dataset whose position triple isn't the first 3 dims (e.g. robocasa EEF = [7,8,9]) can
    set it — all recorded datasets share the generic RecordedEnv and can't carry it in code. See #11 +
    WorldEnv.position_indices. `env` (if given) is queried for the hook; else one is built cheaply on CPU."""
    envcfg = cfg.get("environments", {}) if hasattr(cfg, "get") else {}
    cfg_idx = envcfg.get("position_idx", None)
    if cfg_idx is not None:
        return [int(i) for i in cfg_idx]                       # explicit config override wins
    if env is None:                                            # lazily build a cheap env to read its hook
        try:
            env = make_env(envcfg.get("name", "torus_world"), cfg.environments, 1, "cpu")
        except Exception:
            env = None
    fn = getattr(env, "position_indices", None)
    hook = fn() if callable(fn) else None
    if hook is not None:
        return [int(i) for i in hook]
    if not _POS_IDX_WARNED[0]:
        print("[eval] WARNING: no position_indices (env hook or environments.position_idx set); defaulting "
              "world-xyz viz to obs dims [0,1,2]. Set environments.position_idx if this dataset's position "
              "triple differs (see issue #11).", flush=True)
        _POS_IDX_WARNED[0] = True
    return [0, 1, 2]


def _openloop_split(cfg, model, norm, writer, device, split, R, r, v_scale, prefix, step, coloring="rainbow", fps=60):
    if _is_mm(model):                      # multimodal open-loop is eval_ood_horizon (dict obs); this vector-tensor
        return {}                          # OOD-split path (ood_visual/geometric/dynamics) is a pending MM port
    t0 = time.perf_counter()
    eps = eval_episodes(cfg, norm, split)  # whole split; one batched rollout for all of it
    obs = torch.stack([eps[i]["obs_seq"] for i in range(len(eps))]).to(device)
    act = torch.stack([eps[i]["act_seq"] for i in range(len(eps))]).to(device)
    n_eval = cfg.eval.get("n_episodes", None)  # cap the eval N (config knob); null -> whole split
    if n_eval is not None:
        obs, act = obs[: int(n_eval)], act[: int(n_eval)]
    P, win = cfg.data.P, int(cfg.data.action_smooth_window)
    n_plot = min(int(cfg.eval.n_plot), obs.shape[0])
    _plog(writer, f"[{prefix} @ep{step}] start: {obs.shape[0]} episodes, {obs.shape[1]}-step open-loop rollout, "
                  f"{n_plot} plot/video episodes")
    from omegaconf import OmegaConf
    override = {"R": float(R), "r": float(r)}          # init_speed = v_scale so rollout_metrics match; v_scale
    if v_scale is not None:                            # may be None (env has no init_speed knob) -> leave as-is
        override["init_speed"] = float(v_scale)
    env = make_env(cfg.environments.get("name", "torus_world"),      # THIS split's geometry (may be OOD);
                   OmegaConf.merge(cfg.environments, override), 1, "cpu")
    res = eval_batched(model, norm, env, P, obs, act)
    _plog(writer, f"[{prefix} @ep{step}] rollout done in {time.perf_counter() - t0:.1f}s; rendering...")

    desc = ("Open-loop long-horizon rollout on the torus: a BLACK agent on the TRUE path and a GREY agent on "
            "the model's PREDICTED path. They share the context, then diverge at the fork step. The action "
            "arrow is the applied action along the true path.")
    emit_openloop(writer, prefix, step, env=env, R=R, r=r, coloring=coloring, fps=fps, P=P, smooth_window=win,
                  description=desc, ctx_xyz=res["ctx_xyz"], p_true_xyz=res["p_true_xyz"],
                  p_hat_xyz=res["p_hat_xyz"], actions=res["actions"], curves=res["agg"], n_plot=n_plot,
                  obs_true=norm.denorm_obs(obs[:n_plot]).cpu().numpy(), obs_pred=res["p_hat_obs"][:n_plot],
                  title_fn=lambda i: f"{split} #{i}", log=lambda m: _plog(writer, f"[{prefix} @ep{step}]   {m}"))
    summary = {m: float(res["agg"][m].mean()) for m in res["agg"]}  # mean over the rollout (routine return value)
    _plog(writer, f"[{prefix} @ep{step}] done in {time.perf_counter() - t0:.1f}s")
    return summary


@torch.no_grad()
def eval_ood_horizon(cfg, model, norm, ecfg, writer, device, step=0):
    """The ONE open-loop long-horizon eval for every model (OOD: horizon >> trained). One rollout over
    held-out val episodes decodes proprio (always) + any image head; proprio and image outputs MIRROR each
    other and the code generalizes over arbitrary trunks. All under eval_ood_horizon/:
      - AVERAGED (over episodes, not per-instance) error_vs_step_avg_{linear,log} curves + *_mean scalars, one
        block per head: proprio (obs_error/manifold/pointwise/tangent) and each image <head>/ (psnr/ssim/mse/l1).
      - per-episode visuals for the first n_plot(=4) episodes: proprio trajectory_plot_{i}/video_{i}/scene_{i},
        and each image <head>/filmstrip_{i} + <head>/rollout_{i}.
    Image decode is the cost, so n_ep=8 when an image head is present (else eval.n_episodes)."""
    import numpy as _np

    from ..data.dataset import load_split_episodes_mm
    m = getattr(model, "_orig_mod", model)
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P, fps = cfg.data.P, round(1.0 / ecfg.dt)

    def prog(pct, what):
        _plog(writer, f"[eval_ood_horizon @ep{step}] {pct:3d}% — {what}")

    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                 cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    n_ep = min(8 if img_heads else int(cfg.eval.get("n_episodes", 32) or 32), len(eps))
    eps = eps[:n_ep]
    H = min(int(cfg.eval.get("horizon", 2048)), min(len(o) for o, _, _ in eps) - P - 1)
    prog(0, f"start: {n_ep} eps, H={H}, heads={['proprio'] + img_heads}")

    # ---- one rollout (proprio always; decode image heads too when present) ----
    pro = torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(device)
    ctx = {"proprio": pro}
    for h in img_heads:
        ctx[h] = torch.stack([torch.from_numpy(im[:P]) for _, _, im in eps]).float().div(255.0).to(device)
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[:P + H - 1])) for _, a, _ in eps]).float().to(device)  # normalized (as trained)
    dc = int(cfg.eval.get("decode_chunk", 64) or 0) or None                  # chunk image decode over horizon (PR #8 bug 2)
    out = m.imagine_eval(ctx, acts, H, heads=["proprio"] + img_heads, decode_chunk=dc)
    prog(30, "rollout done")

    n_plot = min(4, n_ep)                                                    # per-episode visuals for the first few

    # ---- AVERAGED error-vs-step curves, one block per head (proprio + each image), mirrored. Averaged over
    #      episodes (NOT per-instance) — same policy as proprio: no per-episode curves. ----
    env = make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1, "cpu")
    pred = out["proprio"]
    p_hat = torch.nan_to_num(norm.denorm_obs(pred), nan=10.0, posinf=10.0, neginf=-10.0)
    p_true = torch.stack([torch.from_numpy(o[P:P + H]) for o, _, _ in eps]).float().to(device)
    per_step = proprio_curves(pred, norm.norm_obs(p_true), p_hat, p_true, env)
    curves = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}        # mean over episodes -> (H,)

    images = {}                                                              # per image head: curves + frames for the emitter
    for head in img_heads:
        ipred = out[head].clamp(0, 1)                                        # (n_ep,H,s,s,3)
        itrue = torch.stack([torch.from_numpy(im[P:P + H]) for _, _, im in eps]).float().div(255.0).to(device)
        images[head] = {"icurves": image_curves(ipred, itrue),               # shared per-step psnr/ssim/mse/l1
                        "full_true": _np.stack([eps[i][2][:P + H].astype(_np.float32) / 255.0 for i in range(n_plot)]),
                        "ipred": ipred[:n_plot].cpu().numpy()}
        # per-step future-accuracy scalars at quarter-horizon steps -> eval_ood_horizon/<head>/<stat>/@+<x>
        emit_horizon_readouts(writer, "eval_ood_horizon", head, images[head]["icurves"], H, step)
    prog(45, f"averaged curves (proprio + {len(img_heads)} image head(s))")

    # ---- everything (curves + per-episode trajectory/image visuals) via the shared open-loop emitter ----
    desc = ("Open-loop long-horizon rollout on the torus: a BLACK agent on the TRUE path and a GREY agent on "
            "the model's PREDICTED path, sharing the context then diverging at the fork.")
    ctx_obs = norm.denorm_obs(pro[:n_plot]).cpu().numpy()
    pos = _pos_idx(cfg, env=env)                                             # world-xyz obs dims (#11; env hook / config)
    emit_openloop(writer, "eval_ood_horizon", step, env=env, R=getattr(ecfg, "R", None),  # R/r only read by the
                  r=getattr(ecfg, "r", None), coloring="hsv", fps=fps, P=P,               # rich (torus) scene path
                  smooth_window=int(cfg.data.action_smooth_window), description=desc,
                  ctx_xyz=ctx_obs[:, :, pos],
                  p_true_xyz=p_true[:n_plot][:, :, pos].cpu().numpy(), p_hat_xyz=p_hat[:n_plot][:, :, pos].cpu().numpy(),
                  actions=[eps[i][1][:P + H].astype(_np.float32) for i in range(n_plot)],
                  curves=curves, n_plot=n_plot, images=(images or None),
                  obs_true=_np.concatenate([ctx_obs, p_true[:n_plot].cpu().numpy()], axis=1),
                  obs_pred=p_hat[:n_plot].cpu().numpy(),
                  title_fn=lambda i: f"eval_ood_horizon #{i} H={H}", log=lambda msg: prog(50, msg))

    if was:
        m.train()
    prog(100, f"done in {time.perf_counter() - t0:.1f}s")
    return {"eval_ood_horizon": float(curves["pointwise_error"].mean())}


@torch.no_grad()
def eval_ae_floor(cfg, model, norm, ecfg, writer, device, step=0):
    """eval_ae_floor — the encode->decode CEILING (issue #12 §4). NO dynamics, NO rollout: encode each REAL
    val frame and decode it straight back, per modality (proprio + each image head). Every downstream image
    metric is bounded by this floor, so it separates "what the tokenizer can represent" from "what the
    dynamics gets wrong". Products MIRROR eval_ood_horizon (same names/layout via emit_openloop) under
    eval_ae_floor/: per image <head> a filmstrip_i (GT vs recon) + rollout_i(=recon) mp4 +
    error_vs_step_avg_{linear,log} + {psnr,ssim,mse,l1}_mean; proprio/{pointwise_error,obs_error}_mean. The
    error-vs-step curve is DELIBERATELY FLAT (each frame independent) — on ood_horizon's axes it reads as the
    ceiling vs the compounding rollout. Env-agnostic (no geometry). With a frozen pretrained AE this is a
    constant across epochs (a cheap "is something training that shouldn't be" detector — #12 §4)."""
    import numpy as _np

    from ..data.dataset import load_split_episodes_mm
    m = getattr(model, "_orig_mod", model)
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P, fps = cfg.data.P, round(1.0 / ecfg.dt)
    dev = device if isinstance(device, str) else device.type

    def prog(pct, what):
        _plog(writer, f"[eval_ae_floor @ep{step}] {pct:3d}% — {what}")

    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                 cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    n_ep = min(int(cfg.eval.get("ae_floor_episodes", 2) or 2), len(eps))
    eps = eps[:n_ep]
    H = min(int(cfg.eval.get("horizon", 2048)), min(len(o) for o, _, _ in eps) - P - 1)
    prog(0, f"start: {n_ep} eps, H={H}, heads={['proprio'] + img_heads} (encode->decode, NO dynamics)")

    pro_full = torch.stack([norm.norm_obs(torch.from_numpy(o[:P + H])) for o, _, _ in eps]).float().to(device)
    obs_full = {"proprio": pro_full}
    for h in img_heads:
        obs_full[h] = torch.stack([torch.from_numpy(im[:P + H]) for _, _, im in eps]).float().div(255.0).to(device)

    # per-frame encode->decode (encode_state is per-frame; chunk over time so image decode memory stays bounded)
    chunk = int(cfg.eval.get("decode_chunk", 64) or 64)
    rec_acc = {}
    for s in range(0, P + H, chunk):
        sub = {k: v[:, s:s + chunk] for k, v in obs_full.items()}
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            rec = m.to_obs(m.encode_state(sub), heads=["proprio"] + img_heads)
        for k, v in rec.items():
            rec_acc.setdefault(k, []).append(v.float())
    recon = {k: torch.cat(v, dim=1) for k, v in rec_acc.items()}
    prog(30, "encode->decode done")

    n_plot = min(4, n_ep)
    env = make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1, "cpu")
    pred = recon["proprio"][:, P:P + H]                              # reconstructed proprio (normalized)
    p_hat = torch.nan_to_num(norm.denorm_obs(pred), nan=10.0, posinf=10.0, neginf=-10.0)
    p_true = torch.stack([torch.from_numpy(o[P:P + H]) for o, _, _ in eps]).float().to(device)
    per_step = proprio_curves(pred, norm.norm_obs(p_true), p_hat, p_true, env)
    curves = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}   # flat over time (per-frame independent)

    images = {}
    for head in img_heads:
        ipred = recon[head][:, P:P + H].clamp(0, 1)
        itrue = obs_full[head][:, P:P + H]
        images[head] = {"icurves": image_curves(ipred, itrue),               # shared per-step psnr/ssim/mse/l1
                        "full_true": _np.stack([eps[i][2][:P + H].astype(_np.float32) / 255.0 for i in range(n_plot)]),
                        "ipred": ipred[:n_plot].cpu().numpy()}
        emit_horizon_readouts(writer, "eval_ae_floor", head, images[head]["icurves"], H, step)
    prog(45, "curves + metrics")

    desc = ("Encode->decode ceiling (NO dynamics): each frame reconstructed independently. The error-vs-step "
            "curve is flat by construction — the floor every rollout image metric is bounded by.")
    ctx_obs = norm.denorm_obs(pro_full[:n_plot, :P]).cpu().numpy()
    pos = _pos_idx(cfg, env=env)
    emit_openloop(writer, "eval_ae_floor", step, env=env, R=getattr(ecfg, "R", None), r=getattr(ecfg, "r", None),
                  coloring="hsv", fps=fps, P=P, smooth_window=int(cfg.data.action_smooth_window), description=desc,
                  ctx_xyz=ctx_obs[:, :, pos],
                  p_true_xyz=p_true[:n_plot][:, :, pos].cpu().numpy(), p_hat_xyz=p_hat[:n_plot][:, :, pos].cpu().numpy(),
                  actions=[eps[i][1][:P + H].astype(_np.float32) for i in range(n_plot)],
                  curves=curves, n_plot=n_plot, images=(images or None),
                  obs_true=_np.concatenate([ctx_obs, p_true[:n_plot].cpu().numpy()], axis=1),
                  obs_pred=p_hat[:n_plot].cpu().numpy(),
                  title_fn=lambda i: f"eval_ae_floor #{i} (encode->decode ceiling)", log=lambda msg: prog(50, msg))
    if was:
        m.train()
    prog(100, f"done in {time.perf_counter() - t0:.1f}s")
    summary = {"eval_ae_floor/proprio/pointwise_error": float(curves["pointwise_error"].mean())}
    summary.update({f"eval_ae_floor/{h}/psnr": float(images[h]["icurves"]["psnr"].mean()) for h in img_heads})
    return summary


def _ood_axis(cfg, model, norm, ecfg, writer, device, step, split):
    """Open-loop on one OOD split, scored on its own geometry + drawn with its coloring (from the
    dataset card). Shared by the visual/geometric/dynamics axes."""
    card = json.load(open(os.path.join(resolve_data_root(cfg), "dataset_card.json")))
    split_env, coloring = card.get("split_env", {}), card.get("coloring", {})
    se = split_env.get(split) or {"R": getattr(ecfg, "R", None), "r": getattr(ecfg, "r", None)}   # lazy + guarded:
    #                            a .get default is eval'd eagerly, so ecfg.R/.r must not be bare (crashes on non-torus envs)
    s = _openloop_split(cfg, model, norm, writer, device, split, se["R"], se["r"],
                        se.get("init_speed", getattr(ecfg, "init_speed", None)), split, step,
                        coloring.get(split, "rainbow"), fps=round(1.0 / ecfg.dt))
    return {split: s}


def eval_ood_visual(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_visual")


def eval_ood_geometric(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_geometric")


def eval_ood_dynamics(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_dynamics")


def eval_control(cfg, model, norm, ecfg, writer, device, step=0):
    """Dual MPPI control (oracle vs learned) through a random sequence of 8 goals. Multimodal models plan
    with an FPV context rendered in the loop (run_and_log_control handles it)."""
    return {"control": run_and_log_control(cfg, model, norm, ecfg, writer, device, step)}


@torch.no_grad()
def eval_manifold(cfg, model, norm, ecfg, writer, device, step=0):
    """Recovered-manifold projections of the carried LATENT space (the flattened token bag), for ANY model.
    Pool the model's COMMITTED next-state prediction (deterministic forward() readout; eps=0 for diffusion)
    over many VAL contexts, then project the latent to 3D + 2D with THREE reducers — PCA (linear, global-
    geometry-faithful), UMAP (nonlinear neighborhoods), t-SNE (local clusters) — each under eval_manifold/<method>/.
    The 3D still is a 6-view (fig_points_6view). Data-space (6D proprio) plots dropped — it's just the torus."""
    from ..data.dataset import load_split_episodes_mm
    from .manifold import manifold_predictions, pad_lims, reduce_dims
    m = getattr(model, "_orig_mod", model)
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    mm_eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                    cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))   # decodes proprio; latent = flattened bag
    _, latents, n_avail = manifold_predictions(m, norm, mm_eps, P=cfg.data.P, n_points=8000,
                                                stride=1, seed=0, device=device)
    sub = (f"each point = one committed 1-step next-state prediction from a real val context "
           f"({latents.shape[0]:,} points over {n_avail:,} contexts)")   # model-agnostic; teacher-forced, not a rollout
    label = f"latent space (full {latents.shape[1]}D z)"
    for method in ("umap", "tsne", "pca"):     # PCA = global truth, UMAP = neighborhoods, t-SNE = local clusters
        for nd in (3, 2):
            e = reduce_dims(latents, method, n_components=nd, seed=0)
            fig_fn = viz.fig_points_9view if nd == 3 else viz.fig_points_2d   # 3D = 9-view (iso abt vertical/horizontal + axial)
            f = fig_fn(e, lims=pad_lims(e), point_size=2.5,            # no color/colorbar (structure only)
                       title=f"recovered manifold — {method.upper()} of {label} to {nd}D, seed=0\n{sub}")
            writer.figure(f"eval_manifold/{method}/latent_space_to_{nd}d", f, step); plt.close(f)
            _plog(writer, f"[manifold @ep{step}] {method} {nd}D done ({time.perf_counter() - t0:.0f}s)")
    if was:
        m.train()
    _plog(writer, f"[manifold @ep{step}] done in {time.perf_counter() - t0:.1f}s")
    return {}


@torch.no_grad()
@torch.no_grad()
def eval_denoising_multistep(cfg, model, norm, ecfg, writer, device, step=0):
    """denoising_multistep (diffusion ONLY; self-skips otherwise): a FIXED agent at one point on a (seed-chosen)
    trajectory while N SEQUENTIAL swarms each denoise the PROPRIO token's rectified flow into decoded ODE paths
    that flow off-surface onto the torus, each leaving a tail that traces the field — then collapse their tails
    onto the next convergence point along the fixed (static) black future line, one swarm after the next. Shows
    the per-step denoising dynamics / flow field at one location. A seed (default: the epoch step; cfg.eval.
    denoising_seed pins it) picks the trajectory + swarm angle so a bad-looking eval won't recur. Scalars:
    eval_flow/std_of_samples + time/*."""
    import numpy as _np
    import torch.nn.functional as F

    from ..data.dataset import load_split_episodes_mm
    from ..models.multimodal import MultiModalFlow
    m = getattr(model, "_orig_mod", model)
    if not isinstance(m, MultiModalFlow):
        return {}
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    R, r = getattr(ecfg, "R", None), getattr(ecfg, "r", None)
    # the torus ATLAS render is torus-ONLY. Gate on the env NAME, NOT on R/r being set: RecordedConfig ships
    # inert R=r=1.0 placeholders (train reads them unconditionally), so "R is not None" is True on recorded too
    # -> would wrongly pick the torus mesh render. Non-torus -> geometry-free plain 3D world-space render (#11).
    has_geom = str(cfg.environments.get("name", "torus_world")) == "torus_world"
    dev = device if isinstance(device, str) else device.type
    P, W, d, K, n_swarm = cfg.data.P, m.window, m.d, m.sampling_steps, 16
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                    cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    # per-eval variety: a seed drives WHICH trajectory + the swarm angle, so a bad-looking eval won't recur (the
    # next eval shows a different one from a different angle) yet stays reproducible. Defaults to the epoch `step`.
    seed = int(step if cfg.eval.get("denoising_seed", None) is None else cfg.eval.denoising_seed)
    rng = _np.random.default_rng(seed)
    o, a, im = eps_ds[int(rng.integers(len(eps_ds)))]               # a seed-chosen episode (raw physical obs/actions)
    Tlen = len(o)
    obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(device)}
    if img_head is not None:
        obs[img_head] = torch.from_numpy(im).float().div(255.0)[None].to(device)
    act = norm.norm_act(torch.from_numpy(a)).float()[None].to(device)
    _ln = lambda x: F.layer_norm(x, (x.shape[-1],))
    dec = m.modalities["proprio"]
    g = torch.Generator(device=device).manual_seed(seed)            # reproducible swarm (seed-varied per eval)
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        z = m.encode_state(obs)                                     # (1, Tlen, n_state, d)

    pos = _pos_idx(cfg)                                             # world-xyz obs dims (#11; default [0,1,2])

    def decode_xyz(z_t, x):                                          # proprio residual x -> physical position (committed = endpoint)
        return norm.denorm_obs(dec.decode(_ln(z_t + x)[:, None, :].float()))[..., pos]

    def step_data(t):                                               # per-step swarm geometry for the quiver
        w = min(W, t + 1)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            h = m.backbone(m._to_input(z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]))[:, -1]   # (1, n_input, d)
        h_t, z_t = h[:, 0, :].float(), z[:, t, 0, :].float()        # proprio-token conditioning + carried token
        ts_ = time.perf_counter()
        m.flow.sample(h_t, steps=K, deterministic=True)             # the committed readout (timed; matches rollout)
        sample_s = time.perf_counter() - ts_
        swarm, ends = [], []
        for _ in range(n_swarm):
            e = torch.randn(1, d, generator=g, device=device)
            _, pth = m.flow.sample(h_t, steps=K, deterministic=False, eps=e, record_path=True)
            sp = _np.stack([decode_xyz(z_t, x)[0].float().cpu().numpy() for x in pth])           # (K+1, 3)
            swarm.append(sp); ends.append(sp[-1])
        return {"swarm": swarm, "ends": ends, "sample_s": sample_s}

    lo, hi = max(P, 60), max(P + 1, Tlen - 2 - 16)                  # seed-varied start; >=60-step history, >=16-step future
    ms_t0 = int(rng.integers(lo, hi)) if hi > lo else lo
    cap = int(cfg.eval.get("denoising_max_steps", None) or 128)     # cap the slow multistep render (default full 128)
    n_ms = min(cap, Tlen - 2 - ms_t0)
    ms_cur = o[ms_t0, pos]
    ms_tail = o[max(0, ms_t0 - 60):ms_t0 + 1, pos]
    ms_future = o[ms_t0:ms_t0 + n_ms + 1, pos]                       # fixed black future line spanning the N steps
    ms_act = viz.action_ambient(ms_cur, a[ms_t0], R, r) if has_geom else None
    _plog(writer, f"[denoising_multistep @ep{step}] seed={seed} {n_ms} steps from t0={ms_t0}, K={K} swarm={n_swarm}")
    ms_steps, spreads, sample_times = [], [], []
    for i in range(n_ms):
        dd = step_data(ms_t0 + i)
        spreads.append(float(_np.linalg.norm(_np.stack(dd["ends"]).std(axis=0))))    # swarm final-position spread
        sample_times.append(dd["sample_s"])
        ms_steps.append({"per_frame": _quiver_round_data(dd["swarm"], grow=10, collapse=5),
                         "true_next": o[ms_t0 + i + 1, pos]})
        if (i + 1) % max(1, n_ms // 5) == 0:
            _plog(writer, f"[denoising_multistep @ep{step}] sampling {int(100 * (i + 1) / n_ms):3d}% "
                          f"({i + 1}/{n_ms} swarms) | elapsed {time.perf_counter() - t0:.0f}s")
    n_frames = sum(len(s["per_frame"]) for s in ms_steps)
    if has_geom:                                                   # torus: rich atlas render (unchanged)
        _plog(writer, f"[denoising_multistep @ep{step}] swarm sampling done in {time.perf_counter() - t0:.0f}s; "
                      f"rendering {n_frames} frames (torus atlas, GPU/EGL)...")
        ms_frames = viz.diffusion_quiver_sequential_frames(R, r, "rainbow", ms_cur, ms_act, ms_tail, ms_future,
                                                           ms_steps, title="denoising multistep",
                                                           log=lambda mm: _plog(writer, f"[denoising_multistep @ep{step}] render {mm}"))
    else:                                                          # #11: geometry-free plain 3D world-space render
        _plog(writer, f"[denoising_multistep @ep{step}] swarm sampling done in {time.perf_counter() - t0:.0f}s; "
                      f"rendering {n_frames} frames (plain 3D world space, autoscaled — no torus mesh)...")
        ms_frames = viz.diffusion_swarm_plain_frames(ms_cur, ms_tail, ms_future, ms_steps,
                                                     title="denoising multistep (world space)",
                                                     log=lambda mm: _plog(writer, f"[denoising_multistep @ep{step}] render {mm}"))
    writer.video("eval_flow/denoising_multistep", ms_frames, 60, step)
    scene = {"description": "Sequential swarms at a FIXED agent: each swarm denoises, then its tails collapse onto the "
                            "convergence point along the (fixed) black future line, before the next swarm; agent, history "
                            "and future do not move.",
             "current_position_xyz": ms_cur, "history_tail_xyz": ms_tail, "future_path_xyz": ms_future,
             "swarm_target_per_step_xyz": [s["true_next"] for s in ms_steps]}
    if has_geom:
        scene["coordinate_system"] = "world xyz, same space as the torus"
        scene["torus"] = {"major_radius_R": float(R), "tube_radius_r": float(r)}
    else:
        scene["coordinate_system"] = f"world xyz from obs dims {pos} (autoscaled)"
    writer.scene("eval_flow/denoising_multistep", scene, step)
    writer.scalars({"eval_flow/std_of_samples": float(_np.mean(spreads)),      # predicted uncertainty
                    "eval_flow/time/sample_s": float(_np.mean(sample_times)),
                    "eval_flow/time/sample_ms_per_euler_step": float(1000.0 * _np.mean(sample_times) / max(1, K))}, step)
    if was:
        m.train()
    _plog(writer, f"[denoising_multistep @ep{step}] done in {time.perf_counter() - t0:.1f}s std_of_samples={_np.mean(spreads):.4f}")
    return {"eval_flow_std_of_samples": float(_np.mean(spreads))}


@torch.no_grad()
def eval_denoising_aggregate(cfg, model, norm, ecfg, writer, device, step=0):
    """denoising_aggregate (diffusion ONLY; self-skips otherwise): denoising ODE paths POOLED over many val
    contexts — a big cloud of predicted next-states collapsing from noise onto the RECOVERED manifold over the K
    flow steps (the aggregate structure the diffusion has learned: the torus emerging from noise). A seed
    (default: the epoch step; cfg.eval.denoising_seed pins it) fixes the sampled contexts + noise."""
    import numpy as _np

    from ..data.dataset import load_split_episodes_mm
    from ..models.multimodal import MultiModalFlow
    from .manifold import manifold_clouds
    m = getattr(model, "_orig_mod", model)
    if not isinstance(m, MultiModalFlow):
        return {}   # flow-in-dynamics is the ONLY precondition now (#11); non-torus renders in plain 3D world space
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    R, r = getattr(ecfg, "R", None), getattr(ecfg, "r", None)
    has_geom = str(cfg.environments.get("name", "torus_world")) == "torus_world"   # torus mesh render is torus-ONLY
    #                                       (RecordedConfig has inert R/r=1.0, so R-is-not-None can't gate this; #11)
    K, P, pos = m.sampling_steps, cfg.data.P, _pos_idx(cfg)
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                    cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    seed = int(step if cfg.eval.get("denoising_seed", None) is None else cfg.eval.denoising_seed)
    _plog(writer, f"[denoising_aggregate @ep{step}] seed={seed} pooling val contexts, K={K}...")
    paths_phys, _, n_avail = manifold_clouds(m, norm, eps_ds, P=P, n_points=5000, cube=3.0, stride=1, seed=seed, device=device)
    paths_xyz = paths_phys[..., pos]                                     # (N, K+1, 3) world positions
    if has_geom:
        Lm, Zm = (R + r) * 1.05, r * 1.6
        lims = ((-Lm, Lm), (-Lm, Lm), (-Zm, Zm))                         # torus-derived box (unchanged)
    else:
        lims = viz._pad3(paths_xyz.reshape(-1, 3))                       # #11: autoscaled from the data
    sub = f"{paths_xyz.shape[0]:,} next-states (of {n_avail:,} val contexts), K={K}"
    _plog(writer, f"[denoising_aggregate @ep{step}] pooled {paths_xyz.shape[0]} paths in {time.perf_counter() - t0:.0f}s; rendering 480 frames...")
    mframes = viz.points_collapse_frames(paths_xyz, lims=lims, n_frames=480,
                                         point_size=2.0, title=f"denoising aggregate — noise -> manifold\n{sub}",
                                         log=lambda mm: _plog(writer, f"[denoising_aggregate @ep{step}] render {mm}"))   # 480 @ 60fps = 8s (0.5x speed)
    writer.video("eval_flow/denoising_aggregate", mframes, 60, step)
    scene = {"description": "Denoising ODE paths pooled over many val contexts: a swarm of predicted next-states "
                            "collapsing from noise onto the recovered manifold over the K flow steps.",
             "denoising_paths_xyz": paths_xyz[:200]}         # 200-path subset (full set is large)
    if has_geom:
        scene["coordinate_system"] = "world xyz, same space as the torus"
        scene["torus"] = {"major_radius_R": float(R), "tube_radius_r": float(r)}
    else:
        scene["coordinate_system"] = f"world xyz from obs dims {pos} (autoscaled)"
    writer.scene("eval_flow/denoising_aggregate", scene, step)
    if was:
        m.train()
    _plog(writer, f"[denoising_aggregate @ep{step}] done in {time.perf_counter() - t0:.1f}s ({paths_xyz.shape[0]} paths)")
    return {}


@torch.no_grad()
def eval_denoising_filmstrip(cfg, model, norm, ecfg, writer, device, step=0):
    """denoising_filmstrip (diffusion ONLY; opt-in): watch the predicted next FRAME resolve over diffusion time.
    Variant (b) — dynamics-side: the predict_next dynamics head is ALWAYS a flow, so it denoises the next
    IMAGE tokens; decode each element of that latent ODE path through the (default MSE) image decoder. Emits
    `denoising_filmstrip_<i>` — one FILE per image (`denoising_filmstrip_images`, each a DIFFERENT (episode,
    step) frame); within each file rows = eps-noise seeds (`denoising_filmstrip_seeds`), cols = [GT | noise |
    denoise step 1..K]. ENV-AGNOSTIC (no geometry/goals) -> the first eval_flow product usable on a recorded
    dataset (#10). K is set LOCALLY (cfg.eval.denoising_filmstrip_steps) so a K=1 (shortcut) training config
    still shows a real trajectory; eps ~ N(0,1) (not the eps=0 committed path) so the 'noise' panel is real noise."""
    import numpy as _np
    import torch.nn.functional as F

    from ..data.dataset import load_split_episodes_mm
    from ..models.multimodal import MultiModalFlow
    m = getattr(model, "_orig_mod", model)
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)
    if not isinstance(m, MultiModalFlow) or img_head is None:
        return {}   # needs the flow dynamics + an image head to decode
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    dev = device if isinstance(device, str) else device.type
    off = 0                                                          # bag offset of the image tokens
    for name, n in m.layout:
        if name == img_head:
            n_img = n; break
        off += n
    P, K = cfg.data.P, int(cfg.eval.get("denoising_filmstrip_steps", 8) or 8)   # LOCAL K (not the training value)
    n_seeds = int(cfg.eval.get("denoising_filmstrip_seeds", 4) or 4)
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                    cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    seed = int(step if cfg.eval.get("denoising_seed", None) is None else cfg.eval.denoising_seed)
    n_images = int(cfg.eval.get("denoising_filmstrip_images", 4) or 4)   # separate FILES, each a different frame
    rng = _np.random.default_rng(seed)
    _ln = lambda x: F.layer_norm(x, (x.shape[-1],))
    _plog(writer, f"[denoising_filmstrip @ep{step}] seed={seed} images={n_images} K={K} seeds={n_seeds} img={img_head}")
    for i in range(n_images):
        o, a, im = eps_ds[int(rng.integers(len(eps_ds)))]           # a DIFFERENT (episode, step) per image
        Tlen = len(o)
        t_ctx = int(rng.integers(max(P, 1), max(P + 1, Tlen - 2)))  # a real context step; predict frame t_ctx+1
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(device),
               img_head: torch.from_numpy(im).float().div(255.0)[None].to(device)}
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(device)
        g = torch.Generator(device=device).manual_seed(seed + i)   # reproducible eps rows, distinct per image
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            z = m.encode_state(obs)                                 # (1, Tlen, n_state, d)
            w = min(m.window, t_ctx + 1)
            h = m.backbone(m._to_input(z[:, t_ctx - w + 1:t_ctx + 1], act[:, t_ctx - w + 1:t_ctx + 1]))[:, -1]  # (1,n_input,d)
        z_bag = z[0, t_ctx].float()                                 # (n_state, d) carried tokens
        h_img = h[0, off:off + n_img, :].float()                    # (n_img, d) conditioning for the image tokens
        z_img = z_bag[off:off + n_img]

        def decode_step(x):                                         # image-token residual -> (H,W,3) float in [0,1]
            bag = z_bag.clone()[None, None]                        # (1,1,n_state,d)
            bag[0, 0, off:off + n_img] = _ln(z_img + x)
            return m.to_obs(bag, heads=[img_head])[img_head][0, 0].clamp(0, 1).float().cpu().numpy()

        gt = (im[t_ctx + 1].astype(_np.float32) / 255.0)           # GT next frame
        rows = []
        for s in range(n_seeds):
            e = torch.randn(n_img, m.d, generator=g, device=device)
            _, path = m.flow.sample(h_img, steps=K, deterministic=False, eps=e, record_path=True)  # K+1 latent states
            rows.append([gt] + [decode_step(x) for x in path])     # [GT, noise(path0), step1..K]
        col_titles = ["GT", "noise"] + [f"k{k}" for k in range(1, len(rows[0]) - 1)]
        ncol = len(rows[0])
        fig, axes = plt.subplots(n_seeds, ncol, figsize=(1.7 * ncol, 1.7 * n_seeds), squeeze=False)
        for ri, row in enumerate(rows):
            for ci, img_np in enumerate(row):
                ax = axes[ri][ci]; ax.imshow(_np.clip(img_np, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(col_titles[ci], fontsize=9)
            axes[ri][0].set_ylabel(f"eps {ri}", fontsize=8)
        fig.suptitle(f"denoising filmstrip {i} — predicted next frame resolving over K={K} dynamics-flow steps "
                     f"(t={t_ctx}, {img_head})", fontsize=10)
        fig.supxlabel("diffusion time →", fontsize=9)   # cols GT | noise | k1..kK read left-to-right as diffusion time
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        writer.figure(f"eval_flow/denoising_filmstrip_{i}", fig, step); plt.close(fig)
    if was:
        m.train()
    _plog(writer, f"[denoising_filmstrip @ep{step}] done in {time.perf_counter() - t0:.1f}s ({n_images} images)")
    return {}


@torch.no_grad()
def eval_interpret(cfg, model, norm, ecfg, writer, device, step=0):
    """VLM-labeled latent interpretability (vision models ONLY; self-skips otherwise). Imagine N short clips
    from val, label each by semantic factor (color/speed/direction) with a VLM (OpenAI, cfg.interpret.vlm),
    embed each clip as ONE latent point (mean over the imagined rollout of the encoded token bag), then recolor
    the recovered-manifold projections (umap/tsne/pca, 2D+3D — the SAME projection per factor) by the VLM
    labels. Also grades the VLM against analytic labels from the imagined proprio (confusion + agreement).
    Config per env: conf/interpret/<env>.yaml. Products under eval_interpret/. See design/interpretability.md."""
    import numpy as _np
    from concurrent.futures import ThreadPoolExecutor

    from omegaconf import OmegaConf

    from ..data.dataset import load_split_episodes_mm
    from . import interpret as I
    m = getattr(model, "_orig_mod", model)
    if not _is_mm(model):
        return {}
    img_head = next((n for n, _ in m.layout if n != "proprio"), None)
    if img_head is None:
        _plog(writer, f"[eval_interpret @ep{step}] no image head — eval_interpret is a vision probe, skipping")
        return {}
    ic = OmegaConf.to_container(cfg.interpret, resolve=True)
    factors = ic["factors"]
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P, H, fps = cfg.data.P, int(ic["clip_len"]), round(1.0 / ecfg.dt)
    dev = device if isinstance(device, str) else device.type
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                 cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))

    # ---- sample N clips (episode, t0): P context frames + H imagined steps ----
    rng = _np.random.RandomState(int(ic["seed"]))
    slices = [(ei, t) for ei in range(len(eps)) for t in range(P, len(eps[ei][0]) - H)]
    rng.shuffle(slices)
    slices = slices[: int(ic["n_clips"])]
    _plog(writer, f"[eval_interpret @ep{step}] start: {len(slices)} clips x {H} frames ({H / fps:.2f}s) from val | vision head={img_head}")

    # ---- imagine each clip (batched); decode EVERY trunk (keyed by trunk id, so multi-trunk models work),
    #      keep raw actions + the mean-pooled internal-state series ----
    heads = [n for n, _ in m.layout]                                    # every modality/trunk id, in bag order
    img_trunks = [n for n in heads if hasattr(m.modalities[n], "ae")]   # image trunks (have a ViT AE) vs vector trunks
    bs = int(ic["batch"])
    bags, clip_acts = [], []
    decoded = {n: [] for n in heads}                                    # per-trunk decoded imaginations, keyed by trunk id
    for c0 in range(0, len(slices), bs):
        chunk = slices[c0:c0 + bs]
        ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(eps[ei][0][t - P:t])) for ei, t in chunk]).float().to(device),
               img_head: torch.stack([torch.from_numpy(eps[ei][2][t - P:t]) for ei, t in chunk]).float().div(255.0).to(device)}
        act = torch.stack([norm.norm_act(torch.from_numpy(eps[ei][1][t - P:t + H - 1])) for ei, t in chunk]).float().to(device)
        # ONE open-loop rollout: bag = the model's INTERNAL predictive state at each imagined step (B,H,n_state,d);
        # each trunk is DECODED from it. So the plotted latent is the state that PRODUCES the prediction, and
        # velocity/color we color by are its open-loop OUTPUTS (decoded) — not encoder inputs. No re-encoding.
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            bag = m._rollout(ctx, act, H, 0.0, None, 0)                                             # (b,H,n_state,d) internal state
            out = m.to_obs(bag, heads=heads)
        bags.extend(bag.float().reshape(len(chunk), H, -1).cpu().numpy())                           # per-clip (H,D) internal-state series
        for n in heads:
            a = out[n].float()
            if n in img_trunks:
                decoded[n].extend((a.clamp(0, 1).cpu().numpy() * 255).astype(_np.uint8))            # (b,H,size,size,3) uint8
            else:
                decoded[n].extend((norm.denorm_obs(a) if n == "proprio" else a).cpu().numpy())      # (b,H,dim) physical (proprio) or raw
        clip_acts.extend([eps[ei][1][t:t + H] for ei, t in chunk])                                  # raw actions in-clip
        _plog(writer, f"[eval_interpret @ep{step}] imagining {min(c0 + bs, len(slices))}/{len(slices)}")
    frames, pro_all = decoded[img_head], decoded["proprio"]             # VLM reads the primary image trunk; analytic reads proprio
    _plog(writer, f"[eval_interpret @ep{step}] imagined {len(slices)} clips ({len(heads)} trunks) in {time.perf_counter() - t0:.0f}s")

    # ---- analytic labels (exact, from the imagined proprio): per-clip scalar -> bucket ----
    # (quantile buckets like speed self-calibrate across the whole clip set; per-clip buckets like color don't)
    ana = {}                                                          # factor -> per-clip bucket list (len == n_clips)
    for f, fc in factors.items():
        if "analytic" in fc:
            kind = fc["analytic"]["kind"]
            ana[f] = I.bucketize(kind, [I.analytic_scalar(kind, p, ecfg.R) for p in pro_all], fc, r=ecfg.r)

    # ---- VLM labels (source: vlm factors — reads the RENDERED image) + N free-form captions (CLIP-style reward
    #      training, same call). ok = clips the VLM successfully returned. ----
    vlm_factors = {f: fc for f, fc in factors.items() if fc.get("source") == "vlm"}
    n_captions = int(ic.get("n_captions", 0))
    vlm = [None] * len(slices)
    ok = list(range(len(slices)))
    if vlm_factors or n_captions:
        key, schema = I.openai_api_key(), I.build_label_schema(vlm_factors, n_captions=n_captions)
        fidx = _np.unique(_np.linspace(0, H - 1, int(ic["vlm_frames"])).round().astype(int))
        vmodel = ic["vlm"]["model"]
        prompt = ic["prompt"]
        if n_captions and ic.get("caption_prompt"):
            prompt = prompt + "\n\n" + ic["caption_prompt"].format(n=n_captions)   # append the caption instructions
        # analytic-sourced factors (e.g. positioning) are EXACT from proprio and the VLM can't read them from the
        # FPV — pass them in as ground truth so captions don't assert the wrong position (color stays visual).
        known_factors = [f for f in ana if factors[f].get("source") == "analytic"]

        def _known(i):
            if not known_factors:
                return ""
            facts = "; ".join(f"{f} = {ana[f][i]}" for f in known_factors)
            return ("\n\nKnown exact facts about this clip (from the simulator — MORE reliable than the frames, do "
                    f"NOT contradict them): {facts}.")

        _plog(writer, f"[eval_interpret @ep{step}] VLM labeling {list(vlm_factors)} + {n_captions} captions "
                      f"(grounding {known_factors}, {vmodel}, {ic['vlm_frames']} frames/clip)...")

        def _label(i):
            return I.label_clip(api_key=key, model=vmodel, prompt=prompt, schema=schema,
                                frames_uint8=[frames[i][k] for k in fidx],
                                action_text=I.build_action_text(clip_acts[i]) + _known(i))

        with ThreadPoolExecutor(max_workers=int(ic["vlm"]["max_workers"])) as ex:
            for i, res in enumerate(ex.map(_label, range(len(slices)))):
                vlm[i] = res
                if (i + 1) % max(1, len(slices) // 8) == 0 or i + 1 == len(slices):
                    _plog(writer, f"[eval_interpret @ep{step}] labeled {i + 1}/{len(slices)}")
        ok = [i for i, v in enumerate(vlm) if v is not None]
        _plog(writer, f"[eval_interpret @ep{step}] VLM labeled {len(ok)}/{len(slices)} clips ({len(slices) - len(ok)} failed)")
        if not ok:
            if was:
                m.train()
            _plog(writer, f"[eval_interpret @ep{step}] no VLM labels returned — aborting (check OPENAI_API_KEY / network)")
            return {}

    # ---- resolve each factor's plotted label from its configured source (aligned to `ok`) ----
    labels_ok = {f: ([vlm[i][f] for i in ok] if fc["source"] == "vlm" else [ana[f][i] for i in ok])
                 for f, fc in factors.items()}
    counts = {f: {b: labels_ok[f].count(b) for b in factors[f]["buckets"]} for f in factors}
    _plog(writer, f"[eval_interpret @ep{step}] label counts: "
          + " | ".join(f"{f}({factors[f]['source']}){counts[f]}" for f in factors))

    # ---- cross-check: only where a VLM (image) reading can be graded vs an analytic (proprio) truth ----
    agree = {}
    for f, fc in factors.items():
        if fc["source"] == "vlm" and f in ana:
            buckets = list(fc["buckets"])
            cm = I.confusion([ana[f][i] for i in ok], labels_ok[f], buckets)
            agree[f] = float(_np.trace(cm) / max(1, cm.sum()))
            cf = viz.fig_confusion(cm, buckets, title=f"{f}: VLM(image) vs analytic(proprio) ({100 * agree[f]:.0f}% agree)")
            writer.figure(f"eval_interpret/crosscheck/{f}_confusion", cf, step); plt.close(cf)
    if agree:
        _plog(writer, f"[eval_interpret @ep{step}] cross-check agreement: "
              + " | ".join(f"{f} {100 * agree[f]:.0f}%" for f in agree))

    # ---- persist per-clip labels.json + crosscheck summary BEFORE the (slow) projections, so a crash in the
    #      reducers never loses the expensive VLM labels (the reward head only needs labels.json + latents) ----
    outdir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_interpret")
    os.makedirs(os.path.join(outdir, "crosscheck"), exist_ok=True)
    recs = [{"episode": int(slices[i][0]), "start": int(slices[i][1]),
             "label": {f: labels_ok[f][j] for f in factors},
             "captions": (vlm[i].get("captions", []) if vlm[i] else []),   # N free-form captions (CLIP reward training)
             "vlm": vlm[i], "analytic": {f: ana[f][i] for f in ana}} for j, i in enumerate(ok)]
    json.dump(recs, open(os.path.join(outdir, "labels.json"), "w"), indent=2)
    json.dump({"agreement": agree, "counts": counts, "n_labeled": len(ok), "n_clips": len(slices),
               "sources": {f: factors[f]["source"] for f in factors}},
              open(os.path.join(outdir, "crosscheck", "summary.json"), "w"), indent=2)
    writer.scalars({**{f"eval_interpret/agreement/{f}": v for f, v in agree.items()},
                    "eval_interpret/n_labeled": float(len(ok))}, step)

    # ---- assemble the points to plot: one clip-mean latent, OR every per-step latent with its clip's label ----
    mode = str(ic.get("point", "mean"))
    if mode == "per_step":                                  # dense, comparable to eval_manifold; clip label broadcast to its H steps
        pts = _np.concatenate([bags[i] for i in ok], axis=0)         # (len(ok)*H, D)
        clip_pos = _np.repeat(_np.arange(len(ok)), H)               # each point -> its clip's index within `ok`
        psize = 2.5
        sub = f"each point = one latent of an imagined rollout, all {H}-steps kept ({len(ok)} clips x {H} = {len(pts):,} points; label broadcast from its clip)"
    else:                                                    # one mean latent per clip (clean)
        pts = _np.stack([bags[i].mean(0) for i in ok])              # (len(ok), D)
        clip_pos = _np.arange(len(ok))
        psize = 6.0
        sub = f"each point = the mean over {H}-steps of an imagined rollout ({len(ok)} clips = {len(pts):,} points)"

    # ---- project + plot every reducer via the shared library (evaluation/projection.py); it saves the fitted
    #      reducers too, so a projection is reusable later (reducer.transform(new_latents) — pca/umap/lda only) ----
    from .projection import project_and_plot
    pdir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_interpret", "saved_projections")
    os.makedirs(pdir, exist_ok=True)
    _np.save(os.path.join(pdir, "clip_index.npy"), clip_pos)                  # each point -> its clip's index within `ok`
    labels_pp = {f: [labels_ok[f][c] for c in clip_pos] for f in factors}     # per-POINT labels (broadcast from clips)
    transform_ok = project_and_plot(writer, "eval_interpret", pts, labels_pp, factors, step=step,
                                    point_size=psize, subtitle=sub, methods=("pca", "tsne", "umap"),
                                    umap_sup_weights=[float(w) for w in ic.get("umap_sup_weights", [0.5, 1.0])],
                                    save_dir=pdir, plots_name="world_model_latent_space_plots",
                                    log=lambda m: _plog(writer, f"[eval_interpret @ep{step}] {m} ({time.perf_counter() - t0:.0f}s)"))
    with open(os.path.join(pdir, "meta.json"), "w") as fh:
        json.dump({"mode": mode, "n_points": int(len(pts)), "latent_dim": int(pts.shape[1]),
                   "transform_available": transform_ok,
                   "note": "load <method>_<nd>d_reducer.pkl and call .transform(new_latents) to project NEW points "
                           "into the same embedding (pca/umap only; tsne has no out-of-sample map)."}, fh, indent=2)

    # ---- example clips per bucket: a 4x4 grid composite (like the dataset composites), ~1s playback ----
    from collections import defaultdict
    grid, fps_ex = 4, max(1, fps // 2)                      # 0.5s clip at half fps -> ~1s (2x slow-mo)
    by_bucket = defaultdict(list)
    for f in factors:
        for j, i in enumerate(ok):
            b = labels_ok[f][j]
            if b in factors[f]["buckets"] and len(by_bucket[(f, b)]) < grid * grid:
                by_bucket[(f, b)].append(frames[i])
    for (f, b), clips in by_bucket.items():
        writer.video(f"eval_interpret/examples/{f}/{b}", viz.tile_clips(clips, grid), fps_ex, step)

    # ---- per-clip imaginations, one dir per clip id, one file per TRUNK (keyed by trunk id, so multi-trunk
    #      models generalize). manifest.json indexes it for a web explorer: click a point -> pop its imagination.
    #      Aligns with projections/ (embedding rows -> clip via clip_index.npy -> manifest clips[] in `ok` order). ----
    if bool(ic.get("imaginations", True)):
        trunk_kind = {n: ("image" if n in img_trunks else "vector") for n in heads}
        imdir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_interpret", "imaginations")
        for j, i in enumerate(ok):
            cd = os.path.join(imdir, str(i)); os.makedirs(cd, exist_ok=True)
            for n in heads:
                if trunk_kind[n] == "image":
                    viz.save_mp4(os.path.join(cd, f"{n}.mp4"), decoded[n][i], fps)      # true 0.5s at native fps
                else:
                    _np.save(os.path.join(cd, f"{n}.npy"), decoded[n][i])               # (H,dim) physical/raw
            _np.save(os.path.join(cd, "actions.npy"), _np.asarray(clip_acts[i], dtype=_np.float32))
        json.dump({"clip_len": H, "fps": fps, "point_mode": mode,
                   "trunks": [{"id": n, "kind": trunk_kind[n],
                               "file": f"{n}.{'mp4' if trunk_kind[n] == 'image' else 'npy'}"} for n in heads],
                   "clips": [{"id": int(i), "episode": int(slices[i][0]), "start": int(slices[i][1]),
                              "labels": {f: labels_ok[f][j] for f in factors}} for j, i in enumerate(ok)]},
                  open(os.path.join(imdir, "manifest.json"), "w"), indent=2)
        _plog(writer, f"[eval_interpret @ep{step}] saved {len(ok)} per-clip imaginations ({len(heads)} trunks) -> imaginations/")

    if was:
        m.train()
    _plog(writer, f"[eval_interpret @ep{step}] done in {time.perf_counter() - t0:.1f}s -> eval_interpret/")
    return {f"eval_interpret_agreement_{f}": v for f, v in agree.items()}


@torch.no_grad()
def eval_action_distribution(cfg, model, norm, ecfg, writer, device, step=0):
    """The learned action PRIOR (action-flow head) vs the TRUE data action distribution. Self-skips unless
    the model has an action head. All under eval_action_distribution/:
      - marginals: PRIMARY product, dataset/env-agnostic per-dim marginals (recorded vs head), one panel/dim.
      - animation_pooled: |a| true(green)|pred(red)|both, POOLED over all episodes/frame, ALL timesteps (no cap).
      - by_state_{true,pred} / animation_byx: |a| split by the env's `action_dist_split` hook (base.py) —
        TORUS-ONLY (the only env implementing it today); envs without it skip these two products entirely.
      - action_true_pred_w1 / w1_mean / w1/dim_*: 1D-Wasserstein, pooled |a| and per-dim (lower=better).
    TRUE = the RECORDED actions (actual history-conditioned data); PRED = one head draw/context (h[k] -> a[k+1],
    leak-free) — the fair comparison for a history-conditioned head. Smoothness scales with #episodes, not samples."""
    m = getattr(model, "_orig_mod", model)
    if not getattr(m, "action_head_enabled", False):
        return {}                                                # no action head -> skip
    from ..data.dataset import load_split_episodes_mm
    was = m.training; m.eval()
    t0 = time.perf_counter()
    def prog(p, w): _plog(writer, f"[eval_action_distribution @ep{step}] {p:3d}% — {w}")

    img_heads = [n for n, _ in m.layout if n != "proprio"]
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    # the action process is a property of the DATASET (recorded in its summary.json), not the cfg default — read
    # it so we always produce the FULL set of products (never a partial run gated on a stale cfg.data.action_sampler).
    asamp = "ornstein_uhlenbeck"
    try:
        asamp = json.load(open(os.path.join(resolve_data_root(cfg), "summary.json"))).get("action_sampler", asamp)
    except Exception:
        pass
    # per-dim names (item 4c): from the DATASET's own meta (not the cfg), null for most datasets -> a[i] fallback.
    action_names = None
    try:
        info = json.load(open(os.path.join(resolve_data_root(cfg), "train", "meta", "info.json")))
        action_names = info.get("features", {}).get("action", {}).get("names") or None
    except Exception:
        pass
    a_max = getattr(ecfg, "a_max", None)   # torus-only histogram x-limit knob; None -> viz derives it from the data
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                 cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    n_ep = min(int(cfg.eval.get("action_dist_episodes", 64) or 64), len(eps))   # default 64 = full val split (max distinct contexts)
    eps = eps[:n_ep]
    L = min(len(o) for o, _, _ in eps)
    prog(0, f"start: {n_ep} val episodes, L={L}, sampler={asamp}")

    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:L])) for o, _, _ in eps]).float().to(device)}
    for h in img_heads:
        ctx[h] = torch.stack([torch.from_numpy(im[:L]) for _, _, im in eps]).float().div(255.0).to(device)
    act = torch.stack([norm.norm_act(torch.from_numpy(a[:L])) for _, a, _ in eps]).float().to(device)  # (E,L,2) normalized

    with torch.no_grad():                                        # no grad: this eval also runs on the TRAINING GPU
        h_ctx = m.action_context(ctx, act)                       # (E,L-1,d): h[k] predicts a[k+1] (leak-free)
        pred_norm = m.sample_action(h_ctx).cpu()                 # (E,L-1,2) head, 1/context
    # obs at each action's state (E,L-1,obs_dim), for the env's OPTIONAL by-state split hook (item 3).
    obs_stack = np.stack([o[1:L] for o, _, _ in eps]).astype(np.float32)
    env = make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1, "cpu")
    split_fn = getattr(env, "action_dist_split", None)
    split_info = split_fn(obs_stack) if split_fn is not None else None   # (labels, low_name, high_name) | None
    prog(30, "context")

    # TRUE = the RECORDED actions (the actual history-conditioned data distribution) — the fair reference for a
    # history-conditioned head. PRED = ONE head draw per context (matching the data's 1-action/context), so both
    # marginals are estimated the same way and are directly comparable. Smoothness comes from #EPISODES (more
    # distinct contexts), NOT more samples/context: the head is sharp per context, so extra draws per context just
    # stack onto the same few spikes. n_ep is capped by the val split (here 64).
    true_a = norm.denorm_act(act[:, 1:].cpu()).numpy()           # (E,L-1,2) recorded a[1..L-1]
    pred_a = norm.denorm_act(pred_norm).numpy()                  # (E,L-1,2) head, 1/context (sampled under no_grad)
    prog(50, "head sampling")

    # PRIMARY product: per-dim marginals (dataset/env-agnostic — no a_max/state-split/geometry needed).
    fig = viz.fig_action_marginals(true_a, pred_a, names=action_names)
    writer.figure("eval_action_distribution/marginals", fig, step); plt.close(fig)
    prog(55, "marginals (primary product)")

    # window: pool +/-w timesteps per frame/tile -> ~(2w+1)x more samples (the dist changes slowly, so bias is
    # tiny). This is the way to densify PAST the #episodes ceiling. w=4 -> ~9x for both true and pred.
    win = int(cfg.eval.get("action_dist_window", 4) or 0)
    if split_info is not None:                                   # by-state products: TORUS-ONLY (item 3)
        labels, low_name, high_name = split_info
        for name, arr in (("true", true_a), ("pred", pred_a)):   # by-state 2-row static: recorded vs head
            fig = viz.fig_action_by_state(arr, labels, a_max, low_name=low_name, high_name=high_name,
                                          sampler_name=f"{asamp} · {name}", window=win)
            writer.figure(f"eval_action_distribution/by_state_{name}", fig, step); plt.close(fig)

    tm, pm = np.linalg.norm(true_a, axis=-1).reshape(-1), np.linalg.norm(pred_a, axis=-1).reshape(-1)
    q = np.linspace(0.0, 1.0, 512)
    w1 = float(np.mean(np.abs(np.quantile(tm, q) - np.quantile(pm, q))))   # 1D-Wasserstein on pooled |a| (vs recorded)
    writer.scalar("eval_action_distribution/true_pred_w1", w1, step)
    # per-dim W1 (item 4b): `live` skips constant dims (W1~=0) so w1_mean isn't flattered by dead dims.
    w1_per_dim = [float(np.mean(np.abs(np.quantile(true_a[..., i], q) - np.quantile(pred_a[..., i], q))))
                  for i in range(true_a.shape[-1])]
    live = [i for i in range(true_a.shape[-1]) if true_a[..., i].std() > 1e-6]
    writer.scalars({f"eval_action_distribution/w1/dim_{i}": w for i, w in enumerate(w1_per_dim)}, step)
    w1_mean = float(np.mean([w1_per_dim[i] for i in live])) if live else 0.0
    writer.scalar("eval_action_distribution/w1_mean", w1_mean, step)
    prog(70, f"distance (w1={w1:.3f}, w1_mean={w1_mean:.3f})")

    fps = round(1.0 / ecfg.dt)
    # POOLED over all episodes per frame (recorded green vs head red), ALL timesteps (no frame cap), +/-win pooled.
    frames = viz.anim_action_distribution(true_a, pred_a, a_max, window=win)
    writer.video("eval_action_distribution/animation_pooled", frames, fps, step)
    prog(85, "animation (pooled)")
    # per-dim marginals VIDEO: the animated companion to the static marginals PNG (same styling + same `win`).
    mframes = viz.anim_action_marginals(true_a, pred_a, names=action_names, window=win)
    writer.video("eval_action_distribution/animation_marginals", mframes, fps, step)
    prog(88, "animation (marginals)")
    if split_info is not None:                                   # by-state animation: TORUS-ONLY (item 3)
        frames_bx = viz.anim_action_by_state(true_a, pred_a, labels, a_max, low_name=low_name,
                                             high_name=high_name, window=win)
        writer.video("eval_action_distribution/animation_byx", frames_bx, fps, step)
    prog(95, "animation (by-state)")

    if was:
        m.train()
    prog(100, f"done in {time.perf_counter() - t0:.1f}s -> eval_action_distribution/")
    return {"action_true_pred_w1": w1}


REGISTRY = {"ood_horizon": eval_ood_horizon, "ood_visual": eval_ood_visual,
            "ood_geometric": eval_ood_geometric, "ood_dynamics": eval_ood_dynamics,
            "control": eval_control, "denoising_multistep": eval_denoising_multistep,
            "denoising_aggregate": eval_denoising_aggregate, "denoising_filmstrip": eval_denoising_filmstrip,
            "ae_floor": eval_ae_floor, "manifold": eval_manifold,
            "interpret": eval_interpret, "action_distribution": eval_action_distribution}


def _quiver_round_data(swarm, grow=10, collapse=5):
    """One round of a sequential swarm: a GROW phase (swarm flows noise->surface, trails growing) then a
    COLLAPSE phase (each tail RETRACTS onto its convergence endpoint, so the swarm ends as a tight knot at
    the predicted point) before the next round begins. grow + collapse frames total."""
    swarm = [np.asarray(s) for s in (swarm or [])]
    if not swarm:
        return [{"swarm": []}]
    K = swarm[0].shape[0] - 1

    def along(path, s):                                # particle at fraction s + growing trail
        fc = s * K; i0 = int(np.floor(fc)); i1 = min(i0 + 1, K); w = fc - i0
        p = (1 - w) * path[i0] + w * path[i1]
        return {"particle": p, "trail": np.vstack([path[: i0 + 1], p[None]]) if w > 1e-6 else path[: i0 + 1]}

    pf = [{"swarm": [along(sp, f / (grow - 1) if grow > 1 else 1.0) for sp in swarm]} for f in range(grow)]
    for f in range(collapse):                          # retract each tail onto its endpoint (the convergence)
        j = int(round((f + 1) / collapse * K))         # trail-start index advances 0 -> K (leaves a tight knot)
        pf.append({"swarm": [{"particle": sp[K], "trail": sp[min(j, K):]} for sp in swarm]})
    return pf
