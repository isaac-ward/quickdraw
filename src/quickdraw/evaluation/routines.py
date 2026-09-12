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
from ..training.setup import eval_episodes, image_head_cams, image_head_sizes, resolve_data_root, step_fps
import torch

from .openloop import emit_horizon_readouts, eval_batched, image_curves, latent_curves, proprio_curves
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
    WorldEnv.position_indices. `env` (if given) is queried for the hook; else one is built cheaply on CPU.
    Returns (pos, explicit): `explicit` is True when pos came from the config or the env hook, False when it
    fell back to the [0,1,2] GUESS (warned). Callers slice proprio pointwise_error to `pos` ONLY when explicit
    (see proprio_curves) — and since pointwise_error is the recorded env's checkpoint_metric, an explicit
    position_idx makes best.ckpt select on POSITION error (intended; e.g. docking)."""
    envcfg = cfg.get("environments", {}) if hasattr(cfg, "get") else {}
    cfg_idx = envcfg.get("position_idx", None)
    if cfg_idx is not None:
        return [int(i) for i in cfg_idx], True                 # explicit config override wins
    if env is None:                                            # lazily build a cheap env to read its hook
        try:
            env = make_env(envcfg.get("name", "torus_world"), cfg.environments, 1, "cpu")
        except Exception:
            env = None
    fn = getattr(env, "position_indices", None)
    hook = fn() if callable(fn) else None
    if hook is not None:
        return [int(i) for i in hook], True                    # explicit env hook
    if not _POS_IDX_WARNED[0]:
        print("[eval] WARNING: no position_indices (env hook or environments.position_idx set); defaulting "
              "world-xyz viz to obs dims [0,1,2]. Set environments.position_idx if this dataset's position "
              "triple differs (see issue #11).", flush=True)
        _POS_IDX_WARNED[0] = True
    return [0, 1, 2], False    # NOT explicit -> a GUESS; callers do NOT slice pointwise_error to it


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
    pos, _ = _pos_idx(cfg, env=env)                                  # world-xyz obs dims (torus hook -> [0,1,2])
    res = eval_batched(model, norm, env, P, obs, act, pos=pos)
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
def ood_horizon_shapes(cfg, has_image_heads: bool, ep_lens, P: int):
    """THE single definition of eval_ood_horizon's shapes. `eval_ood_horizon` and autobatch's `probe_eval` both
    call this, so the memory probe CANNOT drift from the thing it is estimating.

    Added 2026-08-20 after an audit found five separate divergences between the probe and this routine: the
    probe hardcoded n_ep=8 (copying the literal below, so a PROPRIO-ONLY config -- which uses n_episodes, 64 by
    default -- was under-modelled 8x), capped the horizon at an arbitrary 256 instead of applying the
    episode-length clamp (on robocasa val, min length 128 means H is 119 for ANY configured horizon, so the
    probe rolled 256 steps for a routine that can only ever roll 119 -- the whole "91GB" false alarm), and
    modelled only open_loop when the real memory peak is closed_loop_16.

    Returns (n_ep, H, cl_h, modes, calls) where modes = [(name, every, horizon)] and calls =
    [(name, rows, horizon)] is the per-imagine_eval-call shape, i.e. the memory-relevant unit (see
    rollout_regrounded's `cap`)."""
    n_ep = min(8 if has_image_heads else int(cfg.eval.get("n_episodes", 32) or 32), len(ep_lens))
    H = min(int(cfg.eval.get("horizon", 2048)), min(ep_lens[:n_ep]) - P - 1)
    cl_steps = [int(x) for x in cfg.eval.get("closed_loop_steps", [1, 16])]
    cl_h = min(H, int(cfg.eval.get("closed_loop_horizon", 256) or H))
    modes = [("open_loop", H, H)] + [(f"closed_loop_{x}_steps", x, cl_h) for x in cl_steps]
    calls = []
    for name, every, Hm in modes:
        e = min(int(every), Hm)
        n_seg = -(-Hm // e)                                   # ceil
        calls.append((name, min(max(n_ep, 64), n_ep * n_seg), e))   # `cap = max(n_ep, 64)` in rollout_regrounded
    return n_ep, H, cl_h, modes, calls


@torch.no_grad()          # every sibling eval routine has this; ood_horizon did not, so its latent pass was
#                           building a full autograd tape over a 128-step rollout every eval epoch and
#                           discarding it. imagine_eval was already guarded internally; latent_pass was not.
def eval_ood_horizon(cfg, model, norm, ecfg, writer, device, step=0):
    """The ONE long-horizon eval for every model (OOD: horizon >> trained). Held-out val episodes, decoding
    proprio (always) + any image head; the code generalizes over arbitrary trunks. Products are nested under
    a MODE sub-path, one full product set per mode:
      - eval_ood_horizon/open_loop/*                — pure open-loop rollout (context = the first P frames).
      - eval_ood_horizon/closed_loop_{X}_steps/*    — re-inject the GROUND-TRUTH observation as context every X
        steps (one sub-tree per X in eval.closed_loop_steps, default [1, 16]). open_loop IS re-grounding with
        every=H (one segment) — so all modes share ONE rollout fn + ONE emit fn.
    Within each mode, one block per head under <head>/: AVERAGED error_vs_step_avg_{linear,log} + *_mean scalars
    (proprio/ obs_error/manifold/pointwise/tangent, each image <head>/ psnr/ssim/mse/l1) + per-episode visuals
    (proprio/trajectory_*, image <head>/filmstrip_i + rollout_i). n_ep=8 with an image head (decode cost)."""
    import numpy as _np

    from ..data.dataset import load_split_episodes_mm
    m = getattr(model, "_orig_mod", model)
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    heads = ["proprio"] + img_heads
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P, fps = cfg.data.P, step_fps(cfg, ecfg)
    dc = int(cfg.eval.get("decode_chunk", 64) or 0) or None                  # chunk image decode over horizon (PR #8 bug 2)

    def prog(pct, what):
        _plog(writer, f"[eval_ood_horizon @ep{step}] {pct:3d}% — {what}")

    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    n_ep, H, cl_h, _modes, _calls = ood_horizon_shapes(cfg, bool(img_heads),
                                                      [len(o) for o, _, _ in eps], P)
    eps = eps[:n_ep]
    n_plot = min(int(cfg.eval.get("n_plot", 2) or 2), n_ep)   # per-episode visuals; SAME episode indices (0..n_plot-1) across all modes
    env = make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1, "cpu")
    pos, pos_explicit = _pos_idx(cfg, env=env)                              # world-xyz obs dims (#11; env hook / config)
    # GT context (first P frames, normalized) — the emit's ctx_xyz + the fallback obs_true, mode-independent.
    pro0 = torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(device)
    ctx_obs = norm.denorm_obs(pro0[:n_plot]).cpu().numpy()
    p_true = torch.stack([torch.from_numpy(o[P:P + H]) for o, _, _ in eps]).float().to(device)
    # fr is the per-head frame DICT (load_split_episodes_mm). Indexing it by head is what stops one camera
    # being scored as all of them -- this line used to hand EVERY head the same `im` array.
    itrue = {h: torch.stack([torch.from_numpy(fr[h][P:P + H]) for _, _, fr in eps]).float().div(255.0).to(device)
             for h in img_heads}

    def rollout_regrounded(every, Hm, want_bag=False):
        """Hm-step predicted obs (dict per head, (n_ep,Hm,...)), re-grounding on GT every `every` steps. Segment
        the horizon into ceil(Hm/every) chunks; segment s uses GT context obs[s*every:s*every+P] and actions
        a[s*every:s*every+P+every-1], rolls `every` steps via imagine_eval, keeps min(every, Hm-s*every), then
        concatenates the segments in time. Segments are batched over (episode, segment); the batch is chunked
        (context encode is NOT decode_chunk'd) so cl_1 doesn't OOM. every=Hm -> ONE segment == pure open-loop.
        Hm = this mode's horizon (open_loop uses the full H; closed-loop uses min(H, eval.closed_loop_horizon))."""
        every = min(int(every), Hm)
        n_seg = -(-Hm // every)                                             # ceil(Hm/every)
        C = {"proprio": []}
        C.update({h: [] for h in img_heads})
        A = []
        for o, a, fr in eps:                                                # (episode, segment) row order
            for s in range(n_seg):
                st = s * every
                C["proprio"].append(norm.norm_obs(torch.from_numpy(o[st:st + P])))
                for h in img_heads:
                    C[h].append(torch.from_numpy(fr[h][st:st + P]))          # per-head frames, not one shared array
                idx = _np.clip(_np.arange(st, st + P + every - 1), 0, len(a) - 1)   # last seg: pad+clamp (tail discarded)
                A.append(norm.norm_act(torch.from_numpy(a[idx])))
        ctx = {"proprio": torch.stack(C["proprio"]).float().to(device)}
        for h in img_heads:
            ctx[h] = torch.stack(C[h]).float().div(255.0).to(device)
        acts = torch.stack(A).float().to(device)                           # (n_ep*n_seg, P+every-1, act_dim)
        rows = n_ep * n_seg
        cap = max(n_ep, 64)                                                 # per-call batch cap (open_loop: rows=n_ep -> ONE call)
        segs, bag_out = {h: [] for h in heads}, None
        for r0 in range(0, rows, cap):
            sub = {k: v[r0:r0 + cap] for k, v in ctx.items()}
            o_c = m.imagine_eval(sub, acts[r0:r0 + cap], every, heads=heads, decode_chunk=dc, norm=norm,
                                 return_bag=want_bag)
            for h in heads:
                segs[h].append(o_c[h])
            if want_bag and "_bag" in o_c:
                # Rows are ordered (episode, segment). This plain cat is only correct while n_seg == 1, which
                # `want_bag = open_loop` guarantees (open loop IS the single-segment mode). Asserted rather
                # than assumed: with n_seg > 1 the bag would need the same reshape/slice `out[h]` gets below.
                assert n_seg == 1, "latent curves assume the single-segment (open-loop) rollout"
                bag_out = o_c["_bag"] if bag_out is None else torch.cat([bag_out, o_c["_bag"]], 0)
        out = {}
        for h in heads:
            v = torch.cat(segs[h], 0)                                       # (n_ep*n_seg, every, ...)
            v = v.reshape(n_ep, n_seg, *v.shape[1:])
            out[h] = torch.cat([v[:, s, :min(every, Hm - s * every)] for s in range(n_seg)], dim=1)  # (n_ep,Hm,...)
        return (out, bag_out) if want_bag else out

    def latent_pass(Hm, bag):
        """LATENT-space curves for the OPEN-LOOP mode, computed from THE ROLLOUT THAT WAS ALREADY RUN.

        It used to roll a second time. That was wrong twice over: it doubled the rollout cost, and with
        `stochastic_eval: true` (the default) the second rollout draws different eps -- so `latent_cos`
        described a DIFFERENT sample than the psnr/lpips curves it is plotted beside, and pairing them
        compared two draws. Now the bag comes back from the same `imagine_eval` call that produced the
        images (`return_bag`), so every curve on the panel describes one trajectory.

        Open-loop only -- under re-grounding the latent is reset every `every` steps, so a horizon-indexed
        drift curve would not mean what it says. Guarded: an add-on diagnostic must never be able to disable
        the whole ood_horizon routine (2 consecutive failures do that)."""
        try:
            if bag is None:      # LOUD: latent_cos is the primary metric for the dfptf experiments, and a
                #                  silent {} here would make it vanish from the panel with no explanation.
                _plog(writer, f"[eval_ood_horizon @ep{step}] latent curves SKIPPED: imagine_eval returned no "
                              f"`_bag` (return_bag path). The decoded-image products are unaffected.")
                return {}
            gt = {"proprio": norm.norm_obs(p_true[:, :Hm])}
            for h in img_heads:
                gt[h] = itrue[h][:, :Hm]
            anc = m.rel_anchor({"proprio": pro0}) if getattr(m, "_rel_on", lambda: False)() else None
            with torch.autocast(device_type=(device if isinstance(device, str) else device.type),
                                dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                z_gt = m.encode_state(gt, anc)
            return latent_curves(bag[:, :Hm].float(), z_gt.float())
        except Exception as e:                       # fail-soft but NOT silent (design/logging.md)
            _plog(writer, f"[eval_ood_horizon @ep{step}] latent curves SKIPPED ({type(e).__name__}: {e}) — "
                          f"the decoded-image products are unaffected")
            return {}

    def score_and_emit(out, subroutine, desc, Hm, lat=None):
        """Score (image_curves per head + proprio_curves) + emit (emit_openloop) a completed rollout under the
        `subroutine` tag (e.g. eval_ood_horizon/open_loop). Head nesting rides under it via product_tag. `Hm`
        is this mode's horizon; the precomputed full-H GT (p_true/itrue) is sliced to Hm (open_loop: Hm==H)."""
        pred = out["proprio"]
        p_hat = torch.nan_to_num(norm.denorm_obs(pred), nan=10.0, posinf=10.0, neginf=-10.0)
        pt = p_true[:, :Hm]                                                    # GT future sliced to this mode's horizon
        per_step = proprio_curves(pred, norm.norm_obs(pt), p_hat, pt, env,
                                  pos_slice=(pos if pos_explicit else None))    # position-L2 pointwise iff explicit
        curves = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}
        images = {}
        for head in img_heads:
            ipred = out[head].clamp(0, 1)
            ic = image_curves(ipred, itrue[head][:, :Hm])
            ic.update(lat or {})            # latent_motion_ratio / latent_cos ride the head's curve dict, so they
            #                                 reach the SAME panel + the same @+x scalar readouts as motion_ratio
            images[head] = {"icurves": ic,
                            "full_true": _np.stack([eps[i][2][head][:P + Hm].astype(_np.float32) / 255.0 for i in range(n_plot)]),
                            "ipred": ipred[:n_plot].cpu().numpy()}
            emit_horizon_readouts(writer, subroutine, head, images[head]["icurves"], Hm, step)
        emit_openloop(writer, subroutine, step, env=env, R=getattr(ecfg, "R", None), r=getattr(ecfg, "r", None),
                      coloring="hsv", fps=fps, P=P, smooth_window=int(cfg.data.action_smooth_window), description=desc,
                      ctx_xyz=ctx_obs[:, :, pos],
                      p_true_xyz=pt[:n_plot][:, :, pos].cpu().numpy(), p_hat_xyz=p_hat[:n_plot][:, :, pos].cpu().numpy(),
                      actions=[eps[i][1][:P + Hm].astype(_np.float32) for i in range(n_plot)],
                      curves=curves, n_plot=n_plot, images=(images or None),
                      obs_true=_np.concatenate([ctx_obs, pt[:n_plot].cpu().numpy()], axis=1),
                      obs_pred=p_hat[:n_plot].cpu().numpy(), pos_explicit=pos_explicit,
                      title_fn=lambda i: f"{subroutine} #{i} H={Hm}", log=lambda msg: prog(50, msg))
        return {f"{subroutine}/proprio/pointwise_error": float(curves["pointwise_error"].mean())}

    modes = _modes               # from ood_horizon_shapes above -- ONE definition, shared with probe_eval
    prog(0, f"start: {n_ep} eps, H={H} (closed-loop H={cl_h}), heads={heads}, modes={[mn for mn, _, _ in modes]}")
    summary = {}
    for k, (name, every, Hm) in enumerate(modes):
        prog(int(5 + 90 * k / len(modes)), f"mode {name} (re-ground every {every} steps, H={Hm})"
             if every < Hm else f"mode {name} (open-loop, no re-grounding, H={Hm})")
        open_loop = every >= Hm
        out = rollout_regrounded(every, Hm, want_bag=open_loop)
        out, bag = out if open_loop else (out, None)
        desc = (f"Open-loop long-horizon rollout: a BLACK agent on the TRUE path and a GREY agent on the model's "
                f"PREDICTED path, sharing the context then diverging at the fork." if every >= Hm else
                f"Closed-loop rollout: the GROUND-TRUTH observation is re-injected as context every {every} steps "
                f"(over a {Hm}-step horizon), so error resets each re-grounding instead of compounding.")
        lat = latent_pass(Hm, bag) if open_loop else None      # open-loop only (see latent_pass)
        summary.update(score_and_emit(out, f"eval_ood_horizon/{name}", desc, Hm, lat=lat))

    if was:
        m.train()
    prog(100, f"done in {time.perf_counter() - t0:.1f}s")
    return summary


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
    P, fps = cfg.data.P, step_fps(cfg, ecfg)
    dev = device if isinstance(device, str) else device.type

    def prog(pct, what):
        _plog(writer, f"[eval_ae_floor @ep{step}] {pct:3d}% — {what}")

    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    n_ep = min(int(cfg.eval.get("ae_floor_episodes", 2) or 2), len(eps))
    eps = eps[:n_ep]
    H = min(int(cfg.eval.get("horizon", 2048)), min(len(o) for o, _, _ in eps) - P - 1)
    prog(0, f"start: {n_ep} eps, H={H}, heads={['proprio'] + img_heads} (encode->decode, NO dynamics)")

    pro_full = torch.stack([norm.norm_obs(torch.from_numpy(o[:P + H])) for o, _, _ in eps]).float().to(device)
    obs_full = {"proprio": pro_full}
    for h in img_heads:
        obs_full[h] = torch.stack([torch.from_numpy(fr[h][:P + H]) for _, _, fr in eps]).float().div(255.0).to(device)

    # per-frame encode->decode (encode_state is per-frame; chunk over time so image decode memory stays bounded)
    # relative-position: ONE anchor for the whole trajectory (its first frame), threaded to every chunk so the
    # codec is exercised in the SAME relative frame it was trained in; to_obs de-relativizes -> ABSOLUTE recon.
    anchor = m.rel_anchor(obs_full) if m._rel_on() else None
    chunk = int(cfg.eval.get("decode_chunk", 64) or 64)
    rec_acc = {}
    for s in range(0, P + H, chunk):
        sub = {k: v[:, s:s + chunk] for k, v in obs_full.items()}
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            rec = m.to_obs(m.encode_state(sub, anchor), heads=["proprio"] + img_heads, anchor=anchor)
        for k, v in rec.items():
            rec_acc.setdefault(k, []).append(v.float())
    recon = {k: torch.cat(v, dim=1) for k, v in rec_acc.items()}
    prog(30, "encode->decode done")

    n_plot = min(int(cfg.eval.get("n_plot", 2) or 2), n_ep)   # per-episode visuals; SAME episode indices (0..n_plot-1) across all modes
    env = make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1, "cpu")
    pos, pos_explicit = _pos_idx(cfg, env=env)                       # world-xyz obs dims (#11; env hook / config)
    pred = recon["proprio"][:, P:P + H]                              # reconstructed proprio (normalized)
    p_hat = torch.nan_to_num(norm.denorm_obs(pred), nan=10.0, posinf=10.0, neginf=-10.0)
    p_true = torch.stack([torch.from_numpy(o[P:P + H]) for o, _, _ in eps]).float().to(device)
    per_step = proprio_curves(pred, norm.norm_obs(p_true), p_hat, p_true, env,
                              pos_slice=(pos if pos_explicit else None))   # position-L2 pointwise iff explicit
    curves = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}   # flat over time (per-frame independent)

    images = {}
    for head in img_heads:
        ipred = recon[head][:, P:P + H].clamp(0, 1)
        itrue = obs_full[head][:, P:P + H]
        images[head] = {"icurves": image_curves(ipred, itrue),               # shared per-step psnr/ssim/mse/l1
                        "full_true": _np.stack([eps[i][2][head][:P + H].astype(_np.float32) / 255.0 for i in range(n_plot)]),
                        "ipred": ipred[:n_plot].cpu().numpy()}
        emit_horizon_readouts(writer, "eval_ae_floor", head, images[head]["icurves"], H, step)
    prog(45, "curves + metrics")

    desc = ("Encode->decode ceiling (NO dynamics): each frame reconstructed independently. The error-vs-step "
            "curve is flat by construction — the floor every rollout image metric is bounded by.")
    ctx_obs = norm.denorm_obs(pro_full[:n_plot, :P]).cpu().numpy()   # pos/pos_explicit computed above
    emit_openloop(writer, "eval_ae_floor", step, env=env, R=getattr(ecfg, "R", None), r=getattr(ecfg, "r", None),
                  coloring="hsv", fps=fps, P=P, smooth_window=int(cfg.data.action_smooth_window), description=desc,
                  ctx_xyz=ctx_obs[:, :, pos],
                  p_true_xyz=p_true[:n_plot][:, :, pos].cpu().numpy(), p_hat_xyz=p_hat[:n_plot][:, :, pos].cpu().numpy(),
                  actions=[eps[i][1][:P + H].astype(_np.float32) for i in range(n_plot)],
                  curves=curves, n_plot=n_plot, images=(images or None),
                  obs_true=_np.concatenate([ctx_obs, p_true[:n_plot].cpu().numpy()], axis=1),
                  obs_pred=p_hat[:n_plot].cpu().numpy(), pos_explicit=pos_explicit,
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
                        coloring.get(split, "rainbow"), fps=step_fps(cfg, ecfg))
    return {split: s}


def eval_ood_visual(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_visual")


def eval_ood_geometric(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_geometric")


def eval_ood_dynamics(cfg, model, norm, ecfg, writer, device, step=0):
    return _ood_axis(cfg, model, norm, ecfg, writer, device, step, "eval_ood_dynamics")


def eval_control(cfg, model, norm, ecfg, writer, device, step=0):
    """Dual MPPI control (oracle vs learned) through a random sequence of 8 goals. Multimodal models plan
    with an FPV context rendered in the loop (run_and_log_control handles it).

    SKIPS CLEANLY when the env cannot be stepped. MPPI rolls a candidate action sequence through the env,
    so an env with no simulator cannot run control AT ALL -- and the env already says so: RecordedEnv marks
    `reset`/`step` as `not_provided`, which log_env_capabilities prints as `reset x no-sim | step x no-sim`
    at the start of every run. Nothing consulted that, so a recorded run instead dived into MPPI and died
    somewhere inside on an unrelated symptom (a KeyError on the torus-only `fpv["coloring"]`, or before
    that a TypeError from int() on a non-square img_size), which the callback counts toward its fatal
    streak as a BUG. Ask the env first and raise NotImplementedError, the clean-skip the callback already
    understands, so the log says the true reason instead of a misleading traceback."""
    from ..environments.base import env_provides
    from ..environments.registry import make_env
    env = make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1, device)
    missing = [h for h in ("reset", "step") if not env_provides(env, h)]
    if missing:
        raise NotImplementedError(
            f"{type(env).__name__} provides no {'/'.join(missing)} -- MPPI has to roll candidate action "
            f"sequences through a simulator, and a recorded dataset has none. This is the env contract "
            f"working, not a failure; turn the routine off with eval.during_train.evals.control=false to "
            f"stop it being requested at all.")
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
    mm_eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))   # decodes proprio; latent = flattened bag
    # CAP THE EPISODE LENGTH THAT GETS FORWARDED. manifold_predictions runs the model over each selected
    # episode WHOLE, so cost scales with episode length, not with n_points. block-stack val holds 2 very
    # long episodes (3,638 and 3,321 steps at stride 5), so every eval forwarded ~2 GB of resident frames
    # across two image heads plus activations over the full T -- 6.5 GiB on top of a training process
    # already holding 89 GiB, which OOMed and killed the routine every time. A projection needs a
    # REPRESENTATIVE sample of latents, not every timestep: 2 x 1024 steps is ~2,000 points, comfortably
    # more than UMAP/t-SNE need to show structure, and it is the same points these plots would have drawn
    # from anyway (n_points=8000 exceeded the 6,941 available, so it was using all of them).
    max_steps = int(cfg.eval.get("manifold_max_steps", 1024) or 0)
    if max_steps:
        mm_eps = [tuple(x[:max_steps] for x in ep) for ep in mm_eps]
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
    if getattr(getattr(m, "flow", None), "arch", "mlp") == "transformer":     # self-skip (not a failure): the
        _plog(writer, f"[denoising_multistep @ep{step}] skipped — the per-token swarm samples the proprio token "  # per-token
              f"alone, which a JOINT transformer flow can't do (it needs the full bag); not yet ported.")          # swarm
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
    eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    # per-eval variety: a seed drives WHICH trajectory + the swarm angle, so a bad-looking eval won't recur (the
    # next eval shows a different one from a different angle) yet stays reproducible. Defaults to the epoch `step`.
    seed = int(step if cfg.eval.get("denoising_seed", None) is None else cfg.eval.denoising_seed)
    rng = _np.random.default_rng(seed)
    o, a, _fr = eps_ds[int(rng.integers(len(eps_ds)))]              # a seed-chosen episode (raw physical obs/actions)
    im = _fr[img_head]                     # this routine deliberately shows ONE head; name it explicitly
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

    pos, _ = _pos_idx(cfg)                                          # world-xyz obs dims (#11; default [0,1,2])

    def decode_xyz(z_t, x):        # flow output x -> physical position (committed = the path's endpoint)
        # MIRROR predict_next EXACTLY (multimodal.py:978-981), which this used to hardcode:
        #   * `z_t + x` only under predict=residual. Under predict=absolute the flow emits the FULL next
        #     latent, so adding the carried token gave a ~2x-magnitude off-manifold point and this whole
        #     product (swarm quiver, spreads) was silently garbage.
        #   * LN only when latent_norm is on. It used to LN unconditionally, which is ALREADY wrong for
        #     latent_norm=affine runs even in residual mode -- affine's inverse is applied inside
        #     to_obs/decode, so pre-LN'ing here double-normalises.
        nb = (z_t + x) if m.predict_residual else x
        if m.latent_norm:
            nb = _ln(nb)
        return norm.denorm_obs(dec.decode(nb[:, None, :].float()))[..., pos]

    def step_data(t):                                               # per-step swarm geometry for the quiver
        w = min(W, t + 1)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            h = m.backbone(m._to_input(z[:, t - w + 1:t + 1], act[:, t - w + 1:t + 1]))[:, -1]   # (1, n_input, d)
        # Route through m._cond -- NEVER hand-build the conditioning. This used to be `h[:, 0, :]` (width d),
        # which broke the moment the conditioning gained channels: with the action slot + the raw action
        # embedding it is 3*d, so the velocity net's first Linear wanted d_x + time_dim + 3*d = 544 and got
        # 288 -> "mat1 and mat2 shapes cannot be multiplied (1x288 and 544x128)". Same break that killed two
        # runs at ep1 on 2026-08-12; the filmstrip was fixed then, this call site was missed and went
        # unnoticed because flow_arch=transformer makes this routine self-skip. _cond is the single source of
        # truth for that width.
        hc = m._cond(h, act[:, t])                                 # (1, n_state, cond_width)
        h_t, z_t = hc[:, 0, :].float(), z[:, t, 0, :].float()      # proprio-token conditioning + carried token
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
    if getattr(getattr(m, "flow", None), "arch", "mlp") == "transformer":     # self-skip (not a failure): the pooled
        _plog(writer, f"[denoising_aggregate @ep{step}] skipped — the pooled per-token next-state swarm assumes a "  # per-token
              f"factorized flow; the joint transformer flow can't sample the proprio token alone; not yet ported.")  # swarm
        return {}
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    R, r = getattr(ecfg, "R", None), getattr(ecfg, "r", None)
    has_geom = str(cfg.environments.get("name", "torus_world")) == "torus_world"   # torus mesh render is torus-ONLY
    #                                       (RecordedConfig has inert R/r=1.0, so R-is-not-None can't gate this; #11)
    pos, _ = _pos_idx(cfg); K, P = m.sampling_steps, cfg.data.P
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
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
    """denoising_filmstrip (diffusion ONLY; opt-in): does the LATENT flow's refinement actually do anything,
    and does it still do it deep into a rollout?

    REDESIGNED 2026-08-24 (user). The old layout was rows = eps-noise SEEDS at a fixed one-step-ahead
    prediction. Two problems: (a) a well-conditioned flow converges to the same answer from any eps, so the
    seed rows were near-duplicates carrying almost no information; (b) it only ever showed the EASY 1-step
    case, and 1-step at 4 Hz is nearly the identity (record §13: the per-step change is a fraction of the
    frame), so even a perfect prediction looked like the input.

    NOW: rows = ROLLOUT HORIZON h (`denoising_filmstrip_horizons`), cols = [GT | floor | noise | k1..kK]. One
    open-loop chain is rolled from a single context step down the SAME episode, advancing with the committed
    prediction (`predict_next`, exactly as the rollout does); at each requested h we branch off a noisy
    `record_path` sample of that step's flow and decode every element of the latent ODE path. So reading DOWN
    tests whether refinement survives compounding, and reading ACROSS tests whether it refines at all.

    The `floor` column decodes the TRUE latent for that row's frame, so the grid separates the two error
    sources by eye: floor-vs-GT is what the CODEC costs, kK-vs-floor is what the DYNAMICS costs. Only the
    second is the deliverable.

    NO numbers are drawn on the panels (2026-08-25, user: 45 of them is noise). The (h, k) PSNR grid, the
    per-row floor PSNR, and the episode/t_ctx provenance go to `logs/epoch_<step>/eval_flow/
    denoising_filmstrip_<i>.npz`; the first image's grid also goes to scalars `eval_flow/filmstrip/psnr/
    h<h>/{k<k>,floor}`. Trust those over the pictures — the intermediate latents (z + partially-denoised
    residual) are OFF-MANIFOLD for the image decoder, which is trained only on clean latents, so a k panel
    can look arbitrary while still being quantitatively closer; the monotonicity of the k-curve is the
    actual answer to "is the flow refinement doing anything".

    K is set LOCALLY (`denoising_filmstrip_steps`) so a K=1 (shortcut) training config still shows a real
    trajectory; eps ~ N(0,1) (not the eps=0 committed path) so the 'noise' panel is real noise."""
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
    P, K = cfg.data.P, int(cfg.eval.get("denoising_filmstrip_steps", 8) or 8)   # LOCAL K (not the training value)
    hz = list(cfg.eval.get("denoising_filmstrip_horizons", None) or [1, 8, 16, 32, 64])
    hz = sorted({int(h) for h in hz if int(h) >= 1})
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    seed = int(step if cfg.eval.get("denoising_seed", None) is None else cfg.eval.denoising_seed)
    n_images = int(cfg.eval.get("denoising_filmstrip_images", 4) or 4)   # separate FILES, each a different episode
    rng = _np.random.default_rng(seed)
    _ln = lambda x: F.layer_norm(x, (x.shape[-1],))
    scalars = {}
    _plog(writer, f"[denoising_filmstrip @ep{step}] seed={seed} images={n_images} K={K} horizons={hz} img={img_head}")
    for i in range(n_images):
        Hmax = max(hz)
        for _try in range(8):                                    # need an episode long enough for the deepest row
            ep_idx = int(rng.integers(len(eps_ds)))
            o, a, _fr = eps_ds[ep_idx]
            im = _fr[img_head]             # this routine deliberately shows ONE head; name it explicitly
            if len(o) > P + Hmax + 2:
                break
        else:
            _plog(writer, f"[denoising_filmstrip @ep{step}] no val episode longer than P+{Hmax}+2 — skipping image {i}")
            continue
        Tlen = len(o)
        t_ctx = int(rng.integers(max(P, 1), max(P + 1, Tlen - Hmax - 2)))   # context ends here; predict t_ctx+1..+Hmax
        # encode_state indexes EVERY layout name, so obs must carry EVERY image head even though this
        # routine only VISUALISES img_head. Omitting the others is a KeyError, not a silent wrong number.
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(device)}
        obs.update({h: torch.from_numpy(_fr[h]).float().div(255.0)[None].to(device)
                    for h in (n for n, _ in m.layout if n != "proprio")})
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(device)
        g = torch.Generator(device=device).manual_seed(seed + i)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            z = m.encode_state(obs)                              # (1, Tlen, n_state, d) CLEAN encoded episode

        def decode_bag(prev_bag, x_bag):                          # residual x -> (H,W,3), mirrors predict_next EXACTLY
            nb = prev_bag + x_bag if m.predict_residual else x_bag
            if m.latent_norm:
                nb = _ln(nb)
            return m.to_obs(nb[None, None], heads=[img_head])[img_head][0, 0].clamp(0, 1).float().cpu().numpy()

        # ONE open-loop chain from t_ctx, advancing with the COMMITTED prediction. At each requested horizon we
        # branch off a noisy record_path sample of that step's flow purely for the picture; the chain itself is
        # never perturbed by it, so row h really is "the flow at rollout step h".
        hist = z[:, :t_ctx + 1]                                   # (1, T0, n_state, d)
        rows, row_labels, grid = [], [], []
        for hstep in range(1, Hmax + 1):
            t = t_ctx + hstep - 1                                 # action index driving this transition
            w = min(m.window, hist.shape[1])
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                hb = m.backbone(m._to_input(hist[:, -w:], act[:, t - w + 1:t + 1]))[:, -1]   # (1,n_input,d)
            h_state = m._cond(hb, act[:, t])                      # (1,n_state,cond_width) — NEVER hand-build this
            prev = hist[0, -1].float()                            # (n_state,d) carried bag
            if hstep in hz:
                gt = (im[t_ctx + hstep].astype(_np.float32) / 255.0)
                e = torch.randn(m.n_state, m.d, generator=g, device=device)
                _, path = m.flow.sample(h_state[0].float(), steps=K, deterministic=False, eps=e, record_path=True)
                # FLOOR: decode the TRUE latent for this row's frame -- the best this codec can do here, so
                # every k panel reads against a REACHABLE target. k8-vs-floor is dynamics error, floor-vs-GT
                # is codec error; without this column the two are indistinguishable by eye.
                with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    fl = m.to_obs(z[:, t_ctx + hstep][:, None], heads=[img_head])[img_head][0, 0]
                floor = fl.clamp(0, 1).float().cpu().numpy()
                panels = [gt, floor] + [decode_bag(prev, x) for x in path]
                # PSNR of every panel against GT (panel 0 is GT itself -> nan). NOT drawn on the figure any
                # more (45 numbers is noise) -- the grid goes to the .npz beside it, and to scalars.
                psnrs = [float("nan")] + [float(10.0 * _np.log10(1.0 / max(1e-10, float(((p - gt) ** 2).mean()))))
                                          for p in panels[1:]]
                rows.append((panels, psnrs))
                row_labels.append(f"h={hstep}")
                grid.append((hstep, psnrs[1], psnrs[2:]))         # (h, floor, [k0..kK])
                if i == 0:                                        # log the (h,k) grid from the FIRST image only
                    scalars[f"eval_flow/filmstrip/psnr/h{hstep}/floor"] = psnrs[1]
                    for kk, ps in enumerate(psnrs[2:]):           # k=0 is the pure-noise panel
                        scalars[f"eval_flow/filmstrip/psnr/h{hstep}/k{kk}"] = ps
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                nb = m.predict_next(h_state, prev[None])          # (1,n_state,d) committed step
            hist = torch.cat([hist, nb[:, None].to(hist.dtype)], dim=1)
        if not rows:
            continue
        ncol = len(rows[0][0])
        col_titles = ["GT", "floor", "noise"] + [f"k{k}" for k in range(1, ncol - 2)]
        fig, axes = plt.subplots(len(rows), ncol, figsize=(1.7 * ncol, 1.85 * len(rows)), squeeze=False)
        for ri, (panels, _psnrs) in enumerate(rows):
            for ci, img_np in enumerate(panels):
                ax = axes[ri][ci]; ax.imshow(_np.clip(img_np, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(col_titles[ci], fontsize=9)
            axes[ri][0].set_ylabel(row_labels[ri], fontsize=9)
        fig.suptitle(f"denoising filmstrip {i} — latent-flow refinement (cols: K={K} ODE steps) vs ROLLOUT "
                     f"HORIZON (rows), one open-loop chain from t={t_ctx}, {img_head}\n"
                     f"'floor' = decode(TRUE latent) — the codec's own limit for that frame; PSNRs in the .npz",
                     fontsize=9)
        fig.supxlabel("flow refinement (diffusion time) \u2192        |        rows: deeper into the open-loop rollout \u2193",
                      fontsize=8)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        writer.figure(f"eval_flow/denoising_filmstrip_{i}", fig, step); plt.close(fig)
        # the numbers that used to clutter the panels -> logs/epoch_<step>/eval_flow/denoising_filmstrip_<i>.npz.
        # ep_idx/t_ctx are the provenance: seed=step by default, so each EPOCH draws a DIFFERENT frame and
        # epoch-to-epoch PSNR moves for reasons unrelated to the model. Pin cfg.eval.denoising_seed to compare.
        writer.array(f"eval_flow/denoising_filmstrip_{i}", step,
                     horizons=_np.array([g[0] for g in grid], dtype=_np.int32),
                     psnr_floor=_np.array([g[1] for g in grid], dtype=_np.float32),
                     psnr_grid=_np.array([g[2] for g in grid], dtype=_np.float32),   # (n_rows, K+1): k0=noise
                     k_index=_np.arange(K + 1, dtype=_np.int32),
                     ep_idx=_np.int32(ep_idx), t_ctx=_np.int32(t_ctx), K=_np.int32(K), seed=_np.int32(seed))
    if scalars:
        writer.scalars(scalars, step)
    if was:
        m.train()
    _plog(writer, f"[denoising_filmstrip @ep{step}] done in {time.perf_counter() - t0:.1f}s "
                  f"({n_images} images, horizons={hz})")
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
    P, H, fps = cfg.data.P, int(ic["clip_len"]), step_fps(cfg, ecfg)
    dev = device if isinstance(device, str) else device.type
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))

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
        # every image head, not just the one this routine interprets -- encode_state needs the full bag
        ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(eps[ei][0][t - P:t])) for ei, t in chunk]).float().to(device)}
        ctx.update({h: torch.stack([torch.from_numpy(eps[ei][2][h][t - P:t])
                                    for ei, t in chunk]).float().div(255.0).to(device)
                    for h in (n for n, _ in m.layout if n != "proprio")})
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
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    n_ep = min(int(cfg.eval.get("action_dist_episodes", 64) or 64), len(eps))   # default 64 = full val split (max distinct contexts)
    eps = eps[:n_ep]
    L = min(len(o) for o, _, _ in eps)
    prog(0, f"start: {n_ep} val episodes, L={L}, sampler={asamp}")

    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:L])) for o, _, _ in eps]).float().to(device)}
    for h in img_heads:
        ctx[h] = torch.stack([torch.from_numpy(fr[h][:L]) for _, _, fr in eps]).float().div(255.0).to(device)
    act = torch.stack([norm.norm_act(torch.from_numpy(a[:L])) for _, a, _ in eps]).float().to(device)  # (E,L,2) normalized

    with torch.no_grad():                                        # no grad: this eval also runs on the TRAINING GPU
        h_ctx = m.action_context(ctx, act)                       # (E,L-1,d): h[k] predicts a[k+1] (leak-free)
        pred_norm = m.sample_action(h_ctx).cpu()                 # (E,L-1,K*a) head, 1 chunk/context
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
    K = int(getattr(m, "action_head_chunk", 1))
    true_a = norm.denorm_act(act[:, 1:].cpu()).numpy()           # (E,L-1,a) recorded a[1..L-1]
    # A CHUNKED head predicts [a[t], a[t+1], ... a[t+K-1]] per context. The products below are all about the
    # NEXT action, so they use LEAD TIME 0 -- identical to the whole output when K=1. The other lead times are
    # scored as w1/lead_<k> scalars further down rather than folded in here, because pooling them would mix K
    # different prediction problems into one histogram and quietly flatter the head.
    pred_chunk = pred_norm.reshape(*pred_norm.shape[:-1], K, -1) if K > 1 else None
    pred_a = norm.denorm_act(pred_chunk[..., 0, :] if K > 1 else pred_norm).numpy()   # (E,L-1,a) lead time 0
    prog(50, "head sampling" + (f" (chunk K={K}, products use lead time 0)" if K > 1 else ""))

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
    if K > 1:
        # Per-LEAD-TIME W1: does the chunk stay faithful as it reaches further ahead? Slot k is scored against
        # the recorded action k steps later, so each is a like-for-like 1D-Wasserstein on pooled |a|.
        for k in range(K):
            pk = np.linalg.norm(norm.denorm_act(pred_chunk[:, :true_a.shape[1] - k, k, :]).numpy(), axis=-1)
            tk = np.linalg.norm(true_a[:, k:], axis=-1)
            wk = float(np.mean(np.abs(np.quantile(tk.reshape(-1), q) - np.quantile(pk.reshape(-1), q))))
            writer.scalar(f"eval_action_distribution/w1/lead_{k}", wk, step)
    # per-dim W1 (item 4b): `live` skips constant dims (W1~=0) so w1_mean isn't flattered by dead dims.
    w1_per_dim = [float(np.mean(np.abs(np.quantile(true_a[..., i], q) - np.quantile(pred_a[..., i], q))))
                  for i in range(true_a.shape[-1])]
    live = [i for i in range(true_a.shape[-1]) if true_a[..., i].std() > 1e-6]
    writer.scalars({f"eval_action_distribution/w1/dim_{i}": w for i, w in enumerate(w1_per_dim)}, step)
    w1_mean = float(np.mean([w1_per_dim[i] for i in live])) if live else 0.0
    writer.scalar("eval_action_distribution/w1_mean", w1_mean, step)
    prog(70, f"distance (w1={w1:.3f}, w1_mean={w1_mean:.3f})")

    fps = step_fps(cfg, ecfg)
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
