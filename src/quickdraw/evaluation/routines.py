"""Eval routines — one definition each, used both as in-training subscriptions (on a cadence) and
as standalone post-hoc steps. REGISTRY maps name -> routine.

A routine: (cfg, model, norm, ecfg, writer, device, step) -> summary dict. It logs every scalar and
plot through `writer` (one call -> local + wandb identically). Routines are read-only (no grad).
"""

from __future__ import annotations

import json
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

from ..controller.run import _plog, run_and_log_control
from ..environments.registry import make_env
from ..logging import viz
from ..training.setup import (effective_action_dim, eval_episodes, image_head_cams, image_head_sizes,
                             resolve_data_root, step_fps)
import torch

from .conditional import blind_null, energy_score, energy_skill, rank_calibration, rest_skill
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
def ood_horizon_shapes(cfg, has_image_heads: bool, ep_lens, P: int, n_ep_override=None):
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
    # 8 with an image head is a DECODE-COST cap on val's ~1785-step episodes. The OOD/memory splits are
    # ~31 model steps, so decoding all 10-12 of them costs less than 8 of val's -- and with only 10 episodes
    # in a split, throwing 2 away for no reason weakens the only measurement they exist for. The override
    # lets the split-driven routine ask for all of them; None keeps every existing run's shape untouched.
    n_ep = min(int(n_ep_override) if n_ep_override else
               (8 if has_image_heads else int(cfg.eval.get("n_episodes", 32) or 32)), len(ep_lens))
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
def eval_ood_horizon(cfg, model, norm, ecfg, writer, device, step=0, split=None, tag=None):
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
    # THE SPLIT AND THE PRODUCT PREFIX ARE THE ONLY THINGS THAT WERE EVER SPLIT-SPECIFIC HERE. Everything
    # downstream -- the rollout, per-head psnr/ssim/lpips, the proprio obs/manifold/pointwise/tangent
    # error, the filmstrips, the rollout mp4s, the error-vs-step curves -- reads the episodes it is handed.
    # Parameterising these two makes the OOD and memory splits measurable with the IDENTICAL metric code
    # that produced the @+128 numbers, so they are comparable by construction rather than by argument.
    split = str(split or cfg.eval.get("horizon_split", "val"))
    tag = str(tag or "eval_ood_horizon")
    m = getattr(model, "_orig_mod", model)
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    # THE HEAD LIST IS THE MODEL'S, NOT A LITERAL. This used to prepend "proprio" unconditionally, so a
    # model configured without that modality was asked to decode a head it does not have. Every run to
    # date declares proprio, so the assumption was invisible rather than absent; ordering is preserved
    # (proprio first) so this is a no-op for all of them.
    from .timing import Timer
    tm = Timer()                       # every model gets this, ours and external alike -- no opt-in
    has_pro = any(n == "proprio" for n, _ in m.layout)
    heads = (["proprio"] if has_pro else []) + img_heads
    was = m.training
    m.eval()
    t0 = time.perf_counter()
    P, fps = cfg.data.P, step_fps(cfg, ecfg)
    dc = int(cfg.eval.get("decode_chunk", 64) or 0) or None                  # chunk image decode over horizon (PR #8 bug 2)

    def prog(pct, what):
        _plog(writer, f"[{tag} @ep{step}] {pct:3d}% — {what}")

    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), split,
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    # DROP THE REVIEWED-OUT EPISODES before anything is measured. data/ood_windows.py records which OOD
    # clips carry no usable anomaly -- the noodle never entered frame, or the blower produced no motion
    # above the drone's own hover noise -- plus one memory clip whose turn returns too late to fall inside
    # any rollout. Scoring them would average real events together with clips containing no event, which
    # moves the number toward val for a reason that has nothing to do with the model. Splits with no
    # annotation (train/val) are untouched.
    from ..data.ood_windows import kept as _kept_eps
    keep = [i for i in _kept_eps(split) if i < len(eps)]
    if len(keep) < len(eps):
        _plog(writer, f"[{tag} @ep{step}] reviewed exclusions: keeping {len(keep)}/{len(eps)} episodes "
                      f"({[i for i in range(len(eps)) if i not in keep]} dropped, see data/ood_windows.py)")
        eps = [eps[i] for i in keep]
    n_ep, H, cl_h, _modes, _calls = ood_horizon_shapes(cfg, bool(img_heads),
                                                      [len(o) for o, _, _ in eps], P,
                                                      n_ep_override=cfg.eval.get("horizon_n_episodes"))
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
        C = {"proprio": []} if has_pro else {}
        C.update({h: [] for h in img_heads})
        A = []
        for o, a, fr in eps:                                                # (episode, segment) row order
            for s in range(n_seg):
                st = s * every
                if has_pro:
                    C["proprio"].append(norm.norm_obs(torch.from_numpy(o[st:st + P])))
                for h in img_heads:
                    C[h].append(torch.from_numpy(fr[h][st:st + P]))          # per-head frames, not one shared array
                idx = _np.clip(_np.arange(st, st + P + every - 1), 0, len(a) - 1)   # last seg: pad+clamp (tail discarded)
                A.append(norm.norm_act(torch.from_numpy(a[idx])))
        ctx = {"proprio": torch.stack(C["proprio"]).float().to(device)} if has_pro else {}
        for h in img_heads:
            ctx[h] = torch.stack(C[h]).float().div(255.0).to(device)
        acts = torch.stack(A).float().to(device)                           # (n_ep*n_seg, P+every-1, act_dim)
        rows = n_ep * n_seg
        cap = max(n_ep, 64)                                                 # per-call batch cap (open_loop: rows=n_ep -> ONE call)
        segs, bag_out = {h: [] for h in heads}, None
        for r0 in range(0, rows, cap):
            sub = {k: v[r0:r0 + cap] for k, v in ctx.items()}
            with tm.phase("rollout"):
                o_c = m.imagine_eval(sub, acts[r0:r0 + cap], every, heads=heads, decode_chunk=dc,
                                     norm=norm, return_bag=want_bag)
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
                _plog(writer, f"[{tag} @ep{step}] latent curves SKIPPED: imagine_eval returned no "
                              f"`_bag` (return_bag path). The decoded-image products are unaffected.")
                return {}
            gt = {"proprio": norm.norm_obs(p_true[:, :Hm])} if has_pro else {}
            for h in img_heads:
                gt[h] = itrue[h][:, :Hm]
            # the anchor is a PROPRIO position, so it cannot exist without that head
            anc = (m.rel_anchor({"proprio": pro0})
                   if has_pro and getattr(m, "_rel_on", lambda: False)() else None)
            with torch.autocast(device_type=(device if isinstance(device, str) else device.type),
                                dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                z_gt = m.encode_state(gt, anc)
            return latent_curves(bag[:, :Hm].float(), z_gt.float())
        except Exception as e:                       # fail-soft but NOT silent (design/logging.md)
            _plog(writer, f"[{tag} @ep{step}] latent curves SKIPPED ({type(e).__name__}: {e}) — "
                          f"the decoded-image products are unaffected")
            return {}

    def score_and_emit(out, subroutine, desc, Hm, lat=None):
        """Score (image_curves per head + proprio_curves) + emit (emit_openloop) a completed rollout under the
        `subroutine` tag (e.g. eval_ood_horizon/open_loop). Head nesting rides under it via product_tag. `Hm`
        is this mode's horizon; the precomputed full-H GT (p_true/itrue) is sliced to Hm (open_loop: Hm==H)."""
        # THE PROPRIO BLOCK IS SKIPPED WHOLESALE without that head. The error curves, the trajectory
        # plots and the xyz overlays are all derived from proprio, so an image-only model emits its
        # per-head image readouts (the loop below) and nothing else. Computed BEFORE the image loop, as
        # it always was, so the order of writer calls is unchanged for every model that has proprio.
        curves = p_hat = pt = None
        if has_pro:
            pred = out["proprio"]
            p_hat = torch.nan_to_num(norm.denorm_obs(pred), nan=10.0, posinf=10.0, neginf=-10.0)
            pt = p_true[:, :Hm]                                                # GT future sliced to this mode's horizon
            per_step = proprio_curves(pred, norm.norm_obs(pt), p_hat, pt, env,
                                      pos_slice=(pos if pos_explicit else None))  # position-L2 pointwise iff explicit
            curves = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}
        images = {}
        for head in img_heads:
            ipred = out[head].clamp(0, 1)
            with tm.phase("metrics"):
                ic = image_curves(ipred, itrue[head][:, :Hm])
            ic.update(lat or {})            # latent_motion_ratio / latent_cos ride the head's curve dict, so they
            #                                 reach the SAME panel + the same @+x scalar readouts as motion_ratio
            images[head] = {"icurves": ic,
                            "full_true": _np.stack([eps[i][2][head][:P + Hm].astype(_np.float32) / 255.0 for i in range(n_plot)]),
                            "ipred": ipred[:n_plot].cpu().numpy()}
            emit_horizon_readouts(writer, subroutine, head, images[head]["icurves"], Hm, step)
        # EMITTED FOR EVERY MODEL, proprio or not. `emit_openloop` skips each proprio product whose input
        # is None and still writes the image rollouts and filmstrips -- which for an image-only model are
        # the entire visual record, and the reason the external wrapper exists at all.
        emit_openloop(writer, subroutine, step, env=env, R=getattr(ecfg, "R", None), r=getattr(ecfg, "r", None),
                      coloring="hsv", fps=fps, P=P, smooth_window=int(cfg.data.action_smooth_window), description=desc,
                      ctx_xyz=ctx_obs[:, :, pos],
                      p_true_xyz=pt[:n_plot][:, :, pos].cpu().numpy() if has_pro else None,
                      p_hat_xyz=p_hat[:n_plot][:, :, pos].cpu().numpy() if has_pro else None,
                      actions=[eps[i][1][:P + Hm].astype(_np.float32) for i in range(n_plot)],
                      curves=curves, n_plot=n_plot, images=(images or None),
                      obs_true=_np.concatenate([ctx_obs, pt[:n_plot].cpu().numpy()], axis=1) if has_pro else None,
                      obs_pred=p_hat[:n_plot].cpu().numpy() if has_pro else None, pos_explicit=pos_explicit,
                      title_fn=lambda i: f"{subroutine} #{i} H={Hm}", log=lambda msg: prog(50, msg))
        if not has_pro:
            return {}                       # the image products are emitted; the SCALAR here is proprio
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
        summary.update(score_and_emit(out, f"{tag}/{name}", desc, Hm, lat=lat))

    tm.emit(writer, tag, step, horizon=H, n_ep=n_ep, n_heads=max(1, len(img_heads)))
    _plog(writer, f"[{tag} @ep{step}] timing: {tm.summary()}")
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


@torch.no_grad()
def eval_held_out_splits(cfg, model, norm, ecfg, writer, device, step=0):
    """Open-loop prediction error on NAMED held-out splits -- the OOD and memory campaigns.

    starling-2 ships four of them beside train/val (see conf/data/starling2.yaml for the manifest):

        eval_ood_noodle        a novel object enters the frame          -> VISUAL ood, read the image error
        eval_ood_leafblower    airflow pushes the drone                 -> DYNAMIC ood, read the proprio error
        eval_memory_backwall1  turn away from a scene and back again    -> does the rolled state keep it
        eval_memory_backwall2

    THE METRICS ARE NOT NEW, deliberately. This calls `eval_ood_horizon` once per split with its `split`
    and product `tag` swapped, so every number -- per-head psnr/ssim/lpips, proprio obs/manifold/pointwise/
    tangent error, error-vs-step curves, filmstrips, rollout mp4s -- comes from the identical code that
    produced the headline @+128 figures on val. Comparable by construction, not by argument. Run `val`
    alongside in `eval.splits` to get the in-distribution baseline from the same invocation.

    THE HORIZON IS BOUNDED BY THE CLIPS, and it is short: these episodes are ~125 frames at 15 Hz, so at
    the trained stride of 4 they are ~31 model steps and `ood_horizon_shapes` clamps H to about 22 after
    the P=8 context. That is ample for the OOD splits (the novelty arrives inside it) and TIGHT for memory
    -- measured, the away-and-back spans 8-21 model steps against a 32-step attention window, so the
    departed scene is still inside the window when the drone returns. The honest claim on this data is that
    the SELF-ROLLED state preserves the scene through a turn (only P=8 frames are real; the rest of the
    window is the model's own predictions), not that memory outlives the window.
    """
    splits = [str(x) for x in (cfg.eval.get("splits") or [])]
    if not splits:
        _plog(writer, f"[eval_held_out_splits @ep{step}] eval.splits is empty -- nothing to do. Set e.g. "
                      f"eval.splits=[val,eval_ood_noodle,eval_ood_leafblower,eval_memory_backwall1,"
                      f"eval_memory_backwall2]")
        return {}
    root = resolve_data_root(cfg)
    summary = {}
    for sp in splits:
        if not os.path.isdir(os.path.join(root, sp)):
            raise FileNotFoundError(
                f"split {sp!r} is not in the dataset at {root}. Available: "
                f"{sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))}")
        _plog(writer, f"[eval_held_out_splits @ep{step}] === {sp}")
        summary.update(eval_ood_horizon(cfg, model, norm, ecfg, writer, device, step,
                                        split=sp, tag=f"eval_split/{sp}"))
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
        # An mm_eps entry is (obs (T,D), act (T,A), frames) where frames is ALWAYS A DICT
        # {camera: (T,H,W,3)} -- see load_split_episodes_mm. Slicing the tuple elementwise treats
        # that dict as sliceable and raises `KeyError: slice(None, 1024, None)`, which is exactly
        # how this landed broken the first time.
        mm_eps = [(o_[:max_steps], a_[:max_steps], {k: v[:max_steps] for k, v in fr_.items()})
                  for o_, a_, fr_ in mm_eps]
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


def _caption_lines(text: str, px: int, max_lines: int = 4):
    """Wrap `text` to fit `px` wide, MEASURED with cv2.getTextSize rather than guessed from a characters-
    per-line heuristic (which overflowed the bar). Shrinks the font until the whole caption fits in
    max_lines, and returns (lines, scale, line_height) so the caller can size the bar to the text instead
    of the other way round."""
    import cv2
    fnt, pad = cv2.FONT_HERSHEY_SIMPLEX, 6
    for scale in (0.40, 0.36, 0.32, 0.28, 0.24, 0.20):
        th = cv2.getTextSize("Ag", fnt, scale, 1)[0][1]
        lines, cur = [], ""
        for w in str(text).split():
            trial = (cur + " " + w).strip()
            if cv2.getTextSize(trial, fnt, scale, 1)[0][0] <= px - 2 * pad:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines:
            return lines, scale, th + 5
    return lines[:max_lines], scale, th + 5


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
    assert not cfg.interpret.get("_unset"), (
        "no interpretability environment selected: pass `interpret=<env>` explicitly "
        "(starling | torus | pendulum). The factor definitions are per-environment and the default is a "
        "sentinel on purpose -- see conf/interpret/unset.yaml.")
    ic = OmegaConf.to_container(cfg.interpret, resolve=True)
    factors = ic["factors"]
    # REQUIRED per environment: what each raw action axis MEANS. It is written into the VLM prompt, and a
    # number whose meaning is not stated is worse than no number (see interpret.build_action_text).
    _axes = ic.get("action_axes")
    assert _axes, (f"conf/interpret/<env>.yaml must declare `action_axes` -- one {{name, positive, "
                   f"negative}} per RAW action axis, in order. It names the sticks for the VLM prompt.")
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
            _dims = fc["analytic"].get("dims")
            # `from: action` reads the COMMANDED action rather than the imagined proprio. For a world model
            # that is the sharper question -- does the video it paints show the motion it was TOLD to make
            # -- and the actions are exact inputs to the rollout, not predictions.
            _src = clip_acts if fc["analytic"].get("from") == "action" else pro_all
            ana[f] = I.bucketize(kind, [I.analytic_scalar(kind, p, getattr(ecfg, "R", 0.0), _dims, fc)
                                        for p in _src], fc, r=getattr(ecfg, "r", None))

    # ---- VLM labels (source: vlm factors — reads the RENDERED image) + N free-form captions (CLIP-style reward
    #      training, same call). ok = clips the VLM successfully returned. ----
    vlm_factors = {f: fc for f, fc in factors.items() if fc.get("source") == "vlm"}
    n_captions = int(ic.get("n_captions", 0))
    vlm = [None] * len(slices)
    ok = list(range(len(slices)))
    segmented, per_frame, min_frames, seg_cfg = False, [], 3, {}
    if vlm_factors or n_captions:
        key = I.openai_api_key()
        # SEGMENTED MODE: factors marked `per_frame` are asked once per SEGMENT of the clip rather than once
        # for the whole clip, so a label describes the steps it actually covers. Off unless the env config
        # declares `segments:` and marks at least one factor per_frame -- torus/pendulum are unaffected.
        seg_cfg.update(ic.get("segments") or {})
        per_frame[:] = [f for f, fc in vlm_factors.items() if fc.get("per_frame")]
        segmented = bool(seg_cfg) and bool(per_frame)
        min_frames = int(seg_cfg.get("min_frames", 3))
        schema = (I.build_segment_schema(vlm_factors, n_captions, H, int(seg_cfg.get("max", 4)))
                  if segmented else I.build_label_schema(vlm_factors, n_captions=n_captions))
        fidx = _np.unique(_np.linspace(0, H - 1, int(ic["vlm_frames"])).round().astype(int))
        vmodel = ic["vlm"]["model"]
        prompt = ic["prompt"]
        if n_captions and ic.get("caption_prompt"):
            prompt = prompt + "\n\n" + ic["caption_prompt"].format(n=n_captions)   # append the caption instructions
        if segmented:
            prompt = prompt + "\n\n" + (seg_cfg.get("prompt") or "").format(
                H=H, n_max=int(seg_cfg.get("max", 4)), k=min_frames, per_frame=", ".join(per_frame))
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
                                action_text=I.build_action_text(clip_acts[i], _axes) + _known(i))

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

    # ---- SEGMENTS -> per-step labels. Each factor also gets a clip-level MODAL label so the counts, the
    #      cross-check and the manifest keep working exactly as before. ----
    steps_lab, caps_step, n_merged = {}, {}, 0
    if segmented:
        for i in ok:
            segs, mg = I.repair_segments(vlm[i].get("segments") or [], H, min_frames)
            n_merged += mg
            keys = list(per_frame) + (["captions"] if n_captions else [])
            ex = I.expand_segments(segs, H, keys)
            for f in per_frame:
                steps_lab.setdefault(f, {})[i] = ex[f]
                vlm[i][f] = max(set(ex[f]), key=ex[f].count)          # modal label stands in for the clip
            if n_captions:
                caps_step[i] = ex["captions"]
                vlm[i]["captions"] = segs[0].get("captions") or []
            vlm[i]["_segments"] = segs
        _nseg = [len(vlm[i]["_segments"]) for i in ok]
        _plog(writer, f"[eval_interpret @ep{step}] segments: {_np.mean(_nseg):.2f} per clip on average "
                      f"(min {min(_nseg)}, max {max(_nseg)}) | {n_merged} short runs merged into neighbours "
                      f"-- that merge count IS the flicker rate, i.e. how often a claimed change did not last")

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
    # DROP THE FIRST STEP. Step 0's bag is built from the ENCODED REAL CONTEXT; steps 1+ are autoregressive.
    # Measured on starling: the step0 -> step1 centroid jump is 42.3 in latent space against 12.0 for the next
    # step and ~0.6 by step 10, and in UMAP every clip's step 0 lands in one tight blob 15 units from the
    # rollout. It is a different KIND of state, it is 1/H of every bucket, and it drags every projection.
    s0 = 1 if bool(ic.get("drop_first_step", False)) else 0
    Hs = H - s0
    if mode == "per_step":                                  # dense, comparable to eval_manifold
        pts = _np.concatenate([bags[i][s0:] for i in ok], axis=0)    # (len(ok)*Hs, D)
        clip_pos = _np.repeat(_np.arange(len(ok)), Hs)              # each point -> its clip's index within `ok`
        step_pos = _np.tile(_np.arange(s0, H), len(ok))              # each point -> its step within the clip
        psize = 2.5
        sub = (f"each point = one latent of an imagined rollout, steps {s0}..{H - 1} kept ({len(ok)} clips x "
               f"{Hs} = {len(pts):,} points" + (f"; {len(steps_lab)} factor(s) labelled PER STEP"
               if steps_lab else "; label broadcast from its clip") + ")")
    else:                                                    # one mean latent per clip (clean)
        pts = _np.stack([bags[i][s0:].mean(0) for i in ok])         # (len(ok), D)
        clip_pos = _np.arange(len(ok))
        step_pos = _np.zeros(len(ok), dtype=int)
        psize = 6.0
        sub = f"each point = the mean over {H}-steps of an imagined rollout ({len(ok)} clips = {len(pts):,} points)"

    # ---- project + plot every reducer via the shared library (evaluation/projection.py); it saves the fitted
    #      reducers too, so a projection is reusable later (reducer.transform(new_latents) — pca/umap/lda only) ----
    from .projection import project_and_plot
    pdir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_interpret", "saved_projections")
    os.makedirs(pdir, exist_ok=True)
    _np.save(os.path.join(pdir, "clip_index.npy"), clip_pos)                  # each point -> its clip's index within `ok`
    _np.save(os.path.join(pdir, "step_index.npy"), step_pos)                  # each point -> its STEP within the clip,
    #   which a consumer needs to find the SEGMENT a point belongs to (and hence that segment's captions).
    #   Without it the only recourse is assuming the clip-major ordering and dividing, which breaks silently
    #   the moment clips differ in length.
    # PER-POINT labels: a per-step factor gives each point the label of ITS OWN step; everything else is
    # still broadcast from the clip. This is the whole point of segmenting -- under per_step with a clip
    # label, a clip that changed view mislabels most of its own points.
    labels_pp = {f: ([steps_lab[f][ok[c]][t] for c, t in zip(clip_pos, step_pos)]
                     if (f in steps_lab and mode == "per_step")
                     else [labels_ok[f][c] for c in clip_pos]) for f in factors}
    transform_ok = project_and_plot(writer, "eval_interpret", pts, labels_pp, factors, step=step,
                                    point_size=psize, subtitle=sub,
                                    methods=tuple(ic.get("projection_methods", ("pca", "tsne", "umap"))),
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
            if f in steps_lab:
                # PER-STEP FACTOR: the unit of an example is a SEGMENT -- the contiguous run of steps that
                # actually carries the label -- not the clip. One clip that faces two walls contributes a
                # segment to EACH bucket, which is both correct and more informative than forcing it into one.
                lab = steps_lab[f][i]
                t = 0
                while t < len(lab):
                    u = t
                    while u + 1 < len(lab) and lab[u + 1] == lab[t]:
                        u += 1
                    if lab[t] in factors[f]["buckets"] and len(by_bucket[(f, lab[t])]) < grid * grid:
                        by_bucket[(f, lab[t])].append(frames[i][t:u + 1])
                    t = u + 1
            else:
                b = labels_ok[f][j]
                if b in factors[f]["buckets"] and len(by_bucket[(f, b)]) < grid * grid:
                    by_bucket[(f, b)].append(frames[i])
    for (f, b), clips in by_bucket.items():
        # Segments have different lengths, so pad each to the longest with BLACK rather than trimming or
        # looping: a tile that goes dark has simply ended, which reads correctly and loses no frames.
        n = max(len(c) for c in clips)
        clips = [c if len(c) == n else _np.concatenate([c, _np.zeros((n - len(c),) + c.shape[1:], c.dtype)])
                 for c in clips]
        writer.video(f"eval_interpret/examples/{f}/{b}", viz.tile_clips(clips, grid), fps_ex, step)
        if f in steps_lab:
            _lens = [int((c.sum(axis=(1, 2, 3)) > 0).sum()) for c in clips]
            _plog(writer, f"[eval_interpret @ep{step}] examples/{f}/{b}: {len(clips)} segments, "
                          f"median run {int(_np.median(_lens))}/{H} steps")

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
            # SELF-CONTAINED: video + actions + proprio + this clip's own labels/captions/segments, so an
            # imagination directory can be read without joining against labels.json or the manifest.
            json.dump({"id": int(i), "episode": int(slices[i][0]), "start": int(slices[i][1]),
                       "clip_len": H, "fps": fps,
                       "labels": {f: labels_ok[f][j] for f in factors},
                       "labels_per_step": {f: steps_lab[f][i] for f in steps_lab},
                       "segments": (vlm[i] or {}).get("_segments", []),
                       "captions": (vlm[i] or {}).get("captions", []),
                       "reasoning": (vlm[i] or {}).get("reasoning", "")},
                      open(os.path.join(cd, "labels.json"), "w"), indent=2)
        json.dump({"clip_len": H, "fps": fps, "point_mode": mode,
                   "trunks": [{"id": n, "kind": trunk_kind[n],
                               "file": f"{n}.{'mp4' if trunk_kind[n] == 'image' else 'npy'}"} for n in heads],
                   "clips": [{"id": int(i), "episode": int(slices[i][0]), "start": int(slices[i][1]),
                              "labels": {f: labels_ok[f][j] for f in factors}} for j, i in enumerate(ok)]},
                  open(os.path.join(imdir, "manifest.json"), "w"), indent=2)
        _plog(writer, f"[eval_interpret @ep{step}] saved {len(ok)} per-clip imaginations ({len(heads)} trunks) -> imaginations/")

    # ---- examples_captions/: a SAMPLE of clips with their caption written under the frame. The captions are
    #      the only product the reward head actually trains on, and until now they existed solely inside
    #      labels.json -- unreadable alongside the video they describe, which is the one way to judge whether
    #      a caption matches what you see. A sample, not all of them: burning text into video is slow and you
    #      only need to browse. Under segmentation each segment shows its own caption, so the text changes
    #      with the content. ----
    n_cap_ex = int(ic.get("n_caption_examples", 0))
    if n_cap_ex and n_captions:
        import cv2
        cdir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_interpret", "examples_captions")
        os.makedirs(cdir, exist_ok=True)
        for i in ok[:n_cap_ex]:
            clip = frames[i]
            hh, ww = clip.shape[1:3]
            # Lay every frame's caption out FIRST, so the bar is sized to the text that actually has to fit
            # (and one height is used for the whole clip, or the video would change shape mid-play).
            def _txt(t):
                cap = (caps_step.get(i, [None] * len(clip))[t] or (vlm[i] or {}).get("captions") or [""])
                return (cap[0] if isinstance(cap, (list, tuple)) and cap else
                        (cap if isinstance(cap, str) else ""))
            laid = [_caption_lines(_txt(t), ww) for t in range(len(clip))]
            lh = max(l[2] for l in laid)
            bar = max(l[2] * len(l[0]) for l in laid) + 12
            out = _np.zeros((len(clip), hh + bar, ww, 3), dtype=_np.uint8)
            out[:, :hh] = clip
            for t, (lines, scale, _) in enumerate(laid):
                y = hh + lh
                for line in lines:
                    cv2.putText(out[t], line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1,
                                cv2.LINE_AA)
                    y += lh
            viz.save_mp4(os.path.join(cdir, f"{i}.mp4"), out, max(1, fps // 2))
        _plog(writer, f"[eval_interpret @ep{step}] wrote {min(n_cap_ex, len(ok))} caption-annotated clips "
                      f"-> examples_captions/")

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
    # UNDER data.action_aggregate=concat one stored action is `subsample` raw commands laid end to end,
    # TIME-MAJOR (dataset.py: grp.reshape(n, -1) over (n, s, dim)), so dim i is raw axis i % dim at
    # sub-step i // dim. info.json only names the `dim` RAW axes, so the length check in
    # viz.fig_action_marginals rejects them and every panel falls back to a bare "a[i]" -- 16 anonymous
    # panels that cannot be read. Expand the raw names across the sub-steps instead.
    _adim = int(effective_action_dim(cfg))
    _sub = int(cfg.data.get("subsample", 1) or 1)
    if str(cfg.data.get("action_aggregate", "sum")) == "concat" and _sub > 1 and _adim % _sub == 0:
        _raw = _adim // _sub
        # names from the dataset when it has them, else a<axis> -- either way every panel says WHICH raw
        # axis and WHICH sub-step it is, instead of 16 anonymous a[i] tiles.
        base = list(action_names) if action_names and len(action_names) == _raw else [f"a{i}" for i in range(_raw)]
        action_names = [f"{n}·t+{j}" for j in range(_sub) for n in base]
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
    # A CHUNKED head predicts [a[t], a[t+1], ... a[t+K-1]] per context. Every product below is produced ONCE
    # PER LEAD TIME, into its own lead_<k>/ subfolder, because lead times are K DIFFERENT prediction problems
    # and pooling them into one histogram would quietly flatter the head. K=1 keeps the flat, unprefixed
    # layout every pre-chunk run wrote.
    pred_chunk = pred_norm.reshape(*pred_norm.shape[:-1], K, -1)   # (E,L-1,K,a) -- K=1 is the degenerate case
    leads = cfg.eval.get("action_dist_leads", "auto")
    if isinstance(leads, str):                                   # "auto": the near, middle and far end of the chunk
        leads = sorted({0, K // 2, K - 1})
    leads = sorted({int(k) for k in leads if 0 <= int(k) < K})
    win = int(cfg.eval.get("action_dist_window", 4) or 0)        # +/-w timesteps pooled per animation frame
    mx = int(cfg.eval.get("action_dist_max_frames", 0) or 0) or None   # cap the animations (cost ~ frames x dims)
    prog(50, f"head sampling (chunk K={K}, products at lead times {leads})")

    def _dir(k):   # flat layout at K=1 so old runs' product names are untouched; subfoldered once chunked
        return "eval_action_distribution/" + ("" if K == 1 else f"lead_{k:02d}/")

    q = np.linspace(0.0, 1.0, 512)
    fps = step_fps(cfg, ecfg)
    head_w1 = None

    # ---- THE CONDITIONAL SCORE ------------------------------------------------------------------------
    # W1 above compares POOLED histograms, which a model that ignores its context matches perfectly. These
    # need M draws per context instead of one; that costs ~2 ms per 1000 samples against an eval dominated
    # by the animations, so it is effectively free. 0 draws disables the whole block.
    COND = None
    nd = int(cfg.eval.get("action_dist_energy_draws", 32) or 0)
    if nd >= 4 and true_a.shape[1] > K:
        with torch.no_grad():
            dr = torch.stack([m.sample_action(h_ctx) for _ in range(nd)], dim=-2)   # (E,L-1,M,K*a)
        T0 = true_a.shape[1] - K + 1                          # contexts with a full chunk of future
        Y = np.stack([true_a[:, k0:k0 + T0] for k0 in range(K)], axis=2)            # (E,T0,K,a)
        X = norm.denorm_act(dr[:, :T0].reshape(dr.shape[0], T0, nd, K, -1).cpu()).numpy()
        Y = Y.reshape(-1, K, Y.shape[-1])
        X = X.reshape(Y.shape[0], nd, K, Y.shape[-1])
        COND = {"X": X, "Y": Y, "null": blind_null(Y, nd)}
        prog(45, f"conditional score: {nd} draws x {Y.shape[0]} contexts")
    for n_done, k in enumerate(leads):
        # slot k is scored against the recorded action k steps LATER -- the same alignment w1/lead_<k> uses
        T_ = true_a.shape[1] - k
        t_k = true_a[:, k:]
        p_k = norm.denorm_act(pred_chunk[:, :T_, k, :]).numpy()
        d = _dir(k)
        base = 50 + int(45 * n_done / max(1, len(leads)))        # progress budget shared across the lead times

        fig = viz.fig_action_marginals(t_k, p_k, names=action_names)
        writer.figure(f"{d}marginals", fig, step); plt.close(fig)
        if split_info is not None:                               # by-state products: TORUS-ONLY
            labels, low_name, high_name = split_info
            for name, arr in (("true", t_k), ("pred", p_k)):
                fig = viz.fig_action_by_state(arr, labels[:, :T_] if labels.ndim > 1 else labels, a_max,
                                              low_name=low_name, high_name=high_name,
                                              sampler_name=f"{asamp} · {name}", window=win)
                writer.figure(f"{d}by_state_{name}", fig, step); plt.close(fig)

        tm, pm = np.linalg.norm(t_k, axis=-1).reshape(-1), np.linalg.norm(p_k, axis=-1).reshape(-1)
        w1 = float(np.mean(np.abs(np.quantile(tm, q) - np.quantile(pm, q))))
        writer.scalar(f"{d}true_pred_w1", w1, step)
        w1_per_dim = [float(np.mean(np.abs(np.quantile(t_k[..., i], q) - np.quantile(p_k[..., i], q))))
                      for i in range(t_k.shape[-1])]
        live = [i for i in range(t_k.shape[-1]) if t_k[..., i].std() > 1e-6]
        writer.scalars({f"{d}w1/dim_{i}": w for i, w in enumerate(w1_per_dim)}, step)
        w1_mean = float(np.mean([w1_per_dim[i] for i in live])) if live else 0.0
        writer.scalar(f"{d}w1_mean", w1_mean, step)
        if COND is not None:                                     # this lead's slice of the conditional score
            Xk, Yk = COND["X"][:, :, k, :], COND["Y"][:, k, :]
            writer.scalar(f"{d}energy_skill", energy_skill(Xk, Yk, COND["null"][:, :, k, :], ), step)
        if k == leads[0]:
            head_w1 = w1
        prog(base + 2, f"lead {k}: w1={w1:.3f} w1_mean={w1_mean:.3f}")

        frames = viz.anim_action_distribution(t_k, p_k, a_max, window=win, max_frames=mx)
        writer.video(f"{d}animation_pooled", frames, fps, step)
        mframes = viz.anim_action_marginals(t_k, p_k, names=action_names, window=win, max_frames=mx)
        writer.video(f"{d}animation_marginals", mframes, fps, step)
        if split_info is not None:                               # by-state animation: TORUS-ONLY
            frames_bx = viz.anim_action_by_state(t_k, p_k, labels, a_max, low_name=low_name,
                                                 high_name=high_name, window=win)
            writer.video(f"{d}animation_byx", frames_bx, fps, step)
        prog(base + 12, f"lead {k}: animations")

    # EVERY lead time still gets its scalar, even the ones with no figures -- the cost is a quantile, and the
    # shape of w1-vs-lead is the thing that says how far ahead the prior stays faithful.
    if K > 1:
        for k in range(K):
            pk = np.linalg.norm(norm.denorm_act(pred_chunk[:, :true_a.shape[1] - k, k, :]).numpy(), axis=-1)
            tk = np.linalg.norm(true_a[:, k:], axis=-1)
            wk = float(np.mean(np.abs(np.quantile(tk.reshape(-1), q) - np.quantile(pk.reshape(-1), q))))
            writer.scalar(f"eval_action_distribution/w1/lead_{k}", wk, step)
    w1 = head_w1 if head_w1 is not None else 0.0
    writer.scalar("eval_action_distribution/true_pred_w1", w1, step)   # headline = the FIRST lead drawn
    if COND is not None:
        X, Y, nul = COND["X"], COND["Y"], COND["null"]
        N = X.shape[0]
        es = float(energy_score(X.reshape(N, X.shape[1], -1), Y.reshape(N, -1)).mean())
        sk = energy_skill(X.reshape(N, X.shape[1], -1), Y.reshape(N, -1), nul.reshape(N, X.shape[1], -1))
        writer.scalar("eval_action_distribution/energy_score", es, step)
        writer.scalar("eval_action_distribution/energy_skill_vs_blind", sk, step)
        # THE FLOOR OF THE OLD METRIC, logged on every chart: w1_mean below this line is not evidence of
        # anything conditional, because a model that ignores its context reaches it.
        nq = np.linalg.norm(nul[:, :, 0, :], axis=-1).reshape(-1)
        tq = np.linalg.norm(Y[:, 0, :], axis=-1).reshape(-1)
        writer.scalar("eval_action_distribution/w1_blind_null",
                      float(np.mean(np.abs(np.quantile(tq, q) - np.quantile(nq, q)))), step)
        hist, dev, verdict = rank_calibration(X[:, :, 0, :], Y[:, 0, :])
        writer.scalar("eval_action_distribution/rank_calibration_dev", dev, step)
        writer.scalars({f"eval_action_distribution/rank_decile/{i}": float(v) for i, v in enumerate(hist)},
                       step)
        rs = list(rest_skill(X[:, :, 0, :], Y[:, 0, :]))
        writer.scalars({f"eval_action_distribution/rest_auc/dim_{r['dim']}": r["auc"] for r in rs}, step)
        writer.scalars({f"eval_action_distribution/rest_brier_skill/dim_{r['dim']}": r["brier_skill"]
                        for r in rs}, step)
        _plog(writer, f"[eval_action_distribution @ep{step}] conditional: energy skill vs context-blind "
                      f"{sk:+.3f} | calibration {verdict} | rest AUC "
                      + " ".join(f"d{r['dim']}:{r['auc']:.3f}" for r in rs))
    prog(95, "scalars")

    if was:
        m.train()
    prog(100, f"done in {time.perf_counter() - t0:.1f}s -> eval_action_distribution/")
    return {"action_true_pred_w1": w1}


@torch.no_grad()
def eval_steer(cfg, model, norm, ecfg, writer, device, step=0):
    """LANGUAGE-STEERED PLANNING INSIDE THE IMAGINATION. Self-skips unless `steer.head` names a reward head.

    For each request phrase and each starting context from val: draw candidate action chunks from the model's
    own action prior, roll them through the world model, score every imagined latent with the language reward
    head, commit the best chunk, re-plan -- out to `steer.horizon` steps with no environment anywhere.

    TWO NUMBERS PER REQUEST, and the second is the one to trust. `reward_gain` is how much the reward the
    planner was maximising went up, which is nearly circular -- a planner that maximises a number will
    generally raise it. `motion` is derived from the COMMITTED ACTIONS alone, so "the plan scores well" and
    "the plan actually climbs" are separate claims. A request whose reward climbs while its commanded motion
    is unrelated has steered the reward, not the drone."""
    if not (cfg.get("steer", {}) or {}).get("head"):
        return {}
    from omegaconf import OmegaConf as _OC
    sc = _OC.to_container(cfg.steer, resolve=True)
    head = sc["head"]
    m = getattr(model, "_orig_mod", model)
    if not getattr(m, "action_head_enabled", False) and str(sc.get("proposal", "prior")) == "prior":
        _plog(writer, f"[eval_steer @ep{step}] proposal=prior but this checkpoint has NO action head -- "
                      f"load a train_action_model run, or set steer.proposal=gaussian")
        return {}
    from omegaconf import OmegaConf
    from ..data.dataset import load_split_episodes_mm
    from ..language.reward import LanguageReward
    from . import steering as S
    was = m.training
    m.eval()
    t0 = time.perf_counter()

    ic = OmegaConf.to_container(cfg.get("interpret", {}) or {}, resolve=True)
    axes = ic.get("action_axes")
    assert axes, "eval_steer needs `interpret=<env>` for action_axes (what each stick MEANS), see conf/interpret"
    lang = LanguageReward(str(head), device=(device if isinstance(device, str) else device.type))
    P, H = int(cfg.data.P), int(sc.get("horizon", 128))
    look = int(sc.get("lookahead", 0)) or int(getattr(m, "action_head_chunk", 1))
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or img_size,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    _pk = str(sc.get("proposal", "prior"))
    if _pk == "prior":
        prop = S.PriorProposal(m, prefix_guidance=bool(sc.get("prefix_guidance", True)),
                               prefix_freeze=int(sc.get("prefix_freeze", 1)),
                               prefix_decay=float(sc.get("prefix_decay", 0.5)))
    elif _pk == "data":
        # THE BANK: real chunks, from `bank_split` at `bank_stride`. Defaults are train at stride 1, which
        # is 24,737 chunks on starling-2 -- the first version used val at stride 8 and got 365, i.e. 1.5%
        # of what exists, which both under-powered the proposal and made a prefix-retrieval feasibility
        # test look hopeless when it had only been starved. Train also removes an unearned advantage: a
        # val bank contains the literal continuation of the context being planned from.
        from ..data.dataset import load_split_episodes
        _K = int(getattr(m, "action_head_chunk", 1))
        _bs = int(sc.get("bank_stride", 1) or 1)
        _be = load_split_episodes(resolve_data_root(cfg), str(sc.get("bank_split", "train")),
                                  repo_id=cfg.data.get("repo_id", "torus"))
        _ch = [torch.from_numpy(a[i:i + _K]).float()
               for _, a in _be for i in range(0, len(a) - _K, _bs)]
        assert _ch, f"no chunks of {_K} in the bank split -- episodes too short?"
        prop = S.DataProposal(norm.norm_act(torch.stack(_ch)).to(device),
                              prefix_retrieval=bool(sc.get("prefix_retrieval", True)),
                              retrieval_tau=float(sc.get("retrieval_tau", 0.05)))
    else:
        prop = S.GaussianProposal(float(sc.get("noise_sigma", 0.5)))
    prop = S.wrap(prop, sc.get("harness") or [], held_tol=float(sc.get("held_tol", 0.02)),
                  crossfade_decay=float(sc.get("crossfade_decay", 0.5)))
    g = torch.Generator(device=(device if isinstance(device, str) else device.type))
    g.manual_seed(int(sc.get("seed", 0)))
    # SEEDING THE IMAGINATION, not just the proposal. Under model.diffusion.stochastic_eval the dynamics
    # flow SAMPLES at every rolled step, from the GLOBAL rng -- so `g` (which now covers the candidate draw)
    # left the rollouts free, two runs of one config disagreed, and a prior-vs-gaussian comparison could not
    # be paired. Seed the global stream too, and RESTORE it afterwards: this routine also runs as a training
    # callback, where silently reseeding the process would perturb the run it is evaluating.
    _rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.manual_seed(int(sc.get("seed", 0)))
    rng = np.random.RandomState(int(sc.get("seed", 0)))
    starts = [(int(rng.randint(len(eps))), int(rng.randint(P, min(len(eps[0][0]) - 1, 400))))
              for _ in range(int(sc.get("n_contexts", 4)))]
    _plog(writer, f"[eval_steer @ep{step}] {len(sc['requests'])} requests x {len(starts)} contexts | "
                  f"horizon {H} ({H * float(cfg.data.get('subsample', 1)) / max(1e-9, step_fps(cfg, ecfg)):.1f}s) "
                  f"| lookahead {look} | {sc['n_samples']} candidates/block | proposal {prop.name} "
                  f"| objective {sc.get('objective', 'level')} "
                  f"| commit {sc.get('commit', 0) or look} | beta_jerk {sc.get('beta_jerk', 0.0)}"
                  f"{'' if (sc.get('commit', 0) or look) >= look else ' | OVERLAP -> continuity harnesses live'}")

    out, rows = {}, []
    for req in sc["requests"]:
        t_e = lang.text_embedding(req)[0] if lang.text_embedding(req).dim() > 1 else lang.text_embedding(req)
        gains, motions, curves, stills = [], [], [], []
        for ei, t in starts:
            o, a, fr = eps[ei]
            ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t - P:t])).float().unsqueeze(0).to(device)}
            for hh in img_heads:
                ctx[hh] = torch.from_numpy(fr[hh][t - P:t]).float().div(255.0).unsqueeze(0).to(device)
            ca = norm.norm_act(torch.from_numpy(a[t - P:t])).float().unsqueeze(0).to(device)
            bags, acts, sco = S.plan(m, lang, t_e, ctx, ca, prop, horizon=H, lookahead=look,
                                     n_samples=int(sc["n_samples"]), lam=float(sc.get("lam", 0.3)),
                                     objective=str(sc.get("objective", "level")),
                                     commit_rule=str(sc.get("commit_rule", "argmax")),
                                     commit=int(sc.get("commit", 0) or 0),
                                     beta_jerk=float(sc.get("beta_jerk", 0.0)), generator=g)
            gains.append(float(sco[-max(1, len(sco) // 8):].mean() - sco[:max(1, len(sco) // 8)].mean()))
            # DENORMALIZE FIRST. The planner works in normalized actions, and normalizing is (a-mean)/std --
            # the fore/aft stick has a raw mean near -0.46, so normalized zero is NOT stick-centre and the
            # SIGN of a normalized mean does not say which way the stick went. Read the direction off raw
            # stick units or the whole readout is measured about the wrong origin.
            rw = norm.denorm_act(acts.cpu()).numpy()
            motions.append(S.motion_readout(rw, axes, len(axes)))
            # STILLNESS: mean |stick| over axes and steps, in raw units. The per-axis means CANCEL -- a plan
            # that slams left then right averages to zero and reads as motionless -- so a request whose only
            # correct behaviour is a centred stick (`do nothing`) cannot be checked by them, and it must not
            # be checked by the reward either (the head anti-ranks stillness, so maximising R moves). This
            # number is small only if the plan really held still. Read it against the other requests' rows.
            stills.append(float(np.abs(rw.reshape(len(rw), -1, len(axes)).mean(axis=1)).mean()))
            curves.append(sco)
            # ---- ONE FOLDER PER (request, context): the imagined video with the request written under it,
            #      the imagined proprio, the committed actions in RAW stick units, and the numbers. Same
            #      shape as eval_interpret's imaginations/, so the same habits work -- and the request is
            #      IN the frame, which is the only way to check a plan by eye without cross-referencing. ----
            slug = re.sub(r"[^a-z0-9]+", "_", req.lower()).strip("_")
            pdir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_steer", "plans", slug,
                                f"ep{ei:03d}_t{t:04d}")
            os.makedirs(pdir, exist_ok=True)
            dec = m.to_obs(bags.unsqueeze(0), heads=img_heads + ["proprio"])
            pro = norm.denorm_obs(dec["proprio"][0].cpu()).numpy()
            # THE LATENTS THE REWARD ACTUALLY SAW, in the flattened-bag space lang.score consumes -- the
            # same space eval_interpret fit its reducers and the reward head on. Dumped in STEP ORDER so a
            # diagnostic can bin by horizon: the head was fit on 15-step imaginations and the planner scores
            # out to `horizon`, and whether f_z still discriminates that far out is checkable, not a guess.
            np.save(os.path.join(pdir, "latents.npy"), bags.reshape(len(bags), -1).cpu().numpy().astype(np.float32))
            np.save(os.path.join(pdir, "proprio.npy"), pro.astype(np.float32))
            np.save(os.path.join(pdir, "actions.npy"), rw.astype(np.float32))   # RAW stick units
            # ---- THE PROPRIO PRODUCTS, same machinery as the world model's open-loop long-horizon
            #      rollouts (products.emit_openloop -> viz.fig_paths_3d + viz.fig_pos_vs_time), so a plan
            #      reads like every other rollout in the project. The recorded future is drawn for SCALE
            #      ONLY and labelled as such: the plan chose its own actions, so that curve is not the
            #      ground truth of this rollout and calling it GT would be a lie. ----
            pos = list((cfg.environments.get("position_idx") or [0, 1, 2]))
            # `plots` gates the two figures the same way `video` gates the mp4: a many-context sweep writes
            # n_requests x n_contexts of each, and at 26 x 16 that is 832 pngs nobody opens. The .npy and
            # plan.json always land -- they are what the analyses read.
            if len(pos) == 3 and bool(sc.get("plots", True)):
                ctx_xyz = o[t - P:t][:, pos]                       # `o` is RAW (the ctx dict norms it itself)
                rec_xyz = o[t:t + H][:, pos]                       # what the drone ACTUALLY did from here
                pln_xyz = pro[:, pos]
                anch = ctx_xyz[-1:]
                cl = ("recorded (other actions)", "plan")
                rec3 = np.concatenate([anch, rec_xyz])
                pln3 = np.concatenate([anch, pln_xyz])
                f3 = viz.fig_paths_3d(ctx_xyz, rec3, pln3, curve_labels=cl,
                                      title=f"{req} | ep{ei} t{t} | {prop.name}")
                f3.savefig(os.path.join(pdir, "proprio_3d.png"), dpi=110, bbox_inches="tight")
                # The per-axis panels share ONE step axis, so both curves must be the same length -- and the
                # recorded future is SHORTER whenever the start sits within `horizon` of the episode end
                # (t can reach 400 of ~445). Pad it with NaN, which matplotlib simply stops drawing, rather
                # than trimming the plan: the plan out to 128 is the thing being looked at.
                recT = np.full_like(pln3, np.nan)
                recT[:len(rec3)] = rec3[:len(recT)]
                fa = viz.fig_pos_vs_time(ctx_xyz, recT, pln3, fork_step=P, curve_labels=cl,
                                         title=f"{req} | ep{ei} t{t} | {prop.name}")
                fa.savefig(os.path.join(pdir, "proprio_axes.png"), dpi=110, bbox_inches="tight")
                plt.close(f3); plt.close(fa)
            json.dump({"request": req, "episode": int(ei), "start": int(t), "horizon": int(H),
                       "proposal": prop.name, "objective": str(sc.get("objective", "level")),
                       "commit": int(sc.get("commit", 0) or look), "lookahead": int(look),
                       "reward_gain": gains[-1],
                       "reward_curve": [float(x) for x in sco],
                       "commanded_motion": motions[-1], "stillness": stills[-1],
                       "obs_columns": "see data/rosbag.py STATE_COLUMNS (position/velocity/quat/ang-vel/acc)",
                       "alignment": ("actions[i] is applied AT proprio[i] and produces proprio[i+1]; the "
                                     "first imagined frame is driven by the last recorded context action, "
                                     "which the planner did not choose"),
                       "note": ("reward_curve is the objective the planner MAXIMISED, so its rise is near "
                                "circular; commanded_motion is derived from the chosen actions alone and is "
                                "the independent check on whether the plan obeyed the request.")},
                      open(os.path.join(pdir, "plan.json"), "w"), indent=2)
            if bool(sc.get("video", True)):
                import cv2
                fr = (dec[img_heads[0]][0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                hh, ww = fr.shape[1:3]
                laid = [_caption_lines(f"{req}  |  step {i}/{H}  reward {sco[i]:+.3f}", ww)
                        for i in range(len(fr))]
                lh = max(l[2] for l in laid)
                bar = max(l[2] * len(l[0]) for l in laid) + 12
                out_v = np.zeros((len(fr), hh + bar, ww, 3), dtype=np.uint8)
                out_v[:, :hh] = fr
                for i, (lines, scale, _) in enumerate(laid):
                    y = hh + lh
                    for line in lines:
                        cv2.putText(out_v[i], line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                                    (240, 240, 240), 1, cv2.LINE_AA)
                        y += lh
                viz.save_mp4(os.path.join(pdir, "image.mp4"), out_v, step_fps(cfg, ecfg))
        dom = max(motions[0], key=lambda k: abs(motions[0][k]["mean"]))
        mv = {k: float(np.mean([mo[k]["mean"] for mo in motions])) for k in motions[0]}
        rows.append((req, float(np.mean(gains)), float(np.mean(stills)), dom, mv))
        out[f"eval_steer/reward_gain/{req.replace(' ', '_')}"] = float(np.mean(gains))
        out[f"eval_steer/stillness/{req.replace(' ', '_')}"] = float(np.mean(stills))
        _plog(writer, f"[eval_steer @ep{step}] {req!r}: reward_gain {np.mean(gains):+.4f} | "
                      f"stillness {np.mean(stills):.3f} | commanded "
                      + ", ".join(f"{k} {v:+.2f}" for k, v in mv.items()))
    writer.scalars(out, step)
    print(f"\n  {'request':34s} {'reward gain':>12s} {'still':>7s}   commanded motion (mean stick per axis)")
    print("  " + "-" * 104)
    for req, gain, still, dom, mv in rows:
        print(f"  {req:34s} {gain:>+12.4f} {still:>7.3f}   " + "  ".join(f"{k} {v:+.2f}" for k, v in mv.items()))
    print("  reward gain is near circular (the planner maximised it); the stick columns are not. `still` is"
          "\n  mean |stick| and is the ONLY column that can show a request for stillness.")
    if was:
        m.train()
    torch.set_rng_state(_rng_state[0])
    if _rng_state[1] is not None:
        torch.cuda.set_rng_state_all(_rng_state[1])
    _plog(writer, f"[eval_steer @ep{step}] done in {time.perf_counter() - t0:.1f}s -> eval_steer/")
    return {"eval_steer_requests": float(len(rows))}


from .timing import eval_timing as _eval_timing          # fixed-workload benchmark, model-agnostic

REGISTRY = {"steer": eval_steer, "ood_horizon": eval_ood_horizon,
            "held_out_splits": eval_held_out_splits, "ood_visual": eval_ood_visual,
            "ood_geometric": eval_ood_geometric, "ood_dynamics": eval_ood_dynamics,
            "control": eval_control, "denoising_multistep": eval_denoising_multistep,
            "denoising_aggregate": eval_denoising_aggregate, "denoising_filmstrip": eval_denoising_filmstrip,
            "ae_floor": eval_ae_floor, "manifold": eval_manifold,
            "interpret": eval_interpret, "action_distribution": eval_action_distribution,
            "timing": _eval_timing}


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
