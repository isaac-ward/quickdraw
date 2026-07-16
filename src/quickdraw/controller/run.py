"""Run the dual MPPI control eval (oracle vs learned) and log it under eval_control/.

Black = true-dynamics controller (oracle baseline), grey = learned-model controller, both executing
on the true env through the same random 8-goal sequence. Shared by training-time eval and the
standalone `eval_control` entrypoint.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ..logging import viz
from .mppi import MPPIConfig, run_control


def _plog(writer, msg: str):
    """Append a progress line to the run's TOP-LEVEL progress.log (and stdout) so eval timing shows up
    in the SAME file as ProgressPrinter's startup/epoch lines. writer.dir is run_dir/logs (see
    make_writer), so its parent is the run_dir where progress.log lives."""
    msg = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"  # wall-clock stamp so durations are exact
    print(msg, flush=True)
    run_dir = os.path.dirname(writer.dir.rstrip("/"))  # writer.dir = run_dir/logs -> parent is run_dir
    try:
        with open(os.path.join(run_dir, "progress.log"), "a") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _agent(res, i, color, R, r):
    """Build a control_compare_frames agent dict for episode i (pads actions to the path length for the arrow)."""
    path, act = res["paths"][i], res["actions"][i]        # (T,3), (T-1,2)
    act = np.concatenate([act, act[-1:]], axis=0) if len(act) else np.zeros((len(path), 2))
    return {"path": path, "goal_seq": res["goal_seqs"][i], "color": color,
            "avec": viz.action_ambient(path, act, R, r)}  # ambient applied action (T,3)


def run_and_log_control(cfg, model, normalizer, ecfg, writer, device, step=0) -> dict:
    # reuse_render is a RENDER knob living in the control config; strip it before building MPPIConfig
    # (which has no such field) so MPPIConfig(**...) doesn't choke on the extra key.
    mppi_kwargs = {k: v for k, v in cfg.control.items() if k != "reuse_render"}
    fpv = None                                    # image models: render FPV context in the MPPI loop
    core = getattr(model, "_orig_mod", model)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)   # image head name, or None (proprio-only)
    if img_head is not None:                       # only render FPV when there's a real image head
        img_size = next((mod.ae.cfg.img_size for mod in core.modalities.values() if hasattr(mod, "ae")), 128)
        try:
            coloring = json.load(open(os.path.join(cfg.data.root, "dataset_card.json"))).get("coloring", {}).get("train", "rainbow")
        except OSError:
            coloring = "rainbow"
        fpv = {"coloring": coloring, "fov": float(cfg.data.fpv_fov), "size": int(img_size)}
        _plog(writer, f"[eval_control @ep{step}] multimodal: FPV render in the MPPI loop (coloring={coloring}, size={img_size})")

    # language steering: request + reward head -> the LEARNED controller maximizes R(latent, request) with the
    # oracle OFF (a single controller, same code spine). Otherwise the default dual goal race (oracle vs learned).
    lang = cfg.get("language")
    reward, request = None, None
    if lang is not None and lang.get("request") and lang.get("head"):
        from ..language.reward import LanguageReward
        request = str(lang.request)
        reward = LanguageReward(lang.head, device=device)
        reward.text_embedding(request)   # validate up front: raises if the request matches no known buckets (compound OK)
        ov = dict(lang.get("overrides") or {})   # language-mode control knobs (n_episodes/max_steps/...) over `control:`
        mppi_kwargs.update(ov)
        _plog(writer, f"[eval_control @ep{step}] LANGUAGE steering -> '{request}' "
                      f"({os.path.basename(lang.head)}); oracle OFF, single learned controller"
                      + (f"; overrides {ov}" if ov else ""))
    n_ctrl = 1 if reward is not None else 2
    n_plot = int(mppi_kwargs.pop("n_plot", 1))    # NOT an MPPIConfig field: how many episodes to render products for
    _plog(writer, f"[eval_control @ep{step}] start: MPPI {mppi_kwargs['n_episodes']} eps x {n_ctrl} controller(s), "
                  f"{mppi_kwargs['num_samples']} samples, H={mppi_kwargs['horizon']}, max_steps={mppi_kwargs['max_steps']}, "
                  f"render {n_plot} episode(s)")
    t = time.perf_counter()
    res, _ = run_control(model, normalizer, ecfg, MPPIConfig(**mppi_kwargs), device=device,
                         log=lambda m: _plog(writer, f"[eval_control @ep{step}]   {m}"), fpv=fpv,
                         reward=reward, request=request, oracle=(reward is None), n_plot=n_plot)
    t_ctrl = time.perf_counter() - t
    # what matters: cost of ONE MPPI replan (= one action chunk). t_ctrl covers the controller(s) + chunk
    # execution over n_chunks replans, so per-chunk wall time = t_ctrl / n_chunks.
    mppi_chunk_s = t_ctrl / max(1, res["n_chunks"])
    mppi_chunk_hz = 1.0 / mppi_chunk_s if mppi_chunk_s > 0 else 0.0
    _plog(writer, f"[eval_control @ep{step}] MPPI done: {res['n_chunks']} replans over {res['n_steps']} steps "
                  f"-> {mppi_chunk_s * 1000:.0f} ms/chunk ({mppi_chunk_hz:.1f} hz)")
    R, r, fps = ecfg.R, ecfg.r, round(1.0 / ecfg.dt)
    from ..evaluation.products import log_image_head, product_tag

    # controllers present: 'pred' (learned) always; 'true' (oracle) only in the goal race (oracle on).
    kinds = [k for k in ("true", "pred") if k in res]
    colors = {"true": "black", "pred": "dimgray"}
    labels = {"true": "oracle", "pred": "learned"}
    NP = res["n_plot"]

    def _goal_changes(goal_seq):   # steps where the episode's goal marker jumps (explain the distance sawtooth)
        g = np.asarray(goal_seq)
        return (np.where(np.any(g[1:] != g[:-1], axis=-1))[0] + 1).tolist()

    # per-episode products (parallel episodes from different inits): control_video_i (+scene), <head>/rollout_i
    # (+filmstrip), and a distance curve_i. goal race: black oracle vs grey learned; language: single grey agent.
    ctrl_frames = []                       # collected per-episode control videos -> tiled into control_video_combined
    for i in range(NP):
        ti = time.perf_counter()
        agents = [_agent(res[k], i, colors[k], R, r) for k in kinds]
        nf = len(agents[0]["path"])
        _plog(writer, f"[eval_control @ep{step}] rendering control video #{i} ({nf} frames, GPU/EGL)...")
        vtitle = (f'"{request}"  #{i}' if reward is not None else f"control: true vs pred #{i}")
        frames = viz.control_compare_frames(R, r, "hsv", agents, n_frames=nf, title=vtitle,
                                            fan_seq=res["fan_seqs"][i],  # pred's MPPI candidate fan, colored by score
                                            reuse=bool(cfg.control.get("reuse_render", False)),
                                            show_goals=(reward is None),  # language mode has no target -> no goal ring
                                            log=lambda m, i=i: _plog(writer, f"[eval_control @ep{step}]   video #{i} {m}"))
        writer.video(product_tag("eval_control", "control_video", i=i), frames, fps, step)
        ctrl_frames.append(frames)
        scene_desc = (f"Language-steered MPPI on the torus (episode {i}): a single GREY learned-model agent "
                      f"steering to maximize the language reward R(latent, '{request}'). The action arrow per step "
                      f"is the applied control."
                      if reward is not None else
                      f"Dual MPPI control on the torus (episode {i}): a BLACK oracle agent (true dynamics) and a "
                      f"GREY learned-model agent, each navigating to a sequence of goals. Each goal is a RING zone "
                      f"on the surface; the action arrow per step is the applied control.")
        writer.scene(product_tag("eval_control", "control_video", i=i), {
            "description": scene_desc,
            "coordinate_system": "world xyz, same space as the torus",
            "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
            "goal_zone_ring_radius": float(0.0675 * (R + r)),
            "agents": [{"name": labels[k], "color": a["color"], "path_xyz": a["path"], "goal_per_step_xyz": a["goal_seq"],
                        "action_arrow_per_step": {"origins_xyz": a["path"], "vectors_xyz": a["avec"]}}
                       for k, a in zip(kinds, agents)],
        }, step)

        # (MM) predictor-in-the-loop video: the model's imagined FPV for the SELECTED plan vs the actual FPV.
        if res.get("pred_fpv_videos") is not None:
            pv = res["pred_fpv_videos"][i]   # context_len=0 -> top=pred, bottom=actual over the whole run
            log_image_head(writer, "eval_control", img_head, i, pv["actual"], pv["pred"], step, fps,
                           context_len=0, title=f"{img_head} #{i} pred(top)/actual(bottom)")

        # per-episode realized-distance curve
        if reward is not None:   # language: realized distance 1 - R(state, request) (lower = redder, 0 = on target)
            rc = np.asarray(res["pred"]["dist_curves"][i])
            klab = f"learned: distance to '{request}' (1 - R)"
            curve = viz.fig_error_vs_step({klab: rc}, colors={klab: "dimgray"}, yscale="linear")
            writer.figure(product_tag("eval_control", "distance_to_request", i=i), curve, step); plt.close(curve)
            # reward TRACE (4 lines): {imagined, achieved} x {reward head, ground truth}.
            from ..environments import torus as T
            rq, xyz = str(request).lower(), np.asarray(agents[0]["path"])

            def _gt(path):   # torus ground-truth reward on a path for the buckets named in the request
                g = [T.color_reward(path, b) for b in cfg.interpret.factors.color.buckets if str(b).lower() in rq] + \
                    [T.position_reward(path, r, b) for b in cfg.interpret.factors.positioning.buckets if str(b).lower() in rq]
                return np.mean(g, axis=0) if g else None
            lines, cols = {"achieved (reward head)": 1.0 - rc}, {"achieved (reward head)": "tab:blue"}
            gta = _gt(xyz)
            if gta is not None:
                lines["achieved (ground truth)"] = gta; cols["achieved (ground truth)"] = "tab:green"
            if res.get("imag_head_curves") is not None:   # the chosen plan's IMAGINED belief (world-model prediction)
                lines["imagined (reward head)"] = res["imag_head_curves"][i]; cols["imagined (reward head)"] = "tab:cyan"
                igt = _gt(np.asarray(res["imag_paths"][i]))
                if igt is not None:
                    lines["imagined (ground truth)"] = igt; cols["imagined (ground truth)"] = "tab:olive"
            tr = viz.fig_error_vs_step(lines, colors=cols, yscale="linear",
                caption="achieved = on the REAL executed state; imagined = the chosen plan's PREDICTED state (world-model belief).\n"
                        "reward head = cos(f_z, f_t(request)); ground truth = torus reward on the path.\n"
                        "imagined vs achieved  =>  world-model / imagination accuracy (drift).\n"
                        "reward-head vs ground-truth  =>  alignment / grounding accuracy.\n"
                        "all four high and together  =>  the model imagines right, steers there, and the head agrees with truth.")
            writer.figure(product_tag("eval_control", "reward_trace", i=i), tr, step); plt.close(tr)
        else:                    # goal race: distance to current goal, verticals at goal switches
            k_true, k_pred = "true (oracle): dist to current goal", "pred (learned): dist to current goal"
            curve = viz.fig_error_vs_step({k_true: res["true"]["dist_curves"][i], k_pred: res["pred"]["dist_curves"][i]},
                                          colors={k_true: "black", k_pred: "dimgray"},
                                          vlines={"black": _goal_changes(res["true"]["goal_seqs"][i]),
                                                  "dimgray": _goal_changes(res["pred"]["goal_seqs"][i])},
                                          yscale="linear")  # distance to goal is bounded -> linear reads better
            writer.figure(product_tag("eval_control", "distance_to_goal", i=i), curve, step); plt.close(curve)
        _plog(writer, f"[eval_control @ep{step}] episode #{i} rendered in {time.perf_counter() - ti:.1f}s")

    # combined control video: tile the per-episode control videos into one grid — all inits of the request at once.
    if len(ctrl_frames) > 1:
        import math
        Tm = min(len(f) for f in ctrl_frames)
        nc = int(math.ceil(math.sqrt(len(ctrl_frames)))); nr = int(math.ceil(len(ctrl_frames) / nc))
        Hh, Ww = ctrl_frames[0].shape[1:3]
        comb = np.zeros((Tm, nr * Hh, nc * Ww, 3), dtype=ctrl_frames[0].dtype)
        for k, f in enumerate(ctrl_frames):
            rr, cc = divmod(k, nc)
            comb[:, rr * Hh:(rr + 1) * Hh, cc * Ww:(cc + 1) * Ww] = f[:Tm]
        writer.video(product_tag("eval_control", "control_video_combined"), comb, fps, step)
        _plog(writer, f"[eval_control @ep{step}] control_video_combined ({len(ctrl_frames)} eps, {nr}x{nc} grid, {Tm} frames)")

    # (language) world-model latent-space animations: agent moving through the eval_interpret projections toward X_c/X_r.
    if reward is not None and lang.get("interpret_run") and res.get("agent_latents") is not None:
        from omegaconf import OmegaConf

        from ..evaluation.products import load_latent_projection, render_latent_video
        lrun = str(lang.interpret_run)
        for spec in (lang.get("interpret_plots") or [{"method": "lda", "factor": "color", "dims": [2, 3]}]):
            method, factor = str(spec["method"]), str(spec["factor"])
            fc = OmegaConf.to_container(cfg.interpret.factors[factor], resolve=True)
            for dim in [int(d) for d in spec.get("dims", [2, 3])]:
                proj = load_latent_projection(lrun, method, factor, dim, fc=fc, reward=reward, request=request,
                                               hull_frac=float(lang.get("hull_frac", 0.8)))
                if proj is None:
                    _plog(writer, f"[eval_control @ep{step}] latent anim: {method} {dim}d has no out-of-sample map (skip)")
                    continue
                for i in range(min(NP, 1)):   # latent animations are per-mechanism; 1 episode suffices (render cost)
                    ti = time.perf_counter()
                    vid = render_latent_video(proj, res["agent_latents"][i],
                                              title=f'"{request}"  ·  {factor} ({method} {dim}d)  #{i}',
                                              log=lambda m, i=i, method=method, dim=dim:
                                                  _plog(writer, f"[eval_control @ep{step}]   anim {factor}/{method}/{dim}d #{i} {m}"))
                    # eval_control/world_model_latent_space_plots/<categorical>/<reducer>/<nd>d_<i>
                    writer.video(product_tag(f"eval_control/world_model_latent_space_plots/{factor}/{method}", f"{dim}d", i=i),
                                 vid, fps, step)
                    _plog(writer, f"[eval_control @ep{step}] world_model_latent_space_plots/{factor}/{method}/{dim}d_{i} "
                                  f"({len(vid)} frames, {time.perf_counter() - ti:.0f}s)")

    # (language) JOINT-f_z space: reward FIELD (static) + agent animation (concept + reward-field colorings). The
    # aligned space where f_z(latent) meets f_t(text) — where the reward gradient the planner climbs actually lives.
    if reward is not None and lang.get("interpret_run") and res.get("agent_latents") is not None:
        import glob

        import torch
        from omegaconf import OmegaConf

        from ..evaluation.manifold import pad_lims, reduce_dims
        from ..evaluation.projection import animate_joint_space
        base = glob.glob(os.path.join(str(lang.interpret_run), "logs", "epoch_*", "eval_interpret"))
        if base:
            sp = os.path.join(base[0], "saved_projections")
            lat = np.load(os.path.join(sp, "latents.npy")).astype(np.float32)
            clip_idx = np.load(os.path.join(sp, "clip_index.npy"))
            recs = json.load(open(os.path.join(base[0], "labels.json")))
            t_e = reward.text_embedding(request)
            with torch.no_grad():                                    # f_z: WM latent -> joint space; reward field over the cloud
                fz = reward.f_z(torch.from_numpy(lat).to(reward.device)).cpu().numpy()
                rfield = reward.score(torch.from_numpy(lat).to(reward.device), t_e).cpu().numpy()
            t_e_np = t_e.cpu().numpy()
            _plog(writer, f"[eval_control @ep{step}] reward field '{request}' (range {rfield.min():.2f}..{rfield.max():.2f})")
            for factor in ("color", "positioning"):
                fc = OmegaConf.to_container(cfg.interpret.factors[factor], resolve=True)
                labels = [recs[int(c)]["label"][factor] for c in clip_idx]
                yb = {b: k for k, b in enumerate(fc["buckets"])}
                e, red = reduce_dims(fz, "lda", n_components=2, seed=0, return_reducer=True, y=np.array([yb[l] for l in labels]))
                goal2d = red.transform(t_e_np[None])[0][:2]     # the request f_t(request) projected -> its spot in this layout
                # STATIC reward field in THIS factor's LDA layout (readable gradient; the reward_*.mp4 animates the same)
                ff = viz.fig_points_2d(e, color=rfield, cbar_label=f"reward  cos(f_z, f_t('{request}'))",
                                       lims=pad_lims(np.concatenate([e, goal2d[None]])), point_size=3.0,
                                       annotations=[{"pos": goal2d, "text": f"request: {request}"}],
                                       title=f"reward field '{request}' — {factor} LDA")
                writer.figure(product_tag(f"eval_control/joint_latent_space_plots/{factor}/lda", "reward_field"), ff, step)
                plt.close(ff)
                for i in range(min(NP, 1)):   # 1 episode's joint animation suffices; combined control video shows all inits
                    with torch.no_grad():
                        fzt = reward.f_z(torch.from_numpy(np.asarray(res["agent_latents"][i], np.float32)).to(reward.device)).cpu().numpy()
                    for rmode, tag in ((False, "concept"), (True, "reward")):
                        fr = animate_joint_space(fz, fzt, t_e_np, method="lda", labels=labels, factor_cfg=fc,
                                                 reward_mode=rmode, goal_label=request, title=f"'{request}' · {factor} joint ({tag}) #{i}")
                        writer.video(product_tag(f"eval_control/joint_latent_space_plots/{factor}/lda", f"{tag}_2d", i=i),
                                     fr, fps, step)   # <type>_<nd>d_<i>.mp4 (type = concept|reward), matching the other products
                    _plog(writer, f"[eval_control @ep{step}] joint anim {factor}/lda #{i} (concept+reward)")

    # aggregate scalars (over ALL episodes, once)
    summary = {"n_steps": res["n_steps"], "time/mppi_chunk_s": mppi_chunk_s, "time/mppi_chunk_hz": mppi_chunk_hz}
    if reward is not None:   # representative (episode 0) realized-distance start/end/delta
        rc = res["pred"]["dist_curves"][0]
        summary |= {f"distance/{request}/start": float(rc[0]), f"distance/{request}/end": float(rc[-1]),
                    f"distance/{request}/delta": float(rc[-1] - rc[0])}
        _plog(writer, f"[eval_control @ep{step}] language '{request}' distance (ep0) "
                      f"{rc[0]:.3f} -> {rc[-1]:.3f} (delta {rc[-1] - rc[0]:+.3f})")
    else:
        # {true,pred,diff}/{metric} (slash org; diff = true - pred). Only these — no underscore duplicates.
        for m in ("success_rate", "mean_goals_reached", "mean_steps_to_complete", "mean_seconds_to_complete"):
            summary[f"true/{m}"] = res["true"][m]
            summary[f"pred/{m}"] = res["pred"][m]
            summary[f"diff/{m}"] = res["true"][m] - res["pred"][m]
        # goals-reached is meaningful even when seconds-to-complete is NaN (nothing settled all goals)
        _plog(writer, f"[eval_control @ep{step}] goals reached / {res['n_goals']}: "
                      f"oracle={res['true']['mean_goals_reached']:.2f} (success {res['true']['success_rate']:.2f}, "
                      f"{res['true']['mean_seconds_to_complete']:.2f}s) "
                      f"pred={res['pred']['mean_goals_reached']:.2f} (success {res['pred']['success_rate']:.2f})")
    writer.scalars({f"eval_control/{k}": v for k, v in summary.items()}, step)
    # write the raw summary next to THIS epoch's control media (logs/epoch_<i>/eval_control/), not the flat run
    # root — mirrors how writer.video/figure organize by epoch, so it's per-epoch (not clobbered each eval).
    ep_dir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_control")   # writer.dir is already run_dir/logs
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
