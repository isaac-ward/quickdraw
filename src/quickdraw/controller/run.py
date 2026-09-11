"""Run the dual MPPI control eval (oracle vs learned) and log it under eval_control/.

Black = true-dynamics controller (oracle baseline), grey = learned-model controller, both executing
on the true env through the same random 8-goal sequence. Shared by training-time eval and the
standalone `eval_control` entrypoint.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ..environments.base import SceneOverlay, wants_diagnostics
from ..environments.registry import make_env
from ..environments.torus_utils import TorusConfig
from ..logging import viz
from ..training.setup import step_fps
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
    # reuse_render / reward_override are control-config knobs that aren't MPPIConfig fields; strip them
    # before building MPPIConfig so MPPIConfig(**...) doesn't choke on the extra keys.
    mppi_kwargs = {k: v for k, v in cfg.control.items() if k not in ("reuse_render", "reward_override")}
    fpv = None                                    # image models: render FPV context in the MPPI loop
    core = getattr(model, "_orig_mod", model)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)   # image head name, or None (proprio-only)
    if img_head is not None:                       # only render FPV when there's a real image head
        img_size = next((mod.ae.cfg.img_size for mod in core.modalities.values() if hasattr(mod, "ae")), 128)
        # TORUS TEST BY TYPE, not by attribute. `hasattr(ecfg, "R")` was not a torus test: RecordedConfig
        # carries R/r/init_speed as INERT PLACEHOLDERS (environments/recorded.py, "train_world_model reads
        # e.R / e.r / e.init_speed unconditionally"), so it was True for every recorded env and sent drone
        # runs down the torus FPVRenderer path. There they hit int(img_size) on a (112,192) tuple and raised
        # a TypeError, which the callback treats as a BUG (counted toward the fatal streak) instead of the
        # NotImplementedError clean-skip it has for "this env has no simulator" -- the actual reason.
        if isinstance(ecfg, TorusConfig):          # torus: the FPVRenderer fast path (parity-critical, unchanged)
            try:
                from ..training.setup import resolve_data_root
                coloring = json.load(open(os.path.join(resolve_data_root(cfg), "dataset_card.json"))).get("coloring", {}).get("train", "rainbow")
            except OSError:
                coloring = "rainbow"
            fpv = {"coloring": coloring, "fov": float(cfg.data.fpv_fov), "size": int(img_size)}
            _plog(writer, f"[eval_control @ep{step}] multimodal: FPV render in the MPPI loop (coloring={coloring}, size={img_size})")
        else:                                      # generic env: mppi falls back to env.render_obs in the loop
            # NOT int(): a non-square image head carries img_size as an (H, W) TUPLE (starling is (112,192)),
            # and coercing it raised `int() argument must be ... not 'tuple'` BEFORE control ever reached the
            # env -- so every starling run logged a misleading TypeError instead of the real reason (a
            # recorded env has no simulator to step). Only the torus branch above consumes `size`, as a
            # square int for FPVRenderer; the generic path renders through env.render_obs and never reads it.
            fpv = {"size": img_size}
            _plog(writer, f"[eval_control @ep{step}] multimodal: in-loop image context via env.render_obs (size={img_size})")

    # language steering: request + reward head -> the LEARNED controller maximizes R(latent, request) with the
    # oracle OFF (a single controller, same code spine). Otherwise the default dual goal race (oracle vs learned).
    lang = cfg.get("language")
    reward, request, requests = None, None, None
    if lang is not None and lang.get("head") and (lang.get("requests") or lang.get("request")):
        from ..language.reward import LanguageReward
        # MULTI-QUERY: language.requests = a LIST of texts -> one episode PER request, all steered in one batch
        # (so they share the torus). Single language.request -> the classic single-request run (n_episodes inits).
        req_list = [str(x) for x in (lang.get("requests") or [lang.request])]
        request = req_list[0]                     # representative text for the shared joint-space / reward-field plots
        # single request -> requests=None so mppi BROADCASTS it across all n_episodes inits (the classic run);
        # a LIST (>1) -> per-episode targets, one episode PER request steered in one batch.
        requests = req_list if len(req_list) > 1 else None
        reward = LanguageReward(lang.head, device=device)
        for rq in req_list:
            reward.text_embedding(rq)             # validate each up front (raises on an unknown bucket)
        ov = dict(lang.get("overrides") or {})    # language-mode control knobs (n_episodes/max_steps/...) over `control:`
        if requests:                              # multi-query: one episode per request; render them all
            ov["n_episodes"] = ov["n_plot"] = len(requests)
        mppi_kwargs.update(ov)
        _plog(writer, f"[eval_control @ep{step}] LANGUAGE steering -> {req_list} "
                      f"({os.path.basename(lang.head)}); oracle OFF, learned controller"
                      + (f"; overrides {ov}" if ov else ""))
    n_ctrl = 1 if reward is not None else 2
    n_plot = int(mppi_kwargs.pop("n_plot", 1))    # NOT an MPPIConfig field: how many episodes to render products for
    _plog(writer, f"[eval_control @ep{step}] start: MPPI {mppi_kwargs['n_episodes']} eps x {n_ctrl} controller(s), "
                  f"{mppi_kwargs['num_samples']} samples, H={mppi_kwargs['horizon']}, max_steps={mppi_kwargs['max_steps']}, "
                  f"render {n_plot} episode(s)")
    # env-agnostic MPPI scoring (design/gym_refactor.md Phase 4): both controllers score candidates with the
    # TRUE env's reward. Torus: TorusEnv.reward == the old inline goal cost, so torus numbers are unchanged;
    # a generic env brings its own reward. Shaping knobs bound only when the env's reward exposes them.
    mppi_cfg = MPPIConfig(**mppi_kwargs)
    env_factory = lambda: make_env(cfg.environments.get("name", "torus_world"), cfg.environments,
                                   mppi_cfg.n_episodes, device)
    env = env_factory()
    knobs = {k: getattr(mppi_cfg, k) for k in ("beta_vel", "r_settle")
             if k in inspect.signature(env.reward).parameters}
    reward_fn = functools.partial(env.reward, **knobs)
    # OPTIONAL reward override (control.reward_override = dotted path to a callable reward_fn(obs, goal)):
    # replaces env.reward as the scored per-step reward for BOTH controllers. Default null = env.reward
    # (the line above). Orthogonal to language steering — language.head sets `reward` below, untouched.
    ro = cfg.control.get("reward_override", None)
    if ro:
        import importlib
        mod, _, attr = str(ro).rpartition(".")
        reward_fn = getattr(importlib.import_module(mod), attr)
        _plog(writer, f"[eval_control @ep{step}] control.reward_override -> {ro} (replaces env.reward)")
    # NO-GOAL env: an env that supplies no control-goal source (WorldEnv.control_goals absent or -> None,
    # e.g. a gym Pendulum) has nothing to "reach" — run REWARD-ONLY control (maximize env.reward) in a
    # SEPARATE branch. Everything below (the torus goal race + language steering) is untouched.
    goals_fn = getattr(env, "control_goals", None)
    if reward is None and (goals_fn is None
                           or goals_fn(mppi_cfg.n_episodes, mppi_cfg.n_goals, None, device) is None):
        return _run_and_log_control_reward_only(cfg, model, normalizer, ecfg, writer, device, step,
                                                env, mppi_cfg, n_plot, reward_fn)
    t = time.perf_counter()
    res, _ = run_control(model, normalizer, ecfg, mppi_cfg, device=device,
                         log=lambda m: _plog(writer, f"[eval_control @ep{step}]   {m}"), fpv=fpv,
                         reward=reward, request=request, requests=requests, oracle=(reward is None), n_plot=n_plot,
                         reward_fn=reward_fn, env=env, env_factory=env_factory)
    t_ctrl = time.perf_counter() - t
    # what matters: cost of ONE MPPI replan (= one action chunk). t_ctrl covers the controller(s) + chunk
    # execution over n_chunks replans, so per-chunk wall time = t_ctrl / n_chunks.
    mppi_chunk_s = t_ctrl / max(1, res["n_chunks"])
    mppi_chunk_hz = 1.0 / mppi_chunk_s if mppi_chunk_s > 0 else 0.0
    _plog(writer, f"[eval_control @ep{step}] MPPI done: {res['n_chunks']} replans over {res['n_steps']} steps "
                  f"-> {mppi_chunk_s * 1000:.0f} ms/chunk ({mppi_chunk_hz:.1f} hz)")
    R, r = getattr(ecfg, "R", None), getattr(ecfg, "r", None)   # torus geometry; None for a generic env
    fps = step_fps(cfg, ecfg)
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
    ep_requests = res.get("requests")          # per-episode request text (multi-query), or None
    combined_agents = []                        # (language) one pred agent per episode -> ALL on ONE torus (combined)
    for i in range(NP):
        ti = time.perf_counter()
        agents = [_agent(res[k], i, colors[k], R, r) for k in kinds] if R is not None else []
        req_i = ep_requests[i] if ep_requests else request
        nf = len(agents[0]["path"]) if agents else len(res[kinds[0]]["paths"][i])
        _plog(writer, f"[eval_control @ep{step}] rendering control video #{i} ({nf} frames, GPU/EGL)...")
        vtitle = (f'"{req_i}"' if reward is not None else f"control: true vs pred #{i}")
        vids = {}
        if wants_diagnostics(env):   # rich scene via the env's diagnostic renderer (Phase 4.2/5 overlay:
            # agents={true,pred} paths, markers={goal}; torus draws it byte-identical to the legacy viz call).
            # Language mode has no target -> no goal marker -> no goal ring.
            extras = (dict(coloring="hsv", n_frames=nf, title=vtitle,
                           avecs={k: a["avec"] for k, a in zip(kinds, agents)},
                           goal_seqs={k: res[k]["goal_seqs"][i] for k in kinds},
                           fan_seq=res["fan_seqs"][i],  # pred's MPPI candidate fan, colored by score
                           reuse=bool(cfg.control.get("reuse_render", False)),
                           log=lambda m, i=i: _plog(writer, f"[eval_control @ep{step}]   video #{i} {m}"))
                      if R is not None else            # generic env: no torus presentation hints
                      dict(n_frames=nf, title=vtitle,
                           log=lambda m, i=i: _plog(writer, f"[eval_control @ep{step}]   video #{i} {m}")))
            overlay = SceneOverlay(agents={k: res[k]["paths"][i] for k in kinds},
                                   markers=({"goal": res["pred"]["goal_seqs"][i]} if reward is None else {}),
                                   extras=extras)
            vids = env.render_diagnostics(overlay, ["scene"])
        if vids:
            for view, fr in vids.items():   # "scene" keeps the canonical tag; extra views get suffixed
                writer.video(product_tag("eval_control", "control_video" if view == "scene"
                                         else f"control_video_{view}", i=i), fr, fps, step)
        else:   # generic env (no rich scene): pred-vs-true render_obs filmstrip video (Phase 5 fallback)
            import torch
            rend = {k: env.render_obs(torch.as_tensor(np.asarray(res[k]["obs_seqs"][i]), dtype=torch.float32)
                                      ).cpu().numpy().astype(np.float32) / 255.0 for k in kinds}
            vid = (viz.image_rollout_video(rend["true"], rend["pred"], context_len=0).astype(np.uint8)
                   if "true" in rend else (rend["pred"] * 255).astype(np.uint8))
            writer.video(product_tag("eval_control", "control_video", i=i), vid, fps, step)
        if reward is not None:                  # collect this episode's agent for the ONE-torus combined (all BLACK)
            combined_agents.append(_agent(res["pred"], i, "black", R, r))
        scene_desc = (f"Language-steered MPPI on the torus (episode {i}): a single GREY learned-model agent "
                      f"steering to maximize the language reward R(latent, '{req_i}'). The action arrow per step "
                      f"is the applied control."
                      if reward is not None else
                      f"Dual MPPI control on the torus (episode {i}): a BLACK oracle agent (true dynamics) and a "
                      f"GREY learned-model agent, each navigating to a sequence of goals. Each goal is a RING zone "
                      f"on the surface; the action arrow per step is the applied control.")
        if R is not None:   # the scene JSON is torus-specific (worded + R/r geometry); generic envs skip it
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

        # per-episode reward TRACE (4 lines: {imagined, achieved} x {reward head, ground truth}). Replaces the old
        # distance_to_request curve — that was just 1 - achieved(reward head), already a line here.
        if reward is not None:
            rc = np.asarray(res["pred"]["dist_curves"][i])
            from ..environments import torus_utils as T
            rq, xyz = str(req_i).lower(), np.asarray(agents[0]["path"])

            def _gt(path):   # torus ground-truth reward on a path for the buckets named in the request
                g = [T.color_reward(path, b) for b in cfg.interpret.factors.color.buckets if str(b).lower() in rq] + \
                    [T.position_reward(path, r, b) for b in cfg.interpret.factors.positioning.buckets if str(b).lower() in rq]
                return np.mean(g, axis=0) if g else None
            # purple = reward head, green = ground truth; solid = achieved (real), dashed = imagined (predicted)
            lines = {"achieved (reward head)": 1.0 - rc}
            cols = {"achieved (reward head)": "purple"}
            styl = {"achieved (reward head)": "-"}
            gta = _gt(xyz)
            if gta is not None:
                lines["achieved (ground truth)"] = gta; cols["achieved (ground truth)"] = "green"; styl["achieved (ground truth)"] = "-"
            if res.get("imag_head_curves") is not None:   # the chosen plan's IMAGINED belief (world-model prediction)
                lines["imagined (reward head)"] = res["imag_head_curves"][i]; cols["imagined (reward head)"] = "purple"; styl["imagined (reward head)"] = "--"
                igt = _gt(np.asarray(res["imag_paths"][i]))
                if igt is not None:
                    lines["imagined (ground truth)"] = igt; cols["imagined (ground truth)"] = "green"; styl["imagined (ground truth)"] = "--"
            rsteps = res.get("replan_steps", [])          # circle the imagined lines at each MPPI replan
            mk = {k: rsteps for k in lines if k.startswith("imagined")}
            tr = viz.fig_error_vs_step(lines, colors=cols, linestyles=styl, markers=mk, yscale="linear",
                caption="solid = achieved (REAL executed state);  dashed = imagined (chosen plan's PREDICTED state).\n"
                        "purple = reward head cos(f_z, f_t(request));  green = ground truth (torus reward on the path).\n"
                        "imagined vs achieved  =>  world-model / imagination accuracy (drift).\n"
                        "reward-head vs ground-truth  =>  alignment / grounding accuracy.\n"
                        "all four high and together  =>  the model imagines right, steers there, and the head agrees with truth.")
            writer.figure(product_tag("eval_control", "reward_trace", i=i), tr, step); plt.close(tr)
            # raw curve data (npz) next to the figure -> re-plot / average / post-process for the paper without a re-run
            writer.array(product_tag("eval_control", "reward_trace", i=i), step,
                         **{k.replace(" (", "_").replace(")", "").replace(" ", "_"): np.asarray(v) for k, v in lines.items()})
        else:                    # goal race: distance to current goal, verticals at goal switches
            k_true, k_pred = "true (oracle): dist to current goal", "pred (learned): dist to current goal"
            curve = viz.fig_error_vs_step({k_true: res["true"]["dist_curves"][i], k_pred: res["pred"]["dist_curves"][i]},
                                          colors={k_true: "black", k_pred: "dimgray"},
                                          vlines={"black": _goal_changes(res["true"]["goal_seqs"][i]),
                                                  "dimgray": _goal_changes(res["pred"]["goal_seqs"][i])},
                                          yscale="linear")  # distance to goal is bounded -> linear reads better
            writer.figure(product_tag("eval_control", "distance_to_goal", i=i), curve, step); plt.close(curve)
        _plog(writer, f"[eval_control @ep{step}] episode #{i} rendered in {time.perf_counter() - ti:.1f}s")

    # combined control video: ALL agents on ONE torus (one colored dot per request/episode), not a grid.
    if len(combined_agents) > 1:
        nf = max(len(a["path"]) for a in combined_agents)
        ctitle = ("  ·  ".join(dict.fromkeys(ep_requests)) if ep_requests else
                  f"{len(combined_agents)} agents")   # all agents BLACK on one torus
        comb = viz.control_compare_frames(R, r, "hsv", combined_agents, n_frames=nf, title=ctitle,
                                          reuse=bool(cfg.control.get("reuse_render", False)), show_goals=False,
                                          log=lambda m: _plog(writer, f"[eval_control @ep{step}]   combined {m}"))
        writer.video(product_tag("eval_control", "control_video_combined"), comb, fps, step)
        _plog(writer, f"[eval_control @ep{step}] control_video_combined ({len(combined_agents)} agents on one torus)")

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
                for i in range(NP):           # one animation PER control episode (matches control_video_<i>)
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
            with torch.no_grad():                                    # f_z: WM latent -> joint space (request-independent)
                fz = reward.f_z(torch.from_numpy(lat).to(reward.device)).cpu().numpy()
            # which requests get their own reward_field + joint animation. Default: only the representative (#0);
            # language.per_request_joint_plots -> one set PER request (each is an LDA field + 2 anims per factor).
            plot_reqs = list(enumerate(requests)) if lang.get("per_request_joint_plots") else [(0, request)]
            te_cache = {}                                            # per request: (t_e_np, reward field) — factor-independent
            for r_idx, r_txt in plot_reqs:
                t_e = reward.text_embedding(r_txt)
                with torch.no_grad():
                    rf = reward.score(torch.from_numpy(lat).to(reward.device), t_e).cpu().numpy()
                te_cache[r_idx] = (t_e.cpu().numpy(), rf)
                _plog(writer, f"[eval_control @ep{step}] reward field '{r_txt}' (range {rf.min():.2f}..{rf.max():.2f})")
            for factor in ("color", "positioning"):
                fc = OmegaConf.to_container(cfg.interpret.factors[factor], resolve=True)
                labels = [recs[int(c)]["label"][factor] for c in clip_idx]
                yb = {b: k for k, b in enumerate(fc["buckets"])}
                e, red = reduce_dims(fz, "lda", n_components=2, seed=0, return_reducer=True, y=np.array([yb[l] for l in labels]))
                multi = len(plot_reqs) > 1
                # STATIC reward field: one per REQUEST (request-dependent, episode-independent, LDA layout shared)
                for r_idx, r_txt in plot_reqs:
                    t_e_np, rfield = te_cache[r_idx]
                    goal2d = red.transform(t_e_np[None])[0][:2]     # f_t(request) projected -> its spot in this layout
                    ff = viz.fig_points_2d(e, color=rfield, cbar_label=f"reward  cos(f_z, f_t('{r_txt}'))",
                                           lims=pad_lims(np.concatenate([e, goal2d[None]])), point_size=6.0,
                                           annotations=[{"pos": goal2d, "text": r_txt}],
                                           title=f"reward field '{r_txt}' — {factor} LDA")
                    writer.figure(product_tag(f"eval_control/joint_latent_space_plots/{factor}/lda", "reward_field", i=r_idx), ff, step)
                    plt.close(ff)
                # AGENT ANIMATION: one PER EPISODE (matches control_video_<i>). multi-query: episode i steers to
                # request i; single request: all NP episodes share it (each has its own trajectory through the space).
                for i in range(len(plot_reqs) if multi else NP):
                    r_idx = i if multi else 0
                    r_txt, t_e_np = plot_reqs[r_idx][1], te_cache[r_idx][0]
                    with torch.no_grad():
                        fzt = reward.f_z(torch.from_numpy(np.asarray(res["agent_latents"][i], np.float32)).to(reward.device)).cpu().numpy()
                    for rmode, tag in ((False, "concept"), (True, "reward")):
                        fr = animate_joint_space(fz, fzt, t_e_np, method="lda", labels=labels, factor_cfg=fc,
                                                 reward_mode=rmode, goal_label=r_txt, title=f"'{r_txt}' · {factor} joint ({tag}) #{i}")
                        writer.video(product_tag(f"eval_control/joint_latent_space_plots/{factor}/lda", f"{tag}_2d", i=i),
                                     fr, fps, step)   # <type>_<nd>d_<i>.mp4, one per control episode
                    _plog(writer, f"[eval_control @ep{step}] joint anim {factor}/lda #{i} '{r_txt}' (concept+reward)")

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


def _run_and_log_control_reward_only(cfg, model, normalizer, ecfg, writer, device, step,
                                     env, mppi_cfg, n_plot, reward_fn) -> dict:
    """REWARD-ONLY control eval for an env with NO goal source (WorldEnv.control_goals -> None): the env's
    OWN reward is the objective (mppi.run_control_reward_only), so there are no goals/success metrics —
    logs eval_control/{true,pred,diff}/{mean_reward,final_reward} instead, per-episode reward curves, and a
    control video (rich scene if the env draws one — agents only, NO goal marker — else the render_obs
    filmstrip). A SEPARATE branch from run_and_log_control's goal race, which stays untouched."""
    from ..controller.mppi import run_control_reward_only
    from ..evaluation.products import product_tag
    _plog(writer, f"[eval_control @ep{step}] env supplies NO control goals -> REWARD-ONLY control "
                  f"(maximize env.reward); oracle (true dynamics) vs learned")
    env_factory = lambda: make_env(cfg.environments.get("name", "torus_world"), cfg.environments,
                                   mppi_cfg.n_episodes, device)
    t = time.perf_counter()
    res = run_control_reward_only(model, normalizer, env_factory, mppi_cfg, device=device,
                                  log=lambda m: _plog(writer, f"[eval_control @ep{step}]   {m}"),
                                  oracle=True, n_plot=n_plot, reward_fn=reward_fn)
    t_ctrl = time.perf_counter() - t
    mppi_chunk_s = t_ctrl / max(1, res["n_chunks"])
    mppi_chunk_hz = 1.0 / mppi_chunk_s if mppi_chunk_s > 0 else 0.0
    _plog(writer, f"[eval_control @ep{step}] MPPI done: {res['n_chunks']} replans over {res['n_steps']} steps "
                  f"-> {mppi_chunk_s * 1000:.0f} ms/chunk ({mppi_chunk_hz:.1f} hz)")
    fps = step_fps(cfg, ecfg)
    kinds = [k for k in ("true", "pred") if k in res]
    labels = {"true": "oracle", "pred": "learned"}
    NP = res["n_plot"]
    for i in range(NP):
        vids = {}
        if wants_diagnostics(env) and "paths" in res["pred"]:   # rich scene: agents only, NO goal marker
            overlay = SceneOverlay(agents={k: res[k]["paths"][i] for k in kinds},
                                   extras=dict(n_frames=len(res["pred"]["paths"][i]),
                                               title=f"reward-only control #{i}"))
            vids = env.render_diagnostics(overlay, ["scene"])
        if vids:
            for view, fr in vids.items():
                writer.video(product_tag("eval_control", "control_video" if view == "scene"
                                         else f"control_video_{view}", i=i), fr, fps, step)
        else:   # generic env: pred-vs-true render_obs filmstrip video (same fallback as the goal race)
            import torch
            rend = {k: env.render_obs(torch.as_tensor(np.asarray(res[k]["obs_seqs"][i]), dtype=torch.float32)
                                      ).cpu().numpy().astype(np.float32) / 255.0 for k in kinds}
            vid = (viz.image_rollout_video(rend["true"], rend["pred"], context_len=0).astype(np.uint8)
                   if "true" in rend else (rend["pred"] * 255).astype(np.uint8))
            writer.video(product_tag("eval_control", "control_video", i=i), vid, fps, step)
        # per-episode realized-reward curve (oracle black vs learned grey, like the goal-distance curve)
        lines = {f"{labels[k]}: env reward per step": res[k]["reward_curves"][i] for k in kinds}
        cols = {f"{labels[k]}: env reward per step": ("black" if k == "true" else "dimgray") for k in kinds}
        curve = viz.fig_error_vs_step(lines, colors=cols, yscale="linear")
        writer.figure(product_tag("eval_control", "reward_curve", i=i), curve, step)
        plt.close(curve)
    # aggregate scalars: reward-based (no goals -> no success_rate / goals_reached)
    summary = {"n_steps": res["n_steps"], "time/mppi_chunk_s": mppi_chunk_s, "time/mppi_chunk_hz": mppi_chunk_hz}
    for m in ("mean_reward", "final_reward"):
        for k in kinds:
            summary[f"{k}/{m}"] = res[k][m]
        if "true" in res:
            summary[f"diff/{m}"] = res["true"][m] - res["pred"][m]
    _plog(writer, f"[eval_control @ep{step}] reward: " + " ".join(
        f"{labels[k]} mean={res[k]['mean_reward']:.3f} final={res[k]['final_reward']:.3f}" for k in kinds))
    writer.scalars({f"eval_control/{k}": v for k, v in summary.items()}, step)
    ep_dir = os.path.join(writer.dir, f"epoch_{step:04d}", "eval_control")   # writer.dir is already run_dir/logs
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
