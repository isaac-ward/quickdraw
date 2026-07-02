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


def _agent(res, color, R, r):
    """Build a control_compare_frames agent dict (pads actions to the path length for the arrow)."""
    path, act = res["path"], res["actions"]               # (T,3), (T-1,2)
    act = np.concatenate([act, act[-1:]], axis=0) if len(act) else np.zeros((len(path), 2))
    return {"path": path, "goal_seq": res["goal_seq"], "color": color,
            "avec": viz.action_ambient(path, act, R, r)}  # ambient applied action (T,3)


def run_and_log_control(cfg, model, normalizer, ecfg, writer, device, step=0) -> dict:
    _plog(writer, f"[eval_control @ep{step}] start: MPPI {cfg.control.n_episodes} eps x 2 controllers, "
                  f"{cfg.control.num_samples} samples, H={cfg.control.horizon}, max_steps={cfg.control.max_steps}")
    t = time.perf_counter()
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
    res, _ = run_control(model, normalizer, ecfg, MPPIConfig(**mppi_kwargs), device=device,
                         log=lambda m: _plog(writer, f"[eval_control @ep{step}]   {m}"), fpv=fpv)
    t_ctrl = time.perf_counter() - t
    # what matters: cost of ONE MPPI replan (= one action chunk). t_ctrl covers both controllers + chunk
    # execution over n_chunks replans, so per-chunk wall time = t_ctrl / n_chunks.
    mppi_chunk_s = t_ctrl / max(1, res["n_chunks"])
    mppi_chunk_hz = 1.0 / mppi_chunk_s if mppi_chunk_s > 0 else 0.0
    _plog(writer, f"[eval_control @ep{step}] MPPI done: {res['n_chunks']} replans over {res['n_steps']} steps "
                  f"-> {mppi_chunk_s * 1000:.0f} ms/chunk ({mppi_chunk_hz:.1f} hz)")
    R, r, fps = ecfg.R, ecfg.r, round(1.0 / ecfg.dt)

    # one race video: true-dynamics oracle (black) vs learned controller (grey), each with its action
    # arrow and a small current-goal marker in its own colour
    t = time.perf_counter()
    nf = len(res["true"]["path"])
    _plog(writer, f"[eval_control @ep{step}] rendering control video ({nf} frames, GPU/EGL)...")
    agents = [_agent(res["true"], "black", R, r), _agent(res["pred"], "dimgray", R, r)]
    frames = viz.control_compare_frames(R, r, "hsv", agents,
                                        n_frames=nf, title="control: true vs pred",
                                        fan_seq=res["fan_seq"],  # pred's MPPI candidate fan, colored by cost
                                        reuse=bool(cfg.control.get("reuse_render", False)),
                                        log=lambda m: _plog(writer, f"[eval_control @ep{step}]   video {m}"))
    writer.video("eval_control/control_video_0", frames, fps, step)  # _0: we show episode 0 only
    writer.scene("eval_control/control_video_0", {  # 3D geometry for Blender (plain-language keys)
        "description": "Dual MPPI control on the torus: a BLACK oracle agent (true dynamics) and a GREY "
                       "learned-model agent, each navigating to a sequence of goals. Each goal is a RING zone "
                       "on the surface; the action arrow per step is the applied control.",
        "coordinate_system": "world xyz, same space as the torus",
        "torus": {"major_radius_R": float(R), "tube_radius_r": float(r)},
        "goal_zone_ring_radius": float(0.0675 * (R + r)),
        "agents": [{"name": nm, "color": a["color"], "path_xyz": a["path"], "goal_per_step_xyz": a["goal_seq"],
                    "action_arrow_per_step": {"origins_xyz": a["path"], "vectors_xyz": a["avec"]}}
                   for a, nm in zip(agents, ("oracle", "learned"))],
    }, step)
    t_video = time.perf_counter() - t
    _plog(writer, f"[eval_control @ep{step}] video rendered in {t_video:.1f}s")

    # (MM) predictor-in-the-loop video: over the WHOLE control run (ep0), the model's imagined FPV for the
    # SELECTED plan (top) vs the actual FPV (bottom). Shows whether MPPI planned against reality or a fantasy.
    if res.get("pred_fpv_video") is not None:
        pv = res["pred_fpv_video"]
        pvid = viz.image_rollout_video(pv["actual"], pv["pred"], context_len=0)   # context_len=0 -> top=pred, bottom=actual
        writer.video(f"eval_control/{img_head}/prediction_video", pvid, fps, step)   # <head> organization (like eval_ood_horizon)
        _plog(writer, f"[eval_control @ep{step}] {img_head}/prediction_video: {len(pv['pred'])} steps (pred top / actual bottom)")

    # realized cost over time: episode-0 distance to the current goal per control step
    # colours match the race video: oracle = black, learned = dimgray. Dotted verticals mark the steps
    # where episode-0's goal advances (explains the sharp jumps: distance re-targets to the next goal).
    def _goal_changes(goal_seq):
        g = np.asarray(goal_seq)
        return (np.where(np.any(g[1:] != g[:-1], axis=-1))[0] + 1).tolist()
    k_true, k_pred = "true (oracle): dist to current goal", "pred (learned): dist to current goal"
    curve = viz.fig_error_vs_step({k_true: res["true"]["dist_curve"], k_pred: res["pred"]["dist_curve"]},
                                  colors={k_true: "black", k_pred: "dimgray"},
                                  vlines={"black": _goal_changes(res["true"]["goal_seq"]),
                                          "dimgray": _goal_changes(res["pred"]["goal_seq"])},
                                  yscale="linear")  # distance to goal is bounded -> linear reads better
    writer.figure("eval_control/distance_to_goal_0", curve, step)  # _0: episode 0 only
    plt.close(curve)

    # only the MPPI-step timing matters (render times intentionally not logged); + n_steps for context.
    summary = {"n_steps": res["n_steps"], "time/mppi_chunk_s": mppi_chunk_s, "time/mppi_chunk_hz": mppi_chunk_hz}
    # {true,pred,diff}/{metric} (slash org; diff = true - pred). Only these — no underscore duplicates.
    for m in ("success_rate", "mean_goals_reached", "mean_steps_to_complete", "mean_seconds_to_complete"):
        summary[f"true/{m}"] = res["true"][m]
        summary[f"pred/{m}"] = res["pred"][m]
        summary[f"diff/{m}"] = res["true"][m] - res["pred"][m]
    writer.scalars({f"eval_control/{k}": v for k, v in summary.items()}, step)
    # goals-reached is meaningful even when seconds-to-complete is NaN (nothing settled all goals)
    g8 = res["n_goals"]
    _plog(writer, f"[eval_control @ep{step}] goals reached / {g8}: "
                  f"oracle={res['true']['mean_goals_reached']:.2f} (success {res['true']['success_rate']:.2f}, "
                  f"{res['true']['mean_seconds_to_complete']:.2f}s) "
                  f"pred={res['pred']['mean_goals_reached']:.2f} (success {res['pred']['success_rate']:.2f})")
    # write the raw summary next to THIS epoch's control media (logs/epoch_<i>/eval_control/), not the flat run
    # root — mirrors how writer.video/figure organize by epoch, so it's per-epoch (not clobbered each eval).
    ep_dir = os.path.join(writer.dir, "logs", f"epoch_{step:04d}", "eval_control")
    os.makedirs(ep_dir, exist_ok=True)
    with open(os.path.join(ep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
