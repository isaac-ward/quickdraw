"""Step 1: generate all dataset splits + run outputs, all under one logs/ run folder.

Pipeline (the heavy FPV render is decoupled from lerobot so it runs at FULL parallelism):
  1. simulate every split's trajectories (vectors)
  2. render the summary atlas plot + video per split FIRST (eyeball the physics before the long render)
  3. render a 256x256 egocentric clip PER trajectory  -> media/fpv/<split>/ep_<i>.mp4   [80-way]
  4. write each split's lerobot dataset, ingesting those clips as `observation.images.fpv` [per split]
  5. write meta + summary.json, then stitch each eval case's clips into a grid composite
Progress is streamed to stdout AND <run_dir>/progress.log. Point training at data.root=<run_dir>.
"""

from __future__ import annotations

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass, replace
from types import SimpleNamespace

import hydra
import matplotlib.pyplot as plt
import torch

from .data.generate import compute_norm_stats, generate_episodes, write_lerobot_split, write_meta
from .environments.policies import make_policy
from .environments.registry import make_env
from .environments.torus import TorusConfig
from .logging import viz
from .training.setup import env_cfg
from .utils.logging import make_run_dir


def _render_fpv(job: dict):
    """One trajectory's egocentric clip (rendered in parallel, ingested into lerobot later) via the split
    ENV's `render_obs` — the env OWNS its image modality (torus: FPV, byte-identical to the old direct
    viz.fpv_frames call; a generic env, e.g. gym:*, supplies its own frames with no torus params)."""
    env = make_env(job["env_name"], job["scfg"], batch=1)
    frames = env.render_obs(torch.as_tensor(job["obs"])).cpu().numpy()
    viz.save_mp4(job["out"], frames, job["fps"])
    return job["out"]


def _write_lr(job: dict):
    """One split's lerobot dataset (vectors + the pre-rendered FPV clips as observation.images.fpv)."""
    write_lerobot_split(job["root_split"], job["repo_id"], job["obs"], job["act"], job["fps"],
                        fpv_dir=job["fpv_dir"], fpv_size=job["size"])
    return job["name"]


def _render_summary(job: dict):
    """Static atlas PNG + animated atlas MP4 for one split (first n_plot trajectories). Streams its own
    frame progress to the shared progress.log (runs in a worker, so it appends directly)."""
    def slog(msg):
        line = f"[summary:{job['name']}] {msg}"
        print(line, flush=True)
        try:
            with open(job["log_path"], "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
    scfg = TorusConfig(**job["scfg"])
    trajs = [{"xyz": x, "avec": a, "color": "k", "start_scale": 0.5}
             for x, a in zip(job["pos"], job["avec"])]  # start sphere half-size on summary plots
    title = f"{job['name']} (samples)"
    slog("plot...")
    fig = viz.fig_torus_atlas(scfg.R, scfg.r, trajs=trajs[:2], coloring=job["coloring"], title=title,
                              torus_opacity=viz.TORUS_OPACITY)  # static plot: just 2 particles (video keeps all)
    fig.savefig(os.path.join(job["sp"], f"{job['name']}.png"), dpi=viz.DPI)
    plt.close(fig)
    slog("video...")
    frames = viz.animate_frames(scfg.R, scfg.r, job["coloring"], trajs, title=title,
                                smooth_window=job["smooth"], log=slog)
    viz.save_mp4(os.path.join(job["sv"], f"{job['name']}.mp4"), frames, job["fps"])
    slog("done")
    return job["name"]


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_dir = make_run_dir("data_generation", cfg.experiment)
    log_path = os.path.join(run_dir, "progress.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    ecfg = env_cfg(cfg)
    fps, size = round(1.0 / ecfg.dt), viz.FPV_SIZE
    fov, grid, n_plot = float(cfg.data.fpv_fov), int(cfg.data.composite_grid), int(cfg.data.n_plot_trajectories)
    n_cells = grid * grid
    media = os.path.join(run_dir, "media")
    sp, sv, fpv_root, comp = (os.path.join(media, d) for d in
                              ("summary_plots", "summary_videos", "fpv", "composites"))
    for d in (sp, sv, fpv_root, comp):
        os.makedirs(d, exist_ok=True)
    t0 = time.time()

    # 1. simulate every split (cheap, vector only) — env by name from the registry (design/gym_refactor.md
    # Phase 2), rolled with the configured behavior policy. Torus: byte-identical to the pre-refactor loop.
    env_name = cfg.environments.get("name", "torus_world")
    asamp = str(cfg.data.get("action_sampler", "ornstein_uhlenbeck")).lower()
    data = {}
    for name, s in cfg.data.splits.items():
        scfg = replace(ecfg, **dict(s.get("env", {}) or {}))
        env = make_env(env_name, scfg, batch=int(s["n_traj"]))
        obs, act = generate_episodes(env, int(s["n_traj"]), int(s["steps"]), int(s["seed"]),
                                     policy=make_policy(asamp, env))
        data[name] = (scfg, obs, act, s.get("coloring", "hsv"))
        os.makedirs(os.path.join(fpv_root, name), exist_ok=True)
    log(f"[gen] simulated {len(data)} splits, {sum(o.shape[0] for _, o, _, _ in data.values())} trajectories")

    # action-distribution preview (regenerated EVERY run): 8 magnitude-histogram tiles over time, so the
    # data's action distribution (e.g. the two-basin bimodal magnitude) is eyeballable + referenced by the card.
    # Drawn at ACTION_DIST_N_SAMPLES (> dataset size) for clean patterns; the eval samples the head at the same N.
    ad_steps = int(cfg.data.splits["train"]["steps"])
    ad_env = make_env(env_name, ecfg, batch=viz.ACTION_DIST_N_SAMPLES)
    ad_obs, ad_acts = generate_episodes(ad_env, viz.ACTION_DIST_N_SAMPLES, ad_steps,   # roll the env -> reflects the
                                        seed=int(cfg.data.splits["train"]["seed"]),
                                        policy=make_policy(asamp, ad_env))
    # conditioned on ambient x (obs[...,0]) so a STATE-dependent sampler shows its dependence: slow on -x, fast on +x.
    adfig = viz.fig_action_by_state(ad_acts, ad_obs[..., 0], ecfg.a_max, sampler_name=asamp)
    adfig.savefig(os.path.join(media, "action_distribution.png"), dpi=viz.DPI)
    plt.close(adfig)
    log(f"[action-dist] media/action_distribution.png ({viz.ACTION_DIST_N_SAMPLES} traj, by-x, sampler={asamp})")

    workers = int(os.environ.get("GEN_WORKERS") or (os.cpu_count() or 4))   # cap to leave CPU for concurrent training

    # 2. SUMMARY atlas plot + video per split FIRST, so the new physics can be eyeballed before the long
    # FPV render. Each runs in a worker and streams its own frame progress to progress.log. The atlas is
    # TORUS-specific eyeball viz (needs the torus geometry) — skip it for other envs; the FPV clips below
    # (env.render_obs) are the env-agnostic eyeball path.
    if env_name in ("torus_world", "torus"):
        summary_jobs = []
        for name, (scfg, obs, act, coloring) in data.items():
            pos = obs[:n_plot, :, :3]
            avec = viz.action_ambient(pos, act[:n_plot], scfg.R, scfg.r)
            summary_jobs.append({"name": name, "scfg": {"R": scfg.R, "r": scfg.r, "dt": scfg.dt,
                                 "gamma": scfg.gamma, "a_max": scfg.a_max, "init_speed": scfg.init_speed},
                                 "coloring": coloring, "fps": fps, "pos": pos, "avec": avec, "sp": sp, "sv": sv,
                                 "smooth": int(cfg.data.action_smooth_window), "log_path": log_path})
        log(f"[summary] rendering {len(summary_jobs)} split summary plots+videos FIRST (check media/summary_*)...")
        with ProcessPoolExecutor(max_workers=min(len(summary_jobs), workers)) as ex:
            for k, done in enumerate(ex.map(_render_summary, summary_jobs), 1):
                log(f"[summary] {k}/{len(summary_jobs)} complete: {done}  ({time.time() - t0:.0f}s)")
    else:
        log(f"[summary] atlas skipped (torus-specific eyeball viz; env={env_name})")

    # 3. render every trajectory's FPV clip at full parallelism (the heavy step). Each worker rebuilds the
    # SPLIT's env and renders through its `render_obs`: the split's coloring/fov are threaded into the env
    # config (torus honors them, byte-identical to the old direct fpv_frames call; a dataclass-less/generic
    # env cfg passes through untouched and the env renders its own image modality).
    fpv_jobs = []
    for name, (scfg, obs, _, coloring) in data.items():
        rcfg = SimpleNamespace(**asdict(scfg), coloring=coloring, fov=fov) if is_dataclass(scfg) else scfg
        for i in range(obs.shape[0]):
            fpv_jobs.append({"env_name": env_name, "scfg": rcfg, "fps": fps, "obs": obs[i],
                             "out": os.path.join(fpv_root, name, f"ep_{i:04d}.mp4")})
    log(f"[fpv] rendering {len(fpv_jobs)} clips on {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for j, _ in enumerate(ex.map(_render_fpv, fpv_jobs), 1):
            if j % 25 == 0 or j == len(fpv_jobs):  # doubled progress rate (was every 50)
                log(f"[fpv] {j}/{len(fpv_jobs)}  ({time.time() - t0:.0f}s)")

    # 4. write lerobot datasets, ingesting the rendered clips (one dataset per split, in parallel)
    repo = str(cfg.data.get("repo_id", "torus"))   # dataset repo prefix; loaders read the same data.repo_id
    lr_jobs = [{"name": name, "root_split": os.path.join(run_dir, name), "repo_id": f"{repo}/{name}",
                "obs": obs, "act": act, "fps": fps, "size": size,
                "fpv_dir": os.path.join(fpv_root, name)} for name, (_, obs, act, _) in data.items()]
    log(f"[lerobot] writing {len(lr_jobs)} split datasets (vectors + observation.images.fpv): "
        + ", ".join(j["name"] for j in lr_jobs))
    with ProcessPoolExecutor(max_workers=min(len(lr_jobs), workers)) as ex:
        for k, done in enumerate(ex.map(_write_lr, lr_jobs), 1):
            log(f"[lerobot] {k}/{len(lr_jobs)} wrote {done}  ({time.time() - t0:.0f}s)")
    write_meta(run_dir, ecfg, cfg.data.splits, {n: asdict(scfg_) for n, (scfg_, _, _, _) in data.items()},
               {n: c for n, (_, _, _, c) in data.items()}, fps, compute_norm_stats(*data["train"][1:3]))

    # summary.json: counts + simulated duration per split
    P, F, dt = cfg.data.P, cfg.data.F, ecfg.dt
    counts = {}
    for name, (_, obs, _, _) in data.items():
        n, st = obs.shape[0], obs.shape[1]
        tr, sec = n * st, n * st * dt
        c = {"episodes": n, "steps_per_episode": st, "transitions": tr,
             "seconds": round(sec, 2), "minutes": round(sec / 60, 3), "hours": round(sec / 3600, 5)}
        if name in ("train", "val"):
            c["training_windows"] = n * max(0, st - (P + F) + 1)
        counts[name] = c
    card = json.load(open(os.path.join(run_dir, "dataset_card.json")))
    norm = json.load(open(os.path.join(run_dir, "normalization_stats.json")))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"dataset_root": run_dir, "counts": counts, "splits": card["splits"],
                   "split_env": card["split_env"], "coloring": card["coloring"],
                   "action_sampler": asamp,
                   "normalization_stats": norm}, f, indent=2)

    # 4. composites: stitch every split's clips into one grid (reusing the rendered clips)
    for name in data:
        paths = sorted(glob.glob(os.path.join(fpv_root, name, "ep_*.mp4")))[:n_cells]
        viz.stitch_grid_video(paths, os.path.join(comp, f"{name}.mp4"), grid, fps)
        log(f"[composite] {name} ({len(paths)} cells)")

    log(f"[done] {run_dir}  ({time.time() - t0:.0f}s total)")
    log(f"[done] now train with:  data.root={run_dir}")


if __name__ == "__main__":
    main()
