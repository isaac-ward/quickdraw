"""Step 1: generate all dataset splits + run outputs, all under one logs/ run folder.

Everything (the lerobot dataset per split, normalization stats, dataset card, summary, media) is
written under `logs/data_generation_<ts>_<experiment>/`. Point training at it with `data.root=<that>`.
The per-split media (summary plot/video + FPV) render in parallel across processes.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import hydra
import matplotlib.pyplot as plt

from .data.generate import build_all, generate_episodes
from .environments.torus import TorusConfig
from .logging import viz
from .training.setup import env_cfg
from .utils.logging import make_run_dir

N_FPV = 3  # egocentric sample videos per split


def _render_split(task: dict):
    """Render one split's media (atlas PNG, animated atlas MP4, N_FPV egocentric MP4s). Runs in a
    worker process. The PNG and the video frames are produced by the SAME `fig_torus_atlas` call, so
    their layout is identical."""
    t = task
    scfg = TorusConfig(R=t["R"], r=t["r"], dt=t["dt"], gamma=t["gamma"], a_max=t["a_max"], init_speed=t["init_speed"])
    name, steps, coloring = t["name"], t["steps"], t["coloring"]
    obs, act = generate_episodes(scfg, max(t["n_plot"], N_FPV), steps, t["seed"])
    avec = viz.action_ambient(obs[:, :, :3], act, scfg.R, scfg.r)
    trajs = [{"xyz": obs[i, :, :3], "avec": avec[i], "color": "k"} for i in range(t["n_plot"])]
    title = f"{name} (samples)"
    fps = round(1.0 / scfg.dt)  # constant 60 fps, one frame per sim step -> real time for every split

    fig = viz.fig_torus_atlas(scfg.R, scfg.r, trajs=trajs, coloring=coloring, title=title)  # no arrow (static)
    fig.savefig(os.path.join(t["sp"], f"{name}.png"), dpi=viz.DPI)  # no bbox=tight -> matches video frames
    plt.close(fig)

    anim = viz.animate_frames(scfg.R, scfg.r, coloring, trajs, title=title)  # arrow = applied action (tweened)
    viz.save_mp4(os.path.join(t["sv"], f"{name}.mp4"), anim, fps)
    for i in range(N_FPV):
        fpv = viz.fpv_frames(scfg.R, scfg.r, coloring, obs[i], fov=t["fov"])
        viz.save_mp4(os.path.join(t["fp"], f"{name}_{i}.mp4"), fpv, fps)
    return name


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_dir = make_run_dir("data_generation", cfg.experiment)
    ecfg = env_cfg(cfg)
    stats = build_all(ecfg, run_dir, cfg.data.splits, fps=round(1.0 / ecfg.dt), device="cpu")

    # summary: counts + simulated duration (at sim speed dt) per split
    card = json.load(open(os.path.join(run_dir, "dataset_card.json")))
    P, F, dt = cfg.data.P, cfg.data.F, ecfg.dt
    counts = {}
    for name, sp_ in card["splits"].items():
        n, st = int(sp_["n_traj"]), int(sp_["steps"])
        tr = n * st
        sec = tr * dt
        c = {"episodes": n, "steps_per_episode": st, "transitions": tr,
             "seconds": round(sec, 2), "minutes": round(sec / 60, 3), "hours": round(sec / 3600, 5)}
        if name in ("train", "val"):
            c["training_windows"] = n * max(0, st - (P + F) + 1)  # sliced (P+F)-windows, stride 1
        counts[name] = c
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"dataset_root": run_dir, "counts": counts, "splits": card["splits"],
                   "split_env": card["split_env"], "coloring": card["coloring"],
                   "normalization_stats": stats}, f, indent=2)

    sp = os.path.join(run_dir, "media", "summary_plots")
    sv = os.path.join(run_dir, "media", "summary_videos")
    fp = os.path.join(run_dir, "media", "fpv")
    for d in (sp, sv, fp):
        os.makedirs(d, exist_ok=True)

    tasks = []
    for name, s in cfg.data.splits.items():
        scfg = replace(ecfg, **dict(s.get("env", {}) or {}))
        tasks.append({"name": name, "R": scfg.R, "r": scfg.r, "dt": scfg.dt, "gamma": scfg.gamma,
                      "a_max": scfg.a_max, "init_speed": scfg.init_speed,
                      "steps": min(int(s["steps"]), 512), "seed": int(s["seed"]),
                      "coloring": s.get("coloring", "hsv"), "n_plot": int(cfg.data.n_plot_trajectories),
                      "fov": float(cfg.data.fpv_fov), "sp": sp, "sv": sv, "fp": fp})

    workers = min(len(tasks), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for done in ex.map(_render_split, tasks):
            print(f"[data_generation] rendered media for {done}")

    print(f"[data_generation] dataset + outputs at {run_dir}")
    print(f"[data_generation] now train with:  data.root={run_dir}")


if __name__ == "__main__":
    main()
