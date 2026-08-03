"""Convert a RECORDED trajectory dump (no simulator) into the standard lerobot run layout.

  python -m quickdraw.recording_to_lerobot \\
      +recording.dir=~/user_irw/seamstress/assets/flightroom-starling_processed_112x192_30hz \\
      +recording.name=flightroom_starling          # smoke: +recording.max_episodes=4

Recording layout: `data.npz` with `states` (N, obs_dim), `actions` (N, action_dim) and
`episode_indices_mapping` (N,) — episodes are CONTIGUOUS global-index runs; the frame with global
index i is `images/ego/<i>.jpg` (native resolution, e.g. 192w x 112h). Pipeline (mirrors
data_generation.py, minus simulation/summary renders):
  1. group frames by episode; deterministic ~10% val split BY EPISODE (seed 0)
  2. encode each episode's jpgs (global-index order, NATIVE size) -> media/ego/<split>/ep_<i>.mp4  [parallel]
  3. write each split's lerobot dataset, ingesting those clips as `observation.images.ego`  [per split]
  4. write meta (train-only norm stats) + summary.json + dataset_card.json (no torus fields -> generic card)
Point training at data.root=<run_dir> with environments=recorded."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import hydra
import numpy as np

from .data.generate import compute_norm_stats, write_lerobot_split, write_meta
from .environments.recorded import RecordedConfig
from .logging import viz
from .utils.logging import make_run_dir

CAM = "ego"          # the recording's one camera -> observation.images.ego
VAL_FRAC = 0.1       # ~10% of episodes -> val
SPLIT_SEED = 0       # deterministic episode split


def _encode_ep(job: dict):
    """One episode's jpgs (global-index order, native size) -> mp4 (runs in a worker)."""
    import imageio.v2 as imageio
    frames = [imageio.imread(os.path.join(job["img_dir"], f"{i}.jpg"))[..., :3] for i in job["frames"]]
    viz.save_mp4(job["out"], frames, job["fps"])
    return job["out"]


def _write_lr(job: dict):
    """One split's lerobot dataset (vectors + the pre-encoded clips as observation.images.ego)."""
    write_lerobot_split(job["root_split"], job["repo_id"], job["obs"], job["act"], job["fps"],
                        fpv_dir=job["ego_dir"], fpv_size=job["hw"], cam=CAM, task=job["task"])
    return job["name"]


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    rec = cfg.get("recording", None)
    if rec is None or not rec.get("dir"):
        raise ValueError("pass +recording.dir=<processed recording folder> "
                         "(optional: +recording.name=..., +recording.max_episodes=N, +recording.fps=30)")
    src = os.path.expanduser(str(rec.dir))
    name = str(rec.get("name", "recording"))
    max_eps = int(rec.get("max_episodes", 0) or 0)
    fps = int(rec.get("fps", 30))

    run_dir = make_run_dir("recording", name)
    log_path = os.path.join(run_dir, "progress.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    npz = np.load(os.path.join(src, "data.npz"))
    states = npz["states"].astype(np.float32)
    actions = npz["actions"].astype(np.float32)
    ep_map = npz["episode_indices_mapping"]
    img_dir = os.path.join(src, "images", CAM)

    # episodes are CONTIGUOUS runs of ep_map -> per-episode global-index ranges [start, end)
    bounds = np.flatnonzero(np.diff(ep_map)) + 1
    starts = np.concatenate(([0], bounds)).tolist()
    ends = np.concatenate((bounds, [len(ep_map)])).tolist()
    assert len(starts) == len(np.unique(ep_map)), "episode ids are not contiguous runs"
    episodes = list(zip(starts, ends))
    if max_eps:
        episodes = episodes[:max_eps]

    # deterministic ~VAL_FRAC val split BY EPISODE (seed SPLIT_SEED)
    n_val = max(1, round(VAL_FRAC * len(episodes)))
    val_ids = set(np.random.default_rng(SPLIT_SEED).choice(len(episodes), size=n_val, replace=False).tolist())
    split_eps = {"train": [ep for k, ep in enumerate(episodes) if k not in val_ids],
                 "val": [ep for k, ep in enumerate(episodes) if k in val_ids]}
    import imageio.v2 as imageio
    h, w = imageio.imread(os.path.join(img_dir, f"{episodes[0][0]}.jpg")).shape[:2]   # native size (e.g. 112x192)
    log(f"[recording] {src}: {len(episodes)} episodes / {sum(e - s for s, e in episodes)} frames "
        f"({h}x{w} @ {fps} Hz) -> train {len(split_eps['train'])} / val {len(split_eps['val'])} "
        f"(seed {SPLIT_SEED})")
    t0 = time.time()

    # 1. encode every episode's jpgs to a native-size clip at full parallelism (the heavy step)
    workers = int(os.environ.get("GEN_WORKERS") or (os.cpu_count() or 4))
    ego_root = os.path.join(run_dir, "media", CAM)
    enc_jobs = []
    for sp, eps in split_eps.items():
        os.makedirs(os.path.join(ego_root, sp), exist_ok=True)
        for i, (s, e) in enumerate(eps):
            enc_jobs.append({"img_dir": img_dir, "frames": range(s, e), "fps": fps,
                             "out": os.path.join(ego_root, sp, f"ep_{i:04d}.mp4")})
    log(f"[encode] {len(enc_jobs)} episode clips on {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for j, _ in enumerate(ex.map(_encode_ep, enc_jobs), 1):
            if j % 10 == 0 or j == len(enc_jobs):
                log(f"[encode] {j}/{len(enc_jobs)}  ({time.time() - t0:.0f}s)")

    # 2. write lerobot datasets, ingesting the clips (one dataset per split, in parallel)
    data = {sp: ([states[s:e] for s, e in eps], [actions[s:e] for s, e in eps])
            for sp, eps in split_eps.items()}
    lr_jobs = [{"name": sp, "root_split": os.path.join(run_dir, sp), "repo_id": f"{name}/{sp}",
                "obs": obs, "act": act, "fps": fps, "hw": (h, w), "task": name,
                "ego_dir": os.path.join(ego_root, sp)} for sp, (obs, act) in data.items()]
    log(f"[lerobot] writing {len(lr_jobs)} split datasets (vectors + observation.images.{CAM}): "
        + ", ".join(j["name"] for j in lr_jobs))
    with ProcessPoolExecutor(max_workers=min(len(lr_jobs), workers)) as ex:
        for k, done in enumerate(ex.map(_write_lr, lr_jobs), 1):
            log(f"[lerobot] {k}/{len(lr_jobs)} wrote {done}  ({time.time() - t0:.0f}s)")

    # 3. meta (train-only norm stats) + summary.json — same shape as data_generation, minus torus-only fields
    ecfg = RecordedConfig(obs_dim=states.shape[-1], action_dim=actions.shape[-1], dt=1.0 / fps)
    dims = {"obs_dim": ecfg.obs_dim, "action_dim": ecfg.action_dim, "dt": ecfg.dt}
    splits_meta = {sp: {"n_traj": len(eps), "steps": int(round(np.mean([e - s for s, e in eps]))),
                        "seed": SPLIT_SEED} for sp, eps in split_eps.items()}
    write_meta(run_dir, ecfg, splits_meta, {sp: dict(dims) for sp in split_eps},
               {sp: "camera" for sp in split_eps}, fps,
               compute_norm_stats(np.concatenate(data["train"][0]), np.concatenate(data["train"][1])))

    P, F = int(cfg.data.P), int(cfg.data.F)
    counts = {}
    for sp, eps in split_eps.items():
        lens = [e - s for s, e in eps]
        n, tot = len(eps), int(sum(lens))
        sec = tot / fps
        counts[sp] = {"episodes": n, "steps_per_episode": int(round(np.mean(lens))), "transitions": tot,
                      "seconds": round(sec, 2), "minutes": round(sec / 60, 3), "hours": round(sec / 3600, 5),
                      "training_windows": int(sum(max(0, ln - (P + F) + 1) for ln in lens))}
    card = json.load(open(os.path.join(run_dir, "dataset_card.json")))
    norm = json.load(open(os.path.join(run_dir, "normalization_stats.json")))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"dataset_root": run_dir, "source": src, "counts": counts, "splits": card["splits"],
                   "split_env": card["split_env"], "coloring": card["coloring"], "fps": fps,
                   "camera": CAM, "image_hw": [int(h), int(w)],
                   "normalization_stats": norm}, f, indent=2)

    log(f"[done] {run_dir}  ({time.time() - t0:.0f}s total)")
    log(f"[done] now train with:  data.root={run_dir} environments=recorded")


if __name__ == "__main__":
    main()
