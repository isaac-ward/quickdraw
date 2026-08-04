"""Bespoke per-dataset processors -> the standard quickdraw recorded/RecordedEnv run layout.

Each raw dump (starling jpgs, a robocasa lerobot repo, ...) has its own quirks; a small `processor`
parses ONE format into a list of canonical `Episode`s, and the SHARED `build_recorded_dataset`
machinery turns those into a run folder identical in shape to `data_generation` (minus the
simulator/summary renders). Point training at the run folder with `environments=recorded` (the config
GROUP — it carries obs_dim/action_dim/dt; override those for non-starling dims). Verify with `check_dataset`.

    python -m quickdraw.data.processors +processor=starling \\
        +source.dir=~/user_irw/seamstress/assets/flightroom-starling_processed_112x192_30hz \\
        +source.name=flightroom_starling          # smoke: +source.max_episodes=4

    python -m quickdraw.data.processors +processor=robocasa \\
        +source.repo=madang6/quickdraw-robocasa-scene4-4h +source.name=robocasa +source.max_episodes=2

Frames per `Episode` may be per-frame image PATHS (lazy; starling's jpgs), an in-memory (T,H,W,3)
uint8 array (robocasa's decoded clips), or None (no camera -> proprio-only world model). Non-image
datasets skip the media/ encode entirely and write vector-only lerobot splits."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import hydra
import numpy as np

from .generate import compute_norm_stats, write_lerobot_split, write_meta
from ..environments.recorded import RecordedConfig
from ..logging import viz
from ..utils.logging import make_run_dir

VAL_FRAC = 0.1       # ~10% of episodes -> val (deterministic, by episode)
SPLIT_SEED = 0       # deterministic episode split
_P, _F = 8, 64       # conf/data/torus.yaml defaults; only for the informational `training_windows` count


@dataclass
class Episode:
    """The canonical intermediate EVERY processor emits (one recorded trajectory)."""
    states: np.ndarray             # (T, obs_dim) float32
    actions: np.ndarray            # (T, action_dim) float32
    frames: list[str] | np.ndarray | None
    #   list[str]  = per-frame image file paths in temporal order (lazy; starling's jpgs)
    #   np.ndarray = (T, H, W, 3) uint8 in-memory frames (robocasa's decoded clips)
    #   None       = no camera -> proprio-only dataset


# ---------------------------------------------------------------------------
# SHARED build machinery (lifted from recording_to_lerobot)
# ---------------------------------------------------------------------------

def _encode_ep(job: dict):
    """One episode's frames (temporal order, native size) -> mp4 (runs in a worker).

    `frames` is either a list of image file paths (read each) or an in-memory (T,H,W,3) array."""
    import imageio.v2 as imageio

    frames = job["frames"]
    if isinstance(frames, np.ndarray):
        seq = [f[..., :3] for f in frames]
    else:
        seq = [imageio.imread(p)[..., :3] for p in frames]
    viz.save_mp4(job["out"], seq, job["fps"])
    return job["out"]


def _write_lr(job: dict):
    """One split's lerobot dataset (vectors, plus the pre-encoded clips as observation.images.<cam>
    when this dataset has a camera; vector-only when `ego_dir` is None)."""
    write_lerobot_split(job["root_split"], job["repo_id"], job["obs"], job["act"], job["fps"],
                        fpv_dir=job["ego_dir"], fpv_size=job["hw"], cam=job["cam"], task=job["task"])
    return job["name"]


def _frame_hw(frames) -> tuple[int, int]:
    """Native (H, W) of an episode's first frame (path or in-memory array)."""
    if isinstance(frames, np.ndarray):
        return int(frames.shape[1]), int(frames.shape[2])
    import imageio.v2 as imageio
    return tuple(imageio.imread(frames[0]).shape[:2])


def build_recorded_dataset(name: str, episodes: list[Episode], fps: int, cam: str, log=None,
                           extra_splits: dict[str, list[Episode]] | None = None) -> str:
    """Turn canonical `Episode`s into a standard recorded run folder. Returns the run_dir.

    make_run_dir -> deterministic ~10% val split BY EPISODE (seed 0) of `episodes` into train/val ->
    parallel-encode each episode's frames to media/<cam>/<split>/ep_XXXX.mp4 (SKIPPED entirely when
    frames is None) -> write one lerobot dataset per split (with the clips as observation.images.<cam>,
    or vector-only) -> write_meta + summary.json + dataset_card.json (generic; no torus geometry fields).

    `extra_splits` = {split_name: [Episode]} adds EXTRA named splits (e.g. a held-out `eval`
    collection) alongside the normal train/val: each is written to its OWN split directory VERBATIM
    (NO random splitting), encoded/recorded exactly like train/val. Extra splits never affect the
    train/val random split nor the train-only norm stats."""
    log = log or (lambda m: print(m, flush=True))
    has_frames = episodes[0].frames is not None

    run_dir = make_run_dir("recording", name)
    t0 = time.time()

    # deterministic ~VAL_FRAC val split BY EPISODE (seed SPLIT_SEED), preserving processor order
    n_val = max(1, round(VAL_FRAC * len(episodes)))
    val_ids = set(np.random.default_rng(SPLIT_SEED).choice(len(episodes), size=n_val, replace=False).tolist())
    split_eps = {"train": [ep for k, ep in enumerate(episodes) if k not in val_ids],
                 "val": [ep for k, ep in enumerate(episodes) if k in val_ids]}
    # extra named splits (e.g. a held-out `eval` collection): kept in their OWN dir, verbatim, no split
    for sp_name, sp_eps in (extra_splits or {}).items():
        split_eps[sp_name] = list(sp_eps)

    hw = _frame_hw(episodes[0].frames) if has_frames else None
    all_eps = [ep for eps in split_eps.values() for ep in eps]
    split_desc = ", ".join(f"{sp} {len(eps)}" for sp, eps in split_eps.items())
    log(f"[recorded] {name}: {len(all_eps)} episodes / {sum(len(e.states) for e in all_eps)} frames "
        f"({'proprio-only' if not has_frames else f'{hw[0]}x{hw[1]}'} @ {fps} Hz) -> "
        f"{split_desc} (train/val seed {SPLIT_SEED})")

    workers = int(os.environ.get("GEN_WORKERS") or (os.cpu_count() or 4))
    ego_root = os.path.join(run_dir, "media", cam)

    # 1. encode every episode's frames to a native-size clip at full parallelism (SKIP if no camera)
    if has_frames:
        enc_jobs = []
        for sp, eps in split_eps.items():
            os.makedirs(os.path.join(ego_root, sp), exist_ok=True)
            for i, ep in enumerate(eps):
                enc_jobs.append({"frames": ep.frames, "fps": fps,
                                 "out": os.path.join(ego_root, sp, f"ep_{i:04d}.mp4")})
        log(f"[encode] {len(enc_jobs)} episode clips on {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for j, _ in enumerate(ex.map(_encode_ep, enc_jobs), 1):
                if j % 10 == 0 or j == len(enc_jobs):
                    log(f"[encode] {j}/{len(enc_jobs)}  ({time.time() - t0:.0f}s)")

    # 2. write one lerobot dataset per split (parallel), ingesting the clips when present
    lr_jobs = [{"name": sp, "root_split": os.path.join(run_dir, sp), "repo_id": f"{name}/{sp}",
                "obs": [e.states for e in eps], "act": [e.actions for e in eps], "fps": fps,
                "hw": hw, "cam": cam, "task": name,
                "ego_dir": os.path.join(ego_root, sp) if has_frames else None}
               for sp, eps in split_eps.items()]
    log(f"[lerobot] writing {len(lr_jobs)} split datasets "
        f"({'vectors + observation.images.' + cam if has_frames else 'vectors only'}): "
        + ", ".join(j["name"] for j in lr_jobs))
    with ProcessPoolExecutor(max_workers=min(len(lr_jobs), workers)) as ex:
        for k, done in enumerate(ex.map(_write_lr, lr_jobs), 1):
            log(f"[lerobot] {k}/{len(lr_jobs)} wrote {done}  ({time.time() - t0:.0f}s)")

    # 3. meta (train-only norm stats) + summary.json — generic (no torus geometry fields)
    obs_dim = episodes[0].states.shape[-1]
    act_dim = episodes[0].actions.shape[-1]
    ecfg = RecordedConfig(obs_dim=obs_dim, action_dim=act_dim, dt=1.0 / fps)
    dims = {"obs_dim": ecfg.obs_dim, "action_dim": ecfg.action_dim, "dt": ecfg.dt}
    splits_meta = {sp: {"n_traj": len(eps), "seed": SPLIT_SEED,
                        "steps": int(round(np.mean([len(e.states) for e in eps])))}
                   for sp, eps in split_eps.items()}
    train_obs = np.concatenate([e.states for e in split_eps["train"]])
    train_act = np.concatenate([e.actions for e in split_eps["train"]])
    coloring = {sp: ("camera" if has_frames else "vector") for sp in split_eps}
    write_meta(run_dir, ecfg, splits_meta, {sp: dict(dims) for sp in split_eps}, coloring, fps,
               compute_norm_stats(train_obs, train_act))

    counts = {}
    for sp, eps in split_eps.items():
        lens = [len(e.states) for e in eps]
        n, tot = len(eps), int(sum(lens))
        sec = tot / fps
        counts[sp] = {"episodes": n, "steps_per_episode": int(round(np.mean(lens))), "transitions": tot,
                      "seconds": round(sec, 2), "minutes": round(sec / 60, 3), "hours": round(sec / 3600, 5),
                      "training_windows": int(sum(max(0, ln - (_P + _F) + 1) for ln in lens))}
    card = json.load(open(os.path.join(run_dir, "dataset_card.json")))
    norm = json.load(open(os.path.join(run_dir, "normalization_stats.json")))
    summary = {"dataset_root": run_dir, "counts": counts, "splits": card["splits"],
               "split_env": card["split_env"], "coloring": card["coloring"], "fps": fps}
    if has_frames:
        summary["camera"] = cam
        summary["image_hw"] = [int(hw[0]), int(hw[1])]
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({**summary, "normalization_stats": norm}, f, indent=2)

    log(f"[done] {run_dir}  ({time.time() - t0:.0f}s total)")
    log(f"[done] now train with:  data.root={run_dir} environments=recorded "
        f"data.repo_id={name}" + (f" data.cam={cam}" if has_frames else ""))
    return run_dir


# ---------------------------------------------------------------------------
# bespoke processors: parse ONE raw format -> (name, episodes, fps, cam)
# ---------------------------------------------------------------------------

def _parse_starling_dir(src: str, cam: str) -> list[Episode]:
    """Parse one flightroom-starling processed folder into canonical Episodes (CONTIGUOUS runs of
    the episode map; each Episode.frames is that run's jpg paths in temporal order, lazy)."""
    npz = np.load(os.path.join(src, "data.npz"))
    states = npz["states"].astype(np.float32)
    actions = npz["actions"].astype(np.float32)
    ep_map = npz["episode_indices_mapping"]
    img_dir = os.path.join(src, "images", cam)

    # episodes are CONTIGUOUS runs of ep_map -> per-episode global-index ranges [start, end)
    bounds = np.flatnonzero(np.diff(ep_map)) + 1
    starts = np.concatenate(([0], bounds)).tolist()
    ends = np.concatenate((bounds, [len(ep_map)])).tolist()
    assert len(starts) == len(np.unique(ep_map)), "episode ids are not contiguous runs"

    episodes = []
    for s, e in zip(starts, ends):
        frames = [os.path.join(img_dir, f"{i}.jpg") for i in range(s, e)]
        episodes.append(Episode(states=states[s:e], actions=actions[s:e], frames=frames))
    return episodes


def starling(cfg) -> tuple[str, list[Episode], int, str, dict[str, list[Episode]] | None]:
    """Flightroom-starling processed dump: `data.npz` (`states`, `actions`, `episode_indices_mapping`)
    + `images/ego/<global_idx>.jpg`. Episodes are CONTIGUOUS global-index runs of the episode map;
    each `Episode.frames` is that run's jpg paths in temporal order (lazy). Camera: `ego`.

    With `+source.eval_dir=<path>`, that folder is parsed the SAME way and returned as an EXTRA held-out
    `eval` split (its own dir, verbatim; NOT train/val); `dir` still gets the normal seed-0 train/val split.

    Args: +source.dir=<processed folder> +source.name=<name>
          [+source.eval_dir=<path>] [+source.max_episodes=N] [+source.fps=30]."""
    src_cfg = cfg.get("source", None)
    if src_cfg is None or not src_cfg.get("dir"):
        raise ValueError("pass +source.dir=<processed recording folder> "
                         "(optional: +source.name=..., +source.eval_dir=<path>, +source.max_episodes=N, +source.fps=30)")
    src = os.path.expanduser(str(src_cfg.dir))
    name = str(src_cfg.get("name", "starling"))
    max_eps = int(src_cfg.get("max_episodes", 0) or 0)
    fps = int(src_cfg.get("fps", 30))
    cam = "ego"

    episodes = _parse_starling_dir(src, cam)
    if max_eps:
        episodes = episodes[:max_eps]

    extra_splits = None
    if src_cfg.get("eval_dir"):
        eval_src = os.path.expanduser(str(src_cfg.eval_dir))
        eval_episodes = _parse_starling_dir(eval_src, cam)
        if max_eps:
            eval_episodes = eval_episodes[:max_eps]
        extra_splits = {"eval": eval_episodes}
    return name, episodes, fps, cam, extra_splits


def robocasa(cfg) -> tuple[str, list[Episode], int, str, dict[str, list[Episode]] | None]:
    """HF dataset `madang6/quickdraw-robocasa-scene4-4h` — already lerobot-native (one parquet per
    episode under train/data/, one mp4 per episode per camera under train/videos/). Columns:
    `observation.state` (16), `action` (12); fps 20; 3 cameras (256x256x3). We pull only the episodes
    a smoke needs (per-file download), map `states=observation.state`, `actions=action`, decode the
    chosen camera's per-episode mp4 to a (T,H,W,3) uint8 array, and route through the shared builder
    for a uniform, freshly-normalized output. Camera leaf name is used as `cam`.

    Args: +source.repo=<repo> +source.name=<name> [+source.max_episodes=N] [+source.camera=<leaf>]."""
    import imageio.v2 as imageio
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    src_cfg = cfg.get("source", None)
    if src_cfg is None or not src_cfg.get("repo"):
        raise ValueError("pass +source.repo=<hf dataset repo> (optional: +source.name=..., "
                         "+source.max_episodes=N, +source.camera=<camera leaf name>)")
    repo = str(src_cfg.repo)
    name = str(src_cfg.get("name", "robocasa"))
    max_eps = int(src_cfg.get("max_episodes", 0) or 0)
    cam = str(src_cfg.get("camera", "robot0_agentview_left"))   # scene third-person view by default
    vkey = f"observation.images.{cam}"

    info = json.load(open(hf_hub_download(repo, "train/meta/info.json", repo_type="dataset")))
    fps = int(info["fps"])
    n_total = int(info["total_episodes"])
    chunk = int(info["chunks_size"])
    n = min(max_eps, n_total) if max_eps else n_total

    episodes = []
    for idx in range(n):
        c = idx // chunk
        pq_path = hf_hub_download(repo, f"train/data/chunk-{c:03d}/episode_{idx:06d}.parquet", repo_type="dataset")
        vid_path = hf_hub_download(repo, f"train/videos/chunk-{c:03d}/{vkey}/episode_{idx:06d}.mp4", repo_type="dataset")
        t = pq.read_table(pq_path)
        states = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
        rd = imageio.get_reader(vid_path)
        frames = np.stack([np.asarray(f)[..., :3] for f in rd]).astype(np.uint8)
        rd.close()
        frames = frames[:len(states)]   # align 1:1 with the vector rows (drop any trailing decode frame)
        episodes.append(Episode(states=states, actions=actions, frames=frames))
    return name, episodes, fps, cam, None   # robocasa: no extra splits (builder's seed-0 train/val only)


PROCESSORS = {"starling": starling, "robocasa": robocasa}


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg):
    proc = cfg.get("processor", None)
    if proc is None or str(proc) not in PROCESSORS:
        raise ValueError(f"pass +processor=<{'|'.join(PROCESSORS)}> (got {proc!r})")
    name, episodes, fps, cam, extra_splits = PROCESSORS[str(proc)](cfg)

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    run_dir = build_recorded_dataset(name, episodes, fps, cam, log=log, extra_splits=extra_splits)
    with open(os.path.join(run_dir, "progress.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
