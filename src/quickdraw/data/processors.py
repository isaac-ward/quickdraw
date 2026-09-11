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
        # all three cameras into one dataset:  +source.camera=all

    python -m quickdraw.data.processors +processor=block_stack \\
        +source.dir=scratch/longhand +source.name=block_stack   # Swoosh right-arm teleop
        # smoke: +source.max_runs_per_campaign=2 '+source.cameras=[scene_left]' '+source.hw=[96,128]'

Frames per `Episode` may be per-frame image PATHS (lazy; starling's jpgs), an in-memory (T,H,W,3)
uint8 array (robocasa's decoded clips), or None (no camera -> proprio-only world model). Non-image
datasets skip the media/ encode entirely and write vector-only lerobot splits."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import hydra
import numpy as np

from .generate import compute_norm_stats, write_lerobot_split, write_meta
from ..environments.recorded import RecordedConfig
from ..logging import viz
from ..utils.logging import make_run_dir

VAL_FRAC = 0.1       # ~10% -> val. By EPISODE for the random default; by FRAMES where a
                     # processor supplies its own val_ids (see _longest_first_val).
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
    task: str | None = None
    #   WHAT CONDITION THIS EPISODE WAS RECORDED UNDER, e.g. "campaign5_ood_object". Written to
    #   lerobot's per-frame `task` field, which is the only free-text label that survives packaging
    #   (it lands in <split>/meta/tasks.parquet and every episode row references it). None -> the
    #   dataset name, which is what every processor used to do UNCONDITIONALLY.
    #
    #   WHY IT EXISTS (2026-09-08). The published `starling-2` eval split holds 49 episodes drawn from
    #   four separate OOD campaigns and labels every one of them 'starling-2'. Which episodes were the
    #   visual shift and which the dynamics shift is now UNRECOVERABLE from the dataset: the raw
    #   campaign directories say it, the upstream summary.json says it, and this field is where that
    #   survived to -- except it did not exist, so it was dropped. An OOD split you cannot slice by
    #   condition is not an OOD split.


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
    # NO playback_fps: these clips ARE the dataset (data/generate.py reads them back frame-for-frame,
    # and data/dataset.py trains off them), so they stay at the true capture rate. See viz.save_mp4.
    viz.save_mp4(job["out"], seq, job["fps"])
    return job["out"]


def _write_lr(job: dict):
    """One split's lerobot dataset (vectors, plus the pre-encoded clips as observation.images.<cam>
    when this dataset has a camera; vector-only when `ego_dir` is None)."""
    write_lerobot_split(job["root_split"], job["repo_id"], job["obs"], job["act"], job["fps"],
                        fpv_dir=job["ego_dir"], fpv_size=job["hw"], cam=job["cam"], task=job["task"])
    return job["name"]


def _frame_hw(frames) -> tuple[int, int]:
    """Native (H, W) of an episode's first frame (path, in-memory array, or {cam: array})."""
    if isinstance(frames, dict):
        frames = next(iter(frames.values()))
    if isinstance(frames, np.ndarray):
        return int(frames.shape[1]), int(frames.shape[2])
    import imageio.v2 as imageio
    return tuple(imageio.imread(frames[0]).shape[:2])


def build_recorded_dataset(name: str, episodes: list[Episode], fps: int, cam, log=None,
                           extra_splits: dict[str, list[Episode]] | None = None,
                           val_ids: set[int] | None = None,
                           obs_identity: list[int] | None = None) -> str:
    """Turn canonical `Episode`s into a standard recorded run folder. Returns the run_dir.

    make_run_dir -> deterministic ~10% val split BY EPISODE (seed 0) of `episodes` into train/val ->
    parallel-encode each episode's frames to media/<cam>/<split>/ep_XXXX.mp4 (SKIPPED entirely when
    frames is None) -> write one lerobot dataset per split (with the clips as observation.images.<cam>,
    or vector-only) -> write_meta + summary.json + dataset_card.json (generic; no torus geometry fields).

    MULTI-CAMERA: pass `cam` as a list and every `Episode.frames` as a {cam: (T,H,W,3)} dict. Each
    camera gets its own media/<cam>/ tree and its own observation.images.<cam> video key in every
    split. Encoding is one flat job list across (camera x episode), so N cameras cost N times the
    encode but still saturate the pool. A single str + array behaves exactly as before.

    `val_ids` = indices into `episodes` that go to val, REPLACING the random draw. A processor
    passes this when the split has to mean something -- `block_stack` puts the LONGEST trajectories in
    val, because the evaluable open-loop rollout horizon is capped by the SHORTEST val episode, and
    on lego a random split cost 5x the horizon. None -> the historic random VAL_FRAC draw.

    `obs_identity` lists observation dims to leave UNNORMALIZED (mean 0, std 1) -- see
    compute_norm_stats. Pass it for a group of dims that jointly encode one geometric object,
    such as a 6D rotation, where per-dim scaling would break the coupling between them.

    `extra_splits` = {split_name: [Episode]} adds EXTRA named splits (e.g. a held-out `eval`
    collection) alongside the normal train/val: each is written to its OWN split directory VERBATIM
    (NO random splitting), encoded/recorded exactly like train/val. Extra splits never affect the
    train/val random split nor the train-only norm stats.

    EVERY split keeps its per-episode preview clips under media/<cam>/<split>/ -- they are how a human
    (or a Hub visitor) sees what a split actually contains, which is worth the bytes for train and val
    too, not just for a handful of OOD episodes."""
    log = log or (lambda m: print(m, flush=True))
    has_frames = episodes[0].frames is not None

    run_dir = make_run_dir("recording", name)
    t0 = time.time()

    # deterministic ~VAL_FRAC val split BY EPISODE (seed SPLIT_SEED), preserving processor order
    if val_ids is None:
        n_val = max(1, round(VAL_FRAC * len(episodes)))
        val_ids = set(np.random.default_rng(SPLIT_SEED).choice(len(episodes), size=n_val, replace=False).tolist())
        split_rule = f"random, seed {SPLIT_SEED}"
    else:
        val_ids = {int(i) for i in val_ids}
        bad = [i for i in val_ids if not 0 <= i < len(episodes)]
        assert not bad, f"val_ids out of range for {len(episodes)} episodes: {bad}"
        assert val_ids, "val_ids was given but empty -- that would leave val with no episodes"
        assert len(val_ids) < len(episodes), "val_ids covers every episode -- train would be empty"
        split_rule = "explicit (processor-chosen)"
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
        f"{split_desc} (train/val: {split_rule})")

    workers = int(os.environ.get("GEN_WORKERS") or (os.cpu_count() or 4))
    cams = [cam] if isinstance(cam, str) else list(cam)
    ego_roots = {c: os.path.join(run_dir, "media", c) for c in cams}

    # 1. encode every (camera, episode) pair to a native-size clip at full parallelism (SKIP if no camera)
    if has_frames:
        enc_jobs = []
        for c in cams:
            for sp, eps in split_eps.items():
                os.makedirs(os.path.join(ego_roots[c], sp), exist_ok=True)
                for i, ep in enumerate(eps):
                    # single-camera episodes carry a bare array; multi-camera ones a {cam: array} dict
                    fr = ep.frames[c] if isinstance(ep.frames, dict) else ep.frames
                    enc_jobs.append({"frames": fr, "fps": fps,
                                     "out": os.path.join(ego_roots[c], sp, f"ep_{i:04d}.mp4")})
        log(f"[encode] {len(enc_jobs)} episode clips ({len(cams)} camera(s)) on {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for j, _ in enumerate(ex.map(_encode_ep, enc_jobs), 1):
                if j % 10 == 0 or j == len(enc_jobs):
                    log(f"[encode] {j}/{len(enc_jobs)}  ({time.time() - t0:.0f}s)")

    # 2. write one lerobot dataset per split (parallel), ingesting every camera's clips when present
    lr_jobs = [{"name": sp, "root_split": os.path.join(run_dir, sp), "repo_id": f"{name}/{sp}",
                "obs": [e.states for e in eps], "act": [e.actions for e in eps], "fps": fps,
                "hw": hw, "cam": cams,
                # PER-EPISODE task labels, falling back to the dataset name for processors that set none.
                "task": [e.task or name for e in eps],
                "ego_dir": {c: os.path.join(ego_roots[c], sp) for c in cams} if has_frames else None}
               for sp, eps in split_eps.items()]
    log(f"[lerobot] writing {len(lr_jobs)} split datasets "
        f"({'vectors + ' + ', '.join('observation.images.' + c for c in cams) if has_frames else 'vectors only'}): "
        + ", ".join(j["name"] for j in lr_jobs))
    with ProcessPoolExecutor(max_workers=min(len(lr_jobs), workers)) as ex:
        for k, done in enumerate(ex.map(_write_lr, lr_jobs), 1):
            log(f"[lerobot] {k}/{len(lr_jobs)} wrote {done}  ({time.time() - t0:.0f}s)")

    # 3. meta (train-only norm stats) + summary.json — generic (no torus geometry fields)
    obs_dim = episodes[0].states.shape[-1]
    act_dim = episodes[0].actions.shape[-1]
    ecfg = RecordedConfig(obs_dim=obs_dim, action_dim=act_dim, dt=1.0 / fps)
    dims = {"obs_dim": ecfg.obs_dim, "action_dim": ecfg.action_dim, "dt": ecfg.dt}
    splits_meta = {sp: {"n_traj": len(eps), "seed": SPLIT_SEED, "split_rule": split_rule,
                        "steps": int(round(np.mean([len(e.states) for e in eps])))}
                   for sp, eps in split_eps.items()}
    train_obs = np.concatenate([e.states for e in split_eps["train"]])
    train_act = np.concatenate([e.actions for e in split_eps["train"]])
    coloring = {sp: ("camera" if has_frames else "vector") for sp in split_eps}
    write_meta(run_dir, ecfg, splits_meta, {sp: dict(dims) for sp in split_eps}, coloring, fps,
               compute_norm_stats(train_obs, train_act, obs_identity=obs_identity))

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
        summary["camera"] = cams[0] if len(cams) == 1 else cams   # scalar when one, list when several
        summary["cameras"] = cams                                  # always the full list, for tooling
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

def _starling_campaigns(src: str, n_eps: int) -> list[str] | None:
    """Per-episode CAMPAIGN name from the upstream dump's own `summary.json`, or None if unavailable.

    The seamstress exporter writes `episodes: [{episode_index, run_dir, ...}]` where `run_dir` is
    `<raw_root>/<campaign>/<run_timestamp>` -- so the campaign is the run_dir's parent directory, and
    that string is the ONLY record of what condition an episode was flown under (e.g.
    `campaign4_ood_10hz` = a 10 Hz control-rate shift, `campaign5_ood_object` = an object added to the
    flightroom). `data.npz` does NOT carry it.

    Returns None rather than raising when the summary is missing, mismatched, or has no run_dirs: a
    dump without it is still perfectly loadable, it just cannot be sliced by condition afterwards.
    """
    p = os.path.join(src, "summary.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            eps = (json.load(f) or {}).get("episodes") or []
        names = [os.path.basename(os.path.dirname(str(e["run_dir"]))) for e in eps if e.get("run_dir")]
    except Exception:
        return None
    if len(names) != n_eps:
        print(f"[starling] {p} lists {len(names)} run_dirs for {n_eps} parsed episodes -- NOT using it "
              f"for campaign labels (the two must line up 1:1 or the labels would be wrong)", flush=True)
        return None
    return names


def _parse_starling_dir(src: str, cam: str) -> list[Episode]:
    """Parse one flightroom-starling processed folder into canonical Episodes (CONTIGUOUS runs of
    the episode map; each Episode.frames is that run's jpg paths in temporal order, lazy).

    Each Episode also gets its CAMPAIGN as `task` when the dump's summary.json supplies one -- see
    `_starling_campaigns` and `Episode.task` for why that matters."""
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

    camps = _starling_campaigns(src, len(starts))
    if camps:
        import collections
        c = collections.Counter(camps)
        print(f"[starling] {src}: campaign labels from summary.json -> "
              f"{dict(sorted(c.items()))}", flush=True)
    episodes = []
    for k, (s, e) in enumerate(zip(starts, ends)):
        frames = [os.path.join(img_dir, f"{i}.jpg") for i in range(s, e)]
        episodes.append(Episode(states=states[s:e], actions=actions[s:e], frames=frames,
                                task=(camps[k] if camps else None)))
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

    Args: +source.repo=<repo> +source.name=<name> [+source.max_episodes=N]
          [+source.camera=<leaf> | <leaf,leaf,...> | all]   -- several cameras land in ONE dataset as
          separate observation.images.<cam> keys; pick one at train time with `data.cam`."""
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
    # MULTI-CAMERA: comma-separated names, or "all" for every camera the source repo actually has.
    # Default stays the single scene third-person view, so existing invocations are unchanged.
    cam_arg = str(src_cfg.get("camera", "robot0_agentview_left")).strip()
    if cam_arg == "all":
        from huggingface_hub import list_repo_files as _lrf
        cams = sorted({p.split("observation.images.")[1].split("/")[0]
                       for p in _lrf(repo, repo_type="dataset") if "observation.images." in p})
        if not cams:
            raise ValueError(f"{repo}: no observation.images.* video keys found; cannot use camera=all")
    else:
        cams = [c.strip() for c in cam_arg.split(",") if c.strip()]
    multi = len(cams) > 1

    info = json.load(open(hf_hub_download(repo, "train/meta/info.json", repo_type="dataset")))
    fps = int(info["fps"])
    n_total = int(info["total_episodes"])
    chunk = int(info["chunks_size"])
    n = min(max_eps, n_total) if max_eps else n_total

    def _decode(idx: int, c_chunk: int, cam_name: str, n_rows: int) -> np.ndarray:
        vkey = f"observation.images.{cam_name}"
        vp = hf_hub_download(repo, f"train/videos/chunk-{c_chunk:03d}/{vkey}/episode_{idx:06d}.mp4",
                             repo_type="dataset")
        rd = imageio.get_reader(vp)
        fr = np.stack([np.asarray(f)[..., :3] for f in rd]).astype(np.uint8)
        rd.close()
        return fr[:n_rows]   # align 1:1 with the vector rows (drop any trailing decode frame)

    episodes = []
    for idx in range(n):
        c = idx // chunk
        pq_path = hf_hub_download(repo, f"train/data/chunk-{c:03d}/episode_{idx:06d}.parquet", repo_type="dataset")
        t = pq.read_table(pq_path)
        states = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
        # every camera of an episode is rendered from the SAME state trace, so they are frame-aligned
        # by construction and all get truncated to the vector-row count identically.
        frames = ({cm: _decode(idx, c, cm, len(states)) for cm in cams} if multi
                  else _decode(idx, c, cams[0], len(states)))
        episodes.append(Episode(states=states, actions=actions, frames=frames))
    return name, episodes, fps, (cams if multi else cams[0]), None   # no extra splits (seed-0 train/val)


def _bag_one(job: dict):
    """One rosbag run -> Episode. Module-level so ProcessPoolExecutor can pickle it."""
    from .rosbag import read_run
    st, ac, fr = read_run(job["dir"], target_hz=job["hz"], out_hw=tuple(job["hw"]))
    return job["campaign"], job["dir"], Episode(states=st, actions=ac, frames=fr, task=job["campaign"])


def starling_bags(cfg) -> tuple[str, list[Episode], int, str, dict[str, list[Episode]] | None]:
    """RAW rosbag2 flight recordings -> canonical Episodes, campaign labels intact. NO ROS, NO seamstress.

    This replaces a two-stage path that went through another repo: raw bags -> (seamstress) processed
    dump -> (here) lerobot. That indirection is why the published starling-2 dataset lost the one thing
    it most needed -- WHICH CAMPAIGN each episode came from -- and why its `eval` split's provenance took
    a day to reconstruct and still came out ambiguous. Reading the bags here makes the whole path from
    flight to dataset auditable in one repo. See data/rosbag.py for the bag format and the topic rates.

    LAYOUT EXPECTED: <dir>/<campaign>/<run_*>/{metadata.yaml,*_0.db3}. One Episode per run, in sorted
    order, with `Episode.task` set to the campaign directory name -- so the condition survives into
    <split>/meta/tasks.parquet and a consumer can slice by it (see Episode.task).

    SPLIT POLICY, and it is deliberately name-driven rather than positional:
      * a campaign matching `eval_globs` (default: anything with "ood" or "memory" in its name) becomes
        its OWN eval split, named `eval_<campaign suffix>` -- so `campaign21-ood-noodle` lands in
        `eval_ood_noodle`. Separate splits rather than one pooled `eval` because the loader reads
        <root>/<split>/ directly, so each is usable today with no new code, mirroring the torus
        `eval_ood_*` convention; the per-episode task labels then allow finer slicing inside one.
      * a campaign matching `exclude_globs` (default: "*nothing*") is dropped entirely.
      * everything else forms the train/val pool, split deterministically by build_recorded_dataset.

    TARGET_HZ defaults to data/rosbag.py's 15.0, BELOW the ~17 Hz camera, so every step is a distinct
    frame. The published dataset used 30 Hz, which duplicated 44% of consecutive frames (measured) --
    a world model trained on that is asked to predict "no change" on nearly half its steps.

    Args: +source.dir=<tree of campaign dirs> +source.name=<name>
          [+source.target_hz=15] [+source.hw=[112,192]] [+source.workers=16]
          [+source.eval_globs=[*ood*,*memory*]] [+source.exclude_globs=[*nothing*]]
          [+source.max_runs_per_campaign=N]   <- smoke-test escape hatch
    """
    import fnmatch
    from concurrent.futures import ProcessPoolExecutor
    from .rosbag import TARGET_HZ_DEFAULT

    sc = cfg.get("source", None)
    if sc is None or not sc.get("dir"):
        raise ValueError("pass +source.dir=<tree containing campaign*/run_*/ bag dirs> (+source.name=...)")
    root = os.path.expanduser(str(sc.dir))
    name = str(sc.get("name", "starling"))
    hz = float(sc.get("target_hz", TARGET_HZ_DEFAULT))
    hw = tuple(int(x) for x in (sc.get("hw", None) or (112, 192)))
    workers = int(sc.get("workers", 16))
    max_per = int(sc.get("max_runs_per_campaign", 0) or 0)
    eval_globs = [str(g) for g in (sc.get("eval_globs", None) or ["*ood*", "*memory*"])]
    excl_globs = [str(g) for g in (sc.get("exclude_globs", None) or ["*nothing*"])]

    camps = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not camps:
        raise ValueError(f"{root} has no campaign subdirectories")
    jobs, plan = [], {}
    for c in camps:
        if any(fnmatch.fnmatch(c, g) for g in excl_globs):
            plan[c] = "EXCLUDED"
            continue
        runs = sorted(d for d in os.listdir(os.path.join(root, c))
                      if os.path.isdir(os.path.join(root, c, d)))
        if max_per:
            runs = runs[:max_per]
        if not runs:
            plan[c] = "EXCLUDED (no runs)"
            continue
        plan[c] = f"eval_{_split_suffix(c)}" if any(fnmatch.fnmatch(c, g) for g in eval_globs) else "train/val"
        for rd in runs:
            jobs.append({"campaign": c, "dir": os.path.join(root, c, rd), "hz": hz, "hw": hw})

    print(f"[starling_bags] {root}: {len(jobs)} runs across {len(camps)} campaigns "
          f"@ {hz} Hz, {hw[0]}x{hw[1]}", flush=True)
    for c in camps:
        print(f"[starling_bags]   {c:32s} -> {plan[c]}", flush=True)

    by_camp: dict[str, list[Episode]] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for k, (camp, bdir, ep) in enumerate(ex.map(_bag_one, jobs), 1):
            by_camp.setdefault(camp, []).append(ep)
            print(f"[starling_bags] {k}/{len(jobs)} {camp}/{os.path.basename(bdir)} "
                  f"-> {len(ep.states)} steps  ({time.time() - t0:.0f}s)", flush=True)

    main_pool, extra = [], {}
    for c in camps:
        eps = by_camp.get(c)
        if not eps:
            continue
        if plan[c] == "train/val":
            main_pool += eps
        else:
            extra[plan[c]] = eps
    if not main_pool:
        raise ValueError(f"no train/val campaigns matched in {root} (eval_globs={eval_globs})")
    return name, main_pool, int(round(hz)), "ego", (extra or None)


def _longest_first_val(lengths: list[int], frac: float, min_eps: int = 2) -> set[int]:
    """The LONGEST episodes, as the prefix whose frame share lands CLOSEST to `frac`.

    WHY LONGEST-FIRST AND NOT RANDOM. Open-loop rollout evaluation can only run as far as the
    SHORTEST validation episode -- past that there is no ground truth left to score against. On
    the lego corpus a random seed-0 draw happened to pull a short episode into val and capped
    every long-horizon number at a fifth of the horizon the data actually supported. Putting the
    long trajectories in val costs ~the same frames either way and buys the horizon back directly.

    WHY "CLOSEST" AND NOT "UNTIL WE CROSS". Accumulating until the target is exceeded always
    overshoots, and overshoots badly when the longest episodes are much longer than the rest --
    on block-stack it turned a 10% request into 16.8%. Choosing the nearest prefix instead can land
    either side of the target, which is the honest reading of "about 10%".

    WHY A FLOOR OF TWO. A one-episode val set has no across-session variance at all: one
    recording's lighting, object layout and operator mood become the entire validation signal.
    Two is the minimum that can disagree with itself.

    Val will be dominated by whichever campaign recorded long -- for block-stack that is
    campaign5-play-long, and the operator has accepted that. It means val measures long-horizon
    fidelity rather than being a representative i.i.d. sample; read val loss accordingly.
    """
    n = len(lengths)
    if n <= min_eps:
        return {int(np.argmax(lengths))} if n else set()
    total = sum(lengths)
    order = sorted(range(n), key=lambda i: (-lengths[i], i))
    cum, best, best_err = 0, min_eps, None
    for k in range(1, n):                      # k = prefix size; never all n (train must survive)
        cum += lengths[order[k - 1]]
        if k < min_eps:
            continue
        err = abs(cum / total - frac)
        if best_err is None or err < best_err:
            best, best_err = k, err
    return {order[i] for i in range(best)}


def _block_stack_one(job: dict):
    """One block-stack run directory -> Episode. Module-level so ProcessPoolExecutor can pickle it."""
    from .block_stack import read_run
    st, ac, fr, info = read_run(job["dir"], target_hz=job["hz"], out_hw=tuple(job["hw"]),
                                cameras=tuple(job["cams"]))
    if len(job["cams"]) == 1:
        fr = fr[job["cams"][0]]
    return job["campaign"], job["dir"], Episode(states=st, actions=ac, frames=fr,
                                                task=job["campaign"]), info


def block_stack(cfg) -> tuple[str, list[Episode], int, str, dict[str, list[Episode]] | None, dict]:
    """Swoosh right-arm block-stacking teleop -> canonical Episodes. See data/block_stack.py.

    LAYOUT EXPECTED: <dir>/<campaign>/recording_YYYY_MM_DD_HH_MM_SS/{run.json,raw/,video/}.
    One Episode per run, `Episode.task` = the campaign directory name so the condition survives
    into <split>/meta/tasks.parquet.

    SPLIT POLICY for the block-stack corpus, as specified by the operator:
      * campaigns 1-2, and the non-data folders, are EXCLUDED -- 1-2 are bring-up, `shakedown` and
        `audit` are hardware checks, and `_rt*` are stray test artefacts from the collection repo's
        roundtrip test writing into `campaigns/` and being swept into the Drive sync.
      * campaigns 3-7 form the train/val pool, split 90/10 BY FRAMES with the LONGEST trajectories
        reserved for val (see _longest_first_val for why).
      * campaigns 8-9 are held out entirely as `eval_<suffix>` splits, structured identically.

    ACTION = THE XBOX CONTROLLER, five axes. Not the commanded pose and not the SDK arguments --
    those are consequences of the action plus the integrator's state, and conditioning on them
    would hand the model the answer. Both remain in the raw run directories.

    Args: +source.dir=<tree of campaign dirs> [+source.name=block_stack]
          [+source.target_hz=30] [+source.hw=[144,192]] [+source.workers=4]
          [+source.cameras=[scene_left,scene_right,gripper_right_bottom,gripper_right_top]]
          [+source.val_frac=0.1]
          [+source.eval_globs=[campaign8*,campaign9*]]
          [+source.exclude_globs=[campaign1*,campaign2*,_rt*,shakedown,audit]]
          [+source.max_runs_per_campaign=N]   <- smoke-test escape hatch

    WORKERS DEFAULTS TO 4, NOT 16. Each worker holds every decoded camera for a whole episode in
    memory; the longest run here is 639 s, which at 30 Hz and 144x192 is 1.6 GB per camera, so
    four cameras on sixteen workers would ask for ~100 GB. Raise it only alongside fewer cameras
    or a smaller hw.
    """
    import fnmatch
    from concurrent.futures import ProcessPoolExecutor
    from .block_stack import (ALL_CAMERAS, OUT_HW_DEFAULT, STATE_KEYS, TARGET_HZ_DEFAULT,
                              run_seconds)

    sc = cfg.get("source", None)
    if sc is None or not sc.get("dir"):
        raise ValueError("pass +source.dir=<tree containing campaign*/recording_*/ dirs> "
                         "(+source.name=block_stack)")
    root = os.path.expanduser(str(sc.dir))
    name = str(sc.get("name", "block_stack"))
    hz = float(sc.get("target_hz", TARGET_HZ_DEFAULT))
    hw = tuple(int(x) for x in (sc.get("hw", None) or OUT_HW_DEFAULT))
    workers = int(sc.get("workers", 4))
    max_per = int(sc.get("max_runs_per_campaign", 0) or 0)
    val_frac = float(sc.get("val_frac", VAL_FRAC))
    cams = [str(c) for c in (sc.get("cameras", None) or ALL_CAMERAS)]
    eval_globs = [str(g) for g in (sc.get("eval_globs", None) or ["campaign8*", "campaign9*"])]
    excl_globs = [str(g) for g in (sc.get("exclude_globs", None) or
                                   ["campaign1*", "campaign2*", "_rt*", "shakedown", "audit"])]
    bad = [c for c in cams if c not in ALL_CAMERAS]
    if bad:
        raise ValueError(f"unknown camera(s) {bad}; known: {list(ALL_CAMERAS)}")

    camps = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not camps:
        raise ValueError(f"{root} has no campaign subdirectories")
    jobs, plan = [], {}
    for c in camps:
        if any(fnmatch.fnmatch(c, g) for g in excl_globs):
            plan[c] = "EXCLUDED"
            continue
        runs = sorted(d for d in os.listdir(os.path.join(root, c))
                      if os.path.isfile(os.path.join(root, c, d, "run.json")))
        if max_per:
            runs = runs[:max_per]
        if not runs:
            plan[c] = "EXCLUDED (no runs)"
            continue
        plan[c] = f"eval_{_split_suffix(c)}" if any(fnmatch.fnmatch(c, g) for g in eval_globs) else "train/val"
        for rd in runs:
            jobs.append({"campaign": c, "dir": os.path.join(root, c, rd), "hz": hz, "hw": hw,
                         "cams": cams})

    print(f"[block_stack] {root}: {len(jobs)} runs across {len(camps)} campaigns @ {hz} Hz, "
          f"{hw[0]}x{hw[1]}, cameras {cams}", flush=True)
    for c in camps:
        print(f"[block_stack]   {c:26s} -> {plan[c]}", flush=True)
    if not jobs:
        raise ValueError(f"no runs survived the exclude globs in {root}")

    by_camp: dict[str, list[Episode]] = {}
    worst_sync, t0 = 0.0, time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for k, (camp, rdir, ep, info) in enumerate(ex.map(_block_stack_one, jobs), 1):
            by_camp.setdefault(camp, []).append(ep)
            se = info["sync_error_s"]
            w = max(v["max"] for v in se.values())
            over = max(v["over"] for v in se.values())
            worst_sync = max(worst_sync, w)
            print(f"[block_stack] {k}/{len(jobs)} {camp}/{info['run']} -> {info['steps']} steps "
                  f"({info['seconds']:.0f}s)  sync max {w * 1000:.0f}ms, {100 * over:.2f}% past bound"
                  f"{'' if info['validation_all_green'] else '  [!] run.json checks NOT all green'}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    # 16.7 ms is the half-grid bound at 30 Hz -- the best a nearest-neighbour resample can do.
    print(f"[block_stack] worst single-step nearest-neighbour displacement across all runs: "
          f"{worst_sync * 1000:.1f} ms (a stream hiccup shows up here; the per-run "
          f"'past bound' percentage is what indicates a systematic problem)", flush=True)

    main_pool, extra = [], {}
    for c in camps:
        eps = by_camp.get(c)
        if not eps:
            continue
        if plan[c] == "train/val":
            main_pool += eps
        else:
            extra[plan[c]] = eps
    if not main_pool:
        raise ValueError(f"no train/val campaigns matched in {root} (eval_globs={eval_globs})")

    # 90/10 by FRAMES, longest first. Sorting by run_seconds would double-read the timestamps;
    # the episodes are already built, so their true step counts are the exact thing to rank on.
    lens = [len(e.states) for e in main_pool]
    val_ids = _longest_first_val(lens, val_frac)
    v = sorted((lens[i], main_pool[i].task) for i in val_ids)[::-1]
    print(f"[block_stack] val = {len(val_ids)}/{len(main_pool)} episodes, "
          f"{sum(lens[i] for i in val_ids)}/{sum(lens)} frames "
          f"({100 * sum(lens[i] for i in val_ids) / sum(lens):.1f}%), longest first:", flush=True)
    for n, task in v:
        print(f"[block_stack]   {n:6d} steps  {task}", flush=True)

    # dims 3:9 are the 6D rotation: bounded, mutually constrained, and NOT to be z-scored.
    rot = list(range(STATE_KEYS.index("ee_rot6_0"), STATE_KEYS.index("ee_rot6_5") + 1))
    print(f"[block_stack] leaving obs dims {rot} (the 6D rotation) unnormalized -- per-dim "
          f"z-scoring would break |c|=1 and c0.c1=0", flush=True)
    return (name, main_pool, int(round(hz)), (cams if len(cams) > 1 else cams[0]),
            (extra or None), {"val_ids": val_ids, "obs_identity": rot})


def _split_suffix(campaign: str) -> str:
    """`campaign21-ood-noodle` -> `ood_noodle`: drop the campaignNN prefix, dashes to underscores."""
    tail = campaign.split("-", 1)[1] if "-" in campaign else campaign
    return tail.replace("-", "_")


PROCESSORS = {"starling": starling, "robocasa": robocasa, "starling_bags": starling_bags,
              "block_stack": block_stack}


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg):
    proc = cfg.get("processor", None)
    if proc is None or str(proc) not in PROCESSORS:
        raise ValueError(f"pass +processor=<{'|'.join(PROCESSORS)}> (got {proc!r})")
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    # Tee the PROCESSOR's own stdout into progress.log too. Which campaigns were excluded, which
    # episodes went to val and why is the part of the record you actually want six months later,
    # and it was previously terminal-only -- it never reached the dataset folder at all.
    class _Tee(io.TextIOBase):
        def __init__(self, real):
            self.real, self.buf = real, ""

        def write(self, t):
            self.real.write(t)
            self.buf += t
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                log_lines.append(line)
            return len(t)

        def flush(self):
            self.real.flush()

    with contextlib.redirect_stdout(_Tee(sys.stdout)):
        out = PROCESSORS[str(proc)](cfg)
    # 6th element is OPTIONAL: extra kwargs for build_recorded_dataset (block_stack passes its
    # longest-first `val_ids` and the rotation dims to leave unnormalized). Processors that
    # return five keep the historic random split and plain per-dim z-scoring.
    name, episodes, fps, cam, extra_splits = out[:5]
    build_opts = out[5] if len(out) > 5 else {}

    run_dir = build_recorded_dataset(name, episodes, fps, cam, log=log, extra_splits=extra_splits,
                                     **build_opts)
    with open(os.path.join(run_dir, "progress.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
