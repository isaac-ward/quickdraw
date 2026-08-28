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


def _xtcav_san(pv: str) -> str:
    return pv.replace(":", "_").replace(".", "_")


XTCAV_RUNS = ("E300_15671", "E300_15673", "TEST_15668")
XTCAV_BSA_LISTS = ("BSA_List_S10", "BSA_List_S10RF", "BSA_List_S11", "BSA_List_S11RF",
                   "BSA_List_S14", "BSA_List_S20")
XTCAV_DROP_PVS = {"PMT:LI20:3350:QDCRAW", "PMT:LI20:3360:QDCRAW"}  # ragged in E300_15673 (150 rows short)
XTCAV_NAN_DROP = 0.10    # nan_frac <= this -> plain channel (ffill). See XTCAV_MASK_MAX for the band above.
XTCAV_MASK_MAX = 0.90    # nan_frac in (NAN_DROP, MASK_MAX] -> kept as value + validity-mask channel (value
#                          imputed to the head-shot mean = z 0 where invalid; ffill would teach false
#                          freshness once a mask exists). > MASK_MAX -> dropped (dead). v3, critic-reviewed.
XTCAV_SETTLE_MAX = 15    # max leading shots per scan step droppable by the TCAV settle criterion
# --- v5 (record §8.1/§8.2/§8.4): settled-only supervision + S-conditioned actions + obs de-echo ---
XTCAV_OBS_ECHO_PVS = {"TCAV:LI20:2400:A", "TCAV:LI20:2400:P",  # bit-exact copies of action dims (audit §8.1)
                      "WIRE:LI20:3179:POSN"}                   # binary wire park state, zero in-run variance
XTCAV_DEV_MAX = 1.0      # keep TCAV-on shots only when ||phase|-90| < this (deg). §8.2: dev>=1 flags the
#                          re-lock transient with ~97% recall (suppression persists ~14 shots/flip while
#                          dev decays; dev is a LOCK-STATE symptom, not cos(phase) physics).
XTCAV_STREAK_MIN = 0.6   # image streak-consistency: keep TCAV-on shots only when the baseline-subtracted
#                          streak-axis RMS extent >= this x the settled median at the same (sign, step)
#                          (fallback: run median). Catches the ~83 stealth suppressed shots at dev<1.
XTCAV_OFF_AMP = 5.0      # amp below this = TCAV genuinely off -> kept as S=0 anchors (beam present only)
XTCAV_BEAM_MIN_FRAC = 0.2  # image-signal beam gate: cleaned-signal sum >= this x settled median. TMIT is
#                            blind to downstream scraping (audit §8.1) — the gate must be image-side.
XTCAV_FLOOR_ZERO_U8 = 2  # v6 (record §8.16): zero STORED-crop pixels below this. The ~1 u8 camera noise
#                          floor is sub-threshold for every gate and the extractor (all use u8>=5) yet
#                          carries ~35% of total pixel mass — it is the background the mse decoder provably
#                          hedges into block artifacts (§8.14B), and zeroing it reclaims that recon mass
#                          for the beam. Gate: measured-curve byte-invariance vs the v5 conversion.
XTCAV_CROP = (128, 384)  # v4 (2026-08-22): COM-centered crop (rows, cols) of the oriented 184x894 frame,
#                          yiheng preprocess_shot style. Measured on the strict-gated corpus: charge kept
#                          median 100%, p1 94.6%, frames losing >5% = 1.2%, and corr(kept, L2) = -0.01 —
#                          i.e. NO setpoint-correlated clipping (320/256-wide windows fail that test at
#                          +0.12/+0.29; 96 rows doubles the >1%-loss fraction). COM computed on a CLEANED
#                          copy (median-3 + uint8-5 threshold) for stability; crop cut from the RAW frame,
#                          zero-padded at edges. The removed position becomes two obs channels (um).
XTCAV_COM_NOISE_U8 = 5   # cleaning threshold for the COM copy (~ Tier-0's 35 raw counts at 2000/255)
XTCAV_PX_UM = 30.5       # native DTOTR2 pixel pitch (um/px, both axes) — converts COM px -> um channels
XTCAV_SCALE = 2000.0     # bg-subtracted counts -> full uint8. Reviewed 2026-08-18 over 2250 frames: p99 of
#                          per-frame maxima ~2210, camera saturates at 4095 (12-bit); ~1.7% of frames peak
#                          above 2000 (pixel-level clip fraction 3e-7). Raise to ~2500 at the next re-convert
#                          if beam-core structure of the brightest shots ever matters.
XTCAV_BLOCK_STEPS = 6    # scan steps per train/val episode (v3: 3 -> 6; captures ~22 of 26 in-run scan-step
#                          boundaries inside episodes instead of 18 — boundary transitions are the response
#                          signal the run-3 training enriches on)
XTCAV_TAIL_STEPS = 6     # last N scan steps of each run -> one long held-out `eval` episode


def _xtcav_fill(a: np.ndarray) -> np.ndarray:
    """Forward- then backward-fill NaNs along time, per channel (BSA readbacks; probed NaNs are sparse)."""
    for sl in (slice(None), slice(None, None, -1)):
        v = a[sl]
        idx = np.where(np.isfinite(v), np.arange(len(v))[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        a[sl] = v[idx, np.arange(a.shape[1])[None, :]]
    assert np.isfinite(a).all(), "NaNs survived fill (a channel is all-NaN in one run?)"
    return a


def _xtcav_parse_run(run_dir: str):
    """One FACET-II MATLAB-DAQ run -> time-ordered MATCHED-shot arrays. The scalar/frame join uses the
    .mat's 1-based maps (scalars.common_index -> scalar rows, images.DTOTR2.common_index -> frame rows);
    the frame row INSIDE each per-step h5 is found by NDArrayUniqueId == PID, never by position (frames
    get dropped). Frames come out background-subtracted, oriented (energy rows x streak cols), uint8."""
    import h5py
    from scipy.io import loadmat

    run = os.path.basename(os.path.normpath(run_dir))
    ds = loadmat(os.path.join(run_dir, f"{run}.mat"), simplify_cells=True)["data_struct"]

    pvs = [str(pv) for lst in XTCAV_BSA_LISTS for pv in ds["metadata"][lst]["PVs"]
           if str(pv) not in XTCAV_DROP_PVS]
    ci = np.asarray(ds["scalars"]["common_index"], dtype=int) - 1
    steps = np.asarray(ds["pulseID"]["steps"], dtype=int)[ci]
    t_slac = np.asarray(ds["pulseID"]["SLAC_time"], dtype=np.float64)[ci]   # v5: per-shot wall time (s)
    assert np.all(np.diff(steps) >= 0), f"{run}: matched shots not in acquisition order"
    scal = np.stack([np.asarray(ds["scalars"][lst][_xtcav_san(str(pv))], dtype=np.float64)[ci]
                     for lst in XTCAV_BSA_LISTS for pv in ds["metadata"][lst]["PVs"]
                     if str(pv) not in XTCAV_DROP_PVS], axis=1)          # (N, C) raw, NaNs kept

    scan_vals = np.atleast_1d(np.asarray(ds["params"]["scanVals"], dtype=np.float64))
    knobs = np.stack([scan_vals[steps - 1],                              # commanded L2 phase (deg)
                      scal[:, pvs.index("TCAV:LI20:2400:P")],
                      scal[:, pvs.index("TCAV:LI20:2400:A")]], axis=1)

    md = ds["metadata"]["DTOTR2"]
    assert str(md["X_ORIENT"]) == "Positive" and str(md["Y_ORIENT"]) == "Positive" and int(md["IS_ROTATED"]) == 1
    bg = np.asarray(ds["backgrounds"]["DTOTR2"], dtype=np.float32).T     # .mat stores transposed vs the h5
    im = ds["images"]["DTOTR2"]
    fidx = np.asarray(im["common_index"], dtype=int) - 1
    fpid = np.asarray(im["pid"], dtype=int)[fidx]
    fstep = np.asarray(im["step"], dtype=int)[fidx]
    locs = [im["loc"]] if isinstance(im["loc"], str) else [str(p) for p in im["loc"]]
    frames = np.empty((len(ci), bg.shape[1], bg.shape[0]), dtype=np.uint8)   # oriented (N, SizeX, SizeY)
    for s in np.unique(fstep):
        path = os.path.join(run_dir, "images", "DTOTR2", os.path.basename(locs[s - 1]))
        sel = np.flatnonzero(fstep == s)
        with h5py.File(path, "r") as h:
            uid = np.asarray(h["entry/instrument/NDAttributes/NDArrayUniqueId"])
            data = np.asarray(h["entry/data/data"])
        rows = []
        for p in fpid[sel]:
            hits = np.flatnonzero(uid == p)
            if len(hits) > 1:   # reviewed: 4 duplicate PIDs corpus-wide; frames differ, first is arbitrary
                print(f"[xtcav] {run} step {s}: PID {p} x{len(hits)} in {os.path.basename(path)}; taking first")
            rows.append(hits[0])
        rows = np.asarray(rows)
        sub = np.clip(data[rows].astype(np.float32) - bg[None], 0.0, XTCAV_SCALE)
        frames[sel] = (sub.transpose(0, 2, 1) * (255.0 / XTCAV_SCALE) + 0.5).astype(np.uint8)  # IS_ROTATED

    # v5 (record §8.1/§8.2): SETTLED-ONLY supervision. One cleaning pass over every matched frame yields
    # per-shot image stats (COM, cleaned-signal sum, streak-axis RMS extent); the keep rule is then
    # image-verified: TCAV-on shots need locked phase (dev<1 deg) AND a settled-consistent streak AND
    # beam on screen; true-off shots (amp<5) with beam are kept as S=0 anchors. Everything else — the
    # ~14-shot re-lock transients after polarity flips (78% streak-suppressed while dev decays), the
    # phase-slew ramps whose S labels do not match their images, blank/scraped frames — is excised.
    # Mid-step drops create unmarked splices — surfaced to the model via the dt obs channel.
    from scipy.ndimage import center_of_mass, median_filter
    ph, amp = knobs[:, 1], knobs[:, 2]
    dev = np.abs(np.abs(ph) - 90.0)
    mean_on = float(np.median(amp[dev < 1.0]))
    N = len(ci)
    sig = np.empty(N); ext = np.empty(N)
    coms_all = np.empty((N, 2))
    cleaned_all_note = "COM/stats on cleaned copy (median-3, u8>=5, 16-px border zeroed); crop from raw"
    for i in range(N):
        f = frames[i]
        cleaned = median_filter(f.astype(np.float32), size=3)
        cleaned = np.where(cleaned >= XTCAV_COM_NOISE_U8, cleaned, 0.0)
        cleaned[:16, :] = 0.0; cleaned[-16:, :] = 0.0    # border flashes bias COM (verifier 2026-08-25)
        cleaned[:, :16] = 0.0; cleaned[:, -16:] = 0.0
        sig[i] = cleaned.sum(dtype=np.float64)
        p = cleaned.sum(0)                                # streak-axis (cols) projection of cleaned signal
        if p.sum() > 1e-6:
            x = np.arange(len(p), dtype=np.float64)
            c = (p * x).sum() / p.sum()
            ext[i] = np.sqrt((p * (x - c) ** 2).sum() / p.sum())
        else:
            ext[i] = 0.0
        if sig[i] > 0:
            coms_all[i] = center_of_mass(cleaned)         # (cy, cx)
        else:                                             # near-empty frame: geometric center fallback
            coms_all[i] = ((f.shape[0] - 1) / 2.0, (f.shape[1] - 1) / 2.0)

    amp_on = np.abs(amp - mean_on) < 0.5
    settled = amp_on & (dev < XTCAV_DEV_MAX)
    beam = sig >= XTCAV_BEAM_MIN_FRAC * np.median(sig[settled]) if settled.any() else sig > 0
    # settled streak reference per (sign, step); fallback = run-level settled median
    ref = np.full(N, np.median(ext[settled & beam]))
    sgn = np.sign(ph)
    for s in np.unique(steps):
        for g in (-1.0, 1.0):
            m = (steps == s) & (sgn == g)
            base = m & settled & beam
            if base.sum() >= 8:
                ref[m] = np.median(ext[base])
    streak_ok = ext >= XTCAV_STREAK_MIN * ref
    on_ok = settled & beam & streak_ok
    off_ok = (amp < XTCAV_OFF_AMP) & beam
    keep = on_ok | off_ok
    n = {"kept_on": int(on_ok.sum()), "kept_off_anchor": int(off_ok.sum()),
         "drop_dev": int((amp_on & (dev >= XTCAV_DEV_MAX)).sum()),
         "drop_suppressed": int((settled & beam & ~streak_ok).sum()),
         "drop_nobeam": int((~beam).sum()),
         "drop_ramp_other": int((~keep & ~(amp_on & (dev >= XTCAV_DEV_MAX))
                                 & ~(settled & beam & ~streak_ok) & beam).sum())}
    print(f"[xtcav] {run}: v5 gate {n} of {N}", flush=True)
    k = np.flatnonzero(keep)

    # v5 action vector [L2 setpoint, S, dev, amp]: S = amp*sin(phase) signed streak strength; dev =
    # |phase|-90 signed settling coordinate. (S, dev, amp) <-> (phase, amp) is lossless for amp>0;
    # on true-off shots the phase readback wanders (audit §8.1 finding 8) so S and dev are masked to 0.
    S = amp * np.sin(np.deg2rad(ph))
    dsg = np.abs(ph) - 90.0
    off_m = amp < XTCAV_OFF_AMP
    S[off_m] = 0.0
    dsg[off_m] = 0.0
    acts = np.stack([knobs[:, 0], S, dsg, amp], axis=1)

    # v4 crop (unchanged geometry): COM-centered window cut from the RAW frame, zero-padded at edges.
    Hc, Wc = XTCAV_CROP
    crops = np.zeros((len(k), Hc, Wc), dtype=np.uint8)
    coms = np.empty((len(k), 2), dtype=np.float64)
    kept_frac = np.empty(len(k))
    for j, i in enumerate(k):
        f = frames[i]
        cy, cx = coms_all[i]
        r0, c0 = int(round(cy)) - Hc // 2, int(round(cx)) - Wc // 2
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + Hc, f.shape[0]), min(c0 + Wc, f.shape[1])
        crops[j, rs - r0:rs - r0 + (re - rs), cs - c0:cs - c0 + (ce - cs)] = f[rs:re, cs:ce]
        crops[j][crops[j] < XTCAV_FLOOR_ZERO_U8] = 0          # v6 camera-floor zeroing (see constant)
        coms[j] = (cx * XTCAV_PX_UM, cy * XTCAV_PX_UM)              # (streak-x um, energy-y um)
        c_sig = np.where(crops[j] >= XTCAV_COM_NOISE_U8, crops[j], 0).sum(dtype=np.float64)
        f_sig = np.where(f >= XTCAV_COM_NOISE_U8, f, 0).sum(dtype=np.float64)
        kept_frac[j] = c_sig / max(f_sig, 1.0)
    print(f"[xtcav] {run}: crop {Hc}x{Wc} BEAM charge kept median {np.median(kept_frac):.2%} "
          f"p1 {np.percentile(kept_frac, 1):.1%} frames>1%loss {np.mean(kept_frac < 0.99):.1%}", flush=True)

    # v5 dt channel: log1p seconds since the previous KEPT shot in this run — surfaces the splices the
    # gate creates (and the 6-9 s scan-step pauses). First kept shot gets the nominal 10 Hz spacing.
    tk = t_slac[k]
    dt = np.concatenate([[0.1], np.diff(tk)])
    assert (dt > 0).all(), f"{run}: non-increasing SLAC_time on kept shots"
    dt_log = np.log1p(dt).astype(np.float32)

    drops = dict(n, crop_kept_median=float(np.median(kept_frac)),
                 crop_kept_p1=float(np.percentile(kept_frac, 1)), com_note=cleaned_all_note)
    return pvs, steps[k], scal[k], acts[k], crops, len(scan_vals), drops, coms, dt_log


def xtcav(cfg) -> tuple[str, list[Episode], int, str, dict[str, list[Episode]] | None]:
    """FACET-II MATLAB-DAQ XTCAV L2-phase scans (`<dir>/<RUN>/<RUN>.mat` + `images/DTOTR2/*_stepNN.h5`).

    Obs = the 6 BSA scalar lists (136 channels, probed identical across the E300/TEST runs) minus
    dead/ragged channels; NaNs forward-filled per run. Obs are PRE-STANDARDIZED to z-scores using
    head-shot (non-tail) statistics — the raw channels mix TMIT charge counts (~1e9) with phases in
    degrees, and the recorded env's checkpoint metric `pointwise_error` is an L2 over DENORMALIZED
    obs, so raw units would make best.ckpt select on TMIT error alone. The physical per-channel
    mean/std live in the channels json (obs_physical = z * std + mean). Action[t] = [L2 phase
    setpoint, S = amp*sin(phase), dev = |phase|-90, TCAV amp] AT SHOT t+1 (v5, record §8; quickdraw
    semantics: action[t] produces obs[t+1]; final row duplicated, padded/unused; S and dev masked to
    0 on TCAV-off shots); actions stay in PHYSICAL units (degrees/MV) — the loader
    z-scores them and symlog handles the near-constant amp. Frames = DTOTR2, background-subtracted, oriented (energy rows x streak/time cols),
    fixed linear scale -> uint8, grey replicated x3. Episodes = blocks of XTCAV_BLOCK_STEPS scan steps;
    the last XTCAV_TAIL_STEPS of each run become one long held-out `eval` episode (extra split, excluded
    from train/val and norm stats). The kept/dropped channel table + constants are written to
    `<log root>/xtcav_channels_<name>.json` (copy it into the run_dir).

    Args: +source.dir=<XTCAV_analysis root> [+source.name=xtcav_e300] [+source.runs=A,B,...]
          [+source.max_steps=N (smoke: first N scan steps per run, no tail split)] [+source.fps=10]"""
    src_cfg = cfg.get("source", None)
    if src_cfg is None or not src_cfg.get("dir"):
        raise ValueError("pass +source.dir=<XTCAV_analysis root> "
                         "(optional: +source.name=..., +source.runs=A,B, +source.max_steps=N, +source.fps=10)")
    src = os.path.expanduser(str(src_cfg.dir))
    name = str(src_cfg.get("name", "xtcav_e300"))
    runs = [r.strip() for r in str(src_cfg.get("runs", ",".join(XTCAV_RUNS))).split(",") if r.strip()]
    max_steps = int(src_cfg.get("max_steps", 0) or 0)
    fps = int(src_cfg.get("fps", 10))
    cam = "dtotr2"

    raw, run_drops, run_coms, run_dts = [], {}, [], []
    for run in runs:
        p, steps, scal, acts, frames, n_steps, dr, coms, dt_log = _xtcav_parse_run(os.path.join(src, run))
        raw.append((run, p, steps, scal, acts, frames, n_steps))
        run_drops[run] = dr
        run_coms.append(coms)
        run_dts.append(dt_log)

    # v5: obs channel set = the ORDERED INTERSECTION of the runs' BSA lists (E331 shifts may differ,
    # record §8.4), minus the action-echo/dead PVs (§8.1/§8.3: TCAV A/P are bit-exact copies of action
    # dims — keeping them hollows out CFG's unconditional branch; WIRE POSN is a binary park state).
    pvs = [pv for pv in raw[0][1] if all(pv in p for _, p, *_ in raw) and pv not in XTCAV_OBS_ECHO_PVS]
    for run, p, *_ in raw:
        missing = [pv for pv in p if pv not in pvs and pv not in XTCAV_OBS_ECHO_PVS]
        if missing:
            print(f"[xtcav] {run}: {len(missing)} PVs outside the cross-run intersection dropped "
                  f"(e.g. {missing[:3]})", flush=True)
    parsed = [(run, steps, scal[:, [p.index(pv) for pv in pvs]], acts, frames, n_steps)
              for run, p, steps, scal, acts, frames, n_steps in raw]

    # corpus-wide channel policy (v3): <=NAN_DROP plain (ffill); (NAN_DROP, MASK_MAX] value+mask;
    # >MASK_MAX dead. Then zero-variance filter on the plain block.
    pooled = np.concatenate([p[2] for p in parsed])
    nan_frac = np.mean(~np.isfinite(pooled), axis=0)
    keepv = nan_frac <= XTCAV_NAN_DROP
    maskb = (nan_frac > XTCAV_NAN_DROP) & (nan_frac <= XTCAV_MASK_MAX)
    filled = [_xtcav_fill(p[2][:, keepv].astype(np.float32).copy()) for p in parsed]
    live = np.concatenate(filled).std(axis=0) > 0
    filled = [st[:, live] for st in filled]
    mvals = [p[2][:, maskb].astype(np.float32) for p in parsed]          # NaNs kept until standardization
    masks = [np.isfinite(v).astype(np.float32) for v in mvals]
    kept_pvs = [pv for pv, k in zip(pvs, keepv) if k]
    kept_pvs = [pv for pv, l in zip(kept_pvs, live) if l]
    mask_pvs = [pv for pv, mflag in zip(pvs, maskb) if mflag]
    dropped = {pv: f"nan_frac {nan_frac[i]:.3f}" for i, (pv, k) in enumerate(zip(pvs, keepv | maskb)) if not k}
    dropped |= {pv: "zero variance" for pv in set(pvs) - set(kept_pvs) - set(mask_pvs) - set(dropped)}
    dropped |= {pv: "ragged length" for pv in XTCAV_DROP_PVS}
    obs_names = (kept_pvs + [f"{pv} [masked value]" for pv in mask_pvs]
                 + [f"{pv} [validity]" for pv in mask_pvs]
                 + ["DT_LOG1P_S [log1p seconds since previous kept shot]",
                    "DTOTR2_COM_X_um [crop center, streak axis]", "DTOTR2_COM_Y_um [crop center, energy axis]"])

    # pre-standardize on HEAD-shot (non-tail) VALID entries; invalid masked values land at z=0 (= mean).
    # v4: the per-shot crop COM (um) joins the obs — centering moves the positional jitter OUT of the
    # image and INTO these two channels, where the proprio pathway predicts it.
    # v5: the dt channel joins too — splices and scan-step pauses stop being invisible (record §8.1).
    states_per_run = [np.concatenate([f_, v, m, dtc[:, None], cm.astype(np.float32)], axis=1)
                      for f_, v, m, dtc, cm in zip(filled, mvals, masks, run_dts, run_coms)]
    head = np.concatenate([st[p[1] <= (min(p[5], max_steps) if max_steps else p[5] - XTCAV_TAIL_STEPS)]
                           for p, st in zip(parsed, states_per_run)])
    obs_mean, obs_std = np.nanmean(head, axis=0), np.nanstd(head, axis=0)
    assert (obs_std > 0).all(), "zero head-shot std on a surviving channel"
    states_per_run = [np.nan_to_num((st - obs_mean) / obs_std, nan=0.0).astype(np.float32)
                      for st in states_per_run]

    episodes, tails = [], []
    for (run, steps, _, knobs, frames, n_steps), st in zip(parsed, states_per_run):
        knobs = knobs.astype(np.float32)
        assert np.isfinite(knobs).all(), f"{run}: non-finite knob readbacks"

        def _ep(mask):
            idx = np.flatnonzero(mask)                       # contiguous (shots are step-ordered)
            nxt = min(idx[-1] + 1, len(knobs) - 1)           # TRUE next kept shot in the run (dup at run end)
            actions = knobs[np.append(idx[1:], nxt)]         # action[t] = knobs at shot t+1
            return Episode(states=st[idx], actions=actions,
                           frames=np.repeat(frames[idx][..., None], 3, axis=-1))

        head_end = min(n_steps, max_steps) if max_steps else n_steps - XTCAV_TAIL_STEPS
        for b0 in range(1, head_end + 1, XTCAV_BLOCK_STEPS):
            episodes.append(_ep((steps >= b0) & (steps <= min(b0 + XTCAV_BLOCK_STEPS - 1, head_end))))
        if not max_steps:
            tails.append(_ep(steps > head_end))

    log_root = os.environ.get("QUICKDRAW_LOG_ROOT", "logs")
    os.makedirs(log_root, exist_ok=True)
    with open(os.path.join(log_root, f"xtcav_channels_{name}.json"), "w") as f:
        json.dump({"runs": runs, "obs_channels": obs_names, "masked_channels": mask_pvs,
                   "dropped": dropped, "shot_drops": run_drops,
                   "obs_standardization": {"note": "stored obs are z-scores over head (non-tail) VALID "
                                                   "shots; physical = z * std + mean; masked values are "
                                                   "z=0 where their validity channel is 0",
                                           "mean": obs_mean.astype(float).tolist(),
                                           "std": obs_std.astype(float).tolist()},
                   "action": ["L2_PHASE.MKB setpoint (deg), shot t+1",
                              "S = amp*sin(phase) signed streak strength, shot t+1 (0 when TCAV off)",
                              "dev = |phase|-90 deg signed settling coordinate, shot t+1 (0 when off)",
                              "TCAV:LI20:2400:A amp, shot t+1"],
                   "action_note": "phase = sign(S)*(90+dev), amp = amp — lossless for amp>0 (record §8)",
                   "intensity_scale_counts": XTCAV_SCALE, "block_steps": XTCAV_BLOCK_STEPS,
                   "tail_steps": XTCAV_TAIL_STEPS, "settle_max": XTCAV_SETTLE_MAX,
                   "v5_gate": {"dev_max_deg": XTCAV_DEV_MAX, "streak_min_frac": XTCAV_STREAK_MIN,
                               "off_amp": XTCAV_OFF_AMP, "beam_min_frac": XTCAV_BEAM_MIN_FRAC,
                               "obs_echo_excluded": sorted(XTCAV_OBS_ECHO_PVS),
                               "floor_zero_u8": XTCAV_FLOOR_ZERO_U8},
                   "crop": {"hw_native": list(XTCAV_CROP), "px_um": XTCAV_PX_UM,
                            "note": "frames are COM-centered crops of the oriented native frame, "
                                    "zero-padded; absolute position = the DTOTR2_COM_* obs channels"},
                   "orientation": "energy rows x streak cols"}, f, indent=2)
    return name, episodes, fps, cam, ({"eval": tails} if tails else None)


PROCESSORS = {"starling": starling, "robocasa": robocasa, "xtcav": xtcav}


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
