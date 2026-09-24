"""Loading: windowed (train/val) and full-trajectory (eval) access + train-only normalization.

We store with lerobot (generate.py) but read episodes out and window them ourselves, so the
training loader does not depend on lerobot's `delta_timestamps` behavior. The lerobot read is
isolated in `load_split_episodes` (the one version-sensitive spot for loading).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


class Normalizer:
    """Train-only mean/std, applied to every split (so OOD shift stays real)."""

    def __init__(self, stats: dict):
        self.o_mean = torch.tensor(stats["observation_vector"]["mean"])
        self.o_std = torch.tensor(stats["observation_vector"]["std"])
        self.a_mean = torch.tensor(stats["action"]["mean"])
        self.a_std = torch.tensor(stats["action"]["std"])

    @classmethod
    def from_file(cls, root: str, *, obs_keep, obs_fields=("observation_vector",)) -> "Normalizer":
        # obs_keep is a REQUIRED keyword (2026-09-21, was a module global): the stats MUST be subset to the
        # same obs layout the loaders apply, or norm/denorm silently mismatch the data. Pass None for the full
        # vector, or DataConfig.obs_keep. Required-with-no-default so a caller cannot forget it and get the
        # wrong layout -- see DataConfig for why this stopped being ambient state.
        #
        # obs_fields (2026-09-24): the obs is a CONCATENATION of these stored fields, in order (default just
        # observation_vector -> bit-identical). Each field carries its OWN mean/std in normalization_stats.json;
        # we assemble them in the SAME order the loaders concat, then subset by obs_keep POST-CONCAT (obs_keep
        # indexes the FINAL concatenated vector, matching _read_obs_fields). See DataConfig.obs_fields.
        with open(os.path.join(root, "normalization_stats.json")) as f:
            stats = json.load(f)
        n = cls(stats)                                      # sets action + the default observation_vector stats
        means, stds = [], []
        for fld in obs_fields:
            if fld not in stats:
                raise KeyError(f"normalization_stats.json has no stats for obs field {fld!r} (obs_fields={list(obs_fields)}); "
                               f"available: {[k for k in stats if k != 'action']}")
            means.append(torch.tensor(stats[fld]["mean"])); stds.append(torch.tensor(stats[fld]["std"]))
        n.o_mean, n.o_std = torch.cat(means), torch.cat(stds)   # concat in obs_fields order (== observation_vector alone by default)
        return n.subset_obs(obs_keep)                          # obs_keep POST-concat

    def subset_obs(self, obs_keep):
        """Restrict the obs stats to the `obs_keep` index subset, so norm/denorm match the subset the loaders
        apply. Takes the indices EXPLICITLY (was a module global) -- None = full vector, no-op."""
        if obs_keep is not None:
            t = torch.as_tensor(list(obs_keep), dtype=torch.long)
            self.o_mean, self.o_std = self.o_mean[t], self.o_std[t]
        return self

    def tile_act(self, k: int):
        """Repeat the action stats k times, for `data.action_aggregate=concat` where one kept step carries k
        raw actions laid out time-major ([slot0 dims..., slot1 dims..., ...]). Each slot holds the RAW action
        distribution the stats were computed on, so tiling is exactly right -- and it is the reason concat has
        no normalization mismatch, unlike sum. No-op for k<=1."""
        if int(k) > 1:
            self.a_mean, self.a_std = self.a_mean.repeat(int(k)), self.a_std.repeat(int(k))
        return self

    def norm_obs(self, o):
        return (o - self.o_mean.to(o)) / self.o_std.to(o)

    def denorm_obs(self, o):
        return o * self.o_std.to(o) + self.o_mean.to(o)

    def norm_act(self, a):
        return (a - self.a_mean.to(a)) / self.a_std.to(a)

    def denorm_act(self, a):
        return a * self.a_std.to(a) + self.a_mean.to(a)


# ---- TEMPORAL SUBSAMPLING (data.subsample; 1 = OFF = bit-identical) -------------------------------
# WHY: robocasa is 20 Hz, and at 20 Hz the true per-step image change is 0.0389 RMSE against the frozen
# TAESD's own 0.0637 reconstruction RMSE -- the signal is 0.61x the NOISE FLOOR of the codec we predict
# through, with only 3.18% of pixels moving more than that error. Predicting zero motion is then the
# CORRECT solution to the objective, which is exactly what every run did (record §13). At stride 5 the
# per-step change is 1.36x the floor, the action's linear R2 on the observed state change rises 4.7x
# (0.0062 -> 0.0293) and the value of correct action TIMING rises 39x (0.0002 -> 0.0078). Every published
# robot world model that demonstrably controls long rollouts subsamples: V-JEPA-2-AC 4 fps with integrated
# EEF deltas (2506.09985), HMA 2 Hz (2502.04296), IRASim ~4 fps (2406.14540).
#
# NOT `window_stride` -- that is LOCKED at 1 and skips training-window STARTS (thinning coverage). This
# thins the FRAMES INSIDE every window, changing the physical timestep the dynamics models.
#
# Set ONCE per process, and applied INSIDE both episode loaders rather than threaded through their ~12
# call sites. That is deliberate: the catastrophic failure here is a MISSED call site leaving eval at
# 20 Hz while training runs at 4 Hz -- the metrics would be measuring a different problem and would look
# like the model failing. One source of truth makes that inconsistency impossible to write.
# HOW the actions of the skipped frames are folded into the kept step's action. `sum` is the historical
# behaviour and the ONLY correct rule for DELTA actions (robocasa's EEF/rotation deltas compose additively
# over the skipped frames, so the sum IS the net displacement). It is WRONG for ABSOLUTE commands: starling's
# `joy_axis_*` are stick POSITIONS, and a pilot's stick is so autocorrelated that summing s of them scales the
# std by essentially exactly s -- measured on starling-2, normalized |z| std 1.00 / 2.05 / 3.13 / 4.21 at
# stride 1/2/3/4, with excursions to 12.2 sigma, because normalization_stats.json is computed on the RAW
# actions at dataset-generation time and never sees the aggregation.
#   sum     net effect over the window. Delta/velocity actions.                          (historical default)
#   mean    average command over the window. Absolute commands; = sum/s, so it restores z std ~= 1.
#   last    the command in effect at the kept frame. Absolute commands, causal reading.
#   first   the command in effect when the kept transition STARTS.
#   concat  all s raw actions, kept as an s*action_dim vector. LOSSLESS -- no aggregation assumption at
#           all -- and it makes one strided step carry a genuine s-action chunk. Widens the action vector,
#           so training.setup.effective_action_dim derives model action_dim and Normalizer.tile_act tiles
#           the stats to match.
_AGGREGATES = ("sum", "mean", "last", "first", "concat")


@dataclass(frozen=True)
class DataConfig:
    """The four data-loading knobs, carried EXPLICITLY to every loader + the Normalizer. Build once with
    `DataConfig.from_cfg(cfg)` and thread it down.

    WHY THIS REPLACED MODULE GLOBALS (2026-09-21). subsample / action_aggregate / subsample_all_phases /
    obs_keep used to be process-level globals, each set once per entrypoint by a `set_*` call. That design
    has one fatal property: ambient state cannot make "you forgot to set me" a loud error. `eval_checkpoint`
    simply never called `set_subsample`, so it silently evaluated at the DEFAULT subsample=1 against a
    subsample-10 run -- a 10x-wrong rollout that produced a plausible wrong number instead of crashing. The
    fix is to make the config an explicit object that every loader REQUIRES as a keyword-only argument with
    NO default: a missed config is now an immediate TypeError, never a silent wrong-rate load. There is also
    no process state left to desync mid-run (the old globals needed a `_USED` guard for exactly that), because
    every loader reads the object it was handed. Do NOT reintroduce a module-level default for these.

    - subsample: frame stride. Keep every s-th frame; s=1 is no decimation.
    - action_aggregate: how the skipped frames' actions fold into the kept step (see the comment above).
    - subsample_all_phases: emit all s phase offsets as separate TRAIN episodes -- ~s x the windows at the
      SAME rate (uses the frames the decimation throws away). TRAIN ONLY (guarded on the '/train' tag in
      _subsample_episodes): adding phases to VAL would change which episodes eval samples and shift every
      metric, breaking comparability. So it is inert in an eval-only process even when set.
    - obs_keep: obs-dim indices to KEEP (None = full vector). Applied INSIDE the loaders AND the Normalizer
      stats POST-CONCAT (indexes the FINAL obs_fields-concatenated vector), so no call site can load a
      different obs layout than training saw.
    - obs_fields: ordered list of stored parquet features to READ and CONCATENATE into the obs vector. Default
      ("observation_vector",) -> bit-identical to before. e.g. ("observation_vector","observation_vector.cube")
      appends the 32-dim cube state. The loader concats in this order; the Normalizer assembles per-field stats
      in this SAME order; obs_keep then indexes the concatenated result. Keeps the loader dataset-agnostic --
      no field is hardcoded. See design + PROCESSING.md for the cube-state layout.
    """
    subsample: int = 1
    action_aggregate: str = "sum"
    subsample_all_phases: bool = False
    obs_keep: tuple | None = None
    obs_fields: tuple = ("observation_vector",)

    def __post_init__(self):
        if int(self.subsample) < 1:
            raise ValueError(f"data.subsample must be >= 1, got {self.subsample}")
        if self.action_aggregate not in _AGGREGATES:
            raise ValueError(f"data.action_aggregate must be one of {_AGGREGATES}, got {self.action_aggregate!r}")
        if not self.obs_fields:
            raise ValueError("data.obs_fields must be a non-empty list (default ['observation_vector'])")

    @classmethod
    def from_cfg(cls, cfg) -> "DataConfig":
        """Build from a resolved config's `data` group. THE one place that reads these keys off cfg, so the
        defaults live in exactly one spot instead of being re-typed (and mis-typed) at every entrypoint."""
        d = cfg.get("data", {}) or {}
        ok = d.get("obs_keep", None)
        of = d.get("obs_fields", None)
        return cls(subsample=int(d.get("subsample", 1) or 1),
                   action_aggregate=str(d.get("action_aggregate", "sum")),
                   subsample_all_phases=bool(d.get("subsample_all_phases", False)),
                   obs_keep=None if ok is None else tuple(int(i) for i in ok),
                   obs_fields=("observation_vector",) if of is None else tuple(str(f) for f in of))


def _apply_obs_keep(obs_all, obs_keep):
    """Slice loaded obs to the `obs_keep` index subset (no-op if None). Takes the indices EXPLICITLY as an
    argument (2026-09-21, was the module global _OBS_KEEP) -- the loader passes DataConfig.obs_keep. Applied
    POST-CONCAT (indexes the full obs_fields-concatenated vector)."""
    return obs_all if obs_keep is None else obs_all[:, list(obs_keep)]


def _read_obs_fields(hf, obs_fields):
    """Read + CONCATENATE the listed parquet features into one (N, sum_dims) float32 obs matrix, in order
    (2026-09-24). Default obs_fields=('observation_vector',) reproduces the single-field load exactly. A field
    is (N,) scalar or (N,D) vector -> reshaped to (N,-1) before concat. Missing field -> a clear error naming
    what IS available (so a typo/unpushed field fails loudly rather than silently dropping obs). Generic: no
    field name is special-cased -- see DataConfig.obs_fields."""
    parts = []
    for fld in obs_fields:
        if fld not in hf:
            avail = [c for c in getattr(hf, "column_names", list(hf)) if "images" not in c]
            raise KeyError(f"obs field {fld!r} not in the dataset (obs_fields={list(obs_fields)}); available "
                           f"non-image columns: {avail}")
        arr = np.stack(hf[fld]).astype(np.float32)
        parts.append(arr.reshape(arr.shape[0], -1))
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def _subsample_episodes(eps, tag: str, *, dcfg: "DataConfig"):
    """Keep every s-th frame; AGGREGATE the actions that drive each kept transition. act[t] drives
    t -> t+1 (see MultiModalFlow._rollout_step, which reads a_win[:, -1]), so the action for the kept
    step i is the aggregate of act[i*s : (i+1)*s].

    Aggregation is a SUM for delta-like dims (EEF/rotation deltas compose additively over the skipped
    frames) but TAKE-LAST for near-binary dims: summing robocasa's gripper/flag dims would turn +-1 into
    +-5 and destroy their semantics. Binary dims are DETECTED (<=2 unique values), not hardcoded, and
    logged -- on this dataset that is the flag at dim 4 and the gripper at dim 11."""
    # dcfg carries the stride/aggregation/phases EXPLICITLY (2026-09-21, was module globals) -- see DataConfig.
    s, mode = dcfg.subsample, dcfg.action_aggregate
    if s <= 1:
        return eps
    acts = np.concatenate([e[1] for e in eps], 0)
    hold = [d for d in range(acts.shape[1]) if len(np.unique(acts[:, d])) <= 2]
    # PHASE OFFSETS: normally just [0] -- frames 1..s-1 of every group are discarded and never seen. With
    # data.subsample_all_phases (TRAIN only) emit all s of them as separate episodes: ~s x the windows at the
    # SAME rate. See DataConfig.subsample_all_phases.
    all_phases = dcfg.subsample_all_phases and "/train" in tag
    phases = range(s) if all_phases else (0,)
    out, dropped = [], 0
    for ep in eps:
        for ph in phases:
            o, a = ep[0][ph:], ep[1][ph:]
            n = len(o) // s
            if n < 2:                                 # too short to yield even one transition
                dropped += 1
                continue
            grp = a[:n * s].reshape(n, s, -1)
            if mode == "sum":
                aa = grp.sum(axis=1)
            elif mode == "mean":
                aa = grp.mean(axis=1)
            elif mode == "last":
                aa = grp[:, -1].copy()                # .copy(): grp[:, -1] is a VIEW into the episode's array
            elif mode == "first":
                aa = grp[:, 0].copy()
            else:                                     # concat: (n, s, dim) -> (n, s*dim), time-major
                aa = grp.reshape(n, -1).copy()
            if hold and mode in ("sum", "mean"):
                aa[:, hold] = grp[:, -1, hold]        # last raw action in the group, not the aggregate.
                #   Unnecessary for last/first (already one raw action) and wrong for concat (nothing to fix).
            # extra streams (ep[2:]) are sliced identically. A DICT of streams (the multi-camera frame
            # bundle) is sliced VALUE-WISE -- without this branch `x[ph:]` on a dict raises TypeError.
            extra = tuple({k: v[ph:][:n * s:s] for k, v in x.items()} if isinstance(x, dict)
                          else x[ph:][:n * s:s] for x in ep[2:])
            out.append((o[:n * s:s], aa) + extra)
    print(f"[subsample] {tag}: stride {s}{f' x {s} PHASES' if all_phases else ''} | {len(eps)} eps "
          f"{len(acts)} frames -> {len(out)} eps {sum(len(e[0]) for e in out)} frames | actions {mode.upper()}"
          + (f" except take-last on dims {hold}" if hold and mode in ("sum", "mean") else "")
          + (f" -> action_dim x{s}" if mode == "concat" else "")
          + f" | {dropped} eps dropped as too short", flush=True)
    return out


def load_split_episodes(root: str, split: str, *, dcfg: "DataConfig", repo_id: str = "torus"):
    """Return list of (obs (T,D), act (T,A)) float32 arrays. ISOLATED lerobot read. `repo_id` is the
    prefix the split was written with (<repo_id>/<split>; torus datasets = "torus"). `dcfg` (REQUIRED,
    keyword-only) carries the stride / aggregation / obs subset -- applied here, INSIDE the loader, so no call
    site can bypass or forget it (see DataConfig for why this replaced the module globals)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"{repo_id}/{split}", root=os.path.join(root, split))
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    obs_all = _apply_obs_keep(_read_obs_fields(hf, dcfg.obs_fields), dcfg.obs_keep)   # concat obs_fields, obs_keep POST-concat
    act_all = np.stack(hf["action"]).astype(np.float32)
    return _subsample_episodes([(obs_all[ep_idx == e], act_all[ep_idx == e]) for e in np.unique(ep_idx)],
                               f"{repo_id}/{split}", dcfg=dcfg)


def resize_frames_area(x, hw: tuple[int, int]):
    """(N,H,W,3) uint8 -> (N,h,w,3) uint8 by AREA (anti-aliased) downsample. numpy or torch in, same out.

    WHY THIS IS SHARED (2026-09-07). Two places must resize camera frames identically: this module, when it
    builds the `<cam>_<size>.npy` training cache from a dataset's video, and a LIVE environment's
    `render_obs`, which renders at the simulator's native size and must hand the model frames drawn from the
    same distribution. If the two use different filters (area vs bilinear vs nearest) nothing raises -- the
    model simply receives subtly out-of-distribution input and every rollout is quietly worse. So the op
    lives in ONE function that both call, rather than being written twice and allowed to drift.

    AREA specifically, not bilinear: it averages over the full source footprint of each output pixel, which
    is the correct antialiasing filter for a large downsample (256 -> 96 here). Bilinear samples 4 taps and
    aliases thin high-contrast structure -- exactly the ceiling strips and window mullions these datasets
    are full of."""
    was_np = not isinstance(x, torch.Tensor)
    t = torch.from_numpy(np.ascontiguousarray(x)) if was_np else x
    y = torch.nn.functional.interpolate(t.permute(0, 3, 1, 2).float(), size=tuple(hw), mode="area")
    y = y.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
    return y.numpy() if was_np else y


def load_fpv_frames(root: str, split: str, size: int | tuple[int, int] | None = 128,
                    max_frames: int | None = None, cache: bool = True, cam: str = "fpv"):
    """All egocentric frames for a split (lerobot chunked video, camera `cam`), AREA-downsampled ONCE
    and cached to disk (npy next to the split). `size`: int -> size×size (torus default), (H, W) tuple,
    or None -> native resolution (no resize). Returns uint8 (N, H, W, 3). The downsample is the
    only per-frame work and it's cached, so repeat loads are instant (mmap)."""
    import glob as _glob

    import imageio.v2 as imageio
    hw = (size, size) if isinstance(size, int) else (tuple(size) if size is not None else None)
    tag = size if isinstance(size, int) else ("native" if hw is None else f"{hw[0]}x{hw[1]}")
    cache_path = os.path.join(root, split, f"{cam}_{tag}.npy")
    if cache and max_frames is None and os.path.exists(cache_path):
        return np.load(cache_path)
    vid = os.path.join(root, split, "videos", f"observation.images.{cam}")
    mp4s = sorted(_glob.glob(os.path.join(vid, "*", "*.mp4")))
    assert mp4s, f"no {cam} mp4s under {vid}"
    out, buf = [], []

    def _flush():
        if not buf:
            return
        if hw is None:                                                         # native: no resize
            out.append(np.stack(buf))
            buf.clear()
            return
        out.append(resize_frames_area(np.stack(buf), hw))    # THE shared op -- see resize_frames_area
        buf.clear()

    def _have():
        return sum(len(o) for o in out) + len(buf)

    for p in mp4s:
        rd = imageio.get_reader(p)
        for fr in rd:
            buf.append(np.asarray(fr)[..., :3])
            if len(buf) >= 512:
                _flush()
            if max_frames is not None and _have() >= max_frames:
                break
        rd.close()
        if max_frames is not None and _have() >= max_frames:
            break
    _flush()
    frames = np.concatenate(out, 0)
    if max_frames is not None:
        frames = frames[:max_frames]
    elif cache:
        tmp = f"{cache_path}.{os.getpid()}.tmp.npy"   # atomic: write to a per-pid temp then rename, so two
        np.save(tmp, frames)                           # concurrent cold starts can't read a half-written .npy
        os.replace(tmp, cache_path)                    # (12+ GB here; a torn npy silently corrupts training)
    return frames


def load_split_episodes_mm(root: str, split: str, *, dcfg: "DataConfig", img_size=128, cam="fpv", repo_id: str = "torus"):
    """Like load_split_episodes but ALSO returns per-episode camera frames (area-downsampled, uint8),
    aligned 1:1 with obs steps. Returns one entry per episode:

        (obs (T,D), act (T,A), frames)          frames = {key: (T,H,W,3) uint8}   -- ALWAYS a dict

    `cam` may be:
        "robot0_agentview_left"                  -> {"robot0_agentview_left": frames}
        ["cam_a", "cam_b"]                       -> {"cam_a": ..., "cam_b": ...}
        {"cam_scene": "robot0_agentview_left",   -> {"cam_scene": ..., "cam_wrist": ...}
         "cam_wrist": "robot0_eye_in_hand"}         (a MAPPING lets the caller key by MODALITY HEAD)
    `img_size` is one size for every camera, or a mapping/sequence keyed/ordered to match `cam`.

    ELEMENT 2 IS A DICT EVEN FOR ONE CAMERA, DELIBERATELY. It used to be a bare array, so every call site
    wrote `ep[2]` to mean "the camera". With N image heads that pattern silently hands EVERY head the FIRST
    camera's frames -- a head scored against another camera's pixels produces a plausible WRONG NUMBER
    rather than a crash, which is the worst failure mode available. Making it a dict turns every such site
    into an immediate TypeError until it names the head it wants. See design/two_camera_plan.md.

    The chunked video is read in dataset row order (== obs row order), then split by episode_index. `dcfg`
    (REQUIRED, keyword-only) carries the stride / aggregation / obs subset -- applied here, INSIDE the loader,
    so no call site can bypass or forget it (see DataConfig for why this replaced the module globals)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if isinstance(cam, str):
        keys, cams = [cam], [cam]
    elif isinstance(cam, dict):
        keys, cams = list(cam), [cam[k] for k in cam]
    else:
        keys = cams = list(cam)
    if isinstance(img_size, dict):
        sizes = [img_size[k] for k in keys]
    elif img_size is None or isinstance(img_size, (int, tuple)):
        sizes = [img_size] * len(keys)
    else:
        sizes = list(img_size)
    assert len(sizes) == len(keys), f"img_size/cam length mismatch: {len(sizes)} vs {len(keys)}"
    assert len(set(keys)) == len(keys), f"duplicate frame keys {keys}"

    ds = LeRobotDataset(f"{repo_id}/{split}", root=os.path.join(root, split))
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    obs_all = _apply_obs_keep(_read_obs_fields(hf, dcfg.obs_fields), dcfg.obs_keep)   # concat obs_fields, obs_keep POST-concat
    act_all = np.stack(hf["action"]).astype(np.float32)
    per_key = {}
    for k, c, sz in zip(keys, cams, sizes):
        fr = load_fpv_frames(root, split, size=sz, cam=c)                  # (N, H, W, 3), row order
        assert len(fr) == len(obs_all), f"{c}/row count mismatch: {len(fr)} vs {len(obs_all)}"
        per_key[k] = fr
    return _subsample_episodes(
        [(obs_all[ep_idx == e], act_all[ep_idx == e],
          {k: fr[ep_idx == e] for k, fr in per_key.items()}) for e in np.unique(ep_idx)],
        f"{repo_id}/{split}+{'+'.join(cams)}", dcfg=dcfg)


class MMWindowLoader:
    """The ONE GPU-resident window loader for every model. obs/act windows live on `device`; when image
    modalities are present (`image_head` set) each one's FRAME store (uint8, ~3 GB at 128²) is held resident
    too and gathered per batch by a pure GPU index (no host copy). PROPRIO-ONLY passes `image_head=None` ->
    no frames are loaded or gathered (episodes are (obs, act) pairs).

    N HEADS. `image_head` is a name OR a sequence of names; head i reads episode element `e[2 + i]`, which is
    the order `load_split_episodes_mm` returns its cameras in. Yields {obs_seq (B,L,6), act_seq (B,L,2),
    <head> (B,L,H,W,3) in [0,1] per head}. `win_idx` is SHARED across heads -- the same window indices apply
    to every camera, because all streams come from the same episode row order -- so N heads cost N frame
    stores but only one index tensor. Window order matches `stack_windows`, so all streams stay aligned."""

    def __init__(self, episodes, P: int, F: int, normalizer: Normalizer, batch: int, shuffle: bool, device,
                 image_head=None, stride: int = 1):
        L = P + F
        heads = [] if image_head is None else ([image_head] if isinstance(image_head, str) else list(image_head))
        # heads index the frame DICT at episode element 2 by NAME -- see load_split_episodes_mm. Positional
        # access is deliberately impossible: it is what let one camera masquerade as all of them.
        self.image_head = image_head          # kept verbatim for any caller that inspects it
        self.heads = heads
        obs_w, act_w = stack_windows([(e[0], e[1]) for e in episodes], P, F, normalizer, stride)
        self.obs, self.act = obs_w.to(device), act_w.to(device)
        self.frames = None
        if heads:   # per head: concat all episode frames -> one GPU uint8 store; ONE shared per-window index
            self.frames = {}
            avail = set(episodes[0][2]) if isinstance(episodes[0][2], dict) else set()
            for i, h in enumerate(heads):
                assert h in avail, (
                    f"image head {h!r} has no frames: the episodes carry {sorted(avail)}. "
                    f"load_split_episodes_mm must be called with cam={{head: camera}} covering every image "
                    f"modality -- see training/setup.py:window_loaders")
                frames, starts, off = [], [], 0
                for e in episodes:
                    o, img = e[0], e[2][h]
                    frames.append(torch.from_numpy(img))
                    starts.extend(range(off, off + len(o) - L + 1, stride))   # stride matches stack_windows
                    off += len(img)
                self.frames[h] = torch.cat(frames, 0).to(device)              # (N_total,H,W,3) uint8, resident
                if i == 0:                                                    # identical for every head
                    self.win_idx = torch.tensor(starts, device=device)[:, None] + \
                                   torch.arange(L, device=device)[None]
        self.batch, self.shuffle, self.device = batch, shuffle, device
        self.N = self.obs.shape[0]

    def __len__(self):
        return (self.N + self.batch - 1) // self.batch

    def __iter__(self):
        order = torch.randperm(self.N, device=self.device) if self.shuffle else torch.arange(self.N, device=self.device)
        for i in range(0, self.N, self.batch):
            j = order[i: i + self.batch]
            out = {"obs_seq": self.obs.index_select(0, j), "act_seq": self.act.index_select(0, j)}
            if self.frames is not None:
                wi = self.win_idx.index_select(0, j)                          # shared across heads
                for h in self.heads:
                    out[h] = self.frames[h][wi].float().div_(255.0)            # GPU gather, per head
            yield out


def stack_windows(episodes, P: int, F: int, normalizer: Normalizer, stride: int = 1):
    """Pre-build the length-(P+F) windows (every `stride` starts) into two normalized tensors (no per-item
    work later). Returns obs_windows (N,L,6), act_windows (N,L,2). stride>1 drops near-duplicate overlapping
    windows (adjacent starts share L-1 steps) -> fewer batches/epoch, ~no coverage loss over many epochs."""
    L = P + F
    obs_w, act_w = [], []
    for o, a in episodes:
        if len(o) < L:
            continue
        obs_w.append(torch.from_numpy(o).unfold(0, L, stride).permute(0, 2, 1).contiguous())  # (n,L,6)
        act_w.append(torch.from_numpy(a).unfold(0, L, stride).permute(0, 2, 1).contiguous())  # (n,L,2)
    obs, act = torch.cat(obs_w), torch.cat(act_w)
    return normalizer.norm_obs(obs), normalizer.norm_act(act)


class TrajectoryDataset(Dataset):
    """Whole episodes for long-horizon eval. Returns normalized full obs/act sequences."""

    def __init__(self, episodes, normalizer: Normalizer):
        self.eps = episodes
        self.norm = normalizer

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        o, a = self.eps[i]
        return {
            "obs_seq": self.norm.norm_obs(torch.from_numpy(o)),
            "act_seq": self.norm.norm_act(torch.from_numpy(a)),
        }
