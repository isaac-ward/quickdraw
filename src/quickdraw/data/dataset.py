"""Loading: windowed (train/val) and full-trajectory (eval) access + train-only normalization.

We store with lerobot (generate.py) but read episodes out and window them ourselves, so the
training loader does not depend on lerobot's `delta_timestamps` behavior. The lerobot read is
isolated in `load_split_episodes` (the one version-sensitive spot for loading).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict

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
    def from_file(cls, root: str) -> "Normalizer":
        with open(os.path.join(root, "normalization_stats.json")) as f:
            return cls(json.load(f))

    def norm_obs(self, o):
        return (o - self.o_mean.to(o)) / self.o_std.to(o)

    def denorm_obs(self, o):
        return o * self.o_std.to(o) + self.o_mean.to(o)

    def norm_act(self, a):
        return (a - self.a_mean.to(a)) / self.a_std.to(a)

    def denorm_act(self, a):
        return a * self.a_std.to(a) + self.a_mean.to(a)


def _json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def episode_records(root: str, split: str) -> list[dict]:
    """Authoritative episode inventory in on-disk order."""
    path = os.path.join(root, split, "meta", "episodes.jsonl")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    indices = [int(row["episode_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate episode indices in {path}")
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def validate_lerobot_contract(root: str, split: str, schema) -> tuple[dict, list[dict]]:
    """Validate the external package before tensors or videos are allocated."""
    split_root = os.path.join(root, split)
    info = _json(os.path.join(split_root, "meta", "info.json"))
    records = episode_records(root, split)
    card_path = os.path.join(root, "dataset_card.json")
    if os.path.exists(card_path):
        card = _json(card_path)
        if card.get("status") != "complete":
            raise ValueError(f"dataset package is not complete: status={card.get('status')!r}")
    source_stats_path = os.path.join(root, "normalization_stats.json")
    if os.path.exists(source_stats_path):
        source_stats = _json(source_stats_path)
        for key in (str(schema.state_key), str(schema.action_key)):
            item = source_stats.get(key)
            if not item:
                raise ValueError(f"source normalization statistics are missing {key!r}")
            mean, std = np.asarray(item["mean"]), np.asarray(item["std"])
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
                raise ValueError(f"source normalization statistics for {key!r} are invalid")
    if str(info.get("codebase_version")) != "v2.1":
        raise ValueError(f"expected LeRobot v2.1, got {info.get('codebase_version')!r}")
    features = info.get("features", {})
    state_key, action_key = str(schema.state_key), str(schema.action_key)
    image_key = str(schema.image_key) if schema.get("image_key") else None
    vector_contracts = (
        (state_key, int(schema.state_dim), schema.get("state_dtype")),
        (action_key, int(schema.action_dim), schema.get("action_dtype")),
    )
    for key, dim, dtype in vector_contracts:
        if key not in features:
            raise ValueError(f"required feature {key!r} is missing from {split}/meta/info.json")
        shape = list(features[key].get("shape") or [])
        if shape != [dim]:
            raise ValueError(f"{key!r} shape mismatch: metadata={shape}, configured={[dim]}")
        if dtype and str(features[key].get("dtype")) != str(dtype):
            raise ValueError(
                f"{key!r} dtype mismatch: metadata={features[key].get('dtype')!r}, configured={str(dtype)!r}"
            )
    if image_key:
        feature = features.get(image_key)
        if not feature or feature.get("dtype") != "video":
            raise ValueError(f"configured image feature {image_key!r} is not a packaged video")
        shape = list(feature.get("shape") or [])
        source_size = int(schema.get("source_image_size", shape[0] if shape else -1))
        if shape != [source_size, source_size, 3]:
            raise ValueError(
                f"{image_key!r} source shape mismatch: metadata={shape}, "
                f"configured={[source_size, source_size, 3]}"
            )
    fps = int(info.get("fps", -1))
    if fps != int(schema.fps):
        raise ValueError(f"FPS mismatch: metadata={fps}, configured={int(schema.fps)}")
    if int(info.get("total_episodes", -1)) != len(records):
        raise ValueError("episode inventory count disagrees with info.json")
    if int(info.get("total_frames", -1)) != sum(int(row["length"]) for row in records):
        raise ValueError("episode frame total disagrees with info.json")
    if schema.get("expected_episodes") and int(info["total_episodes"]) != int(schema.expected_episodes):
        raise ValueError(
            f"episode total mismatch: metadata={info['total_episodes']}, expected={schema.expected_episodes}"
        )
    if schema.get("expected_frames") and int(info["total_frames"]) != int(schema.expected_frames):
        raise ValueError(
            f"frame total mismatch: metadata={info['total_frames']}, expected={schema.expected_frames}"
        )
    if not info.get("data_path") or (image_key and not info.get("video_path")):
        raise ValueError("LeRobot metadata is missing a data/video path template")
    return info, records


def derive_episode_partition(records: list[dict], validation) -> dict:
    """Deterministic task-stratified episode holdout. Task text is inventory only, never model input."""
    groups = defaultdict(list)
    for row in records:
        tasks = row.get("tasks") or []
        if len(tasks) != 1:
            raise ValueError(f"episode {row['episode_index']} must have exactly one top-level task")
        groups[str(tasks[0])].append(row)
    fraction = float(validation.fraction)
    minimum = int(validation.minimum_per_group)
    seed = int(validation.seed)
    if not bool(validation.stratify_by_episode_task):
        raise ValueError("the supported RoboCasa holdout must be stratified by top-level episode task")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation fraction must be in (0,1), got {fraction}")
    rng = np.random.default_rng(seed)
    train, val, summary = [], [], {}
    for task, rows in sorted(groups.items()):
        ids = np.asarray(sorted(int(row["episode_index"]) for row in rows), dtype=np.int64)
        n_val = max(minimum, round(len(ids) * fraction))
        n_val = min(n_val, len(ids) - 1)
        if n_val < 1:
            raise ValueError(f"task group {task!r} cannot supply disjoint train/validation episodes")
        val_ids = sorted(int(x) for x in rng.permutation(ids)[:n_val])
        val_set = set(val_ids)
        train_ids = sorted(int(x) for x in ids if int(x) not in val_set)
        by_id = {int(row["episode_index"]): row for row in rows}
        train.extend(train_ids)
        val.extend(val_ids)
        summary[task] = {
            "episodes": len(ids),
            "frames": sum(int(row["length"]) for row in rows),
            "train_episodes": len(train_ids),
            "train_frames": sum(int(by_id[i]["length"]) for i in train_ids),
            "val_episodes": len(val_ids),
            "val_frames": sum(int(by_id[i]["length"]) for i in val_ids),
        }
    all_ids = {int(row["episode_index"]) for row in records}
    if set(train) & set(val) or set(train) | set(val) != all_ids:
        raise AssertionError("derived episode partition is not disjoint and exhaustive")
    return {
        "schema_version": 1,
        "policy": {
            "fraction": fraction,
            "seed": seed,
            "stratify_by_episode_task": bool(validation.stratify_by_episode_task),
            "minimum_per_group": minimum,
        },
        "train_episode_indices": sorted(train),
        "val_episode_indices": sorted(val),
        "groups": summary,
    }


def _vector_column(table, key: str) -> np.ndarray:
    values = table[key].combine_chunks().to_pylist()
    out = np.asarray(values, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{key!r} must decode as a vector column, got shape {out.shape}")
    return np.ascontiguousarray(out)


def load_split_episodes(
    root: str,
    split: str,
    state_key: str = "observation_vector",
    action_key: str = "action",
    episode_indices=None,
):
    """Read LeRobot episode parquet files into canonical contiguous float32 (state, action) pairs."""
    import pyarrow.parquet as pq

    info = _json(os.path.join(root, split, "meta", "info.json"))
    records = episode_records(root, split)
    by_id = {int(row["episode_index"]): row for row in records}
    selected = sorted(by_id) if episode_indices is None else sorted(int(i) for i in episode_indices)
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"unknown episode indices requested: {sorted(unknown)}")
    out = []
    for episode_index in selected:
        episode_chunk = episode_index // int(info["chunks_size"])
        rel = info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)
        path = os.path.join(root, split, rel)
        table = pq.read_table(path, columns=["episode_index", state_key, action_key])
        stored_ids = np.asarray(table["episode_index"].combine_chunks().to_pylist()).reshape(-1)
        if len(stored_ids) and not np.all(stored_ids == episode_index):
            raise ValueError(f"{path} contains rows from another episode")
        obs, act = _vector_column(table, state_key), _vector_column(table, action_key)
        expected = int(by_id[episode_index]["length"])
        if len(obs) != expected or len(act) != expected:
            raise ValueError(
                f"episode {episode_index} length mismatch: metadata={expected}, state={len(obs)}, action={len(act)}"
            )
        out.append((obs, act))
    return out


def normalization_stats(episodes, state_key: str, action_key: str, eps: float = 1e-6) -> dict:
    """Canonical train-only statistics for the existing QuickDraw Normalizer."""
    if not episodes:
        raise ValueError("cannot compute normalization statistics from an empty episode set")
    obs = np.concatenate([episode[0] for episode in episodes], axis=0).astype(np.float64, copy=False)
    act = np.concatenate([episode[1] for episode in episodes], axis=0).astype(np.float64, copy=False)
    o_mean, o_std = obs.mean(0), np.maximum(obs.std(0), eps)
    a_mean, a_std = act.mean(0), np.maximum(act.std(0), eps)
    for name, value in (("observation mean", o_mean), ("observation std", o_std),
                        ("action mean", a_mean), ("action std", a_std)):
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {name} in train-derived normalization")
    return {
        "observation_vector": {"mean": o_mean.tolist(), "std": o_std.tolist()},
        "action": {"mean": a_mean.tolist(), "std": a_std.tolist()},
        "source_keys": {"observation_vector": state_key, "action": action_key},
    }


def episode_inventory(episodes, P: int, F: int, stride: int = 1) -> dict:
    lengths = [len(episode[0]) for episode in episodes]
    L = P + F
    short = [n for n in lengths if n < L]
    if short:
        raise ValueError(f"{len(short)} episodes are shorter than the required P+F={L}: {short[:8]}")
    return {
        "episodes": len(lengths),
        "frames": sum(lengths),
        "transitions": sum(max(0, n - 1) for n in lengths),
        "windows": sum((n - L) // stride + 1 for n in lengths),
        "window_stride": stride,
    }


def load_fpv_frames(root: str, split: str, size: int = 128, max_frames: int | None = None, cache: bool = True):
    """All egocentric FPV frames for a split (lerobot chunked video), AREA-downsampled to size×size ONCE
    and cached to disk (npy next to the split). Returns uint8 (N, size, size, 3). The downsample is the
    only per-frame work and it's cached, so repeat loads are instant (mmap)."""
    import glob as _glob

    import imageio.v2 as imageio
    cache_path = os.path.join(root, split, f"fpv_{size}.npy")
    if cache and max_frames is None and os.path.exists(cache_path):
        return np.load(cache_path)
    vid = os.path.join(root, split, "videos", "observation.images.fpv")
    mp4s = sorted(_glob.glob(os.path.join(vid, "*", "*.mp4")))
    assert mp4s, f"no FPV mp4s under {vid}"
    out, buf = [], []

    def _flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float()       # (b,3,H,W)
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="area")  # anti-aliased downsample
        out.append(x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy())
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
        np.save(cache_path, frames)
    return frames


def _frame_cache_path(root: str, split: str, image_key: str, size: int, records: list[dict],
                      cache_root: str | None, cache_identity: str | None) -> str:
    identity = {
        "root": os.path.realpath(root),
        "split": split,
        "image_key": image_key,
        "size": int(size),
        "logical_split": cache_identity,
        "episodes": [(int(row["episode_index"]), int(row["length"])) for row in records],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_key)
    base = cache_root or os.environ.get("QUICKDRAW_CACHE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "quickdraw")
    path = os.path.join(os.path.expanduser(base), "frames")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{safe_key}_{size}_{digest}.npy")


def load_video_frame_store(root: str, split: str, image_key: str, size: int, records: list[dict],
                           cache_root: str | None = None, cache_identity: str | None = None):
    """Decode one configured camera in episode order into an immutable, mmap-backed uint8 cache."""
    import imageio.v2 as imageio

    info = _json(os.path.join(root, split, "meta", "info.json"))
    cache_path = _frame_cache_path(root, split, image_key, size, records, cache_root, cache_identity)
    expected_total = sum(int(row["length"]) for row in records)
    expected_shape = (expected_total, int(size), int(size), 3)
    if os.path.exists(cache_path):
        frames = np.load(cache_path, mmap_mode="c")
        if frames.dtype == np.uint8 and tuple(frames.shape) == expected_shape:
            return frames, cache_path
        raise ValueError(
            f"invalid frame cache {cache_path}: shape={frames.shape}, dtype={frames.dtype}; "
            f"expected {expected_shape} uint8"
        )

    tmp = f"{cache_path}.tmp-{os.getpid()}.npy"
    frames = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=expected_shape)
    offset = 0
    try:
        for row in records:
            episode_index = int(row["episode_index"])
            episode_chunk = episode_index // int(info["chunks_size"])
            rel = info["video_path"].format(
                episode_chunk=episode_chunk, episode_index=episode_index, video_key=image_key
            )
            path = os.path.join(root, split, rel)
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing configured video for episode {episode_index}: {path}")
            expected = int(row["length"])
            reader = imageio.get_reader(path)
            buf, decoded = [], 0

            def flush():
                nonlocal offset
                if not buf:
                    return
                x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float()
                x = torch.nn.functional.interpolate(x, size=(size, size), mode="area")
                chunk = x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
                frames[offset: offset + len(chunk)] = chunk
                offset += len(chunk)
                buf.clear()

            try:
                for frame in reader:
                    buf.append(np.asarray(frame)[..., :3])
                    decoded += 1
                    if len(buf) >= 256:
                        flush()
            finally:
                reader.close()
            flush()
            if decoded != expected:
                raise ValueError(
                    f"episode {episode_index} video/frame mismatch: decoded={decoded}, metadata={expected}"
                )
        if offset != expected_total:
            raise AssertionError(f"decoded frame total {offset} != expected {expected_total}")
        frames.flush()
        del frames
        os.replace(tmp, cache_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return np.load(cache_path, mmap_mode="c"), cache_path


def load_split_episodes_mm(
    root: str,
    split: str,
    img_size: int = 128,
    *,
    state_key: str = "observation_vector",
    action_key: str = "action",
    image_key: str | None = None,
    episode_indices=None,
    cache_root: str | None = None,
    cache_identity: str | None = None,
):
    """Canonical aligned (state, action, selected RGB camera) episodes."""
    episodes = load_split_episodes(root, split, state_key, action_key, episode_indices)
    if image_key is None:  # existing torus layout/cache path
        frames_all = load_fpv_frames(root, split, size=img_size)
        if len(frames_all) != sum(len(episode[0]) for episode in episodes):
            raise ValueError("FPV/row count mismatch")
        out, offset = [], 0
        for obs, act in episodes:
            out.append((obs, act, frames_all[offset: offset + len(obs)]))
            offset += len(obs)
        return out

    records = episode_records(root, split)
    all_frames, _ = load_video_frame_store(
        root, split, image_key, img_size, records, cache_root, cache_identity
    )
    offsets, offset = {}, 0
    for row in records:
        idx, length = int(row["episode_index"]), int(row["length"])
        offsets[idx] = (offset, offset + length)
        offset += length
    selected = sorted(offsets) if episode_indices is None else sorted(int(i) for i in episode_indices)
    out = []
    for (obs, act), idx in zip(episodes, selected, strict=True):
        start, stop = offsets[idx]
        frames = all_frames[start:stop]
        if len(obs) != len(frames):
            raise ValueError(f"episode {idx} state/video alignment mismatch: {len(obs)} vs {len(frames)}")
        out.append((obs, act, frames))
    return out


class MMWindowLoader:
    """The ONE GPU-resident window loader for every model. obs/act windows live on `device`; when an image
    modality is present (`image_head` set) its FRAME store (uint8, ~3 GB at 128²) is held resident too and
    gathered per batch by a pure GPU index (no host copy). PROPRIO-ONLY passes `image_head=None` -> no frames
    are loaded or gathered (episodes are (obs, act) pairs). Yields {obs_seq (B,L,6), act_seq (B,L,2)[,
    <image_head> (B,L,H,W,3) in [0,1]]}. Window order matches `stack_windows`, so all streams stay aligned."""

    def __init__(self, episodes, P: int, F: int, normalizer: Normalizer, batch: int, shuffle: bool, device,
                 image_head: str | None = None, stride: int = 1):
        L = P + F
        self.image_head = image_head
        obs_w, act_w = stack_windows([(e[0], e[1]) for e in episodes], P, F, normalizer, stride)
        self.obs, self.act = obs_w.to(device), act_w.to(device)
        self.frames = None
        if image_head is not None:   # concat all episode frames -> one GPU uint8 store + per-window GLOBAL frame idx
            frames, starts, off = [], [], 0
            for e in episodes:
                o, img = e[0], e[2]
                frames.append(torch.from_numpy(img))
                starts.extend(range(off, off + len(o) - L + 1, stride))   # stride matches stack_windows -> streams stay aligned
                off += len(img)
            self.frames = torch.cat(frames, 0).to(device)                    # (N_total,H,W,3) uint8, GPU-resident
            starts = torch.tensor(starts, device=device)
            self.win_idx = starts[:, None] + torch.arange(L, device=device)[None]
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
                out[self.image_head] = self.frames[self.win_idx.index_select(0, j)].float().div_(255.0)  # GPU gather
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
