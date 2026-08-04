#!/usr/bin/env python3
"""Build/finalize the QuickDraw-style Hugging Face package for certified RoboCasa.

The package contains one LeRobot v2.1 train split beneath a single run root,
plus root-level QuickDraw metadata. Source observation/action arrays are copied
unchanged; only dataset-global indices and integer annotation codebooks are
remapped to prevent cross-task label aliasing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / ".cache/robocasa-data-generation/render-episodes"
DEFAULT_SUMMARY = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-eight-artifact-v1/summary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/"
    "quickdraw-hf-robocasa-scene4-certified-4h"
)
FPS = 20
TARGET_FRAMES = 4 * 60 * 60 * FPS
ANNOTATION_CODE_COLUMNS = (
    "annotation.human.task_description",
    "annotation.human.task_name",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
    "task_index",
)
OPTIONAL_TEXT_ANNOTATIONS = (
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
)
VIDEO_KEYS = (
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def source_root(task: str) -> Path:
    return DATA_ROOT / task / "lerobot"


def source_parquet(task: str, episode: int) -> Path:
    matches = list(
        (source_root(task) / "data").glob(
            f"chunk-*/episode_{episode:06d}.parquet"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one parquet for {task}/{episode}, found {matches}"
        )
    return matches[0]


def source_extra(task: str, episode: int, name: str) -> Path:
    return (
        source_root(task)
        / "extras"
        / f"episode_{episode:06d}"
        / name
    )


def read_final_certificate(path: Path) -> dict[str, Any]:
    final = None
    for line in path.read_text(errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") in {"result", "error"}:
            final = value
    if final is None:
        raise ValueError(f"certificate has no final record: {path}")
    return final


def choose_selection(
    summary_path: Path, target_frames: int
) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    candidates = []
    seen = set()
    for item in summary["results"]:
        if item.get("accepted") is not True:
            continue
        identity = (str(item["task"]), int(item["episode"]))
        if identity in seen:
            continue
        seen.add(identity)
        certificate_path = Path(item["certificate"])
        if not certificate_path.is_absolute():
            certificate_path = REPO_ROOT / certificate_path
        certificate = read_final_certificate(certificate_path)
        if not (
            certificate.get("accepted") is True
            and certificate.get("complete_trace") is True
            and certificate.get("initialization_compatible") is True
            and certificate.get("collision_absent") is True
        ):
            raise ValueError(f"summary references unaccepted certificate {certificate_path}")
        frozen = certificate.get("render_artifact")
        if not isinstance(frozen, dict):
            raise ValueError(
                f"certificate lacks a same-process render artifact: {certificate_path}"
            )
        artifact_path = Path(frozen["path"])
        if (
            not artifact_path.exists()
            or file_sha256(artifact_path) != frozen["sha256"]
        ):
            raise ValueError(f"invalid render artifact for {certificate_path}")
        candidates.append(
            {
                "task": identity[0],
                "source_episode": identity[1],
                "frames": int(item["frames"]),
                "certificate": str(certificate_path.resolve()),
                "certificate_sha256": file_sha256(certificate_path),
                "render_artifact": {
                    **frozen,
                    "path": str(artifact_path.resolve()),
                },
            }
        )
    tasks = sorted({item["task"] for item in candidates})
    expected_tasks = sorted(summary.get("selected_tasks", tasks))
    if tasks != expected_tasks:
        raise ValueError(
            f"certified pool task coverage differs: {tasks} vs {expected_tasks}"
        )
    leaders = []
    selected_ids = set()
    for task in tasks:
        leader = max(
            (item for item in candidates if item["task"] == task),
            key=lambda item: (item["frames"], -item["source_episode"]),
        )
        leaders.append(leader)
        selected_ids.add((leader["task"], leader["source_episode"]))
    selected = list(leaders)
    frames = sum(item["frames"] for item in selected)
    for item in sorted(
        candidates,
        key=lambda value: (-value["frames"], value["task"], value["source_episode"]),
    ):
        identity = (item["task"], item["source_episode"])
        if frames >= target_frames:
            break
        if identity in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(identity)
        frames += item["frames"]
    if frames < target_frames:
        raise ValueError(
            f"only {frames / FPS / 3600:.3f} certified hours are available"
        )
    selected.sort(key=lambda value: (value["task"], value["source_episode"]))
    offset = 0
    for episode_index, item in enumerate(selected):
        item["episode_index"] = episode_index
        item["index_start"] = offset
        item["index_stop"] = offset + item["frames"]
        offset += item["frames"]
    return selected


def load_or_create_selection(
    summary_path: Path, output_root: Path, target_frames: int
) -> dict[str, Any]:
    path = output_root / "selection.json"
    if path.exists():
        selection = json.loads(path.read_text())
        if int(selection["target_frames"]) != target_frames:
            raise ValueError("existing selection has a different target")
        return selection
    episodes = choose_selection(summary_path, target_frames)
    selection = {
        "scene": {"layout_id": 4, "style_id": 4},
        "fps": FPS,
        "target_frames": target_frames,
        "target_hours": target_frames / FPS / 3600,
        "selected_frames": sum(item["frames"] for item in episodes),
        "selected_hours": sum(item["frames"] for item in episodes) / FPS / 3600,
        "selected_episodes": len(episodes),
        "tasks": sorted({item["task"] for item in episodes}),
        "selection_rule": (
            "longest accepted episode per task, then longest remaining accepted "
            "episodes until the target; full episodes only"
        ),
        "episodes": episodes,
        "certificate_summary": str(summary_path.resolve()),
        "certificate_summary_sha256": file_sha256(summary_path),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return selection


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total: np.ndarray | None = None
        self.square_total: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        values = values.astype(np.float64)
        batch_count = values.shape[0]
        batch_total = values.sum(axis=0)
        batch_square_total = np.square(values).sum(axis=0)
        batch_minimum = values.min(axis=0)
        batch_maximum = values.max(axis=0)
        if self.total is None:
            self.total = batch_total
            self.square_total = batch_square_total
            self.minimum = batch_minimum
            self.maximum = batch_maximum
        else:
            self.total += batch_total
            self.square_total += batch_square_total
            self.minimum = np.minimum(self.minimum, batch_minimum)
            self.maximum = np.maximum(self.maximum, batch_maximum)
        self.count += batch_count

    def record(self, count_override: int | None = None) -> dict[str, Any]:
        if not self.count or self.total is None or self.square_total is None:
            raise ValueError("empty moments")
        mean = self.total / self.count
        variance = np.maximum(self.square_total / self.count - np.square(mean), 0)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(self.count if count_override is None else count_override)],
        }


def column_numpy(table: pa.Table, name: str) -> np.ndarray:
    values = table[name].to_pylist()
    first = values[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.asarray(values)
    return np.asarray(values)


def table_stats(table: pa.Table) -> tuple[dict[str, Any], dict[str, Moments]]:
    records = {}
    accumulators = {}
    for name in table.column_names:
        values = column_numpy(table, name)
        moments = Moments()
        moments.update(values)
        records[name] = moments.record()
        accumulators[name] = moments
    return records, accumulators


def merge_moments(target: Moments, source: Moments) -> None:
    if source.total is None:
        return
    if target.total is None:
        target.total = source.total.copy()
        target.square_total = source.square_total.copy()
        target.minimum = source.minimum.copy()
        target.maximum = source.maximum.copy()
    else:
        target.total += source.total
        target.square_total += source.square_total
        target.minimum = np.minimum(target.minimum, source.minimum)
        target.maximum = np.maximum(target.maximum, source.maximum)
    target.count += source.count


def build_codebook(tasks: list[str]) -> tuple[list[dict[str, Any]], dict[str, dict[int, int]]]:
    global_records = []
    value_to_index = {}
    local_maps = {}
    for task in tasks:
        records = json_lines(source_root(task) / "meta/tasks.jsonl")
        local_map = {}
        for record in records:
            text = str(record["task"])
            if text not in value_to_index:
                value_to_index[text] = len(global_records)
                global_records.append(
                    {"task_index": value_to_index[text], "task": text}
                )
            local_map[int(record["task_index"])] = value_to_index[text]
        local_maps[task] = local_map
    return global_records, local_maps


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    field = table.schema.field(index)
    array = pa.array(values, type=field.type)
    return table.set_column(index, field, array)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_package(
    summary_path: Path, output_root: Path, target_frames: int
) -> dict[str, Any]:
    selection = load_or_create_selection(summary_path, output_root, target_frames)
    split_root = output_root / "train"
    data_dir = split_root / "data/chunk-000"
    meta_dir = split_root / "meta"
    extras_dir = split_root / "extras"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    extras_dir.mkdir(parents=True, exist_ok=True)

    codebook, local_maps = build_codebook(selection["tasks"])
    with (meta_dir / "tasks.jsonl").open("w") as stream:
        for record in codebook:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    aggregate: dict[str, Moments] = {}
    episode_stats = []
    episode_records = []
    source_records = []
    canonical_schema_metadata = None
    descriptions = {
        task: json_lines(source_root(task) / "meta/tasks.jsonl")[0]["task"]
        for task in selection["tasks"]
    }
    for item in selection["episodes"]:
        task = item["task"]
        source_episode = int(item["source_episode"])
        episode_index = int(item["episode_index"])
        source_path = source_parquet(task, source_episode)
        table = pq.read_table(source_path)
        if table.num_rows != int(item["frames"]):
            raise ValueError(
                f"frame mismatch for {task}/{source_episode}: "
                f"{table.num_rows} vs {item['frames']}"
            )
        original_state = column_numpy(table, "observation.state")
        original_action = column_numpy(table, "action")
        local_map = local_maps[task]
        for name in OPTIONAL_TEXT_ANNOTATIONS:
            if name not in table.column_names:
                table = table.append_column(
                    pa.field(name, pa.int64()),
                    pa.array(
                        np.zeros(table.num_rows, dtype=np.int64),
                        type=pa.int64(),
                    ),
                )
        if "subtask_idx" not in table.column_names:
            table = table.append_column(
                pa.field("subtask_idx", pa.int64()),
                pa.array(
                    np.full(table.num_rows, -1, dtype=np.int64),
                    type=pa.int64(),
                ),
            )
        for name in ANNOTATION_CODE_COLUMNS:
            if name not in table.column_names:
                continue
            old = column_numpy(table, name).astype(np.int64)
            try:
                new = np.asarray([local_map[int(value)] for value in old], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"{task} missing annotation code {exc}") from exc
            table = replace_column(table, name, new)
        table = replace_column(
            table, "episode_index", np.full(table.num_rows, episode_index)
        )
        table = replace_column(
            table, "frame_index", np.arange(table.num_rows, dtype=np.int64)
        )
        table = replace_column(
            table,
            "index",
            np.arange(
                int(item["index_start"]), int(item["index_stop"]), dtype=np.int64
            ),
        )
        table = replace_column(
            table,
            "timestamp",
            np.arange(table.num_rows, dtype=np.float32) / FPS,
        )
        if not np.array_equal(original_state, column_numpy(table, "observation.state")):
            raise ValueError("observation.state changed during reindexing")
        if not np.array_equal(original_action, column_numpy(table, "action")):
            raise ValueError("action changed during reindexing")
        if canonical_schema_metadata is None:
            canonical_schema_metadata = table.schema.metadata
        table = table.replace_schema_metadata(canonical_schema_metadata)

        output_path = data_dir / f"episode_{episode_index:06d}.parquet"
        temporary_path = output_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(output_path)

        stats, moments = table_stats(table)
        episode_stats.append({"episode_index": episode_index, "stats": stats})
        for name, value in moments.items():
            merge_moments(aggregate.setdefault(name, Moments()), value)
        episode_records.append(
            {
                "episode_index": episode_index,
                "tasks": [descriptions[task]],
                "length": table.num_rows,
            }
        )

        certificate_path = Path(item["certificate"])
        certificate_copy = (
            extras_dir / "certificates" / f"episode_{episode_index:06d}.jsonl"
        )
        certificate_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(certificate_path, certificate_copy)
        ep_meta_path = source_extra(task, source_episode, "ep_meta.json")
        model_path = source_extra(task, source_episode, "model.xml.gz")
        states_path = source_extra(task, source_episode, "states.npz")
        source_record = {
            **item,
            "source_parquet": {
                "path": str(source_path.resolve()),
                "sha256": file_sha256(source_path),
            },
            "packaged_parquet": {
                "path": str(output_path.resolve()),
                "sha256": file_sha256(output_path),
            },
            "source_ep_meta": {
                "path": str(ep_meta_path.resolve()),
                "sha256": file_sha256(ep_meta_path),
            },
            "source_model_xml_gz": {
                "path": str(model_path.resolve()),
                "sha256": file_sha256(model_path),
            },
            "source_states_npz": {
                "path": str(states_path.resolve()),
                "sha256": file_sha256(states_path),
            },
            "observation_state_sha256": array_sha256(original_state),
            "action_sha256": array_sha256(original_action),
            "annotation_remap": {
                str(key): value for key, value in sorted(local_map.items())
            },
        }
        source_records.append(source_record)
        write_json(
            extras_dir
            / "source-records"
            / f"episode_{episode_index:06d}.json",
            source_record,
        )

    with (meta_dir / "episodes.jsonl").open("w") as stream:
        for record in episode_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(extras_dir / "base-episode-stats.json", episode_stats)
    write_json(
        extras_dir / "source-records.json",
        {"episodes": source_records},
    )

    exemplar_task = selection["tasks"][0]
    exemplar_info = json.loads(
        (source_root(exemplar_task) / "meta/info.json").read_text()
    )
    info = {
        **exemplar_info,
        "codebase_version": "v2.1",
        "robot_type": "PandaOmron",
        "total_episodes": len(selection["episodes"]),
        "total_frames": int(selection["selected_frames"]),
        "total_tasks": len(selection["tasks"]),
        "total_videos": len(selection["episodes"]) * len(VIDEO_KEYS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{len(selection['episodes'])}"},
    }
    write_json(meta_dir / "info.json", info)
    for name in ("modality.json", "embodiment.json"):
        source = source_root(exemplar_task) / "meta" / name
        if source.exists():
            shutil.copy2(source, meta_dir / name)

    dataset_meta = {
        "dataset_kind": "robocasa_scene4_certified_clutter",
        "scene": selection["scene"],
        "tasks": selection["tasks"],
        "fps": FPS,
        "source_robot": "PandaOmron",
        "certificate_version": "quickdraw-scene4-direct-occupancy-v1",
        "clutter_protocol_version": "quickdraw-scene4-composite-v10",
        "action_application_during_certification": "none",
        "action_application_during_rendering": "none",
        "source_observation_state_unchanged": True,
        "source_actions_unchanged": True,
        "annotation_codes_globally_remapped": True,
    }
    write_json(extras_dir / "dataset_meta.json", dataset_meta)

    global_stats = {
        name: moments.record() for name, moments in sorted(aggregate.items())
    }
    write_json(extras_dir / "base-stats.json", global_stats)
    normalization = {
        name: {
            "mean": global_stats[name]["mean"],
            "std": [
                float(value) + 1e-6 for value in global_stats[name]["std"]
            ],
        }
        for name in ("observation.state", "action")
    }
    write_json(output_root / "normalization_stats.json", normalization)
    card = {
        "dataset_kind": dataset_meta["dataset_kind"],
        "scene": selection["scene"],
        "splits": {
            "train": {
                "episodes": len(selection["episodes"]),
                "frames": selection["selected_frames"],
                "hours": selection["selected_hours"],
            }
        },
        "tasks": selection["tasks"],
        "fps": FPS,
        "features": info["features"],
        "status": "awaiting-render-finalization",
    }
    write_json(output_root / "dataset_card.json", card)
    summary = {
        "dataset_kind": dataset_meta["dataset_kind"],
        "dataset_root": str(output_root.resolve()),
        "status": "prepared",
        "counts": {
            "train": {
                "episodes": len(selection["episodes"]),
                "transitions": selection["selected_frames"],
                "seconds": selection["selected_frames"] / FPS,
                "hours": selection["selected_hours"],
            }
        },
        "tasks": selection["tasks"],
        "scene": selection["scene"],
        "selection": str((output_root / "selection.json").resolve()),
    }
    write_json(output_root / "summary.json", summary)
    return summary


def image_stats_from_result(
    result: dict[str, Any], key: str
) -> tuple[dict[str, Any], Moments]:
    raw = result["image_moments_uint8"][key]
    pixel_count = int(raw["pixel_count"])
    moments = Moments()
    moments.count = pixel_count
    moments.total = np.asarray(raw["sum"], dtype=np.float64) / 255.0
    moments.square_total = (
        np.asarray(raw["sum_squares"], dtype=np.float64) / (255.0 ** 2)
    )
    moments.minimum = np.asarray(raw["min"], dtype=np.float64) / 255.0
    moments.maximum = np.asarray(raw["max"], dtype=np.float64) / 255.0
    record = moments.record(count_override=int(result["frames"]))
    record = {
        field: [[[value]] for value in values]
        if field != "count"
        else values
        for field, values in record.items()
    }
    return record, moments


def finalize_package(output_root: Path) -> dict[str, Any]:
    selection = json.loads((output_root / "selection.json").read_text())
    split_root = output_root / "train"
    meta_dir = split_root / "meta"
    extras_dir = split_root / "extras"
    base_episode_stats = {
        int(record["episode_index"]): record
        for record in json.loads(
            (extras_dir / "base-episode-stats.json").read_text()
        )
    }
    global_stats = json.loads((extras_dir / "base-stats.json").read_text())
    global_image_moments = {key: Moments() for key in VIDEO_KEYS}
    render_records = []
    for item in selection["episodes"]:
        index = int(item["episode_index"])
        result_path = (
            extras_dir / "render-results" / f"episode_{index:06d}.json"
        )
        if not result_path.exists():
            raise FileNotFoundError(f"missing render result {result_path}")
        result = json.loads(result_path.read_text())
        if (
            result.get("status") != "complete"
            or int(result["frames"]) != int(item["frames"])
            or result["certificate_sha256"] != item["certificate_sha256"]
        ):
            raise ValueError(f"invalid render result {result_path}")
        record = base_episode_stats[index]
        for key in VIDEO_KEYS:
            image_record, moments = image_stats_from_result(result, key)
            record["stats"][key] = image_record
            merge_moments(global_image_moments[key], moments)
        render_records.append(result)

    with (meta_dir / "episodes_stats.jsonl").open("w") as stream:
        for index in sorted(base_episode_stats):
            stream.write(json.dumps(base_episode_stats[index], sort_keys=True) + "\n")
    for key, moments in global_image_moments.items():
        record = moments.record(count_override=int(selection["selected_frames"]))
        global_stats[key] = {
            field: [[[value]] for value in values]
            if field != "count"
            else values
            for field, values in record.items()
        }
    write_json(meta_dir / "stats.json", global_stats)

    video_bytes = sum(
        int(video["bytes"])
        for result in render_records
        for video in result["videos"].values()
    )
    info = json.loads((meta_dir / "info.json").read_text())
    expected_videos = info["total_episodes"] * len(VIDEO_KEYS)
    actual_videos = len(
        list((split_root / "videos").glob("chunk-*/*/episode_*.mp4"))
    )
    if actual_videos != expected_videos:
        raise ValueError(
            f"video count mismatch: {actual_videos} vs {expected_videos}"
        )

    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "status": "complete",
            "video_streams": actual_videos,
            "video_bytes": video_bytes,
            "certification": {
                "certificate_version": "quickdraw-scene4-direct-occupancy-v1",
                "clutter_protocol_version": "quickdraw-scene4-composite-v10",
                "interpolation_factor": 4,
                "effective_hz": 80,
                "penetration_tolerance_m": 0.001,
                "all_tasks_covered": True,
            },
        }
    )
    write_json(summary_path, summary)
    card_path = output_root / "dataset_card.json"
    card = json.loads(card_path.read_text())
    card["status"] = "complete"
    write_json(card_path, card)

    task_rows = "\n".join(f"- `{task}`" for task in selection["tasks"])
    readme = f"""---
license: cc-by-4.0
pretty_name: QuickDraw RoboCasa Scene 4 Certified Clutter
tags:
- robotics
- robocasa
- lerobot
- world-models
configs:
  - config_name: train
    data_files: train/data/**/*.parquet
---

# QuickDraw RoboCasa Scene 4 Certified Clutter

A {selection['selected_hours']:.3f}-hour, {len(selection['episodes'])}-episode
LeRobot v2.1 training split in exact RoboCasa scene `(4, 4)`.

Derived from the RoboCasa assets and target-human demonstrations, distributed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Generated
with QuickDraw; the QuickDraw and RoboCasa codebases are MIT-licensed.

The active source observation states and 12-dimensional recorded actions are
unchanged. Other task inventories were sampled by their own deterministic
RoboCasa initializers. Each episode passed direct 80 Hz swept-occupancy
inspection with a 1 mm numerical penetration tolerance. Rendering loads stored
states directly and applies no action or simulation step.

## Tasks

{task_rows}

See `selection.json`, `train/extras/certificates/`, and
`train/extras/source-records/` for episode-level provenance.
"""
    (output_root / "README.md").write_text(readme)
    (output_root / "LICENSE_DATA.md").write_text(
        "RoboCasa assets and datasets are licensed under Creative Commons "
        "Attribution 4.0 International (CC BY 4.0).\n\n"
        "RoboCasa Team, RoboCasa: Large-Scale Simulation of Everyday Tasks "
        "for Generalist Robots. https://robocasa.ai/\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--target-frames", type=int, default=TARGET_FRAMES)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_package(
            args.summary.resolve(), args.output.resolve(), args.target_frames
        )
    else:
        result = finalize_package(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
