#!/usr/bin/env python3
"""Validate a completed QuickDraw RoboCasa package and write validation.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ANNOTATION_COLUMNS = (
    "annotation.human.task_description",
    "annotation.human.task_name",
    "annotation.human.subtask",
    "annotation.human.subtask_name",
    "annotation.human.subtask_stage",
    "task_index",
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


def column_numpy(table, name: str) -> np.ndarray:
    values = table[name].to_pylist()
    if values and isinstance(values[0], (list, tuple, np.ndarray)):
        return np.asarray(values)
    return np.asarray(values)


def final_certificate(path: Path) -> dict[str, Any]:
    result = None
    for line in path.read_text(errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") in {"result", "error"}:
            result = value
    if result is None:
        raise ValueError(f"certificate has no final record: {path}")
    return result


def video_info(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}")
    return streams[0]


def validate(root: Path) -> dict[str, Any]:
    selection = json.loads((root / "selection.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    info = json.loads((root / "train/meta/info.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("package summary is not complete")
    if int(selection["selected_frames"]) < int(selection["target_frames"]):
        raise ValueError("selection is shorter than target")
    if len(selection["episodes"]) != int(info["total_episodes"]):
        raise ValueError("episode count mismatch")
    if int(selection["selected_frames"]) != int(info["total_frames"]):
        raise ValueError("frame count mismatch")

    task_records = [
        json.loads(line)
        for line in (root / "train/meta/tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    valid_codes = {int(record["task_index"]) for record in task_records}
    if valid_codes != set(range(len(task_records))):
        raise ValueError("task codebook is not contiguous")

    expected_index = 0
    canonical_schema = None
    video_count = 0
    video_bytes = 0
    certificate_hashes = []
    artifact_hashes = []
    for item in selection["episodes"]:
        episode_index = int(item["episode_index"])
        frames = int(item["frames"])
        parquet_path = (
            root
            / "train/data/chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet_path)
        if table.num_rows != frames:
            raise ValueError(f"row count mismatch in {parquet_path}")
        if canonical_schema is None:
            canonical_schema = table.schema
        elif not table.schema.equals(canonical_schema, check_metadata=True):
            raise ValueError(f"schema mismatch in {parquet_path}")
        if set(table["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"episode_index mismatch in {parquet_path}")
        if table["frame_index"].to_pylist() != list(range(frames)):
            raise ValueError(f"frame_index mismatch in {parquet_path}")
        if table["index"].to_pylist() != list(
            range(expected_index, expected_index + frames)
        ):
            raise ValueError(f"global index mismatch in {parquet_path}")
        timestamps = column_numpy(table, "timestamp")
        expected_timestamps = np.arange(frames, dtype=np.float32) / int(
            info["fps"]
        )
        if not np.array_equal(timestamps, expected_timestamps):
            raise ValueError(f"timestamp mismatch in {parquet_path}")
        for name in ANNOTATION_COLUMNS:
            if name not in table.column_names:
                raise ValueError(f"missing normalized annotation {name}")
            if not set(table[name].to_pylist()) <= valid_codes:
                raise ValueError(f"unmapped annotation code in {name}")

        source_record_path = (
            root
            / "train/extras/source-records"
            / f"episode_{episode_index:06d}.json"
        )
        source_record = json.loads(source_record_path.read_text())
        source_table = pq.read_table(source_record["source_parquet"]["path"])
        if file_sha256(Path(source_record["source_parquet"]["path"])) != (
            source_record["source_parquet"]["sha256"]
        ):
            raise ValueError("source parquet hash mismatch")
        output_state = column_numpy(table, "observation.state")
        output_action = column_numpy(table, "action")
        source_state = column_numpy(source_table, "observation.state")
        source_action = column_numpy(source_table, "action")
        if not np.array_equal(output_state, source_state):
            raise ValueError(f"state changed in episode {episode_index}")
        if not np.array_equal(output_action, source_action):
            raise ValueError(f"action changed in episode {episode_index}")
        if array_sha256(output_state) != source_record[
            "observation_state_sha256"
        ]:
            raise ValueError("state provenance hash mismatch")
        if array_sha256(output_action) != source_record["action_sha256"]:
            raise ValueError("action provenance hash mismatch")

        certificate_path = Path(item["certificate"])
        if file_sha256(certificate_path) != item["certificate_sha256"]:
            raise ValueError("certificate hash mismatch")
        certificate = final_certificate(certificate_path)
        if not (
            certificate.get("accepted") is True
            and certificate.get("complete_trace") is True
            and certificate.get("initialization_compatible") is True
            and certificate.get("collision_absent") is True
        ):
            raise ValueError("unaccepted certificate in selection")
        artifact_path = Path(item["render_artifact"]["path"])
        if file_sha256(artifact_path) != item["render_artifact"]["sha256"]:
            raise ValueError("frozen artifact hash mismatch")
        if certificate["render_artifact"]["sha256"] != item[
            "render_artifact"
        ]["sha256"]:
            raise ValueError("certificate/artifact binding mismatch")
        certificate_hashes.append(item["certificate_sha256"])
        artifact_hashes.append(item["render_artifact"]["sha256"])

        render_result_path = (
            root
            / "train/extras/render-results"
            / f"episode_{episode_index:06d}.json"
        )
        render_result = json.loads(render_result_path.read_text())
        if (
            render_result.get("status") != "complete"
            or int(render_result["frames"]) != frames
            or render_result["certificate_sha256"] != item["certificate_sha256"]
        ):
            raise ValueError("render result mismatch")
        for key in VIDEO_KEYS:
            video_path = Path(render_result["videos"][key]["path"])
            if file_sha256(video_path) != render_result["videos"][key]["sha256"]:
                raise ValueError(f"video hash mismatch: {video_path}")
            stream = video_info(video_path)
            expected = {
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 256,
                "height": 256,
                "avg_frame_rate": "20/1",
                "nb_frames": str(frames),
            }
            for name, value in expected.items():
                if stream.get(name) != value:
                    raise ValueError(
                        f"{video_path} {name}: {stream.get(name)} != {value}"
                    )
            video_count += 1
            video_bytes += video_path.stat().st_size
        expected_index += frames

    if expected_index != int(selection["selected_frames"]):
        raise ValueError("validated frame total mismatch")
    if video_count != int(info["total_videos"]):
        raise ValueError("validated video total mismatch")
    if not (root / "train/meta/stats.json").exists():
        raise ValueError("missing aggregate stats")
    normalization = json.loads((root / "normalization_stats.json").read_text())
    for name in ("observation.state", "action"):
        values = np.asarray(normalization[name]["mean"] + normalization[name]["std"])
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite normalization values for {name}")

    report = {
        "status": "valid",
        "root": str(root),
        "episodes": len(selection["episodes"]),
        "frames": expected_index,
        "seconds": expected_index / int(info["fps"]),
        "hours": expected_index / int(info["fps"]) / 3600,
        "tasks": selection["tasks"],
        "parquet_schemas_identical_with_metadata": True,
        "source_states_unchanged": True,
        "source_actions_unchanged": True,
        "annotation_codes_global_and_non_aliasing": True,
        "certificates_accepted_and_hash_bound": len(certificate_hashes),
        "frozen_artifacts_hash_bound": len(artifact_hashes),
        "videos": video_count,
        "video_bytes": video_bytes,
        "video_format": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "width": 256,
            "height": 256,
            "fps": 20,
        },
    }
    temporary = root / "validation.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "validation.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
