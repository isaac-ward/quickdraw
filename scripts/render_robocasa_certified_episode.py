#!/usr/bin/env python3
"""Render an exact frozen RoboCasa composite artifact into three MP4 streams.

This EGL process never imports a RoboCasa initializer. It loads the compiled
model produced in certification mode, applies each stored source qpos by the
frozen joint-name mapping, calls MuJoCo forward kinematics, and renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

CAMERAS = (
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
)
VIDEO_KEYS = tuple(f"observation.images.{camera}" for camera in CAMERAS)
FPS = 20
WIDTH = 256
HEIGHT = 256
RENDERER_VERSION = "robocasa-visual-geoms-v1"
VISUAL_GEOM_GROUPS = (0, 1, 1, 0, 0, 0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(states: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(states.dtype).encode())
    digest.update(np.asarray(states.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(states).tobytes())
    return digest.hexdigest()


def ffmpeg_command(path: Path) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-g", str(FPS),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]


def completed_result(
    result_path: Path, artifact_sha256: str, episode_index: int
) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("artifact_sha256") != artifact_sha256
        or result.get("episode_index") != episode_index
        or result.get("status") != "complete"
        or result.get("renderer_version") != RENDERER_VERSION
    ):
        return None
    for video in result.get("videos", {}).values():
        path = Path(video["path"])
        if not path.exists() or file_sha256(path) != video["sha256"]:
            return None
    return result


def render(
    artifact_path: Path,
    output_root: Path,
    episode_index: int,
    certificate_path: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_sha256 = file_sha256(artifact_path)
    artifact = json.loads(artifact_path.read_text())
    certificate_sha256 = None
    certificate_record = None
    if certificate_path is not None:
        certificate_sha256 = file_sha256(certificate_path)
        for line in certificate_path.read_text(errors="ignore").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("event") in {"result", "error"}:
                certificate_record = value
        if certificate_record is None or certificate_record.get("accepted") is not True:
            raise ValueError("render certificate is missing or unaccepted")
        frozen = certificate_record.get("render_artifact", {})
        if (
            Path(frozen.get("path", "")).resolve() != artifact_path
            or frozen.get("sha256") != artifact_sha256
        ):
            raise ValueError("certificate does not identify this frozen artifact")
    if artifact.get("status") != "complete":
        raise ValueError("render artifact is incomplete")
    if (
        "episode_index" in artifact
        and int(artifact["episode_index"]) != episode_index
    ):
        raise ValueError("artifact episode index mismatch")
    for field in ("model", "mapping", "states"):
        path = Path(artifact[field]["path"])
        if not path.exists() or file_sha256(path) != artifact[field]["sha256"]:
            raise ValueError(f"artifact {field} hash mismatch")

    result_path = (
        output_root / "extras" / "render-results" / f"episode_{episode_index:06d}.json"
    )
    prior = completed_result(result_path, artifact_sha256, episode_index)
    if prior is not None:
        print(json.dumps({"event": "render_reused", **prior}, sort_keys=True), flush=True)
        return prior

    model = mujoco.MjModel.from_binary_path(artifact["model"]["path"])
    data = mujoco.MjData(model)
    mapping_data = np.load(artifact["mapping"]["path"])
    baseline_qpos = mapping_data["baseline_qpos"]
    source_indices = mapping_data["source_indices"]
    composite_indices = mapping_data["composite_indices"]
    source_nq = int(mapping_data["source_nq"])
    source_nv = int(mapping_data["source_nv"])
    source_na = int(mapping_data["source_na"])
    states = np.load(artifact["states"]["path"])["states"]
    if state_sha256(states) != artifact["states"]["state_array_sha256"]:
        raise ValueError("source state array hash mismatch")
    expected_width = 1 + source_nq + source_nv + source_na
    if states.shape != (int(artifact["frames"]), expected_width):
        raise ValueError(
            f"state shape {states.shape} does not match artifact "
            f"({artifact['frames']}, {expected_width})"
        )

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    scene_option = mujoco.MjvOption()
    # RoboCasa uses geom group 0 for simplified collision proxies. Rendering
    # those together with groups 1 and 2 produces brightly colored overlays
    # and z-fighting wherever collision and visual surfaces are coplanar.
    scene_option.geomgroup[:] = VISUAL_GEOM_GROUPS
    encoders: dict[str, subprocess.Popen] = {}
    temporary_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    stats = {
        key: {
            "sum": np.zeros(3, dtype=np.float64),
            "sum_squares": np.zeros(3, dtype=np.float64),
            "min": np.full(3, 255, dtype=np.uint8),
            "max": np.zeros(3, dtype=np.uint8),
            "pixel_count": 0,
        }
        for key in VIDEO_KEYS
    }
    try:
        available_cameras = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(model.ncam)
        }
        missing = set(CAMERAS) - available_cameras
        if missing:
            raise ValueError(f"compiled model is missing cameras: {sorted(missing)}")
        for key in VIDEO_KEYS:
            directory = output_root / "videos" / "chunk-000" / key
            directory.mkdir(parents=True, exist_ok=True)
            final_path = directory / f"episode_{episode_index:06d}.mp4"
            temporary_path = directory / f"episode_{episode_index:06d}.partial.mp4"
            temporary_path.unlink(missing_ok=True)
            process = subprocess.Popen(
                ffmpeg_command(temporary_path),
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            encoders[key] = process
            temporary_paths[key] = temporary_path
            final_paths[key] = final_path

        data.qpos[:] = baseline_qpos
        for frame_index, state in enumerate(states):
            source = state[1 : 1 + source_nq]
            data.qpos[composite_indices] = source[source_indices]
            mujoco.mj_forward(model, data)
            for camera, key in zip(CAMERAS, VIDEO_KEYS):
                renderer.update_scene(
                    data, camera=camera, scene_option=scene_option
                )
                frame = np.asarray(renderer.render(), dtype=np.uint8)
                if frame.shape != (HEIGHT, WIDTH, 3):
                    raise ValueError(f"unexpected frame shape {frame.shape} for {key}")
                stream = encoders[key].stdin
                if stream is None:
                    raise RuntimeError(f"ffmpeg stdin unavailable for {key}")
                stream.write(frame.tobytes())
                flat = frame.reshape(-1, 3)
                stats[key]["sum"] += flat.sum(axis=0, dtype=np.uint64)
                stats[key]["sum_squares"] += np.square(
                    flat.astype(np.uint64)
                ).sum(axis=0, dtype=np.uint64)
                stats[key]["min"] = np.minimum(stats[key]["min"], flat.min(axis=0))
                stats[key]["max"] = np.maximum(stats[key]["max"], flat.max(axis=0))
                stats[key]["pixel_count"] += len(flat)
            if (frame_index + 1) % 250 == 0 or frame_index + 1 == len(states):
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "event": "render_progress",
                            "task": artifact["task"],
                            "source_episode": artifact["source_episode"],
                            "episode_index": episode_index,
                            "frames": frame_index + 1,
                            "total_frames": len(states),
                            "frames_per_second": (frame_index + 1) / elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        failures = {}
        for key, process in encoders.items():
            if process.stdin is not None:
                process.stdin.close()
                process.stdin = None
            returncode = process.wait()
            stderr = b"" if process.stderr is None else process.stderr.read()
            if returncode:
                failures[key] = stderr.decode(errors="replace")
        if failures:
            raise RuntimeError(f"ffmpeg failures: {failures}")
        for key in VIDEO_KEYS:
            temporary_paths[key].replace(final_paths[key])

        result = {
            "status": "complete",
            "renderer_version": RENDERER_VERSION,
            "visible_geom_groups": list(VISUAL_GEOM_GROUPS),
            "episode_index": episode_index,
            "task": artifact["task"],
            "source_episode": artifact["source_episode"],
            "frames": len(states),
            "fps": FPS,
            "duration_seconds": len(states) / FPS,
            "state_application": "frozen joint-name qpos map plus MuJoCo forward kinematics",
            "action_application": "none",
            "dynamics_steps": 0,
            "artifact": str(artifact_path.resolve()),
            "artifact_sha256": artifact_sha256,
            "certificate": (
                str(certificate_path)
                if certificate_path is not None
                else artifact.get("certificate")
            ),
            "certificate_sha256": (
                certificate_sha256
                if certificate_sha256 is not None
                else artifact.get("certificate_sha256")
            ),
            "certificate_version": artifact["certificate_version"],
            "clutter_protocol_version": artifact["clutter_protocol_version"],
            "placement_provenance_sha256": artifact[
                "placement_provenance_sha256"
            ],
            "videos": {
                key: {
                    "path": str(final_paths[key].resolve()),
                    "sha256": file_sha256(final_paths[key]),
                    "bytes": final_paths[key].stat().st_size,
                }
                for key in VIDEO_KEYS
            },
            "image_moments_uint8": {
                key: {
                    "sum": values["sum"].tolist(),
                    "sum_squares": values["sum_squares"].tolist(),
                    "min": values["min"].astype(int).tolist(),
                    "max": values["max"].astype(int).tolist(),
                    "pixel_count": int(values["pixel_count"]),
                }
                for key, values in stats.items()
            },
            "wall_seconds": time.perf_counter() - started,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_result = result_path.with_suffix(".json.tmp")
        temporary_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary_result.replace(result_path)
        print(json.dumps({"event": "render_complete", **result}, sort_keys=True), flush=True)
        return result
    except BaseException:
        for process in encoders.values():
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("episode_index", type=int)
    parser.add_argument("--certificate", type=Path)
    args = parser.parse_args()
    if args.episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    render(
        args.artifact.resolve(),
        args.output_root.resolve(),
        args.episode_index,
        None if args.certificate is None else args.certificate.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
