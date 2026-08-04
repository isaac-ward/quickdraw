#!/usr/bin/env python3
"""Render a frozen certified RoboCasa package in parallel, then finalize metadata."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / ".cache/robocasa-data-generation/render-runtime/bin/python"
RENDERER = REPO_ROOT / "scripts/render_robocasa_certified_episode.py"
PACKAGER = REPO_ROOT / "scripts/package_robocasa_certified.py"
DEFAULT_PACKAGE = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/"
    "quickdraw-hf-robocasa-scene4-certified-4h"
)
DEFAULT_LOGS = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/"
    "quickdraw-hf-robocasa-scene4-certified-4h-render-logs"
)


def environment(**updates: str) -> dict[str, str]:
    value = dict(os.environ)
    value.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        **updates,
    )
    return value


def run_command(
    command: list[str], env: dict[str, str], log_path: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument(
        "--egl-devices",
        default="0,1",
        help="EGL device IDs corresponding positionally to --gpus",
    )
    args = parser.parse_args()
    package = args.package.resolve()
    logs = args.logs.resolve()
    selection = json.loads((package / "selection.json").read_text())
    episodes = selection["episodes"]
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    egl_devices = [
        value.strip() for value in args.egl_devices.split(",") if value.strip()
    ]
    if not gpus:
        raise ValueError("at least one GPU is required")
    if len(egl_devices) != len(gpus):
        raise ValueError("--gpus and --egl-devices must have the same length")

    jobs = []
    for ordinal, item in enumerate(episodes):
        index = int(item["episode_index"])
        gpu = gpus[ordinal % len(gpus)]
        egl_device = egl_devices[ordinal % len(egl_devices)]
        jobs.append(
            (
                [
                    str(RUNTIME),
                    str(RENDERER),
                    item["render_artifact"]["path"],
                    str(package / "train"),
                    str(index),
                    "--certificate",
                    item["certificate"],
                ],
                environment(
                    MUJOCO_GL="egl",
                    CUDA_VISIBLE_DEVICES=gpu,
                    MUJOCO_EGL_DEVICE_ID=egl_device,
                ),
                logs / f"episode_{index:06d}.log",
            )
        )

    started = time.perf_counter()
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_command, command, env, log): index
            for index, (command, env, log) in enumerate(jobs)
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            result = future.result()
            completed_count += 1
            if result["returncode"]:
                failures.append({"index": index, **result})
            print(
                json.dumps(
                    {
                        "event": "render_pool_progress",
                        "completed": completed_count,
                        "total": len(jobs),
                        "failures": len(failures),
                        "elapsed_seconds": time.perf_counter() - started,
                        "latest_index": index,
                        "latest_wall_seconds": result["wall_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        raise RuntimeError(f"render failures: {json.dumps(failures[:5], indent=2)}")

    completed = subprocess.run(
        [
            str(RUNTIME),
            str(PACKAGER),
            "finalize",
            "--output",
            str(package),
        ],
        cwd=REPO_ROOT,
        env=environment(MUJOCO_GL="disable"),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"package finalization exited {completed.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
