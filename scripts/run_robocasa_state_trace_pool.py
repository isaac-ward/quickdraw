#!/usr/bin/env python3
"""Run direct RoboCasa state-trace certificates in isolated parallel workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / ".cache/robocasa-data-generation/render-runtime/bin/python"
CERTIFIER = REPO_ROOT / "scripts/robocasa_state_trace_certificate.py"
NATIVE_RESULT_DIRS = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/candidate-v25-current-nine-native-all",
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/heat-scene4-native-all",
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/direct-state-trace-pool-v1"
)


def json_records(path: Path):
    for line in path.read_text(errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def final_record(path: Path) -> dict[str, Any] | None:
    result = None
    for value in json_records(path):
        if value.get("event") in {"result", "error"}:
            result = value
    return result


def native_pool() -> list[dict[str, Any]]:
    jobs = []
    seen = set()
    for root in NATIVE_RESULT_DIRS:
        for path in sorted(root.glob("*.jsonl")):
            result = final_record(path)
            if not result or result.get("accepted") is not True:
                continue
            identity = (result["task"], int(result["episode"]))
            if identity in seen:
                continue
            seen.add(identity)
            jobs.append(
                {
                    "task": identity[0],
                    "episode": identity[1],
                    "frames": int(result["frames"]),
                    "native_result": str(path),
                }
            )
    return sorted(jobs, key=lambda job: (-job["frames"], job["task"], job["episode"]))


def command(job: dict[str, Any], attempt: int, interpolation_factor: int) -> list[str]:
    return [
        str(RUNTIME),
        str(CERTIFIER),
        job["task"],
        str(job["episode"]),
        "--augmentation-attempt",
        str(attempt),
        "--interpolation-factor",
        str(interpolation_factor),
    ]


def run_attempt(
    job: dict[str, Any],
    attempt: int,
    interpolation_factor: int,
    output: Path,
) -> dict[str, Any]:
    stem = f"{job['task']}-{job['episode']}-attempt{attempt}"
    result_path = output / f"{stem}.jsonl"
    stderr_path = output / f"{stem}.stderr"
    existing = final_record(result_path) if result_path.exists() else None
    if existing is not None:
        return existing
    temporary = result_path.with_suffix(".jsonl.tmp")
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MUJOCO_GL="disable",
        SDL_VIDEODRIVER="dummy",
        PYTHONHASHSEED="0",
    )
    with temporary.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            command(job, attempt, interpolation_factor),
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    temporary.replace(result_path)
    result = final_record(result_path)
    if result is None:
        result = {
            "event": "error",
            "kind": "pool-worker",
            "task": job["task"],
            "episode": job["episode"],
            "attempt": attempt,
            "returncode": completed.returncode,
            "error": "certificate process produced no final record",
        }
    return result


def run_job(
    job: dict[str, Any],
    attempts: range,
    interpolation_factor: int,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    attempted = []
    for attempt in attempts:
        result = run_attempt(job, attempt, interpolation_factor, output)
        attempted.append(attempt)
        if result.get("accepted") is True:
            return {
                **job,
                "accepted": True,
                "accepted_attempt": attempt,
                "attempted": attempted,
                "certificate": str(
                    output / f"{job['task']}-{job['episode']}-attempt{attempt}.jsonl"
                ),
                "wall_seconds": time.perf_counter() - started,
            }
        if result.get("event") == "error" and result.get("error_type") in {
            "PlacementError",
            "RandomizationError",
        }:
            continue
        if result.get("event") == "error":
            return {
                **job,
                "accepted": False,
                "attempted": attempted,
                "error": result.get("error"),
                "wall_seconds": time.perf_counter() - started,
            }
    return {
        **job,
        "accepted": False,
        "attempted": attempted,
        "wall_seconds": time.perf_counter() - started,
    }


def write_summary(output: Path, completed: list[dict[str, Any]], total: int) -> None:
    accepted = [result for result in completed if result.get("accepted")]
    accepted_by_task = Counter(result["task"] for result in accepted)
    frames_by_task = Counter()
    for result in accepted:
        frames_by_task[result["task"]] += result["frames"]
    payload = {
        "jobs_total": total,
        "jobs_completed": len(completed),
        "jobs_accepted": len(accepted),
        "accepted_frames": sum(result["frames"] for result in accepted),
        "accepted_seconds": sum(result["frames"] for result in accepted) / 20,
        "accepted_by_task": dict(sorted(accepted_by_task.items())),
        "accepted_frames_by_task": dict(sorted(frames_by_task.items())),
        "results": sorted(completed, key=lambda value: (value["task"], value["episode"])),
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--start-attempt", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=16)
    parser.add_argument("--interpolation-factor", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.interpolation_factor < 1:
        raise ValueError("worker, attempt, and interpolation counts must be positive")
    jobs = native_pool()
    if args.limit is not None:
        jobs = jobs[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native-pool.json").write_text(
        json.dumps({"jobs": jobs, "frames": sum(job["frames"] for job in jobs)}, indent=2)
        + "\n"
    )
    attempts = range(args.start_attempt, args.start_attempt + args.max_attempts)
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_job, job, attempts, args.interpolation_factor, args.output
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            write_summary(args.output, completed, len(jobs))
            accepted_frames = sum(
                item["frames"] for item in completed if item.get("accepted")
            )
            print(
                json.dumps(
                    {
                        "event": "pool_progress",
                        "completed": len(completed),
                        "total": len(jobs),
                        "accepted": sum(bool(item.get("accepted")) for item in completed),
                        "accepted_frames": accepted_frames,
                        "accepted_hours": accepted_frames / 20 / 3600,
                        "elapsed_seconds": time.perf_counter() - started,
                        "latest": result,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    write_summary(args.output, completed, len(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
