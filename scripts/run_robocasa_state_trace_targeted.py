#!/usr/bin/env python3
"""Certify the RoboCasa pool by resampling only the conflicting task inventory."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_robocasa_state_trace_pool import final_record, native_pool
from scripts.robocasa_state_trace_certificate import SELECTED_TASKS
RUNTIME = REPO_ROOT / ".cache/robocasa-data-generation/render-runtime/bin/python"
CERTIFIER = REPO_ROOT / "scripts/robocasa_state_trace_certificate.py"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/direct-state-trace-targeted-v1"
)
REUSABLE_ROOTS = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/direct-state-trace-pool-v1",
)


def prioritized_pool() -> list[dict[str, Any]]:
    jobs = native_pool()
    leaders = []
    remaining = []
    seen_tasks = set()
    for job in jobs:
        if job["task"] not in seen_tasks:
            seen_tasks.add(job["task"])
            leaders.append(job)
        else:
            remaining.append(job)
    return sorted(leaders, key=lambda value: value["task"]) + remaining


def reusable_certificate(job: dict[str, Any]) -> Path | None:
    for root in REUSABLE_ROOTS:
        for path in sorted(root.glob(f"{job['task']}-{job['episode']}-attempt*.jsonl")):
            result = final_record(path)
            if result and result.get("accepted") is True:
                return path
    return None


def configuration_key(attempts: dict[str, int]) -> str:
    payload = json.dumps(attempts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def certificate_command(
    job: dict[str, Any], attempts: dict[str, int], interpolation_factor: int
) -> list[str]:
    command = [
        str(RUNTIME),
        str(CERTIFIER),
        job["task"],
        str(job["episode"]),
        "--augmentation-attempt",
        "0",
        "--interpolation-factor",
        str(interpolation_factor),
    ]
    for task, attempt in sorted(attempts.items()):
        command.extend(["--inactive-attempt", f"{task}={attempt}"])
    return command


def run_configuration(
    job: dict[str, Any],
    attempts: dict[str, int],
    proposal: int,
    interpolation_factor: int,
    output: Path,
) -> tuple[dict[str, Any], Path]:
    key = configuration_key(attempts)
    stem = f"{job['task']}-{job['episode']}-proposal{proposal:02d}-{key}"
    path = output / f"{stem}.jsonl"
    stderr_path = output / f"{stem}.stderr"
    existing = final_record(path) if path.exists() else None
    if existing is not None:
        return existing, path
    temporary = path.with_suffix(".jsonl.tmp")
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
            certificate_command(job, attempts, interpolation_factor),
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    temporary.replace(path)
    result = final_record(path)
    if result is None:
        result = {
            "event": "error",
            "kind": "targeted-pool-worker",
            "task": job["task"],
            "episode": job["episode"],
            "returncode": completed.returncode,
            "error": "certificate process produced no final record",
        }
    return result, path


def task_from_geom(record: dict[str, Any]) -> str | None:
    for field in ("extra_geom", "geom1", "geom2"):
        value = record.get(field)
        if isinstance(value, str) and value.startswith("qdclutter__"):
            return value.split("__", 2)[1]
    return None


def repair_task(
    result: dict[str, Any], inactive_tasks: list[str], attempts: dict[str, int]
) -> str | None:
    inactive = set(inactive_tasks)
    explicit = result.get("inactive_task")
    if explicit in inactive:
        return explicit
    scores: Counter[str] = Counter()
    for pair in result.get("initial_collision_pair_counts", []):
        weight = int(pair.get("contacts", 1))
        for field in ("participant1", "participant2"):
            participant = pair.get(field)
            if participant in inactive:
                scores[participant] += weight
    for intersection in result.get("trajectory_intersections", []):
        for field in ("owner1", "owner2"):
            owner = intersection.get(field)
            if owner in inactive:
                scores[owner] += 1
        owner = task_from_geom(intersection)
        if owner in inactive:
            scores[owner] += 1
    if scores:
        return min(
            scores,
            key=lambda task: (-scores[task], attempts[task], task),
        )
    if result.get("event") == "error" and result.get("error_type") not in {
        "PlacementError",
        "RandomizationError",
    }:
        return None
    return min(inactive_tasks, key=lambda task: (attempts[task], task))


def run_job(
    job: dict[str, Any],
    max_proposals: int,
    interpolation_factor: int,
    output: Path,
) -> dict[str, Any]:
    reused = reusable_certificate(job)
    if reused is not None:
        return {
            **job,
            "accepted": True,
            "reused": True,
            "proposals": 0,
            "certificate": str(reused),
            "wall_seconds": 0.0,
        }
    inactive_tasks = [task for task in SELECTED_TASKS if task != job["task"]]
    attempts = {task: 0 for task in inactive_tasks}
    history = []
    started = time.perf_counter()
    for proposal in range(max_proposals):
        result, path = run_configuration(
            job, attempts, proposal, interpolation_factor, output
        )
        history.append(
            {
                "proposal": proposal,
                "attempts": dict(attempts),
                "certificate": str(path),
                "accepted": bool(result.get("accepted")),
                "event": result.get("event"),
                "error_type": result.get("error_type"),
                "initialization_compatible": result.get("initialization_compatible"),
                "trajectory_intersection_count": result.get(
                    "trajectory_intersection_count"
                ),
            }
        )
        if result.get("accepted") is True:
            return {
                **job,
                "accepted": True,
                "reused": False,
                "proposals": proposal + 1,
                "attempts": dict(attempts),
                "certificate": str(path),
                "history": history,
                "wall_seconds": time.perf_counter() - started,
            }
        task = repair_task(result, inactive_tasks, attempts)
        if task is None:
            return {
                **job,
                "accepted": False,
                "proposals": proposal + 1,
                "error": result.get("error"),
                "history": history,
                "wall_seconds": time.perf_counter() - started,
            }
        attempts[task] += 1
    return {
        **job,
        "accepted": False,
        "proposals": max_proposals,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def summary_payload(
    jobs_total: int, completed: list[dict[str, Any]], target_frames: int
) -> dict[str, Any]:
    accepted = [result for result in completed if result.get("accepted")]
    by_task = Counter(result["task"] for result in accepted)
    frames_by_task = Counter()
    for result in accepted:
        frames_by_task[result["task"]] += result["frames"]
    accepted_frames = sum(result["frames"] for result in accepted)
    return {
        "jobs_total": jobs_total,
        "jobs_completed": len(completed),
        "jobs_accepted": len(accepted),
        "target_frames": target_frames,
        "accepted_frames": accepted_frames,
        "accepted_seconds": accepted_frames / 20,
        "accepted_hours": accepted_frames / 20 / 3600,
        "accepted_by_task": dict(sorted(by_task.items())),
        "accepted_frames_by_task": dict(sorted(frames_by_task.items())),
        "all_tasks_covered": set(SELECTED_TASKS).issubset(by_task),
        "results": sorted(completed, key=lambda value: (value["task"], value["episode"])),
    }


def write_summary(
    output: Path, jobs_total: int, completed: list[dict[str, Any]], target_frames: int
) -> dict[str, Any]:
    payload = summary_payload(jobs_total, completed, target_frames)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "summary.json")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--interpolation-factor", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=288_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.max_proposals < 1 or args.interpolation_factor < 1:
        raise ValueError("worker, proposal, and interpolation counts must be positive")
    jobs = prioritized_pool()
    if args.limit is not None:
        jobs = jobs[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native-pool.json").write_text(
        json.dumps({"jobs": jobs, "frames": sum(job["frames"] for job in jobs)}, indent=2)
        + "\n"
    )
    completed: list[dict[str, Any]] = []
    pending = iter(jobs)
    futures: dict[concurrent.futures.Future, dict[str, Any]] = {}
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(min(args.workers, len(jobs))):
            job = next(pending, None)
            if job is not None:
                futures[
                    executor.submit(
                        run_job,
                        job,
                        args.max_proposals,
                        args.interpolation_factor,
                        args.output,
                    )
                ] = job
        while futures:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                futures.pop(future)
                result = future.result()
                completed.append(result)
                summary = write_summary(
                    args.output, len(jobs), completed, args.target_frames
                )
                print(
                    json.dumps(
                        {
                            "event": "targeted_pool_progress",
                            "elapsed_seconds": time.perf_counter() - started,
                            "latest": result,
                            **{key: summary[key] for key in (
                                "jobs_completed",
                                "jobs_accepted",
                                "accepted_frames",
                                "accepted_hours",
                                "accepted_by_task",
                                "all_tasks_covered",
                            )},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                target_met = (
                    summary["accepted_frames"] >= args.target_frames
                    and summary["all_tasks_covered"]
                )
                if not target_met:
                    job = next(pending, None)
                    if job is not None:
                        futures[
                            executor.submit(
                                run_job,
                                job,
                                args.max_proposals,
                                args.interpolation_factor,
                                args.output,
                            )
                        ] = job
    write_summary(args.output, len(jobs), completed, args.target_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
