#!/usr/bin/env python3
"""Certify the conservative eight-task Scene-4 corpus and freeze accepted models."""

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
from scripts.run_robocasa_state_trace_targeted import repair_task

RUNTIME = REPO_ROOT / ".cache/robocasa-data-generation/render-runtime/bin/python"
CERTIFIER = REPO_ROOT / "scripts/robocasa_state_trace_certificate.py"
SELECTED_TASKS = (
    "GetToastedBread",
    "DeliverStraw",
    "WashFruitColander",
    "PrepareCoffee",
    "MakeIceLemonade",
    "GatherTableware",
    "HeatKebabSandwich",
    "LoadDishwasher",
)
WARM_ROOTS = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-targeted-v1",
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-pool-v1",
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-eight-artifact-v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_artifact(result: dict[str, Any]) -> bool:
    frozen = result.get("render_artifact")
    if not isinstance(frozen, dict):
        return False
    path = Path(frozen.get("path", ""))
    return path.exists() and file_sha256(path) == frozen.get("sha256")


def jobs() -> list[dict[str, Any]]:
    values = [
        job for job in native_pool() if job["task"] in SELECTED_TASKS
    ]
    leaders = []
    remaining = []
    seen = set()
    for job in values:
        if job["task"] in seen:
            remaining.append(job)
        else:
            seen.add(job["task"])
            leaders.append(job)
    return sorted(leaders, key=lambda value: value["task"]) + remaining


def warm_certificate(job: dict[str, Any]) -> dict[str, Any] | None:
    patterns = (
        f"{job['task']}-{job['episode']}-proposal*.jsonl",
        f"{job['task']}-{job['episode']}-attempt*.jsonl",
    )
    for root in WARM_ROOTS:
        for pattern in patterns:
            for path in sorted(root.glob(pattern), reverse=True):
                result = final_record(path)
                if result and result.get("accepted") is True:
                    return result
    return None


def configuration_key(attempts: dict[str, int]) -> str:
    payload = json.dumps(attempts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def certificate_command(
    job: dict[str, Any],
    inactive_tasks: list[str],
    attempts: dict[str, int],
    interpolation_factor: int,
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
    for task in inactive_tasks:
        command.extend(["--inactive-task", task])
    for task, attempt in sorted(attempts.items()):
        command.extend(["--inactive-attempt", f"{task}={attempt}"])
    return command


def run_configuration(
    job: dict[str, Any],
    inactive_tasks: list[str],
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
    if existing is not None and (
        existing.get("accepted") is not True or valid_artifact(existing)
    ):
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
            certificate_command(
                job,
                inactive_tasks,
                attempts,
                interpolation_factor,
            ),
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
            "error_type": "EightTaskPoolWorkerError",
            "error": f"certificate exited {completed.returncode} without result",
            "task": job["task"],
            "episode": job["episode"],
        }
    return result, path


def run_job(
    job: dict[str, Any],
    max_proposals: int,
    interpolation_factor: int,
    output: Path,
) -> dict[str, Any]:
    inactive_tasks = [task for task in SELECTED_TASKS if task != job["task"]]
    attempts = {task: 0 for task in inactive_tasks}
    warm = warm_certificate(job)
    if warm is not None:
        for task in inactive_tasks:
            attempts[task] = int(warm["inactive_attempts"].get(task, 0))
    history = []
    started = time.perf_counter()
    for proposal in range(max_proposals):
        result, path = run_configuration(
            job,
            inactive_tasks,
            attempts,
            proposal,
            interpolation_factor,
            output,
        )
        history.append(
            {
                "proposal": proposal,
                "attempts": dict(attempts),
                "certificate": str(path),
                "accepted": bool(result.get("accepted")),
                "artifact": valid_artifact(result),
                "event": result.get("event"),
                "error_type": result.get("error_type"),
                "initialization_compatible": result.get(
                    "initialization_compatible"
                ),
                "trajectory_intersection_count": result.get(
                    "trajectory_intersection_count"
                ),
            }
        )
        if result.get("accepted") is True and valid_artifact(result):
            return {
                **job,
                "accepted": True,
                "certificate": str(path.resolve()),
                "render_artifact": result["render_artifact"],
                "attempts": dict(attempts),
                "warm_started": warm is not None,
                "proposals": proposal + 1,
                "history": history,
                "wall_seconds": time.perf_counter() - started,
            }
        task = repair_task(result, inactive_tasks, attempts)
        if task is None:
            break
        attempts[task] += 1
    return {
        **job,
        "accepted": False,
        "warm_started": warm is not None,
        "proposals": len(history),
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def summary_payload(
    total: int, completed: list[dict[str, Any]], target_frames: int
) -> dict[str, Any]:
    accepted = [item for item in completed if item.get("accepted")]
    by_task = Counter(item["task"] for item in accepted)
    frames_by_task = Counter()
    for item in accepted:
        frames_by_task[item["task"]] += int(item["frames"])
    frames = sum(int(item["frames"]) for item in accepted)
    return {
        "selected_tasks": list(SELECTED_TASKS),
        "excluded_structural_conflict": {
            "task": "StackBowlsCabinet",
            "with": "GatherTableware",
            "reason": (
                "Stack destination cabinet can be occupied by GatherTableware; "
                "coexistence depends on favorable episode fixture choices"
            ),
        },
        "jobs_total": total,
        "jobs_completed": len(completed),
        "jobs_accepted": len(accepted),
        "target_frames": target_frames,
        "accepted_frames": frames,
        "accepted_seconds": frames / 20,
        "accepted_hours": frames / 20 / 3600,
        "accepted_by_task": dict(sorted(by_task.items())),
        "accepted_frames_by_task": dict(sorted(frames_by_task.items())),
        "all_tasks_covered": set(SELECTED_TASKS).issubset(by_task),
        "all_accepted_have_same_process_artifacts": all(
            valid_artifact(final_record(Path(item["certificate"])) or {})
            for item in accepted
        ),
        "results": sorted(
            completed, key=lambda item: (item["task"], int(item["episode"]))
        ),
    }


def write_summary(
    output: Path,
    total: int,
    completed: list[dict[str, Any]],
    target_frames: int,
) -> dict[str, Any]:
    payload = summary_payload(total, completed, target_frames)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "summary.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--interpolation-factor", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=288_000)
    args = parser.parse_args()
    pool = jobs()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native-pool.json").write_text(
        json.dumps(
            {
                "selected_tasks": list(SELECTED_TASKS),
                "jobs": pool,
                "frames": sum(int(job["frames"]) for job in pool),
            },
            indent=2,
        )
        + "\n"
    )
    completed = []
    pending = iter(pool)
    futures = {}
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(min(args.workers, len(pool))):
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
                    args.output, len(pool), completed, args.target_frames
                )
                print(
                    json.dumps(
                        {
                            "event": "eight_task_progress",
                            "elapsed_seconds": time.perf_counter() - started,
                            "latest": {
                                key: result.get(key)
                                for key in (
                                    "task",
                                    "episode",
                                    "frames",
                                    "accepted",
                                    "warm_started",
                                    "proposals",
                                    "wall_seconds",
                                )
                            },
                            **{
                                key: summary[key]
                                for key in (
                                    "jobs_completed",
                                    "jobs_accepted",
                                    "accepted_frames",
                                    "accepted_hours",
                                    "accepted_by_task",
                                    "all_tasks_covered",
                                )
                            },
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
    summary = write_summary(
        args.output, len(pool), completed, args.target_frames
    )
    return 0 if (
        summary["accepted_frames"] >= args.target_frames
        and summary["all_tasks_covered"]
        and summary["all_accepted_have_same_process_artifacts"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
