#!/usr/bin/env python3
"""Re-certify accepted traces while freezing the exact accepted model in-process."""

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

from scripts.run_robocasa_state_trace_pool import final_record
from scripts.run_robocasa_state_trace_targeted import repair_task
from scripts.robocasa_state_trace_certificate import SELECTED_TASKS

RUNTIME = REPO_ROOT / ".cache/robocasa-data-generation/render-runtime/bin/python"
CERTIFIER = REPO_ROOT / "scripts/robocasa_state_trace_certificate.py"
DEFAULT_INPUT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-targeted-v1/summary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-artifact-targeted-v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_artifact(record: dict[str, Any]) -> bool:
    frozen = record.get("render_artifact")
    if not isinstance(frozen, dict):
        return False
    path = Path(frozen.get("path", ""))
    return (
        path.exists()
        and file_sha256(path) == frozen.get("sha256")
        and json.loads(path.read_text()).get("status") == "complete"
    )


def configuration_key(attempts: dict[str, int]) -> str:
    payload = json.dumps(attempts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def command(
    task: str,
    episode: int,
    attempts: dict[str, int],
    interpolation_factor: int,
) -> list[str]:
    value = [
        str(RUNTIME),
        str(CERTIFIER),
        task,
        str(episode),
        "--augmentation-attempt",
        "0",
        "--interpolation-factor",
        str(interpolation_factor),
    ]
    for inactive_task, attempt in sorted(attempts.items()):
        value.extend(["--inactive-attempt", f"{inactive_task}={attempt}"])
    return value


def run_configuration(
    job: dict[str, Any],
    attempts: dict[str, int],
    proposal: int,
    interpolation_factor: int,
    output: Path,
) -> tuple[dict[str, Any], Path]:
    key = configuration_key(attempts)
    stem = (
        f"{job['task']}-{int(job['episode'])}-"
        f"proposal{proposal:02d}-{key}"
    )
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
            command(
                job["task"],
                int(job["episode"]),
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
            "error_type": "ArtifactPoolWorkerError",
            "error": f"certificate exited {completed.returncode} without a result",
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
    original_path = Path(job["certificate"])
    if not original_path.is_absolute():
        original_path = REPO_ROOT / original_path
    original = final_record(original_path)
    if original is None or original.get("accepted") is not True:
        raise ValueError(f"invalid original certificate {original_path}")
    if valid_artifact(original):
        return {
            **job,
            "accepted": True,
            "reused": True,
            "certificate": str(original_path.resolve()),
            "render_artifact": original["render_artifact"],
            "proposals": 0,
        }

    inactive_tasks = [task for task in SELECTED_TASKS if task != job["task"]]
    attempts = {
        task: int(original["inactive_attempts"][task]) for task in inactive_tasks
    }
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
                "reused": False,
                "certificate": str(path.resolve()),
                "render_artifact": result["render_artifact"],
                "attempts": dict(attempts),
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
        "proposals": len(history),
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def write_summary(
    output: Path,
    jobs: list[dict[str, Any]],
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [item for item in completed if item.get("accepted")]
    by_task = Counter(item["task"] for item in accepted)
    frames_by_task = Counter()
    for item in accepted:
        frames_by_task[item["task"]] += int(item["frames"])
    frames = sum(int(item["frames"]) for item in accepted)
    payload = {
        "jobs_total": len(jobs),
        "jobs_completed": len(completed),
        "jobs_accepted": len(accepted),
        "accepted_frames": frames,
        "accepted_seconds": frames / 20,
        "accepted_hours": frames / 20 / 3600,
        "accepted_by_task": dict(sorted(by_task.items())),
        "accepted_frames_by_task": dict(sorted(frames_by_task.items())),
        "all_tasks_covered": set(SELECTED_TASKS).issubset(by_task),
        "all_accepted_have_same_process_artifacts": all(
            bool(item.get("render_artifact")) for item in accepted
        ),
        "results": sorted(
            completed, key=lambda item: (item["task"], int(item["episode"]))
        ),
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "summary.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--interpolation-factor", type=int, default=4)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    jobs = [item for item in source["results"] if item.get("accepted") is True]
    args.output.mkdir(parents=True, exist_ok=True)
    completed = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_job,
                job,
                args.max_proposals,
                args.interpolation_factor,
                args.output,
            ): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            summary = write_summary(args.output, jobs, completed)
            print(
                json.dumps(
                    {
                        "event": "artifact_certificate_progress",
                        "elapsed_seconds": time.perf_counter() - started,
                        "latest": {
                            key: result.get(key)
                            for key in (
                                "task",
                                "episode",
                                "frames",
                                "accepted",
                                "reused",
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
    summary = write_summary(args.output, jobs, completed)
    if not (
        summary["accepted_frames"] >= 288_000
        and summary["all_tasks_covered"]
        and summary["all_accepted_have_same_process_artifacts"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
