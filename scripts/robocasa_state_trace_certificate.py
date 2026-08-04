#!/usr/bin/env python3
"""Certify clutter-augmented RoboCasa demos from recorded state geometry.

This checker never applies an action and never advances MuJoCo dynamics. It
constructs the deterministic clutter composite, maps each archived source
state into that composite by joint name, calls forward kinematics / collision
detection, and rejects only observed intersections or initialization aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.robocasa_acceptance import DATA_ROOT, emit, load_episode, make_env, reset_to_source
from scripts.robocasa_composite_acceptance import (
    ALL_TARGET_TASKS,
    PROTOCOL_VERSION as CLUTTER_PROTOCOL_VERSION,
    collision_pair_counts,
    parse_inactive_attempts,
    prepare_augmented_episode,
)

CERTIFICATE_VERSION = "quickdraw-scene4-direct-occupancy-v1"
SELECTED_TASKS = (
    "GetToastedBread",
    "DeliverStraw",
    "WashFruitColander",
    "PrepareCoffee",
    "MakeIceLemonade",
    "GatherTableware",
    "StackBowlsCabinet",
    "HeatKebabSandwich",
    "LoadDishwasher",
)
# Ignore contact-depth noise below 1 mm; this is not a positive clearance
# requirement and does not inflate any source geometry.
CONTACT_EPSILON = 1e-3
POSE_EPSILON = 1e-7
ARTIFACT_ROOT = (
    REPO_ROOT
    / ".cache/robocasa-data-generation/acceptance-results/"
    "direct-state-trace-render-artifacts-v1"
)


@dataclass(frozen=True)
class JointLayout:
    name: str
    joint_type: int
    qpos_start: int
    qpos_stop: int


@dataclass(frozen=True)
class StateMap:
    source_nq: int
    source_nv: int
    source_na: int
    source_indices: np.ndarray
    composite_indices: np.ndarray
    quaternion_slices: tuple[slice, ...]


def state_sha256(states: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(states.dtype).encode())
    digest.update(np.asarray(states.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(states).tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_render_artifact(
    env,
    mapping: StateMap,
    result: dict[str, Any],
) -> dict[str, Any]:
    identity = canonical_sha256(
        {
            "certificate_version": result["certificate_version"],
            "clutter_protocol_version": result["clutter_protocol_version"],
            "task": result["task"],
            "episode": result["episode"],
            "inactive_attempts": result["inactive_attempts"],
            "state_sha256": result["state_sha256"],
        }
    )[:16]
    directory = ARTIFACT_ROOT / (
        f"{result['task']}-{int(result['episode']):06d}-{identity}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.mjb"
    mapping_path = directory / "mapping.npz"
    manifest_path = directory / "artifact.json"

    temporary_model = model_path.with_suffix(".mjb.tmp")
    mujoco.mj_saveModel(env.sim.model._model, str(temporary_model), None)
    temporary_model.replace(model_path)

    temporary_mapping = mapping_path.with_suffix(".npz.tmp")
    with temporary_mapping.open("wb") as stream:
        np.savez_compressed(
            stream,
            baseline_qpos=np.array(env.sim.data.qpos, copy=True),
            source_indices=mapping.source_indices,
            composite_indices=mapping.composite_indices,
            source_nq=np.array(mapping.source_nq, dtype=np.int64),
            source_nv=np.array(mapping.source_nv, dtype=np.int64),
            source_na=np.array(mapping.source_na, dtype=np.int64),
        )
    temporary_mapping.replace(mapping_path)

    states_path = (
        DATA_ROOT
        / result["task"]
        / "lerobot"
        / "extras"
        / f"episode_{int(result['episode']):06d}"
        / "states.npz"
    ).resolve()
    artifact = {
        "status": "complete",
        "task": result["task"],
        "source_episode": result["episode"],
        "frames": result["source_frames"],
        "certificate_version": result["certificate_version"],
        "clutter_protocol_version": result["clutter_protocol_version"],
        "inactive_attempts": result["inactive_attempts"],
        "state_sha256": result["state_sha256"],
        "placement_provenance_sha256": canonical_sha256(
            result["placement_provenance"]
        ),
        "model": {
            "path": str(model_path.resolve()),
            "sha256": file_sha256(model_path),
        },
        "mapping": {
            "path": str(mapping_path.resolve()),
            "sha256": file_sha256(mapping_path),
        },
        "states": {
            "path": str(states_path),
            "sha256": file_sha256(states_path),
            "state_array_sha256": result["state_sha256"],
        },
        "action_application": "none",
        "dynamics_steps": 0,
        "emission": "same process and model instance as direct certificate",
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    temporary_manifest.replace(manifest_path)
    return {
        "path": str(manifest_path.resolve()),
        "sha256": file_sha256(manifest_path),
        "model_sha256": artifact["model"]["sha256"],
        "mapping_sha256": artifact["mapping"]["sha256"],
        "placement_provenance_sha256": artifact[
            "placement_provenance_sha256"
        ],
    }


def joint_layout(model) -> tuple[JointLayout, ...]:
    layouts = []
    for joint_id, name in enumerate(model.joint_names):
        start = int(model.jnt_qposadr[joint_id])
        stop = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < model.njnt
            else int(model.nq)
        )
        layouts.append(JointLayout(name, int(model.jnt_type[joint_id]), start, stop))
    return tuple(layouts)


def build_state_map(source_model, composite_model) -> StateMap:
    source_indices: list[int] = []
    composite_indices: list[int] = []
    quaternion_slices: list[slice] = []
    for layout in joint_layout(source_model):
        composite_joint_id = composite_model.joint_name2id(layout.name)
        composite_start = int(composite_model.jnt_qposadr[composite_joint_id])
        composite_stop = (
            int(composite_model.jnt_qposadr[composite_joint_id + 1])
            if composite_joint_id + 1 < composite_model.njnt
            else int(composite_model.nq)
        )
        width = layout.qpos_stop - layout.qpos_start
        if composite_stop - composite_start != width:
            raise ValueError(
                f"joint width changed for {layout.name}: "
                f"source={width}, composite={composite_stop - composite_start}"
            )
        source_indices.extend(range(layout.qpos_start, layout.qpos_stop))
        composite_indices.extend(range(composite_start, composite_stop))
        # MuJoCo joint types: FREE=0, BALL=1, SLIDE=2, HINGE=3.
        if layout.joint_type == 0:
            quaternion_slices.append(slice(layout.qpos_start + 3, layout.qpos_stop))
        elif layout.joint_type == 1:
            quaternion_slices.append(slice(layout.qpos_start, layout.qpos_stop))
    if len(source_indices) != int(source_model.nq):
        raise ValueError(
            f"source joint map covers {len(source_indices)} of {source_model.nq} qpos values"
        )
    return StateMap(
        source_nq=int(source_model.nq),
        source_nv=int(source_model.nv),
        source_na=int(source_model.na),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        composite_indices=np.asarray(composite_indices, dtype=np.int64),
        quaternion_slices=tuple(quaternion_slices),
    )


def source_qpos(state: np.ndarray, mapping: StateMap) -> np.ndarray:
    expected = 1 + mapping.source_nq + mapping.source_nv + mapping.source_na
    if state.shape != (expected,):
        raise ValueError(f"expected flattened state width {expected}, got {state.shape}")
    return np.asarray(state[1 : 1 + mapping.source_nq], dtype=np.float64)


def slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = first + alpha * (second - first)
        return value / np.linalg.norm(value)
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return (
        np.sin((1.0 - alpha) * theta) / sin_theta * first
        + np.sin(alpha * theta) / sin_theta * second
    )


def interpolate_qpos(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
    quaternion_slices: tuple[slice, ...],
) -> np.ndarray:
    result = first + alpha * (second - first)
    for quaternion_slice in quaternion_slices:
        result[quaternion_slice] = slerp_wxyz(
            first[quaternion_slice], second[quaternion_slice], alpha
        )
    return result


def apply_source_qpos(env, mapping: StateMap, qpos: np.ndarray) -> None:
    env.sim.data.qpos[mapping.composite_indices] = qpos[mapping.source_indices]
    env.sim.forward()


def geom_name(model, geom_id: int) -> str:
    return model.geom_id2name(int(geom_id)) or f"geom#{int(geom_id)}"


def source_geometry_alignment_error(source_env, composite_env) -> float:
    """Verify that named source geometry is unchanged by adding clutter."""
    maximum = 0.0
    composite_names = set(composite_env.sim.model.geom_names)
    for source_id, name in enumerate(source_env.sim.model.geom_names):
        if not name or name not in composite_names:
            continue
        composite_id = composite_env.sim.model.geom_name2id(name)
        position_error = float(
            np.linalg.norm(
                source_env.sim.data.geom_xpos[source_id]
                - composite_env.sim.data.geom_xpos[composite_id]
            )
        )
        rotation_error = float(
            np.linalg.norm(
                source_env.sim.data.geom_xmat[source_id]
                - composite_env.sim.data.geom_xmat[composite_id]
            )
        )
        maximum = max(maximum, position_error, rotation_error)
    return maximum


def direct_initial_collisions(collisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep observed penetrations; ignore merely close contact candidates."""
    return [
        collision
        for collision in collisions
        if float(collision.get("distance", 0.0)) < -CONTACT_EPSILON
    ]


def geometry_partitions(env, placements: list[dict[str, Any]]) -> dict[str, Any]:
    model = env.sim.model
    extra_owners: dict[int, str] = {}
    for object_model in env.objects.values():
        if not object_model.name.startswith("qdclutter__"):
            continue
        owner = object_model.name.split("__", 2)[1]
        for name in object_model.contact_geoms:
            extra_owners[int(model.geom_name2id(name))] = owner
    active_ids = set()
    for object_model in env.objects.values():
        if object_model.name.startswith("qdclutter__"):
            continue
        active_ids.update(
            int(model.geom_name2id(name)) for name in object_model.contact_geoms
        )
    robot_ids = {
        int(value) for value in ([] if env.robot_geom_ids is None else env.robot_geom_ids)
    }
    expected_extra_names = {placement["name"] for placement in placements}
    actual_extra_names = {
        object_model.name
        for object_model in env.objects.values()
        if object_model.name.startswith("qdclutter__")
    }
    if expected_extra_names != actual_extra_names:
        raise ValueError(
            "added-object identity mismatch: "
            f"missing={sorted(expected_extra_names - actual_extra_names)}, "
            f"unexpected={sorted(actual_extra_names - expected_extra_names)}"
        )
    return {
        "extra_owners": extra_owners,
        "extra_ids": set(extra_owners),
        "active_ids": active_ids,
        "robot_ids": robot_ids,
    }


def extra_contact_records(env, partitions: dict[str, Any]) -> list[dict[str, Any]]:
    model = env.sim.model
    records = []
    for contact_index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[contact_index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        if first not in partitions["extra_ids"] and second not in partitions["extra_ids"]:
            continue
        records.append(
            {
                "geom1_id": first,
                "geom2_id": second,
                "geom1": geom_name(model, first),
                "geom2": geom_name(model, second),
                "owner1": partitions["extra_owners"].get(first),
                "owner2": partitions["extra_owners"].get(second),
                "distance": float(contact.dist),
            }
        )
    return records


def initial_supports(env, partitions: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    supports = {}
    for record in extra_contact_records(env, partitions):
        first = record["geom1_id"]
        second = record["geom2_id"]
        if first in partitions["extra_ids"] and second not in partitions["extra_ids"]:
            extra, other = first, second
        elif second in partitions["extra_ids"] and first not in partitions["extra_ids"]:
            extra, other = second, first
        else:
            continue
        if other in partitions["active_ids"] or other in partitions["robot_ids"]:
            continue
        if record["distance"] > CONTACT_EPSILON:
            continue
        supports[(extra, other)] = {
            "other_position": np.array(env.sim.data.geom_xpos[other], copy=True),
            "other_rotation": np.array(env.sim.data.geom_xmat[other], copy=True),
        }
    return supports


def observed_intersections(
    env,
    partitions: dict[str, Any],
    supports: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for record in extra_contact_records(env, partitions):
        if record["distance"] >= -CONTACT_EPSILON:
            continue
        first = record["geom1_id"]
        second = record["geom2_id"]
        if first in partitions["extra_ids"] and second not in partitions["extra_ids"]:
            extra, other = first, second
        elif second in partitions["extra_ids"] and first not in partitions["extra_ids"]:
            extra, other = second, first
        elif first in partitions["extra_ids"] and second in partitions["extra_ids"]:
            if record["owner1"] == record["owner2"]:
                continue
            record["reason"] = "cross-task-added-object-intersection"
            failures.append(record)
            continue
        else:
            continue
        if (extra, other) in supports:
            continue
        if other in partitions["robot_ids"]:
            reason = "added-object-robot-intersection"
        elif other in partitions["active_ids"]:
            reason = "added-object-source-object-intersection"
        else:
            reason = "added-object-moving-fixture-intersection"
        record["reason"] = reason
        failures.append(record)
    return failures


def moved_supports(
    env, supports: dict[tuple[int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    model = env.sim.model
    failures = []
    for (extra, other), initial in supports.items():
        position_delta = float(
            np.linalg.norm(env.sim.data.geom_xpos[other] - initial["other_position"])
        )
        rotation_delta = float(
            np.linalg.norm(env.sim.data.geom_xmat[other] - initial["other_rotation"])
        )
        if position_delta <= POSE_EPSILON and rotation_delta <= POSE_EPSILON:
            continue
        failures.append(
            {
                "reason": "added-object-support-moved",
                "extra_geom": geom_name(model, extra),
                "support_geom": geom_name(model, other),
                "position_delta": position_delta,
                "rotation_matrix_delta": rotation_delta,
            }
        )
    return failures


def placement_provenance(placements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task": placement["task"],
            "name": placement["name"],
            "seed": placement["seed"],
            "inventory_source_episode": placement["inventory_source_episode"],
            **placement["provenance"],
        }
        for placement in placements
    ]


def certify_one(
    task: str,
    episode_number: int,
    inactive_tasks: list[str],
    augmentation_attempt: int,
    inactive_attempt_overrides: dict[str, int],
    interpolation_factor: int,
    max_states: int | None,
) -> int:
    started = time.perf_counter()
    source_env = None
    env = None
    try:
        episode = load_episode(task, episode_number)
        states = episode["states"]
        source_env = make_env(episode["dataset_meta"])
        reset_to_source(source_env, episode)
        inactive_attempts = {
            inactive_task: augmentation_attempt for inactive_task in inactive_tasks
        }
        inactive_attempts.update(inactive_attempt_overrides)
        env, composite_episode, placements, raw_initial_collisions = prepare_augmented_episode(
            task, episode_number, inactive_tasks, inactive_attempts
        )
        mapping = build_state_map(source_env.sim.model, env.sim.model)
        partitions = geometry_partitions(env, placements)
        initial_collisions = direct_initial_collisions(raw_initial_collisions)
        state_count = len(states) if max_states is None else min(len(states), max_states)
        complete_trace = state_count == len(states)
        common = {
            "event": "result",
            "kind": "augmented-direct-state-trace-certificate",
            "certificate_version": CERTIFICATE_VERSION,
            "clutter_protocol_version": CLUTTER_PROTOCOL_VERSION,
            "task": task,
            "episode": episode_number,
            "augmentation_attempt": augmentation_attempt,
            "inactive_attempts": inactive_attempts,
            "inactive_tasks": inactive_tasks,
            "added_objects": len(placements),
            "source_frames": len(composite_episode["actions"]),
            "source_states": len(states),
            "checked_states": state_count,
            "complete_trace": complete_trace,
            "interpolation_method": "linear-qpos-with-quaternion-slerp",
            "interpolation_factor": interpolation_factor,
            "effective_hz": 20 * interpolation_factor,
            "penetration_tolerance_m": CONTACT_EPSILON,
            "state_sha256": state_sha256(states),
            "placement_provenance": placement_provenance(placements),
        }
        if initial_collisions:
            result = {
                **common,
                "initialization_compatible": False,
                "initial_collision_count": len(initial_collisions),
                "initial_collision_pair_counts": collision_pair_counts(
                    initial_collisions, task
                ),
                "initial_collisions": initial_collisions[:20],
                "checked_samples": 0,
                "trajectory_intersections": [],
                "accepted": False,
                "wall_seconds": time.perf_counter() - started,
            }
            print(json.dumps(result, sort_keys=True), flush=True)
            return 3

        first_qpos = source_qpos(states[0], mapping)
        apply_source_qpos(env, mapping, first_qpos)
        alignment_error = source_geometry_alignment_error(source_env, env)
        if alignment_error > POSE_EPSILON:
            raise ValueError(
                f"source geometry changed in composite model: max error {alignment_error}"
            )
        supports = initial_supports(env, partitions)
        failures = observed_intersections(env, partitions, supports)
        failures.extend(moved_supports(env, supports))
        checked_samples = 1
        failure_state = 0 if failures else None
        failure_alpha = 0.0 if failures else None
        for state_index in range(max(0, state_count - 1)):
            if failures:
                break
            first = source_qpos(states[state_index], mapping)
            second = source_qpos(states[state_index + 1], mapping)
            for interpolation_index in range(1, interpolation_factor + 1):
                alpha = interpolation_index / interpolation_factor
                qpos = interpolate_qpos(
                    first, second, alpha, mapping.quaternion_slices
                )
                apply_source_qpos(env, mapping, qpos)
                checked_samples += 1
                sample_failures = moved_supports(env, supports)
                sample_failures.extend(
                    observed_intersections(env, partitions, supports)
                )
                if sample_failures:
                    failures.extend(sample_failures[:20])
                    failure_state = state_index
                    failure_alpha = alpha
                    break
            if (state_index + 1) % 250 == 0 or state_index + 2 == state_count:
                elapsed = time.perf_counter() - started
                emit(
                    "certificate_progress",
                    task=task,
                    episode=episode_number,
                    state=state_index + 2,
                    states=state_count,
                    checked_samples=checked_samples,
                    samples_per_second=checked_samples / elapsed,
                )
        result = {
            **common,
            "initialization_compatible": True,
            "initial_collision_count": 0,
            "support_contact_count": len(supports),
            "source_geometry_alignment_max_error": alignment_error,
            "checked_samples": checked_samples,
            "trajectory_intersection_count": len(failures),
            "trajectory_intersections": failures[:20],
            "failure_state": failure_state,
            "failure_alpha": failure_alpha,
            "collision_absent": not failures,
            "accepted": complete_trace and not failures,
        }
        if result["accepted"]:
            result["render_artifact"] = write_render_artifact(
                env, mapping, result
            )
        result["wall_seconds"] = time.perf_counter() - started
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["accepted"] or not complete_trace else 2
    except BaseException as exc:
        emit(
            "error",
            task=task,
            episode=episode_number,
            error_type=type(exc).__name__,
            error=str(exc),
            inactive_task=getattr(exc, "inactive_task", None),
            traceback=traceback.format_exc(),
            wall_seconds=time.perf_counter() - started,
        )
        return 1
    finally:
        if env is not None:
            env.close()
        if source_env is not None:
            source_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=ALL_TARGET_TASKS)
    parser.add_argument("episode", type=int)
    parser.add_argument("--augmentation-attempt", type=int, default=0)
    parser.add_argument("--inactive-task", action="append", choices=ALL_TARGET_TASKS)
    parser.add_argument(
        "--inactive-attempt", action="append", default=[], metavar="TASK=INDEX"
    )
    parser.add_argument("--interpolation-factor", type=int, default=4)
    parser.add_argument("--max-states", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interpolation_factor < 1:
        raise ValueError("--interpolation-factor must be positive")
    if args.max_states is not None and args.max_states < 1:
        raise ValueError("--max-states must be positive")
    inactive_tasks = args.inactive_task or [
        task for task in SELECTED_TASKS if task != args.task
    ]
    if args.task in inactive_tasks:
        raise ValueError("the active task cannot also be inactive clutter")
    return certify_one(
        args.task,
        args.episode,
        inactive_tasks,
        args.augmentation_attempt,
        parse_inactive_attempts(args.inactive_attempt, inactive_tasks),
        args.interpolation_factor,
        args.max_states,
    )


if __name__ == "__main__":
    raise SystemExit(main())
