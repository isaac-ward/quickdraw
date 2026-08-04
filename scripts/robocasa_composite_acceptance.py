#!/usr/bin/env python3
"""Construct deterministic RoboCasa clutter composites.

The active source episode is immutable. Each inactive task is instantiated in
scene (4, 4) with a deterministic per-source seed so its own task initializer
selects object instances, fixture references, reset regions, and placements.
Those placements are transferred through the sampled reset-region frame into
the active source's exact archived fixture geometry and articulated state.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import robocasa  # noqa: F401: registers environments
import robosuite
from robocasa.utils import env_utils as EnvUtils
from robosuite.models.base import MujocoXML
from robosuite.utils import transform_utils as T

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from scripts.robocasa_acceptance import (
    DATA_ROOT,
    REPO_ROOT,
    load_episode,
    make_env,
)


ALL_TARGET_TASKS = tuple(
    sorted(
        path.stem
        for path in (
            REPO_ROOT / ".cache/robocasa-data-generation/target-composite"
        ).glob("*.json")
    )
)
PROTOCOL_VERSION = "quickdraw-scene4-composite-v10"
TEMP_ROOT = REPO_ROOT / ".cache/robocasa-data-generation/composite-runtime"


def deterministic_seed(
    active_task: str,
    episode: int,
    inactive_task: str,
    augmentation_attempt: int = 0,
) -> int:
    identity = (
        f"{PROTOCOL_VERSION}|target-human|{active_task}|{episode}|"
        f"{inactive_task}|attempt={augmentation_attempt}"
    )
    digest = hashlib.sha256(identity.encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


@lru_cache(maxsize=None)
def scene4_source_metadata(task: str) -> tuple[dict[str, Any], ...]:
    path = REPO_ROOT / ".cache/robocasa-data-generation/target-composite" / f"{task}.json"
    with path.open() as handle:
        metadata = json.load(handle)
    episodes = tuple(
        episode
        for episode in metadata["ep_meta"]
        if episode.get("layout_id") == 4 and episode.get("style_id") == 4
    )
    if not episodes:
        raise ValueError(f"no Scene (4, 4) source metadata for {task}")
    return episodes


def inactive_source_metadata(
    active_task: str, episode: int, inactive_task: str, source_attempt: int
) -> tuple[dict[str, Any], int]:
    candidates = scene4_source_metadata(inactive_task)
    identity = (
        f"{PROTOCOL_VERSION}|inventory-source|{active_task}|{episode}|{inactive_task}|attempt={source_attempt}"
    )
    index = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    selected = candidates[index % len(candidates)]
    return deepcopy(selected), int(selected["episode_index"])


def install_deterministic_default_rng(seed: int):
    """Give RoboCasa's otherwise-unseeded child generators stable seeds."""
    original_default_rng = np.random.default_rng
    deterministic_child_seed = seed ^ 0xA5A5A5A5

    def default_rng(child_seed=None):
        if child_seed is None:
            child_seed = deterministic_child_seed
        return original_default_rng(child_seed)

    np.random.default_rng = default_rng
    return original_default_rng


def task_dataset_meta(task: str) -> dict[str, Any]:
    path = DATA_ROOT / task / "lerobot/extras/dataset_meta.json"
    if not path.exists():
        path = DATA_ROOT / "GetToastedBread/lerobot/extras/dataset_meta.json"
    with path.open() as handle:
        metadata = json.load(handle)
    metadata["env_args"]["env_name"] = task
    metadata["env_args"]["env_kwargs"]["env_name"] = task
    return metadata


def make_inventory_env(task: str, seed: int):
    kwargs = dict(task_dataset_meta(task)["env_args"]["env_kwargs"])
    kwargs.update(
        layout_ids=None,
        style_ids=None,
        layout_and_style_ids=[[4, 4]],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        seed=seed,
    )
    return robosuite.make(**kwargs)


def reset_to_source(env, episode: dict[str, Any]) -> str:
    env.set_ep_meta(episode["ep_meta"])
    env.reset()
    source_xml = env.edit_model_xml(episode["model_xml"])
    env.reset_from_xml_string(source_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(episode["states"][0])
    env.sim.forward()
    env.update_state()
    return source_xml


def namespace_object_cfg(
    cfg: dict[str, Any], inactive_task: str, object_names: set[str]
) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    prefix = f"qdclutter__{inactive_task}__"

    def rename(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, tuple):
            return tuple(rename(item) for item in value)
        if isinstance(value, str) and value in object_names:
            return prefix + value
        return value

    return rename(cfg)


def geom_transform(env, geom_name: str) -> tuple[np.ndarray, np.ndarray]:
    geom_id = env.sim.model.geom_name2id(geom_name)
    position = np.array(env.sim.data.geom_xpos[geom_id], dtype=float)
    rotation = np.array(env.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    return position, rotation


def fixture_body_transform(
    env, fixture
) -> tuple[np.ndarray, np.ndarray, str]:
    body_name = fixture.root_body
    body_id = env.sim.model.body_name2id(body_name)
    position = np.array(env.sim.data.body_xpos[body_id], dtype=float)
    rotation = np.array(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return position, rotation, body_name


def fixture_region_transform(
    env, fixture, region_name: str
) -> tuple[np.ndarray, np.ndarray, str]:
    geom_name = f"{fixture.naming_prefix}reg_{region_name}"
    if geom_name in env.sim.model.geom_names:
        position, rotation = geom_transform(env, geom_name)
        return position, rotation, geom_name
    return fixture_body_transform(env, fixture)


def placement_in_source_region(
    source_env,
    inventory_env,
    cfg: dict[str, Any],
    sampled_position: np.ndarray,
    sampled_quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    placement = cfg.get("placement", {})
    fixture_name = placement.get("fixture")
    reset_region = cfg.get("reset_region")
    if not isinstance(fixture_name, str) or not reset_region:
        raise NotImplementedError(
            f"only fixture-root placements are supported: {cfg['name']}"
        )
    region_name = reset_region["name"]
    reference_fixture_name = (
        placement.get("sample_region_kwargs", {}).get("ref") or fixture_name
    )
    inventory_fixture = inventory_env.get_fixture(reference_fixture_name)
    source_fixture = source_env.get_fixture(reference_fixture_name)
    if inventory_fixture is None or source_fixture is None:
        raise ValueError(
            f"fixture {reference_fixture_name!r} is not shared by source and inactive task"
        )
    if reference_fixture_name == fixture_name:
        inventory_position, inventory_rotation, inventory_reference = (
            fixture_region_transform(inventory_env, inventory_fixture, region_name)
        )
        source_position, source_rotation, source_reference = (
            fixture_region_transform(source_env, source_fixture, region_name)
        )
    else:
        inventory_position, inventory_rotation, inventory_reference = (
            fixture_body_transform(inventory_env, inventory_fixture)
        )
        source_position, source_rotation, source_reference = fixture_body_transform(
            source_env, source_fixture
        )

    object_rotation = T.quat2mat(
        T.convert_quat(sampled_quaternion_wxyz, to="xyzw")
    )
    relative_position = inventory_rotation.T @ (
        sampled_position - inventory_position
    )
    relative_rotation = inventory_rotation.T @ object_rotation
    mapped_position = source_position + source_rotation @ relative_position
    mapped_rotation = source_rotation @ relative_rotation
    mapped_quaternion_wxyz = T.convert_quat(
        T.mat2quat(mapped_rotation), to="wxyz"
    )
    provenance = {
        "fixture": fixture_name,
        "region": region_name,
        "inventory_reference": inventory_reference,
        "source_reference": source_reference,
        "relative_position": relative_position.tolist(),
    }
    return mapped_position, mapped_quaternion_wxyz, provenance


def placement_on_mapped_object(
    inventory_env,
    inactive_task: str,
    cfg: dict[str, Any],
    sampled_position: np.ndarray,
    sampled_quaternion_wxyz: np.ndarray,
    mapped_by_original: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = cfg.get("placement", {}).get("sample_args", {}).get("reference")
    if not isinstance(reference, str) or reference not in mapped_by_original:
        raise ValueError(f"nested placement reference is unavailable: {cfg['name']}")
    parent_position, parent_quaternion, _ = inventory_env.object_placements[
        reference
    ]
    parent_rotation = T.quat2mat(
        T.convert_quat(np.asarray(parent_quaternion), to="xyzw")
    )
    object_rotation = T.quat2mat(
        T.convert_quat(sampled_quaternion_wxyz, to="xyzw")
    )
    relative_position = parent_rotation.T @ (
        sampled_position - np.asarray(parent_position)
    )
    relative_rotation = parent_rotation.T @ object_rotation
    mapped_parent = mapped_by_original[reference]
    mapped_parent_rotation = T.quat2mat(
        T.convert_quat(mapped_parent["quaternion"], to="xyzw")
    )
    mapped_position = (
        mapped_parent["position"] + mapped_parent_rotation @ relative_position
    )
    mapped_rotation = mapped_parent_rotation @ relative_rotation
    mapped_quaternion_wxyz = T.convert_quat(
        T.mat2quat(mapped_rotation), to="wxyz"
    )
    return mapped_position, mapped_quaternion_wxyz, {
        "fixture": None,
        "region": None,
        "inventory_reference": f"object:{reference}",
        "source_reference": f"object:qdclutter__{inactive_task}__{reference}",
        "relative_position": relative_position.tolist(),
    }


def sample_inactive_inventory(
    source_env,
    active_task: str,
    episode: int,
    inactive_task: str,
    augmentation_attempt: int,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    seed = deterministic_seed(
        active_task, episode, inactive_task, augmentation_attempt
    )
    random.seed(seed)
    np.random.seed(seed)
    original_default_rng = install_deterministic_default_rng(seed)
    inventory_env = None
    try:
        inventory_env = make_inventory_env(inactive_task, seed)
        source_metadata, source_episode = inactive_source_metadata(
            active_task, episode, inactive_task, augmentation_attempt
        )
        inventory_env.set_ep_meta(source_metadata)
        # Environment construction consumes RNG before the metadata-driven
        # reset. Rewind every source used by fixture placement so reconstructing
        # this proposal in another process produces the same fixture obstacles.
        random.seed(seed)
        np.random.seed(seed)
        inventory_env.rng = np.random.default_rng(seed)
        inventory_env.reset()
        # Model construction and object placement normally share env.rng. Some
        # fixture / asset construction paths consume a process-dependent number
        # of samples, which can shift otherwise identical object coordinates.
        # Rebuild and run the task's own placement initializer with an isolated
        # stream so the source metadata still defines the inventory and rules,
        # while a proposal's fresh coordinates are reproducible in parallel.
        inventory_env.rng = np.random.default_rng(seed)
        inventory_env.placement_initializer = EnvUtils._get_placement_initializer(
            inventory_env, inventory_env.object_cfgs
        )
        # Match Kitchen._load_model exactly: the inactive task object sampler
        # sees every fixture from its own Scene-4 source environment. These are
        # collision obstacles only; placement regions and coordinates still
        # come exclusively from the inactive task recorded object configs.
        placement_obstacles = inventory_env.fxtr_placements
        inventory_env.object_placements = inventory_env.placement_initializer.sample(
            placed_objects=placement_obstacles
        )
        scene = (int(inventory_env.layout_id), int(inventory_env.style_id))
        if scene != (4, 4):
            raise ValueError(f"inactive inventory initialized in scene {scene}")
        serialized_cfgs = inventory_env.get_ep_meta()["object_cfgs"]
        object_names = {cfg["name"] for cfg in serialized_cfgs}
        models: list[Any] = []
        namespaced_cfgs: list[dict[str, Any]] = []
        placements: list[dict[str, Any]] = []
        mapped_by_original: dict[str, dict[str, np.ndarray]] = {}
        geom_owners: dict[str, str] = {}
        for serialized_cfg in serialized_cfgs:
            original_name = serialized_cfg["name"]
            if original_name not in inventory_env.object_placements:
                raise KeyError(
                    f"inactive placement missing for {inactive_task}/{original_name}"
                )
            sampled_position, sampled_quaternion, _ = (
                inventory_env.object_placements[original_name]
            )
            reference = (
                serialized_cfg.get("placement", {})
                .get("sample_args", {})
                .get("reference")
            )
            if reference is None:
                mapped_position, mapped_quaternion, provenance = placement_in_source_region(
                    source_env,
                    inventory_env,
                    serialized_cfg,
                    np.asarray(sampled_position, dtype=float),
                    np.asarray(sampled_quaternion, dtype=float),
                )
            else:
                mapped_position, mapped_quaternion, provenance = placement_on_mapped_object(
                    inventory_env,
                    inactive_task,
                    serialized_cfg,
                    np.asarray(sampled_position, dtype=float),
                    np.asarray(sampled_quaternion, dtype=float),
                    mapped_by_original,
                )
            mapped_by_original[original_name] = {
                "position": mapped_position,
                "quaternion": mapped_quaternion,
            }
            cfg = namespace_object_cfg(
                serialized_cfg, inactive_task, object_names
            )
            cfg["type"] = "object"
            model, info = EnvUtils.create_obj(source_env, cfg)
            cfg["info"] = info
            models.append(model)
            namespaced_cfgs.append(cfg)
            placements.append(
                {
                    "task": inactive_task,
                    "original_name": original_name,
                    "name": model.name,
                    "joint": model.joints[0],
                    "position": mapped_position,
                    "quaternion": mapped_quaternion,
                    "seed": seed,
                    "inventory_source_episode": source_episode,
                    "placement_obstacles": sorted(placement_obstacles),
                    "provenance": provenance,
                }
            )
            for geom_name in model.contact_geoms:
                geom_owners[geom_name] = inactive_task
        return models, namespaced_cfgs, placements, geom_owners
    finally:
        try:
            if inventory_env is not None:
                inventory_env.close()
        finally:
            np.random.default_rng = original_default_rng


def snapshot_named_state(env) -> dict[str, Any]:
    return {
        "joints": {
            name: (
                np.array(env.sim.data.get_joint_qpos(name), copy=True),
                np.array(env.sim.data.get_joint_qvel(name), copy=True),
            )
            for name in env.sim.model.joint_names
        },
        "time": float(env.sim.data.time),
        "act": np.array(env.sim.data.act, copy=True),
        "ctrl": np.array(env.sim.data.ctrl, copy=True),
        "mocap_pos": np.array(env.sim.data.mocap_pos, copy=True),
        "mocap_quat": np.array(env.sim.data.mocap_quat, copy=True),
    }


def restore_named_state(env, snapshot: dict[str, Any]) -> None:
    available = set(env.sim.model.joint_names)
    for name, (qpos, qvel) in snapshot["joints"].items():
        if name not in available:
            raise KeyError(f"source joint missing from composite model: {name}")
        env.sim.data.set_joint_qpos(name, qpos)
        env.sim.data.set_joint_qvel(name, qvel)
    env.sim.data.time = snapshot["time"]
    if env.sim.data.act.shape == snapshot["act"].shape:
        env.sim.data.act[:] = snapshot["act"]
    if env.sim.data.ctrl.shape == snapshot["ctrl"].shape:
        env.sim.data.ctrl[:] = snapshot["ctrl"]
    if env.sim.data.mocap_pos.shape == snapshot["mocap_pos"].shape:
        env.sim.data.mocap_pos[:] = snapshot["mocap_pos"]
        env.sim.data.mocap_quat[:] = snapshot["mocap_quat"]


def merge_extra_models(source_xml: str, models: list[Any]) -> str:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", dir=TEMP_ROOT, delete=False
        ) as handle:
            handle.write(source_xml)
            path = Path(handle.name)
        merged = MujocoXML(str(path))
        # Attach the free-jointed subtree directly to worldbody, matching
        # robosuite's Task.merge_objects() topology.
        for model in models:
            merged.merge_assets(model)
            merged.worldbody.append(model.get_obj())
        return merged.get_xml()
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def initial_collision_report(
    env,
    active_contact_geoms: set[str],
    extra_geom_owners: dict[str, str],
) -> list[dict[str, Any]]:
    robot_geoms = set()
    if env.robot_geom_ids is not None:
        robot_geoms = {
            env.sim.model.geom_id2name(int(geom_id))
            for geom_id in env.robot_geom_ids
        }
    failures = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        geom1 = env.sim.model.geom_id2name(contact.geom1)
        geom2 = env.sim.model.geom_id2name(contact.geom2)
        owner1 = extra_geom_owners.get(geom1)
        owner2 = extra_geom_owners.get(geom2)
        if owner1 is None and owner2 is None:
            continue
        reason = None
        if owner1 is not None and owner2 is not None and owner1 != owner2:
            reason = "cross-task-extra-contact"
        elif owner1 is not None and geom2 in active_contact_geoms:
            reason = "extra-active-object-contact"
        elif owner2 is not None and geom1 in active_contact_geoms:
            reason = "extra-active-object-contact"
        elif owner1 is not None and geom2 in robot_geoms:
            reason = "extra-robot-contact"
        elif owner2 is not None and geom1 in robot_geoms:
            reason = "extra-robot-contact"
        elif float(contact.dist) < -0.005:
            reason = "deep-initial-penetration"
        if reason is not None:
            failures.append(
                {
                    "reason": reason,
                    "geom1": geom1,
                    "geom2": geom2,
                    "owner1": owner1,
                    "owner2": owner2,
                    "distance": float(contact.dist),
                }
            )
    return failures


def collision_pair_counts(
    collisions: list[dict[str, Any]], active_task: str
) -> list[dict[str, Any]]:
    """Reduce every rejected contact to an untruncated participant-pair count."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for collision in collisions:
        reason = collision["reason"]
        owner1 = collision.get("owner1")
        owner2 = collision.get("owner2")
        if reason == "extra-active-object-contact":
            owner1 = owner1 or active_task
            owner2 = owner2 or active_task
        elif reason == "extra-robot-contact":
            owner1 = owner1 or "robot"
            owner2 = owner2 or "robot"
        elif reason == "deep-initial-penetration":
            owner1 = owner1 or "scene-fixture"
            owner2 = owner2 or "scene-fixture"
        participant1, participant2 = sorted((str(owner1), str(owner2)))
        counts[(participant1, participant2, reason)] += 1
    return [
        {
            "participant1": participant1,
            "participant2": participant2,
            "reason": reason,
            "contacts": count,
        }
        for (participant1, participant2, reason), count in sorted(counts.items())
    ]


def prepare_augmented_episode(
    task: str,
    episode_number: int,
    inactive_tasks: list[str],
    inactive_attempts: dict[str, int],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode = load_episode(task, episode_number)
    env = make_env(episode["dataset_meta"])
    try:
        source_xml = reset_to_source(env, episode)
        source_snapshot = snapshot_named_state(env)
        active_contact_geoms = {
            geom_name
            for model in env.objects.values()
            for geom_name in model.contact_geoms
        }
        extra_models: list[Any] = []
        extra_cfgs: list[dict[str, Any]] = []
        placements: list[dict[str, Any]] = []
        extra_geom_owners: dict[str, str] = {}
        for inactive_task in inactive_tasks:
            try:
                models, cfgs, task_placements, geom_owners = (
                    sample_inactive_inventory(
                        env,
                        task,
                        episode_number,
                        inactive_task,
                        inactive_attempts[inactive_task],
                    )
                )
            except BaseException as exc:
                exc.inactive_task = inactive_task
                raise
            extra_models.extend(models)
            extra_cfgs.extend(cfgs)
            placements.extend(task_placements)
            extra_geom_owners.update(geom_owners)
        composite_xml = merge_extra_models(source_xml, extra_models)
        for model in extra_models:
            env.objects[model.name] = model
        env.object_cfgs.extend(extra_cfgs)
        env.reset_from_xml_string(composite_xml)
        env.sim.reset()
        restore_named_state(env, source_snapshot)
        for placement in placements:
            env.sim.data.set_joint_qpos(
                placement["joint"],
                np.concatenate(
                    [placement["position"], placement["quaternion"]]
                ),
            )
            env.sim.data.set_joint_qvel(placement["joint"], np.zeros(6))
        env.sim.forward()
        env.update_state()
        collisions = initial_collision_report(
            env, active_contact_geoms, extra_geom_owners
        )
        return env, episode, placements, collisions
    except BaseException:
        env.close()
        raise


def parse_inactive_attempts(
    values: list[str], inactive_tasks: list[str]
) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for value in values:
        task, separator, index_text = value.rpartition("=")
        if not separator or task not in inactive_tasks:
            raise ValueError(f"invalid inactive attempt {value!r}")
        index = int(index_text)
        if index < 0 or task in overrides:
            raise ValueError(f"invalid inactive attempt {value!r}")
        overrides[task] = index
    return overrides
