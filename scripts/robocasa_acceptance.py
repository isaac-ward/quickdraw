#!/usr/bin/env python3
"""Load immutable RoboCasa source episodes for direct-state certification."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Importing robocasa registers its environments with robosuite.
import robocasa  # noqa: F401
import robosuite


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / ".cache" / "robocasa-data-generation" / "render-episodes"


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def episode_paths(task: str, episode: int) -> dict[str, Path]:
    base = DATA_ROOT / task / "lerobot"
    ep_name = f"episode_{episode:06d}"
    parquet_matches = sorted((base / "data").glob(f"**/{ep_name}.parquet"))
    if len(parquet_matches) != 1:
        raise FileNotFoundError(
            f"expected one parquet for {task} episode {episode}, got {parquet_matches}"
        )
    extra = base / "extras" / ep_name
    return {
        "base": base,
        "dataset_meta": base / "extras" / "dataset_meta.json",
        "ep_meta": extra / "ep_meta.json",
        "model": extra / "model.xml.gz",
        "states": extra / "states.npz",
        "actions": parquet_matches[0],
    }


def load_episode(task: str, episode: int) -> dict[str, Any]:
    paths = episode_paths(task, episode)
    with paths["dataset_meta"].open() as handle:
        dataset_meta = json.load(handle)
    with paths["ep_meta"].open() as handle:
        ep_meta = json.load(handle)
    with gzip.open(paths["model"], "rt") as handle:
        model_xml = handle.read()
    with np.load(paths["states"]) as archive:
        states = archive["states"]
    action_series = pd.read_parquet(paths["actions"], columns=["action"])["action"]
    actions_lerobot = np.vstack(action_series.to_numpy())
    with (paths["base"] / "meta" / "modality.json").open() as handle:
        modality_dict = json.load(handle)
    native_ranges = {
        "end_effector_position": (0, 3),
        "end_effector_rotation": (3, 6),
        "gripper_close": (6, 7),
        "base_motion": (7, 11),
        "control_mode": (11, 12),
    }
    actions = np.zeros_like(actions_lerobot)
    for key, details in modality_dict["action"].items():
        native_start, native_end = native_ranges[key]
        actions[:, native_start:native_end] = actions_lerobot[
            :, details["start"] : details["end"]
        ]
    return {
        "dataset_meta": dataset_meta,
        "ep_meta": ep_meta,
        "model_xml": model_xml,
        "states": states,
        "actions": actions,
    }


def make_env(dataset_meta: dict[str, Any]):
    kwargs = dict(dataset_meta["env_args"]["env_kwargs"])
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    return robosuite.make(**kwargs)


def reset_to_source(env, episode: dict[str, Any]) -> None:
    env.set_ep_meta(episode["ep_meta"])
    env.reset()
    model_xml = env.edit_model_xml(episode["model_xml"])
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(episode["states"][0])
    env.sim.forward()
    env.update_state()
