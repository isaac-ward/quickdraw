#!/usr/bin/env python3
"""Render initial, middle, and final overhead views of selected RoboCasa episodes.

The script loads the immutable recorded MuJoCo states directly. It does not step the
environment, replay actions, or alter task state. Absolute asset paths embedded by the
original collection machine are rewritten only so MuJoCo can find identical assets in
the project-local cache.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


SELECTED_EPISODES = {
    "GetToastedBread": 116,
    "DeliverStraw": 199,
    "ArrangeTea": 90,
    "LoadDishwasher": 490,
    "PackIdenticalLunches": 2,
    "StirVegetables": 108,
}

def _body_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx) or ""
        for idx in range(model.nbody)
    ]


def _resolve_body_ids(names: list[str], requested: list[str]) -> list[int]:
    resolved: list[int] = []
    for item in requested:
        preferred = (f"{item}_main", item)
        matches = [idx for idx, name in enumerate(names) if name in preferred]
        if not matches:
            matches = [
                idx for idx, name in enumerate(names) if name.startswith(f"{item}_")
            ]
        if matches:
            resolved.append(matches[0])
    return sorted(set(resolved))


def _fixture_names(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        names: list[str] = []
        for child in value.values():
            names.extend(_fixture_names(child))
        return names
    if isinstance(value, list):
        names = []
        for child in value:
            names.extend(_fixture_names(child))
        return names
    return []


def _set_recorded_state(
    model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray
) -> None:
    expected = 1 + model.nq + model.nv + model.na
    if state.size != expected:
        raise ValueError(f"recorded state has {state.size} values; model expects {expected}")
    cursor = 0
    data.time = state[cursor]
    cursor += 1
    data.qpos[:] = state[cursor : cursor + model.nq]
    cursor += model.nq
    data.qvel[:] = state[cursor : cursor + model.nv]
    cursor += model.nv
    if model.na:
        data.act[:] = state[cursor : cursor + model.na]
    mujoco.mj_forward(model, data)


def _rewrite_asset_paths(xml: str, cache_dir: Path) -> str:
    robosuite_assets = (cache_dir / "robosuite/robosuite/models/assets").resolve()
    robocasa_assets = (
        cache_dir / "source/robocasa/models/assets"
    ).resolve()
    xml = re.sub(
        r'file="[^"]*/robosuite/models/assets/',
        f'file="{robosuite_assets}/',
        xml,
    )
    return re.sub(
        r'file="[^"]*/robocasa/models/assets/',
        f'file="{robocasa_assets}/',
        xml,
    )


def render_task(
    task: str,
    cache_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[str, object]:
    episode_index = SELECTED_EPISODES[task]
    episode_dir = (
        cache_dir
        / "render-episodes"
        / task
        / "lerobot"
        / "extras"
        / f"episode_{episode_index:06d}"
    )
    with gzip.open(episode_dir / "model.xml.gz", "rt") as stream:
        xml = _rewrite_asset_paths(stream.read(), cache_dir)
    with (episode_dir / "ep_meta.json").open() as stream:
        episode_meta = json.load(stream)
    states = np.load(episode_dir / "states.npz")["states"]

    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)

    frame_indices = [0, len(states) // 2, len(states) - 1]
    frame_labels = ["initial", "middle", "end"]
    names = _body_names(model)
    task_object_names = [cfg["name"] for cfg in episode_meta.get("object_cfgs", [])]
    fixture_names = _fixture_names(episode_meta.get("fixture_refs", {}))
    body_ids = _resolve_body_ids(names, task_object_names + fixture_names)
    mobile_base_ids = _resolve_body_ids(names, ["mobilebase0_base"])
    framing_ids = sorted(set(body_ids + mobile_base_ids))
    if not framing_ids:
        raise RuntimeError(f"could not resolve any framing bodies for {task}")

    _set_recorded_state(model, data, states[0])
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor_room_g0_vis"
    )
    if floor_id < 0:
        raise RuntimeError("could not resolve the visual room floor")
    room_center = data.geom_xpos[floor_id].copy()
    room_span = float(2 * np.max(model.geom_size[floor_id, :2]))

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [room_center[0], room_center[1], 0.75]
    camera.distance = room_span * 1.75
    camera.azimuth = 0.0
    camera.elevation = -82.0

    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[1] = 1
    scene_option.sitegroup[:] = 0

    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    rendered: list[Image.Image] = []
    try:
        for label, frame_index in zip(frame_labels, frame_indices):
            _set_recorded_state(model, data, states[frame_index])
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            image = Image.fromarray(renderer.render())
            image.save(task_output_dir / f"{label}.png")
            rendered.append(image)
    finally:
        renderer.close()

    label_height = 48
    triptych = Image.new(
        "RGB", (width * len(rendered), height + label_height), color=(250, 250, 250)
    )
    draw = ImageDraw.Draw(triptych)
    for panel, (label, frame_index, image) in enumerate(
        zip(frame_labels, frame_indices, rendered)
    ):
        x = panel * width
        triptych.paste(image, (x, label_height))
        draw.text(
            (x + 16, 15),
            f"{label.upper()}  frame {frame_index:,}",
            fill=(20, 20, 20),
        )
    triptych.save(task_output_dir / "triptych.png")

    provenance = {
        "task": task,
        "episode_index": episode_index,
        "layout_id": episode_meta["layout_id"],
        "style_id": episode_meta["style_id"],
        "state_count": len(states),
        "frame_indices": dict(zip(frame_labels, frame_indices)),
        "task_object_names": task_object_names,
        "fixture_names": fixture_names,
        "framing_body_names": [names[idx] for idx in framing_ids],
        "camera": {
            "projection": "perspective",
            "coverage": "full room from floor_room_g0_vis",
            "azimuth_degrees": camera.azimuth,
            "elevation_degrees": camera.elevation,
            "lookat": camera.lookat.tolist(),
            "distance": camera.distance,
            "width": width,
            "height": height,
        },
        "visible_geom_groups": [1],
        "visible_site_groups": [],
        "state_source": "immutable recorded MuJoCo states",
    }
    with (task_output_dir / "provenance.json").open("w") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(SELECTED_EPISODES),
        default=list(SELECTED_EPISODES),
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / ".cache/robocasa-data-generation"
    output_dir = args.output_dir or cache_dir / "renders/scene-4-selected-tasks"
    for task in args.tasks:
        provenance = render_task(
            task=task,
            cache_dir=cache_dir,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
        )
        print(
            f"{task}: episode {provenance['episode_index']}, "
            f"frames {provenance['frame_indices']}"
        )


if __name__ == "__main__":
    main()
