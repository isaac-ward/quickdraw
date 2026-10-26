"""One reader for the block-stack corpus, shared by the two xarm7 figure scripts.

`isaac-ronald-ward/block-stack`: an xArm 7 right arm mounted 45 deg clockwise from vertical,
teleoperated with an Xbox pad, four USB cameras, everything resampled onto one 30 Hz grid. It is
already in the on-disk format the loader wants (lerobot v3.0 splits), and the HF cache holds the whole
68 GB of it, so nothing is downloaded here.

FRAMES ARE CUT WITH FFMPEG, NOT DECODED FROM THE TOP. One mp4 per camera holds many episodes end to
end, and the figures want a window in the middle of a long run -- reading sequentially to frame 90,000
costs minutes per camera. meta/episodes carries each episode's `from_timestamp` INSIDE its file, so a
global row maps to a file-local frame exactly, and `-ss` lands on it.

The one thing worth stating out loud: this is NOT the starling format. 17-D state, 5-D action, 144x192
frames at 30 Hz, four cameras rather than one.
"""
from __future__ import annotations

import glob
import os
import subprocess

import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq

SNAP = sorted(glob.glob("/caches/hf/hub/datasets--isaac-ronald-ward--block-stack/snapshots/*/"))[-1]
SPLIT = "train"
FPS = 30.0
H, W = 144, 192                       # the stored frame, 4:3 like the 640x480 source

# THE PAIR THE 2-CAMERA RECIPE TRAINS ON (conf/model/vl128_blockstack_2cam.yaml), not a fresh choice:
# `scene_right` is the fixed third-person view (26% more frame-to-frame change than scene_left) and
# `gripper_right_top` is the highest-motion of the four, the gripper-mounted analogue of robocasa's
# eye-in-hand. The other two cameras exist in the corpus and are one string away.
CAMS = ("scene_right", "gripper_right_top")
CAM_LABELS = {"scene_right": "scene camera", "gripper_right_top": "gripper camera"}

# PROCESSING.md section 5. 17 dims, and the 6-D rotation block is six components of a rotation matrix
# rather than six angles -- which is why the strip charts default to leaving it out.
STATE_KEYS = (["ee_x_mm", "ee_y_mm", "ee_z_mm"]
              + [f"ee_rot6_{i}" for i in range(6)]
              + ["gripper"]
              + [f"joint{i + 1}_rad" for i in range(7)])
# PROCESSING.md section 1, the teleop table: left stick = planar motion, right stick Y = height,
# right stick X = rotation about the WORLD vertical, right trigger = gripper (proportional).
ACTION_KEYS = ["move_x", "move_y", "height", "yaw", "gripper"]


def table():
    """-> (observation_vector (N,17), action (N,5), episode_index (N,)) for the whole split."""
    t = pq.read_table(sorted(glob.glob(f"{SNAP}{SPLIT}/data/**/*.parquet", recursive=True))[0]).to_pydict()
    obs = np.stack([np.asarray(x, np.float32) for x in t["observation_vector"]])
    act = np.stack([np.asarray(x, np.float32) for x in t["action"]])
    return obs, act, np.asarray(t["episode_index"])


def episodes():
    """-> list of dicts: the row span of each episode and, per camera, which file it is in and where."""
    t = pq.read_table(sorted(glob.glob(f"{SNAP}{SPLIT}/meta/episodes/**/*.parquet",
                                       recursive=True))[0]).to_pydict()
    out = []
    for i in range(len(t["episode_index"])):
        e = dict(ep=int(t["episode_index"][i]), lo=int(t["dataset_from_index"][i]),
                 hi=int(t["dataset_to_index"][i]), cams={})
        for c in CAMS:
            k = f"videos/observation.images.{c}"
            e["cams"][c] = (int(t[f"{k}/chunk_index"][i]), int(t[f"{k}/file_index"][i]),
                            float(t[f"{k}/from_timestamp"][i]))
        out.append(e)
    return out


def _locate(eps, row):
    for e in eps:
        if e["lo"] <= row < e["hi"]:
            return e
    raise IndexError(f"row {row} is in no episode")


def frames(cam, row0, count, eps=None):
    """`count` frames of `cam` starting at GLOBAL row `row0` -> (count, H, W, 3) uint8.

    The window may not cross an episode boundary: two episodes are two different scenes, and a clip that
    straddles the cut is a jump cut nobody asked for."""
    eps = eps or episodes()
    e = _locate(eps, row0)
    assert row0 + count <= e["hi"], (f"a {count}-frame window from row {row0} runs past episode "
                                     f"{e['ep']} (ends at {e['hi']})")
    chunk, fidx, t0 = e["cams"][cam]
    path = f"{SNAP}{SPLIT}/videos/observation.images.{cam}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
    start = (round(t0 * FPS) + (row0 - e["lo"])) / FPS
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                          "-ss", f"{start:.6f}", "-i", path, "-frames:v", str(count),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(out, np.uint8)
    assert a.size == count * H * W * 3, f"{cam}: got {a.size / (H * W * 3):.1f} frames, wanted {count}"
    return a.reshape(count, H, W, 3)
