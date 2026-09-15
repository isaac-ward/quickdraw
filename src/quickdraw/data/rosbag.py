"""Read a rosbag2 flight recording into (states, actions, frames) — NO ROS, NO seamstress.

WHY THIS EXISTS. The starling flight data arrives as rosbag2 sqlite bags (`metadata.yaml` +
`*_0.db3`), and the only path from those to a trainable dataset used to be a processor living in a
different repo. That is a bad dependency for the thing that defines our data: it means the dataset's
provenance (which campaign an episode came from, what rate the images really ran at, how velocity was
derived) lives somewhere we do not read, and every one of those turned out to matter. This module reads
the bags directly with `rosbags` — a pure-Python rosbag2 reader, no ROS install — so the whole path from
raw flight to lerobot dataset is in this repo and auditable.

WHAT IS IN THE BAGS (measured on campaign15-24, 2026-09-08):

    /vrpn_mocap/fmu/pose    geometry_msgs/PoseStamped   ~276 Hz   position + orientation quaternion
    /imu_apps               sensor_msgs/Imu             ~86 Hz    angular velocity + linear acceleration
    /joy                    sensor_msgs/Joy             ~69 Hz    7 axes (the action is axes 0..3)
    /hires_small_color      sensor_msgs/Image           ~17 Hz    360x640, encoding 'yuv422' (UYVY)

THE CAMERA IS THE SLOWEST STREAM AND THAT IS THE WHOLE STORY. Everything else runs 4-16x faster, so the
image rate sets what a "step" can mean. Resampling to a grid FASTER than ~17 Hz does not create
information, it duplicates frames: the published starling-2 dataset was built at 30 Hz and its own
upstream summary admits `num_unique_images_used: 2730` for a `num_frames: 3500` episode -- i.e. 22% of
its frames are repeats of the previous one, and a world model trained on it is being asked to predict
"nothing changed" on a fifth of its steps. `TARGET_HZ_DEFAULT` below is therefore BELOW the camera rate,
so every step is a distinct observation. Raise it only if you want to reproduce the old dataset.
"""

from __future__ import annotations

import numpy as np

# Topics. Pose is listed as candidates because the mocap subject name varies across campaigns
# (`fmu` vs `ministarling`); the first present one wins.
POSE_TOPICS = ("/vrpn_mocap/fmu/pose", "/vrpn_mocap/ministarling/pose")
IMU_TOPIC = "/imu_apps"
JOY_TOPIC = "/joy"
IMAGE_TOPIC = "/hires_small_color"

ACTION_AXES = 4          # /joy carries 7 axes; the recorded action is the first four
TARGET_HZ_DEFAULT = 15.0  # BELOW the ~17 Hz camera -> every step is a distinct frame. See the module note.

# The 16-dim observation vector, in order. Named here because the published dataset ships it with NO
# `names` metadata, which cost a day of reverse-engineering on the robocasa side; never do that again.
STATE_COLUMNS = (
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z",
)
ACTION_COLUMNS = tuple(f"joy_axis_{i}" for i in range(ACTION_AXES))


def uyvy_to_rgb(buf: bytes, h: int, w: int) -> np.ndarray:
    """ROS `yuv422` (== UYVY 4:2:2, 2 bytes/pixel) -> (h, w, 3) uint8 RGB.

    Byte order per pixel PAIR is U Y0 V Y1: one chroma sample shared by two luma samples. Decoded with
    the BT.601 full-range coefficients, which is what the ROS/OpenCV `yuv422 -> rgb8` path uses; getting
    the standard wrong shifts every colour slightly and silently, so it is spelled out rather than
    delegated to a library that may pick differently.
    """
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, w // 2, 4).astype(np.int16)
    u, y0, v, y1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    y = np.empty((h, w), dtype=np.int16)
    y[:, 0::2], y[:, 1::2] = y0, y1
    cb = np.repeat(u - 128, 2, axis=1).astype(np.int32)
    cr = np.repeat(v - 128, 2, axis=1).astype(np.int32)
    yy = y.astype(np.int32)
    rgb = np.stack([yy + ((1436 * cr) >> 10),                        # + 1.402 * cr
                    yy - ((352 * cb + 731 * cr) >> 10),              # - 0.344*cb - 0.714*cr
                    yy + ((1815 * cb) >> 10)], axis=-1)              # + 1.772 * cb
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _nearest(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Index of the nearest `source` time for each `target` time (both sorted, seconds)."""
    j = np.searchsorted(source, target)
    j = np.clip(j, 1, len(source) - 1)
    left, right = source[j - 1], source[j]
    return np.where(target - left <= right - target, j - 1, j)


def read_run(bag_dir: str, target_hz: float = TARGET_HZ_DEFAULT,
             out_hw: tuple[int, int] = (112, 192), want_frames: bool = True):
    """One bag directory -> (states (T,16) float32, actions (T,4) float32, frames (T,h,w,3) uint8|None).

    All four streams are resampled onto ONE uniform grid at `target_hz` spanning the IMAGE timespan --
    images define the usable window, since a step with no frame is useless to a visual world model.
    Pose/imu/joy are sampled nearest-neighbour (they run 4-16x faster than the grid, so interpolation
    would be inventing precision), images likewise. Velocity is DERIVED from position by central
    difference (`np.gradient`), because the mocap publishes pose only.
    """
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)

    raws: dict[str, list] = {}
    times: dict[str, list] = {}
    with Reader(bag_dir) as r:
        topics = {c.topic for c in r.connections}
        pose_topic = next((t for t in POSE_TOPICS if t in topics), None)
        if pose_topic is None:
            raise ValueError(f"{bag_dir}: no pose topic among {POSE_TOPICS}; has {sorted(topics)}")
        keep = {pose_topic, IMU_TOPIC, JOY_TOPIC, IMAGE_TOPIC}
        for conn, t, raw in r.messages():
            if conn.topic in keep:
                raws.setdefault(conn.topic, []).append((raw, conn.msgtype))
                times.setdefault(conn.topic, []).append(t * 1e-9)
    for t in (pose_topic, IMU_TOPIC, JOY_TOPIC, IMAGE_TOPIC):
        if not raws.get(t):
            raise ValueError(f"{bag_dir}: topic {t} is empty")

    T = {k: np.asarray(v, dtype=np.float64) for k, v in times.items()}
    # THE GRID IS DEFINED BY THE IMAGES. dt from target_hz; the span is the camera's first..last frame.
    it = T[IMAGE_TOPIC]
    dt = 1.0 / float(target_hz)
    n = max(1, int(np.floor((it[-1] - it[0]) / dt)) + 1)
    grid = it[0] + dt * np.arange(n)

    pi = _nearest(grid, T[pose_topic])
    ii = _nearest(grid, T[IMU_TOPIC])
    ji = _nearest(grid, T[JOY_TOPIC])
    mi = _nearest(grid, it)

    de = lambda k, idx: [ts.deserialize_cdr(raws[k][j][0], raws[k][j][1]) for j in idx]
    pose = de(pose_topic, pi)
    pos = np.array([[m.pose.position.x, m.pose.position.y, m.pose.position.z] for m in pose], dtype=np.float64)
    quat = np.array([[m.pose.orientation.x, m.pose.orientation.y,
                      m.pose.orientation.z, m.pose.orientation.w] for m in pose], dtype=np.float32)
    vel = (np.gradient(pos, dt, axis=0) if len(pos) > 1 else np.zeros_like(pos)).astype(np.float32)
    imu = de(IMU_TOPIC, ii)
    ang = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in imu], dtype=np.float32)
    acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in imu], dtype=np.float32)
    joy = de(JOY_TOPIC, ji)
    act = np.array([list(m.axes[:ACTION_AXES]) + [0.0] * max(0, ACTION_AXES - len(m.axes)) for m in joy],
                   dtype=np.float32)
    states = np.concatenate([pos.astype(np.float32), vel, quat, ang, acc], axis=1)
    assert states.shape[1] == len(STATE_COLUMNS), f"{states.shape[1]} != {len(STATE_COLUMNS)}"

    frames = None
    if want_frames:
        from .dataset import resize_frames_area
        # Decode each SOURCE image at most once, then index -- the grid may reuse a frame if target_hz
        # exceeds the camera rate (which TARGET_HZ_DEFAULT deliberately avoids).
        uniq, inv = np.unique(mi, return_inverse=True)
        small = []
        for j in uniq:
            m = ts.deserialize_cdr(raws[IMAGE_TOPIC][int(j)][0], raws[IMAGE_TOPIC][int(j)][1])
            if m.encoding != "yuv422":
                raise ValueError(f"{bag_dir}: unexpected image encoding {m.encoding!r} (expected yuv422)")
            small.append(resize_frames_area(uyvy_to_rgb(m.data, m.height, m.width)[None], out_hw)[0])
        frames = np.stack(small)[inv]
    return states, act, frames


def run_info(bag_dir: str) -> dict:
    """Cheap per-bag report (topics, counts, rates) with no deserialization -- for auditing a tree."""
    from rosbags.rosbag2 import Reader
    with Reader(bag_dir) as r:
        out = {"topics": {}}
        for c in r.connections:
            out["topics"][c.topic] = {"type": c.msgtype, "count": c.msgcount}
        out["duration_s"] = round((r.end_time - r.start_time) * 1e-9, 2)
    for t, d in out["topics"].items():
        d["hz"] = round(d["count"] / out["duration_s"], 1) if out["duration_s"] else None
    return out
