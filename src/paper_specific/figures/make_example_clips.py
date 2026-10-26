"""Search the whole starling-2 flight log for two kinds of clip, and export them as GIFs.

    exports_examples/approach_<x>.gif   the drone closes on something and then holds it in frame
    exports_examples/axes_<x>.gif       a stretch covering several distinct directions of travel
    exports_examples/_approach.png      contact sheet, 8 frames per clip
    exports_examples/_axes.png          contact sheet, 8 frames per clip
    exports_examples/manifest.json      start row, episode and stats for every clip, so any of them can be
                                        promoted to a full fpv/controller/proprio set later

BOTH SEARCHES RUN ON BODY-FRAME VELOCITY, not on the joystick. What the pilot asked for and what the drone
did are different signals, and "flew at the ladder" is a claim about the second. World velocity is rotated
into the body frame through the recorded quaternion, with the heading negated -- starling's yaw is 180 deg
out, a correction that has now been rediscovered three times (see also evaluation/steering calibration).

WHAT THE DATA ACTUALLY HAS, measured rather than hoped for: of 10,648 scanned 15 s windows, ZERO contain
all six directions and exactly fourteen contain five, even with the thresholds wound down to 0.20 m/s.
This is undirected play -- the pilot flew the room, they did not run an axis check -- so `axes_a` is the
only five-direction clip in two hours of flight and the rest are the best four-direction ones. Loosening
further does not find more sequences, it just relabels noise.

    python src/paper_specific/figures/make_example_clips.py
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import imageio.v3 as iio3
import matplotlib
import numpy as np
import pyarrow.parquet as pq

import make_sync_gifs as G

OUT = "/app/logs/paper_icra_2027/exports_examples"
NAMES = ("forward", "back", "right", "left", "up", "down")      # indexed by axis*2 + (component < 0)

# --- approach-and-hold: APPR_A frames closing, then APPR_L - APPR_A frames parked, heading steady throughout
APPR_L, APPR_A = 135, 90        # 9 s total, 6 s of it closing
APPR_SPEED, APPR_DIST = 0.45, 2.0      # mean forward speed (m/s) and ground closed (m) over the approach
APPR_DWELL, APPR_DRIFT, APPR_YAWRMS = 0.40, 25.0, 0.30   # m/s while parked, deg of heading drift, rad/s RMS

# --- direction sequence: a frame is labelled only when ONE body axis clearly dominates
AXES_L, AXES_MAG, AXES_RATIO, AXES_RUN, AXES_MIN = 225, 0.20, 1.15, 6, 4
N_EACH = 5


def body_frame(obs):
    """World velocity -> (forward, lateral, up) in the body frame."""
    vel, (qx, qy, qz, qw) = obs[:, 3:6], (obs[:, 6], obs[:, 7], obs[:, 8], obs[:, 9])
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.stack([-(np.cos(yaw) * vel[:, 0] + np.sin(yaw) * vel[:, 1]),
                     -(-np.sin(yaw) * vel[:, 0] + np.cos(yaw) * vel[:, 1]), vel[:, 2]], 1), yaw


def spread(cands, length, per_ep=2):
    """Keep the best, non-overlapping, at most `per_ep` from any one flight -- otherwise one good minute
    of flying supplies every 'example' and they are all the same room from the same angle."""
    out = []
    for c in cands:
        if any(abs(c["start"] - o["start"]) < length for o in out):
            continue
        if sum(1 for o in out if o["episode"] == c["episode"]) >= per_ep:
            continue
        out.append(c)
        if len(out) == N_EACH:
            break
    return out


def find_approach(obs, ep, comp, yaw):
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    fwd, yr, speed = comp[:, 0], obs[:, 12], np.linalg.norm(obs[:, 3:6], axis=1)
    out = []
    for s in range(0, len(obs) - APPR_L, 15):
        if ep[s] != ep[s + APPR_L - 1]:
            continue
        a, d = slice(s, s + APPR_A), slice(s + APPR_A, s + APPR_L)
        dist = float(fwd[a].sum() / G.SRC_FPS)
        if fwd[a].mean() < APPR_SPEED or dist < APPR_DIST:
            continue
        if speed[d].mean() > APPR_DWELL:
            continue
        if abs(wrap(yaw[s + APPR_L - 1] - yaw[s])) > np.deg2rad(APPR_DRIFT):
            continue
        if np.sqrt((yr[s:s + APPR_L] ** 2).mean()) > APPR_YAWRMS:
            continue
        out.append({"start": int(s), "episode": int(ep[s]), "closes_m": round(dist, 2),
                    "dwell_speed_mps": round(float(speed[d].mean()), 3)})
    out.sort(key=lambda c: -c["closes_m"])
    return out


def find_axes(ep, comp):
    mag = np.abs(comp)
    order = np.argsort(-mag, axis=1)
    rows = np.arange(len(mag))
    m1, m2 = mag[rows, order[:, 0]], mag[rows, order[:, 1]]
    top = order[:, 0]
    lab = np.where((m1 > AXES_MAG) & (m1 > AXES_RATIO * m2),
                   top * 2 + (comp[rows, top] < 0), -1)
    out = []
    for s in range(0, len(lab) - AXES_L, 10):
        if ep[s] != ep[s + AXES_L - 1]:
            continue
        w, runs, i = lab[s:s + AXES_L], {}, 0
        while i < AXES_L:                                   # only runs held for AXES_RUN frames count:
            j = i                                           # a single frame of "left" is drift, not a move
            while j < AXES_L and w[j] == w[i]:
                j += 1
            if w[i] >= 0 and j - i >= AXES_RUN:
                runs[int(w[i])] = runs.get(int(w[i]), 0) + (j - i)
            i = j
        if len(runs) >= AXES_MIN:
            out.append({"start": int(s), "episode": int(ep[s]), "n_directions": len(runs),
                        "labelled_frames": int(sum(runs.values())),
                        "directions": [NAMES[k] for k in sorted(runs)]})
    out.sort(key=lambda c: (-c["n_directions"], -c["labelled_frames"]))
    return out


def sheet(clips, raw, length, caption, path):
    rows = []
    for c in clips:
        st = np.hstack([cv2.resize(raw[c["start"] + k * (length // 7)], (224, 131)) for k in range(8)])
        cv2.putText(st, caption(c), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        rows.append(st)
    cv2.imwrite(path, np.vstack(rows)[..., ::-1])
    print(f"  {os.path.basename(path)}")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    tab = pq.read_table(sorted(glob.glob(f"{G.ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
    ep = np.asarray(tab["episode_index"])
    obs = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])
    comp, yaw = body_frame(obs)
    print(f"  scanning {len(obs)} steps ({len(obs) / G.SRC_FPS / 60:.0f} min of flight)")

    appr_all, axes_all = find_approach(obs, ep, comp, yaw), find_axes(ep, comp)
    appr, axes = spread(appr_all, APPR_L), spread(axes_all, AXES_L)
    print(f"  approach: {len(appr_all)} windows qualify -> {len(appr)} kept")
    for c in appr:
        print(f"     ep{c['episode']:3d} @{c['start']:6d}  closes {c['closes_m']:5.1f} m, "
              f"then {c['dwell_speed_mps']:.2f} m/s")
    print(f"  axes: {len(axes_all)} windows with >= {AXES_MIN} directions -> {len(axes)} kept")
    for c in axes:
        print(f"     ep{c['episode']:3d} @{c['start']:6d}  {c['n_directions']} dirs: "
              + ", ".join(c["directions"]))

    jobs = ([("approach", chr(97 + i), c, APPR_L) for i, c in enumerate(appr)]
            + [("axes", chr(97 + i), c, AXES_L) for i, c in enumerate(axes)])
    need, bases = set(), {}
    for kind, letter, c, length in jobs:
        idx = c["start"] + np.round(np.arange(int(length / G.SRC_FPS * G.GIF_FPS))
                                    / G.GIF_FPS * G.SRC_FPS).astype(int)
        bases[(kind, letter)] = idx
        need |= set(int(i) for i in idx) | {c["start"] + k * (length // 7) for k in range(8)}

    vid = sorted(glob.glob(f"{G.ROOT}/train/videos/**/*.mp4", recursive=True))[0]
    stop, raw = max(need), {}
    for j, f in enumerate(iio3.imiter(vid, plugin="pyav")):
        if j in need:
            raw[j] = f
        if j >= stop:
            break

    green = tuple(int(255 * v) for v in matplotlib.colors.to_rgb(G.VIS))
    for kind, letter, c, length in jobs:
        frames = []
        for i in bases[(kind, letter)]:
            im = cv2.resize(raw[int(i)], (G.TILE_W - 2 * G.BORDER, G.TILE_H - 2 * G.BORDER),
                            interpolation=cv2.INTER_LANCZOS4)
            f = np.full((G.TILE_H, G.TILE_W, 3), green, np.uint8)
            f[G.BORDER:-G.BORDER, G.BORDER:-G.BORDER] = im
            frames.append(f)
        G.dump(frames, f"{kind}_{letter}", photographic=True, out_dir=OUT)

    sheet(appr, raw, APPR_L, lambda c: f"approach  ep{c['episode']} @{c['start']}  closes {c['closes_m']} m",
          f"{OUT}/_approach.png")
    sheet(axes, raw, AXES_L, lambda c: f"axes  ep{c['episode']} @{c['start']}  "
          + ",".join(c["directions"]), f"{OUT}/_axes.png")
    json.dump({"fps": G.GIF_FPS, "source_fps": G.SRC_FPS,
               "approach": {chr(97 + i): dict(c, frames=APPR_L) for i, c in enumerate(appr)},
               "axes": {chr(97 + i): dict(c, frames=AXES_L) for i, c in enumerate(axes)}},
              open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"  manifest.json")
    import shutil
    shutil.rmtree(G.TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
