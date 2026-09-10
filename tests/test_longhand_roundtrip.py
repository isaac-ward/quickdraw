"""A built longhand dataset must agree with the source runs it was built from.

Run against a dataset produced by `+processor=longhand`:

    QD_LONGHAND_ROOT=logs/recording_..._longhand \\
    QD_LONGHAND_SRC=scratch/longhand \\
    .venv/bin/python tests/test_longhand_roundtrip.py

WHY THE SHIFT SWEEP IS THE POINT. Comparing built frames to source frames and getting a small
number proves nothing on its own -- a re-encoded video is a few grey levels off everywhere, and
a dataset misaligned by one row would ALSO score "small" on a slow-moving scene. The acceptance
criterion is that the error is a strict MINIMUM at shift 0 with a clear rise either side. A flat
curve means the comparison is insensitive and the test is vacuous.

This is precisely the check the lego corpus did not have, where a silent misalignment between
images and states survived all the way into training.
"""
import glob
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quickdraw.data.swoosh import read_run          # noqa: E402

DS = os.environ.get("QD_LONGHAND_ROOT", "")
SRC_ROOT = os.environ.get("QD_LONGHAND_SRC", "scratch/longhand")
# val episode 0 = the longest run in the pool, which is what the longest-first split guarantees.
SRC_RUN = os.environ.get("QD_LONGHAND_RUN",
                         "campaign5-play-long/recording_2026_09_10_06_43_58")
SPLIT, EP = "val", 0


def _skip_if_unbuilt():
    if not DS or not os.path.isdir(DS):
        print(f"    SKIP: set QD_LONGHAND_ROOT to a built dataset (got {DS!r})")
        return True
    return False


def _source():
    return read_run(os.path.join(SRC_ROOT, SRC_RUN))


def test_vectors_are_bit_exact():
    """The parquet must hold EXACTLY what the reader produced -- no silent recast or reorder."""
    if _skip_if_unbuilt():
        return
    import pandas as pd
    st, ac, _, info = _source()
    pq = sorted(glob.glob(os.path.join(DS, SPLIT, "data", "*", "*.parquet")))
    assert pq, f"no parquet under {DS}/{SPLIT}/data"
    df = pd.concat([pd.read_parquet(p) for p in pq])
    e = df[df.episode_index == EP].sort_values("frame_index")
    assert len(e) == info["steps"], f"{len(e)} rows vs {info['steps']} source steps"
    o = np.stack(e.observation_vector.values).astype(np.float32)
    a = np.stack(e.action.values).astype(np.float32)
    print(f"    {len(e)} rows;  max |state| diff {np.abs(o - st).max():.6g}, "
          f"max |action| diff {np.abs(a - ac).max():.6g}")
    assert np.abs(o - st).max() == 0.0
    assert np.abs(a - ac).max() == 0.0


def test_frames_align_with_rows():
    """mp4 frame i must BE row i: a strict error minimum at shift 0, rising either side."""
    if _skip_if_unbuilt():
        return
    import imageio.v2 as imageio
    _, _, fr, info = _source()
    a, b, shifts = 9000, 9120, range(-3, 4)
    assert b + max(shifts) < info["steps"], "probe window runs past the episode"
    for cam in ("scene_left", "gripper_right_top"):
        p = sorted(glob.glob(os.path.join(
            DS, SPLIT, "videos", f"observation.images.{cam}", "*", "*.mp4")))[0]
        rd, got = imageio.get_reader(p), []
        for i, f in enumerate(rd):
            if a <= i < b:
                got.append(np.asarray(f)[..., :3].astype(np.float32))
            if i >= b:
                break
        rd.close()
        got = np.stack(got)
        curve = [float(np.abs(got - fr[cam][a + s:b + s].astype(np.float32)).mean())
                 for s in shifts]
        best = list(shifts)[int(np.argmin(curve))]
        print(f"    {cam:20} {[round(v, 2) for v in curve]}  argmin={best:+d}")
        assert best == 0, f"{cam} aligns best at shift {best}, not 0"
        # the sweep must actually discriminate, or the minimum above is meaningless
        assert min(curve[0], curve[-1]) > 1.5 * curve[3], (
            f"{cam} shift curve is too flat to prove anything: {curve}")


if __name__ == "__main__":
    import traceback

    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fs:
        try:
            print(f"  {f.__name__}")
            f()
        except Exception:
            bad += 1
            print("    FAILED")
            traceback.print_exc()
    print(f"\n{len(fs) - bad}/{len(fs)} passed")
    raise SystemExit(1 if bad else 0)
