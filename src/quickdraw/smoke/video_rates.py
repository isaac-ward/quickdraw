"""Three rates, kept apart: dataset CAPTURE, model STEP, and mp4 PLAYBACK.

The measured failure this guards: `s2_sub4` trains at `data.subsample=4` on 15 Hz data, so one
autoregressive step spans 267 ms -- but every eval video was encoded at `round(1/ecfg.dt)` = 15 fps,
i.e. FOUR TIMES real speed, with nothing on screen to give it away. A 34 s prediction played in 8.5 s.
`training.setup.step_fps` fixes the rate; `viz.pace` then retimes for viewing by REPEATING frames, so
duration stays exact and no pixel the model never produced ever appears in evidence about the model.

    python -m quickdraw.smoke.video_rates
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from ..logging import viz
from ..logging.writer import RunWriter
from ..training.setup import step_fps

ok = bad = 0


def check(n, cond, extra=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{('  ' + extra) if extra else ''}", flush=True)


class _Sink:
    def __init__(self):
        self.got = []

    def video(self, tag, frames, fps, step):
        self.got.append((len(frames), fps))


def main() -> int:
    blank = [np.zeros((4, 4, 3), np.uint8)]

    # 1. the model's step rate folds in the frame stride -- the actual bug
    e = SimpleNamespace(dt=1.0 / 15.0)
    for sub, want in [(1, 15.0), (2, 7.5), (3, 5.0), (4, 3.75), (5, 3.0), (None, 15.0)]:
        got = step_fps(OmegaConf.create({"data": {"subsample": sub}}), e)
        check(f"step_fps at subsample={sub} is {want} Hz", abs(got - want) < 1e-9, f"got {got}")

    # 2. pacing preserves WALL-CLOCK DURATION, including for non-integer ratios
    for n, fps, pb in [(128, 3.75, 30), (128, 15.0, 30), (100, 7.5, 30), (128, 5.0, 24), (128, 3.75, None)]:
        out, of = viz.pace(blank * n, fps, pb)
        check(f"{n}f @{fps}Hz -> @{of}Hz keeps duration {n / fps:.3f}s",
              abs(len(out) / of - n / fps) < 1.0 / fps, f"got {len(out) / of:.3f}s")

    # 3. it REPEATS, never invents or reorders or drops -- the reason a rollout video stays evidence
    f = [np.full((2, 2, 3), i, np.uint8) for i in range(10)]
    out, _ = viz.pace(f, 3.0, 30)
    vals = [int(x[0, 0, 0]) for x in out]
    check("every output frame is an input frame (nothing blended)", set(vals) <= set(range(10)))
    check("time order preserved", vals == sorted(vals))
    check("no unique frame is dropped", len(set(vals)) == 10, f"{len(set(vals))}/10")
    check("endpoints preserved", vals[0] == 0 and vals[-1] == 9, f"{vals[0]}..{vals[-1]}")
    out, of = viz.pace(f, 60.0, 30)
    check("a source FASTER than playback is left alone (never decimated)", out is f and of == 60.0)
    check("playback == true rate is a no-op", viz.pace(f, 30.0, 30)[0] is f)
    check("playback None is a no-op (true rate survives)", viz.pace(f, 7.0, None) == (f, 7.0))

    # 4. RunWriter paces ONCE, so the local mirror and wandb cannot diverge
    a, b = _Sink(), _Sink()
    RunWriter("/tmp", [a, b], playback_fps=30).video("t", blank * 128, 3.75, 0)
    check("RunWriter retimes 128f @3.75Hz -> 1024f @30Hz", a.got == [(1024, 30.0)], str(a.got))
    check("both backends receive the IDENTICAL paced video", a.got == b.got)
    s = _Sink()
    RunWriter("/tmp", [s], playback_fps=None).video("t", blank * 128, 3.75, 0)
    check("environments.preview_fps: null -> encode at the true rate", s.got == [(128, 3.75)], str(s.got))

    # 5. it survives an actual encode (a fractional rate used to be rounded: 3.75 -> 4, 6.7% fast)
    import imageio.v2 as iio
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.mp4")
        fr = [np.random.randint(0, 255, (112, 192, 3), dtype=np.uint8) for _ in range(40)]
        viz.save_mp4(p, fr, 3.75, playback_fps=30)
        m = iio.get_reader(p).get_meta_data()
        check("encoded at 30 fps with the true duration",
              abs(m["fps"] - 30) < 0.1 and abs(m["duration"] - 40 / 3.75) < 0.2,
              f"{m['fps']} fps, {m['duration']}s")
        viz.save_mp4(p, fr, 3.75)
        check("an unpaced fractional rate is NOT rounded",
              abs(iio.get_reader(p).get_meta_data()["fps"] - 3.75) < 0.05)

    # 5b. the WATCHABILITY FLOOR: a crawl is sped up by a stated factor, a fast clip is untouched
    for true_fps, floor, want in [(3.75, 12, 12.0), (5.0, 12, 12.0), (15.0, 12, 15.0), (30.0, 12, 30.0),
                                  (3.75, 0, 3.75), (3.75, None, 3.75)]:
        got = viz.playback_rate(true_fps, floor)
        check(f"playback_rate({true_fps} Hz, floor {floor}) = {want}", got == want, f"got {got}")
    check("a clip above the floor is NEVER slowed down", viz.playback_rate(30.0, 12) == 30.0)
    r = viz.playback_rate(3.75, 12)
    check("128 steps at 3.75 Hz: 34.1 s real time -> 10.7 s at the floor",
          abs(128 / 3.75 - 34.13) < 0.1 and abs(128 / r - 10.67) < 0.1, f"{128 / r:.2f}s")
    a, b = _Sink(), _Sink()
    w = RunWriter("/tmp", [a, b], playback_fps=30, preview_min_fps=12)
    w.video("t", blank * 128, 3.75, 0)
    check("RunWriter applies the floor: 128f @3.75Hz -> 30 fps over 10.7 s",
          a.got == [(320, 30.0)], str(a.got))
    check("and announces the speed-up exactly once", w._said_speed)
    s2 = _Sink()
    RunWriter("/tmp", [s2], playback_fps=30, preview_min_fps=0).video("t", blank * 128, 3.75, 0)
    check("preview_min_fps=0 -> honest real time (34.1 s)", s2.got == [(1024, 30.0)], str(s2.got))

    # 6. dataset videos are DATA: the ingestion paths must not pass playback_fps
    import inspect
    from ..data import processors
    from .. import data_generation
    for mod, fn in [(data_generation, "_render_fpv"), (processors, "_encode_ep")]:
        src = inspect.getsource(getattr(mod, fn))
        check(f"{mod.__name__}.{fn} does NOT pace (its clips are ingested into the dataset)",
              "playback_fps" not in src.split("viz.save_mp4")[-1].split(")")[0])

    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
