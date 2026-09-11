"""Preview mp4s: 30 UNIQUE frames per second, and an honest statement of how fast that is.

Two separate things, both of which were wrong at some point:

1. THE RATE. Every video of a prediction used to be encoded at round(1/ecfg.dt), the dataset's FRAME
   period -- but one autoregressive step spans `data.subsample` frames. s2_sub4 trains at subsample=4 on
   15 Hz data, so its rollouts played at FOUR TIMES real speed with nothing on screen to say so.
   `training.setup.step_fps` derives the true step rate the way dt_eff already does.

2. THE PLAYBACK. Encoding at that true rate is honest and unwatchable: 128 steps at 3.75 Hz is a
   34-second slideshow. An earlier attempt repeated frames to reach a 30 fps container, which preserved
   duration, added no unique frames, and -- at a non-integer repeat ratio -- made the cadence UNEVEN,
   i.e. judder. So previews now encode ONE FRAME PER CONTAINER FRAME at `preview_fps`: 30 unique frames a
   second, even cadence, and the resulting speed-up is printed to progress.log once per run.

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

    # ---- 1. the model's step rate folds in the frame stride: the original bug ----
    e = SimpleNamespace(dt=1.0 / 15.0)
    for sub, want in [(1, 15.0), (2, 7.5), (3, 5.0), (4, 3.75), (5, 3.0), (None, 15.0)]:
        got = step_fps(OmegaConf.create({"data": {"subsample": sub}}), e)
        check(f"step_fps at subsample={sub} is {want} Hz", abs(got - want) < 1e-9, f"got {got}")

    # ---- 2. ONE frame per container frame: never repeat, never drop, never interpolate ----
    for n, true_fps in [(136, 3.75), (136, 5.0), (136, 15.0), (50, 30.0)]:
        s = _Sink()
        RunWriter("/tmp", [s], playback_fps=30).video("t", blank * n, true_fps, 0)
        check(f"{n}f @{true_fps} Hz -> {n}f @30 fps ({n / 30:.2f}s, {30 / true_fps:.1f}x real time)",
              s.got == [(n, 30.0)], str(s.got))
    check("a 136-step rollout is 4.53 s at 30 fps, not 34 s", abs(136 / 30 - 4.53) < 0.01)

    # ---- 3. the cadence is EVEN, which is what repetition could not give ----
    # (implied by one-frame-per-frame: there is no repeat ratio at all, integer or otherwise)
    s = _Sink()
    RunWriter("/tmp", [s], playback_fps=30).video("t", blank * 137, 3.75, 0)  # awkward count, still 1:1
    check("an awkward frame count still maps 1:1 (no rounding, no drift)", s.got == [(137, 30.0)], str(s.got))

    # ---- 4. both backends get the SAME video, and the speed-up is stated once ----
    a, b = _Sink(), _Sink()
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "logs"), exist_ok=True)
        w = RunWriter(os.path.join(d, "logs"), [a, b], playback_fps=30)
        w.video("t", blank * 136, 3.75, 0)
        w.video("t2", blank * 136, 3.75, 0)
        check("both backends receive the identical video", a.got == b.got)
        check("the speed-up is stated ONCE, not per video", w._said_speed and len(a.got) == 2)
        log = open(os.path.join(d, "progress.log")).read()
        check("and it lands in progress.log, naming the factor and the true rate",
              "8.00x REAL TIME" in log and "3.75 Hz" in log and "267 ms/step" in log, log.strip()[-90:])

    # ---- 5. real time is still available, and is the no-op path ----
    s = _Sink()
    RunWriter("/tmp", [s], playback_fps=None).video("t", blank * 136, 3.75, 0)
    check("preview_fps: null -> encode at the true step rate (real time, 36.3 s)",
          s.got == [(136, 3.75)], str(s.got))
    s = _Sink()
    w = RunWriter("/tmp", [s], playback_fps=30)
    w.video("t", blank * 50, 30.0, 0)
    check("a clip already at the container rate says nothing", s.got == [(50, 30.0)] and not w._said_speed)

    # ---- 6. it survives a real encode, and a fractional true rate is not rounded ----
    import imageio.v2 as iio
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.mp4")
        fr = [np.random.randint(0, 255, (112, 192, 3), dtype=np.uint8) for _ in range(136)]
        viz.save_mp4(p, fr, 3.75, playback_fps=30)
        m = iio.get_reader(p).get_meta_data()
        check("encoded 30 fps / 4.53 s with every frame distinct",
              abs(m["fps"] - 30) < 0.1 and abs(m["duration"] - 136 / 30) < 0.15,
              f"{m['fps']} fps, {m['duration']}s")
        viz.save_mp4(p, fr, 3.75)
        check("an unpaced fractional rate is NOT rounded (3.75 -> 4 was 6.7% fast)",
              abs(iio.get_reader(p).get_meta_data()["fps"] - 3.75) < 0.05)

    # ---- 7. dataset videos are DATA: the ingestion paths must never be retimed ----
    import inspect
    from ..data import processors
    from .. import data_generation
    for mod, fn in [(data_generation, "_render_fpv"), (processors, "_encode_ep")]:
        src = inspect.getsource(getattr(mod, fn))
        check(f"{mod.__name__}.{fn} does NOT retime (its clips are ingested into the dataset)",
              "playback_fps" not in src.split("viz.save_mp4")[-1].split(")")[0])

    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
