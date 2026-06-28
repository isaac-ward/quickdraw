"""Golden-frame test for the renderer refactor: guarantees NO visual change.

Renders a fixed, deterministic canned scene through every public video/figure producer and compares to
saved reference frames pixel-for-pixel. Used to prove the TorusRenderer consolidation is byte-identical
to the legacy per-frame-plotter code.

Workflow:
  capture:  uv run python -m quickdraw.smoke.render_golden capture <dir>   # save refs (run on OLD code)
  compare:  uv run python -m quickdraw.smoke.render_golden compare <dir>   # assert new == refs
A capture-then-compare on the SAME code also serves as a run-to-run determinism check.
"""
import os
import sys

import numpy as np

from quickdraw.logging import viz

R, r = 0.75, 0.25


def _traj(T, seed):
    rng = np.random.default_rng(seed)
    th = np.cumsum(np.full(T, 0.18)) + 0.5 + 0.05 * rng.standard_normal(T).cumsum()
    ph = np.cumsum(np.full(T, 0.32))
    x = (R + r * np.cos(ph)) * np.cos(th)
    y = (R + r * np.cos(ph)) * np.sin(th)
    z = r * np.sin(ph)
    p = np.stack([x, y, z], 1)
    act = rng.standard_normal((T, 2)) * 0.4
    return p, act


def _cases():
    import matplotlib.pyplot as plt
    p, act = _traj(10, 0)
    avec = viz.action_ambient(p, act, R, r)

    def static():  # static atlas PNG path — exercises targets + start/end markers (videos don't)
        fig = viz.fig_torus_atlas(R, r, trajs=[{"xyz": p, "color": "black"}],
                                  targets=[("g", p[-1])], arrows=[(p[-1], avec[-1])],
                                  coloring="hsv", markers=True)
        a = viz._fig_rgb(fig); plt.close(fig); return a[None]

    def animate():
        return viz.animate_frames(R, r, "hsv", [{"xyz": p, "avec": avec.tolist(), "color": "black"}], n_frames=5)

    def traj():
        p2, _ = _traj(10, 1)
        return viz.traj_compare_frames(R, r, "hsv", p, p2, avec, P=4, n_frames=5)

    def control():
        pa, aa = _traj(10, 2); pb, ab = _traj(10, 3)
        ava = viz.action_ambient(pa, aa, R, r); avb = viz.action_ambient(pb, ab, R, r)
        gs = np.tile(p[-1], (len(pa), 1))
        base = p[:4]
        fan = [{"pts": base[None] + 0.04 * np.arange(8)[:, None, None], "ret": np.arange(8.0)}
               for _ in range(len(pa))]
        agents = [{"color": "black", "path": pa, "avec": ava, "goal_seq": gs},
                  {"color": "dimgray", "path": pb, "avec": avb, "goal_seq": gs}]
        return viz.control_compare_frames(R, r, "hsv", agents, n_frames=5, fan_seq=fan)

    def fpv():
        v = np.gradient(p, axis=0) * 60.0
        return viz.fpv_frames(R, r, "hsv", np.concatenate([p, v], 1), n_frames=5)

    return {"static": static, "animate": animate, "traj": traj, "control": control, "fpv": fpv}


def main():
    mode, refdir = sys.argv[1], sys.argv[2]
    os.makedirs(refdir, exist_ok=True)
    ok = True
    for name, fn in _cases().items():
        frames = np.ascontiguousarray(np.asarray(fn()))
        path = os.path.join(refdir, name + ".npy")
        if mode == "capture":
            np.save(path, frames); print(f"  captured {name:8s} {frames.shape}")
            continue
        ref = np.load(path)
        if ref.shape == frames.shape and np.array_equal(ref, frames):
            print(f"  [OK]   {name:8s} {frames.shape} pixel-identical")
        else:
            ok = False
            if ref.shape == frames.shape:
                d = np.abs(ref.astype(np.int32) - frames.astype(np.int32))
                print(f"  [DIFF] {name:8s} maxabs={d.max()} meanabs={d.mean():.5f} diff_px={(d > 0).sum()}/{d.size}")
            else:
                print(f"  [DIFF] {name:8s} shape ref{ref.shape} != new{frames.shape}")
    if mode == "capture":
        print("CAPTURED"); return 0
    print("ALL PIXEL-IDENTICAL" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
