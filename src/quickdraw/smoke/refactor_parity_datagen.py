"""Parity gate for the Phase-2 data-generation refactor (design/gym_refactor.md): the env-agnostic
`generate_episodes(env, ..., policy=...)` must produce BYTE-IDENTICAL torus obs/act to the pre-refactor
loop. The old code is loaded from `git show HEAD:src/quickdraw/data/generate.py` — run BEFORE committing
Phase 2 for a true old-vs-new diff (after commit it degrades to a run-to-run determinism check).
Frames are NOT re-diffed here: Phase 2 does not touch the FPV render path (same renderer, same obs ->
same frames; renderer identity is covered by smoke/render_golden).

Run (CPU only): CUDA_VISIBLE_DEVICES="" uv run python -m quickdraw.smoke.refactor_parity_datagen
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

from ..data.generate import generate_episodes
from ..environments.policies import make_policy
from ..environments.registry import make_env
from ..environments.torus import TorusConfig

N_TRAJ, STEPS, SEED = 2, 16, 0


def _old_generate_episodes():
    """Import the pre-Phase-2 data/generate.py from git HEAD as a throwaway module. In the container the
    repo's .git is not mounted — there, point OLD_GENERATE_PY at a host-side `git show` extract instead."""
    path = os.environ.get("OLD_GENERATE_PY")
    if not path:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        src = subprocess.check_output(["git", "-C", root, "show", "HEAD:src/quickdraw/data/generate.py"],
                                      text=True)
        path = os.path.join(tempfile.mkdtemp(), "generate_old.py")
        with open(path, "w") as f:
            f.write(src)
    spec = importlib.util.spec_from_file_location("quickdraw.data._generate_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # package context so its relative imports resolve
    spec.loader.exec_module(mod)
    return mod.generate_episodes


def main():
    old_gen = _old_generate_episodes()
    tc = TorusConfig()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    for sampler in ("ou", "bimodal"):
        o_old, a_old = old_gen(tc, N_TRAJ, STEPS, SEED, action_sampler=sampler)
        # new path 1: legacy TorusConfig signature (delegates to env+policy internally)
        o_leg, a_leg = generate_episodes(tc, N_TRAJ, STEPS, SEED, action_sampler=sampler)
        # new path 2: fully env-agnostic — make_env + make_policy, exactly as data_generation.py calls it
        env = make_env("torus_world", tc, batch=N_TRAJ)
        o_new, a_new = generate_episodes(env, N_TRAJ, STEPS, SEED, policy=make_policy(sampler, env))
        for tag, (o, a) in {"legacy-sig": (o_leg, a_leg), "env+policy": (o_new, a_new)}.items():
            eq = np.array_equal(o_old, o) and np.array_equal(a_old, a)
            check(f"{sampler:8s} {tag:10s} obs+act EXACTLY equal", eq, f"obs {o.shape}, act {a.shape}")
            if not eq:
                print(f"         max|d_obs|={np.abs(o_old - o).max():.3e}  max|d_act|={np.abs(a_old - a).max():.3e}")

    # random policy is NEW in Phase 2 (no old reference): check shape, action range, and determinism.
    o1, a1 = generate_episodes(make_env("torus_world", tc, batch=N_TRAJ), N_TRAJ, STEPS, SEED,
                               action_sampler="random")
    o2, a2 = generate_episodes(make_env("torus_world", tc, batch=N_TRAJ), N_TRAJ, STEPS, SEED,
                               action_sampler="random")
    check("random   policy     shape/range/deterministic",
          a1.shape == (N_TRAJ, STEPS, 2) and float(np.abs(a1).max()) <= tc.a_max
          and np.array_equal(o1, o2) and np.array_equal(a1, a2))

    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
