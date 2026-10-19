"""Generate the MBARI dataset: nominal + OOD, in parallel, as per-episode npz shards.

PARALLEL BECAUSE RENDERING IS THE WHOLE COST. One 30 s episode is 900 control steps x 4 cameras x 2
rasterisations = 7,200 renders, and a 144x192 EGL render costs ~8 ms no matter what the quality
settings say -- measured: shadows off, no multisampling and building the scene once each bought
something, but the floor is the rasteriser. 240 nominal episodes alone is 4 hours single-threaded.

Workers are PROCESSES, each with its own MuJoCo model, EGL context and renderer, pinned round-robin
across the visible GPUs. The work is embarrassingly parallel -- episodes share nothing but the XML --
so throughput scales until the GPUs saturate.

Each episode lands as its own `.npz` so a crashed worker costs one episode, not the run.
"""
from __future__ import annotations

import glob
import os
import pathlib
import time

import numpy as np

FPS = 30


def _ep_path(out, split, idx):
    return pathlib.Path(out) / split / f"ep_{idx:05d}.npz"


def _worker(args):
    """One episode, in its own process. Returns (ok, split, idx, seconds, note)."""
    split, idx, kind, seed, steps, size, out, gpu, glitch_p, intruder = args
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    try:
        from . import intruders, record
        model = _worker.model if getattr(_worker, "model", None) is not None else None
        if model is None:
            model = record.load_model() if intruder is None else intruders.load_model(intruder)
            _worker.model = model
            _worker.kind_key = intruder
        elif getattr(_worker, "kind_key", None) != intruder:
            model = record.load_model() if intruder is None else intruders.load_model(intruder)
            _worker.model = model
            _worker.kind_key = intruder
        extra = intruders.mover(model, intruder, seed) if intruder else None
        ep = record.rollout(model, kind, seed, steps=steps, size=size,
                            glitch_p=glitch_p, extra=extra)
        if ep is None:
            return (False, split, idx, time.time() - t0, "rollout returned None")
        p = _ep_path(out, split, idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, obs=ep["obs"], act=ep["act"],
                            **{f"frames_{c}": v for c, v in ep["frames"].items()},
                            meta=np.array(str(ep["meta"])))
        return (True, split, idx, time.time() - t0, "")
    except Exception as exc:                                  # noqa: BLE001 - one episode must not kill the run
        return (False, split, idx, time.time() - t0, f"{type(exc).__name__}: {exc}")


def plan_nominal(n=720, steps=900, seed0=1000):
    """720 episodes = 6.00 h at 30 Hz, 240 per behaviour class, interleaved so ANY PREFIX is
    class-balanced -- which is what lets a bigger run reuse a smaller one's episodes unchanged:
    job i always has seed 1000+i and class CLASSES[i%3], whatever n is."""
    from .trajectories import CLASSES

    jobs = []
    for i in range(n):
        kind = CLASSES[i % len(CLASSES)]
        jobs.append(("train", i, kind, seed0 + i, steps, None, 0.0, None))
    return jobs


def plan_ood(n_per=50, steps=360, seed0=50000):
    """Three OOD sets. 50 trajectories each, 12 s apiece = 10 minutes per set.

    Behaviour classes are drawn from the SAME nominal mix, so the only thing that changes is the
    condition under test -- a glitching thruster, a shark, a purple buoy. If the trajectories
    also differed the detector could separate them on motion statistics alone and tell us nothing."""
    from .trajectories import CLASSES

    jobs = []
    for s, (glitch_p, intruder) in enumerate([(0.30, None), (0.0, "fish"), (0.0, "buoy")]):
        split = ["ood_thruster", "ood_fish", "ood_buoy"][s]
        for i in range(n_per):
            jobs.append((split, i, CLASSES[i % len(CLASSES)], seed0 + 1000 * s + i, steps,
                         None, glitch_p, intruder))
    return jobs


def steps_by_class(out, split):
    """{class: total steps on disk} for one split, read from each episode's own meta."""
    import ast
    import collections

    got = collections.Counter()
    for f in glob.glob(str(pathlib.Path(out) / split / "ep_*.npz")):
        if ".tmp." in f:
            continue
        try:
            m = ast.literal_eval(str(np.load(f)["meta"]))
        except Exception:                                      # noqa: BLE001 - a partial file
            continue
        got[m["kind"]] += int(m["steps"])
    return got


def plan_topup(out, split, target_steps, steps, glitch_p, intruder, seed_base, max_jobs=4000):
    """Jobs that fill each behaviour class up to `target_steps`, counting what is ALREADY on disk.

    WHY THIS EXISTS. Episodes now end on contact, so a 900-step job does not necessarily yield 900
    steps -- measured on v6, the random class keeps about half. Generating a fixed episode COUNT
    therefore silently delivers less data than asked for, and unevenly: the random class truncates
    far more than the planned circles, so an episode-balanced set is no longer a STEP-balanced one.
    The spec was a third of the time in each class, so the target is expressed in STEPS and each
    class is topped up until it gets there.

    Seeds come from a reserved band per split so a top-up can never collide with the base plan.
    """
    from .trajectories import CLASSES

    got = steps_by_class(out, split)
    have = {c: int(got.get(c, 0)) for c in CLASSES}
    jobs, k = [], 0
    for ci, c in enumerate(CLASSES):
        deficit = target_steps - have[c]
        while deficit > 0 and k < max_jobs:
            idx = 900000 + ci * 100000 + k                    # episode index, well clear of the plan
            jobs.append((split, idx, c, seed_base + ci * 100000 + k, steps, None, glitch_p, intruder))
            deficit -= steps                                   # assume full length; a short one is
            k += 1                                             # caught on the next pass
    return jobs, have


def run_serial(jobs, out, size=(144, 192), log=print, every=5):
    """Run `jobs` in THIS process, skipping episodes already on disk.

    Deliberately not multiprocessing. A spawn Pool inside this container hung before starting a single
    worker -- no output, no GPU use, two live processes -- and the shard-per-process shell pattern in
    `shard()` gets the same parallelism with nothing to deadlock: independent interpreters, one EGL
    context each, no pickling, and a crashed shard costs its own episodes and nothing else.
    """
    out = pathlib.Path(out)
    todo = [j for j in jobs if not _ep_path(out, j[0], j[1]).exists()]
    log(f"[gen] {len(jobs)} planned, {len(jobs)-len(todo)} on disk, {len(todo)} to run")
    t0, fails = time.time(), []
    from . import intruders, record
    model_cache = {}
    for n, (split, idx, kind, seed, steps, _sz, glitch_p, intruder) in enumerate(todo, 1):
        try:
            if intruder not in model_cache:
                model_cache[intruder] = (record.load_model() if intruder is None
                                         else intruders.load_model(intruder))
            model = model_cache[intruder]
            extra = intruders.mover(model, intruder, seed) if intruder else None
            # RETRY ONCE ON AN EGL FAULT. "Offscreen framebuffer is not complete" is GPU memory
            # pressure, not a bad episode -- it happened 24 shards deep while two training runs
            # held 160 GB across both cards. Dropping the episode silently is the wrong response:
            # rebuild the renderer and try again, and only then give up.
            try:
                ep = record.rollout(model, kind, seed, steps=steps, size=size,
                                    glitch_p=glitch_p, extra=extra)
            except Exception as exc:                          # noqa: BLE001
                if "framebuffer" not in str(exc).lower():
                    raise
                model_cache.pop(intruder, None)
                time.sleep(5.0)
                model = (record.load_model() if intruder is None
                         else intruders.load_model(intruder))
                model_cache[intruder] = model
                extra = intruders.mover(model, intruder, seed) if intruder else None
                ep = record.rollout(model, kind, seed, steps=steps, size=size,
                                    glitch_p=glitch_p, extra=extra)
            if ep is None:
                fails.append((split, idx, "rollout returned None")); continue
            p = _ep_path(out, split, idx)
            p.parent.mkdir(parents=True, exist_ok=True)
            # ATOMIC, for the same reason tank.xml is: a reader (the packer, a progress check, the
            # top-up planner counting steps) must never see a half-written episode. np.savez writes
            # straight to the path, so a concurrent reader gets `BadZipFile`.
            tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, obs=ep["obs"], act=ep["act"],
                                **{f"frames_{c}": v for c, v in ep["frames"].items()},
                                meta=np.array(str(ep["meta"])))
            os.replace(tmp, p)
        except Exception as exc:                              # noqa: BLE001
            fails.append((split, idx, f"{type(exc).__name__}: {exc}"))
        if n % every == 0 or n == len(todo):
            el = time.time() - t0
            log(f"[gen] {n}/{len(todo)}  {el/60:.1f} min, {(len(todo)-n)*el/n/60:.1f} min left, "
                f"{len(fails)} failed")
    for f in fails[:10]:
        log(f"[gen] FAILED {f[0]}/{f[1]}: {f[2]}")
    return fails


def shard(jobs, k, n):
    """Episodes for shard k of n, round-robin so every shard gets a mix of splits and lengths."""
    return [j for i, j in enumerate(jobs) if i % n == k]


def run(jobs, out, size=(144, 192), workers=12, gpus=(0, 1), log=print):
    import multiprocessing as mp

    out = pathlib.Path(out)
    todo = [(sp, i, k, sd, st, size, str(out), gpus[n % len(gpus)], gp, intr)
            for n, (sp, i, k, sd, st, _sz, gp, intr) in enumerate(jobs)
            if not _ep_path(out, sp, i).exists()]
    log(f"[gen] {len(jobs)} episodes planned, {len(jobs)-len(todo)} already on disk, {len(todo)} to run "
        f"on {workers} workers across GPUs {gpus}")
    if not todo:
        return []
    t0 = time.time()
    done, fails = 0, []
    with mp.get_context("spawn").Pool(workers) as pool:
        for ok, sp, i, dt, note in pool.imap_unordered(_worker, todo, chunksize=1):
            done += 1
            if not ok:
                fails.append((sp, i, note))
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                log(f"[gen] {done}/{len(todo)}  {el/60:.1f} min elapsed, "
                    f"{(len(todo)-done)*el/max(1,done)/60:.1f} min left, {len(fails)} failed")
    for sp, i, note in fails[:10]:
        log(f"[gen] FAILED {sp}/{i}: {note}")
    return fails


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate MBARI episodes (one shard per process).")
    ap.add_argument("--out", default="/app/logs/mbari/data/v6")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--nominal", type=int, default=720)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--ood-per", type=int, default=50)
    ap.add_argument("--ood-steps", type=int, default=360)
    ap.add_argument("--topup", action="store_true",
                    help="emit only the jobs needed to reach the per-class STEP targets")
    ap.add_argument("--per-class", type=int, default=216000,   # 2.00 h each -> 6.00 h nominal
                    help="target steps per behaviour class in train")
    ap.add_argument("--ood-per-class", type=int, default=6000,  # 3.33 min each -> 10 min per set
                    help="target steps per behaviour class in each OOD split")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if a.topup:
        all_jobs = []
        for split, gp, intr, st, tgt in (("train", 0.0, None, a.steps, a.per_class),
                                         ("ood_thruster", 0.30, None, a.ood_steps, a.ood_per_class),
                                         ("ood_fish", 0.0, "fish", a.ood_steps, a.ood_per_class),
                                         ("ood_buoy", 0.0, "buoy", a.ood_steps, a.ood_per_class)):
            j, have = plan_topup(a.out, split, tgt, st, gp, intr,
                                 seed_base=5000000 + 1000000 * len(all_jobs) % 7000000)
            all_jobs += j
            if a.shard == 0:
                short = {c: tgt - v for c, v in have.items()}
                print(f"[topup] {split:13s} have {dict(have)}  short {short}", flush=True)
    else:
        all_jobs = plan_nominal(n=a.nominal, steps=a.steps) + plan_ood(n_per=a.ood_per,
                                                                      steps=a.ood_steps)
    mine = shard(all_jobs, a.shard, a.shards)
    run_serial(mine, a.out, log=lambda m: print(f"[s{a.shard}] {m}", flush=True))
