"""Do per-episode condition labels survive packaging into `<split>/meta/tasks.parquet`?

That question has a measured answer of NO for every dataset published so far: `starling-2`'s eval split
holds 49 episodes from four OOD campaigns and labels all of them 'starling-2', so it cannot be sliced by
condition. This checks the fix end to end -- including an actual LeRobotDataset write and read-back,
because the failure was not in deriving the labels but in them being dropped on the way to disk.

    python -m quickdraw.smoke.episode_tasks
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import numpy as np

from ..data.generate import write_lerobot_split
from ..data.processors import Episode, _parse_starling_dir, _starling_campaigns

ok = bad = 0


def check(n, cond, extra=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {n}{('  ' + extra) if extra else ''}", flush=True)


def _fake_dump(d: str, per_campaign: dict[str, int], T: int = 4):
    """A minimal starling-shaped dump: data.npz + summary.json with run_dirs, no images needed."""
    eps, ep_map, rows = [], [], []
    for camp, n in per_campaign.items():
        for r in range(n):
            i = len(eps)
            ep_map += [i] * T
            rows.append({"episode_index": i, "num_frames": T,
                         "run_dir": f"/raw/{camp}/run_{i:04d}"})
            eps.append(camp)
    N = len(ep_map)
    np.savez(os.path.join(d, "data.npz"),
             states=np.random.randn(N, 16).astype(np.float32),
             actions=np.random.randn(N, 4).astype(np.float32),
             episode_indices_mapping=np.asarray(ep_map))
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({"episodes": rows}, f)
    return eps


def main() -> int:
    LAYOUT = {"campaign21_ood_a": 3, "campaign22_ood_b": 2, "campaign23_ood_c": 2}
    with tempfile.TemporaryDirectory() as d:
        want = _fake_dump(d, LAYOUT)
        eps = _parse_starling_dir(d, "ego")
        check("parser returns one Episode per run", len(eps) == len(want), f"{len(eps)} vs {len(want)}")
        check("every Episode carries its CAMPAIGN as task", [e.task for e in eps] == want,
              str([e.task for e in eps]))
        check("campaign comes from run_dir's PARENT, not the run timestamp",
              all("run_" not in (e.task or "") for e in eps))

        # a dump with no summary.json must still load, just unlabelled
        os.remove(os.path.join(d, "summary.json"))
        check("no summary.json -> loads fine, task=None (not a crash)",
              all(e.task is None for e in _parse_starling_dir(d, "ego")))
        # a MISMATCHED summary must be refused rather than mislabel
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump({"episodes": [{"run_dir": "/raw/wrong/run_0"}]}, f)
        check("summary.json that does not line up 1:1 is REFUSED, not applied",
              _starling_campaigns(d, len(want)) is None)

    # the part that actually failed before: does it reach tasks.parquet?
    with tempfile.TemporaryDirectory() as out:
        obs = [np.random.randn(4, 16).astype(np.float32) for _ in want]
        act = [np.random.randn(4, 4).astype(np.float32) for _ in want]
        write_lerobot_split(os.path.join(out, "eval"), "smoke/eval", obs, act, 30,
                            fpv_dir=None, cam="ego", task=want)
        import pyarrow.parquet as pq
        tp = glob.glob(os.path.join(out, "eval", "meta", "tasks.parquet"))
        check("tasks.parquet written", bool(tp))
        got = pq.read_table(tp[0]).to_pydict()
        names = set(next(v for k, v in got.items() if any(isinstance(x, str) for x in v)))
        check("ALL campaign names survive into tasks.parquet", names == set(LAYOUT),
              f"{sorted(names)}")
        epf = glob.glob(os.path.join(out, "eval", "meta", "episodes", "**", "*.parquet"), recursive=True)
        per_ep = pq.read_table(epf[0], columns=["episode_index", "tasks"]).to_pydict()
        flat = [t[0] if isinstance(t, (list, np.ndarray)) else t for t in per_ep["tasks"]]
        check("each EPISODE row references its own campaign", flat == want, str(flat))
        # and the whole point: a consumer can slice by condition
        idx = [i for i, t in enumerate(flat) if t == "campaign22_ood_b"]
        check("consumer can select one condition's episodes", idx == [3, 4], str(idx))

    # a str task still behaves exactly as before (every other processor relies on this)
    with tempfile.TemporaryDirectory() as out:
        obs = [np.random.randn(3, 6).astype(np.float32) for _ in range(2)]
        act = [np.random.randn(3, 2).astype(np.float32) for _ in range(2)]
        write_lerobot_split(os.path.join(out, "train"), "smoke/train", obs, act, 30,
                            fpv_dir=None, cam="fpv", task="torus")
        import pyarrow.parquet as pq
        got = pq.read_table(glob.glob(os.path.join(out, "train", "meta", "tasks.parquet"))[0]).to_pydict()
        names = set(next(v for k, v in got.items() if any(isinstance(x, str) for x in v)))
        check("a bare str task is unchanged (backwards compatible)", names == {"torus"}, str(names))
    check("Episode(task=...) defaults to None so every other processor is untouched",
          Episode(states=np.zeros((1, 2), np.float32), actions=np.zeros((1, 1), np.float32),
                  frames=None).task is None)
    print(f"\n{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
