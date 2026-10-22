"""Per-channel proprio prediction error, measured against the simulator rather than inferred from control.

WHY THIS EXISTS. The learned control arm fails, and integrating the model's predicted velocities from
the true state fixed ATTITUDE (93 deg -> 39) while barely moving POSITION (3.80 m -> 3.21). That
asymmetry says the angular-rate channel is usable and the linear-velocity channel is not -- but it
says it indirectly, through a closed loop with a planner in it. This measures the channels directly.

IT SEPARATES BIAS FROM VARIANCE, which is the part that changes what you do next. Integration turns a
constant velocity bias into linear position drift, and a bias is usually cheap -- a normalisation or
offset problem. Variance is not: it needs capacity or more training. `mean_err` is the signed bias per
axis, `rmse` the total, and `std_err` what is left after the bias is removed.

    python -m quickdraw.sim.mbari.proprio_error <run_dir_or_ckpt> [--horizon 72] [--episodes 8]
"""
from __future__ import annotations

import argparse

import numpy as np

NAMES = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"]
GROUPS = {"position": (0, 3), "quaternion": (3, 7), "lin_vel": (7, 10), "ang_vel": (10, 13)}


def measure(ckpt, horizon=72, episodes=8, stride=64, dev="cuda", log=print):
    """Open-loop rollout from real context; per-dim error at a ladder of horizons."""
    import torch

    from ...data.dataset import DataConfig, load_split_episodes_mm
    from ...training.setup import build_model, normalizer, resolve_data_root
    from .run_uncertainty import _cfg_for, held_out_ids

    cfg = _cfg_for(ckpt)
    core = build_model(cfg).to(dev).eval()
    from ...evaluation.standalone import load_checkpoint
    load_checkpoint(core, ckpt)
    norm = normalizer(cfg)
    P = int(cfg.data.P)
    val_ids, _ = held_out_ids(cfg.data.root)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", dcfg=DataConfig.from_cfg(cfg),
                                 img_size=(144, 192), cam="fpv", repo_id=cfg.data.get("repo_id", "torus"))
    eps = eps[:episodes]
    log(f"  {len(eps)} held-out episodes, horizon {horizon}, context P={P}")

    err = []                                             # (n_windows, horizon, 13) signed error
    for o, a, fr in eps:
        key = next(iter(fr))
        for st in range(0, len(o) - P - horizon, stride):
            ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[st:st + P]))[None].float().to(dev),
                   key: torch.from_numpy(fr[key][st:st + P])[None].float().div(255.0).to(dev)}
            acts = norm.norm_act(torch.from_numpy(a[st:st + P + horizon - 1]))[None].float().to(dev)
            with torch.no_grad():
                out = core.imagine_shared(ctx, acts, horizon, 1, heads=["proprio"], norm=norm)
            pred = norm.denorm_obs(out["proprio"].cpu())[0].numpy()        # (horizon,13)
            err.append(pred - o[st + P:st + P + horizon])
    e = np.stack(err)                                                      # (N,H,13)
    log(f"  {len(e)} rollout windows\n")

    for k in [1, 8, 16, 32, min(horizon, 72)]:
        if k > horizon:
            continue
        ek = e[:, k - 1]                                                   # (N,13) error at step k
        log(f"  --- @+{k} ---")
        log(f"    {'group':11s} {'rmse':>9} {'|bias|':>9} {'std':>9}   per-axis bias")
        for g, (i, j) in GROUPS.items():
            v = ek[:, i:j]
            rmse = float(np.sqrt((v ** 2).mean()))
            bias = v.mean(0)
            std = float(v.std(0).mean())
            log(f"    {g:11s} {rmse:9.4f} {np.abs(bias).mean():9.4f} {std:9.4f}   "
                + " ".join(f"{NAMES[i + d]}={bias[d]:+.4f}" for d in range(j - i)))
        log("")
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("--horizon", type=int, default=72)
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args()
    measure(a.ckpt, horizon=a.horizon, episodes=a.episodes)


if __name__ == "__main__":
    main()
