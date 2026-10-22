"""Side-by-side AR-rollout comparison of dynamics WMs (no prior vs kinematic-prior variants A/B).

The scalar val metrics are NOT comparable across runs (setting position_idx makes the prior runs'
pointwise_error position-only while the prior-free run's is full-obs), so the honest comparison is the
autoregressive trajectory itself: roll each model forward on the SAME val episodes with the SAME true action
sequence from the SAME context, and look at where the predicted path goes vs ground truth.

Each model is (label, ckpt). Pass as many as you like; --free/--A/--B are a convenience for the standard
three (no prior, owm-style observed-body-rate attitude, no-attitude-prior).

Outputs (logs/cave/prior_compare/):
  drift_vs_step.png     per-step position / velocity / attitude error vs rollout time, every model, mean±sd.
  trajectories_xy.png   top-down x-y path, true vs each model, small-multiples over episodes.
  trajectories_axes.png x(t), y(t), z(t) per episode, true vs each model.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

POS, QUAT, VEL, ANGVEL = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13)
# stable color per label; true is always black
COLORS = {"no prior": "#dd8452", "A: obs body-rate": "#4c72b0", "B: no attitude prior": "#55a868"}
_FALLBACK = ["#8172b3", "#c44e52", "#937860", "#da8bc3"]


def _load(ckpt, root, dev):
    from ...evaluation.standalone import load_checkpoint
    from ...training.setup import build_model, normalizer
    from ..mbari.run_uncertainty import _cfg_for

    cfg = _cfg_for(ckpt)
    cfg.data.root = root                      # saved config has the container path; use the host path here
    core = build_model(cfg).to(dev)
    load_checkpoint(core, ckpt)
    core.eval()
    return core, normalizer(cfg), int(cfg.data.P)


@torch.no_grad()
def rollout(core, norm, o, a, P, H, dev):
    """AR rollout H steps from the first P-step context on the episode's true actions -> (H,13) pred."""
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[:P]))[None].float().to(dev)}
    acts = norm.norm_act(torch.from_numpy(a[:P - 1 + H]))[None].float().to(dev)
    out = core.imagine_eval(ctx, acts, H, heads=["proprio"], norm=norm)
    return norm.denorm_obs(out["proprio"][0].cpu()).numpy()


def _rot_deg(qp, qt):
    qp = qp / (np.linalg.norm(qp, axis=-1, keepdims=True) + 1e-9)
    qt = qt / (np.linalg.norm(qt, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(2 * np.arccos(np.clip(np.abs((qp * qt).sum(-1)), 0, 1)))


def _color(label, i):
    return COLORS.get(label, _FALLBACK[i % len(_FALLBACK)])


def run(models, root, out="/app/logs/cave/prior_compare", n_ep=6, H=90, dev="cuda", log=print):
    """models: list of (label, ckpt). Rolls every model on the same n_ep val episodes and plots the comparison."""
    from ...data.dataset import DataConfig, load_split_episodes
    from ..mbari.run_uncertainty import _cfg_for

    loaded = []
    for label, ckpt in models:
        log(f"  loading [{label}]: {ckpt}")
        loaded.append((label, _load(ckpt, root, dev)))
    P = max(m[1][2] for m in loaded)

    # dcfg is REQUIRED keyword-only since 2026-09-21; take it from the first model's own
    # training config rather than assuming defaults, so a rate change cannot silently desync.
    _c = _cfg_for(models[0][1]); _c.data.root = root
    eps = load_split_episodes(root, "val", dcfg=DataConfig.from_cfg(_c),
                              repo_id="cave-grapevine")[:n_ep]
    truth = []
    preds = {label: [] for label, _ in loaded}
    for o, a in eps:
        Hc = min(H, len(o) - P)
        truth.append(o[P:P + Hc])
        for label, (core, norm, p) in loaded:
            preds[label].append(rollout(core, norm, o, a, p, Hc, dev))
        log(f"    rolled ep {len(truth)}/{len(eps)}  ({Hc} steps)")

    outp = pathlib.Path(out); outp.mkdir(parents=True, exist_ok=True)
    labels = [lab for lab, _ in loaded]
    _plot_drift(truth, preds, labels, outp, log)
    _plot_xy(truth, preds, labels, outp)
    _plot_axes(truth, preds, labels, outp)
    log(f"  wrote 3 figures to {outp}")
    return outp


def _curves(truth, preds):
    """Per-step mean position / velocity / attitude error across episodes (ragged -> min length)."""
    L = min(min(len(t) for t in truth), min(len(p) for p in preds))
    pos = np.stack([np.linalg.norm(p[:L, POS] - t[:L, POS], axis=1) for t, p in zip(truth, preds)])
    vel = np.stack([np.linalg.norm(p[:L, VEL] - t[:L, VEL], axis=1) for t, p in zip(truth, preds)])
    rot = np.stack([_rot_deg(p[:L, QUAT], t[:L, QUAT]) for t, p in zip(truth, preds)])
    return pos, vel, rot


def _plot_drift(truth, preds, labels, outp, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {lab: _curves(truth, preds[lab]) for lab in labels}
    L = min(c[0].shape[1] for c in curves.values())
    t = np.arange(L) / 30.0
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (ci, name, unit) in zip(axes, [(0, "Position", "m"), (1, "Velocity", "m/s"), (2, "Attitude", "deg")]):
        for i, lab in enumerate(labels):
            m = curves[lab][ci][:, :L]
            mu, sd = m.mean(0), m.std(0)
            c = _color(lab, i)
            ax.plot(t, mu, color=c, lw=2, label=lab)
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.15)
        ax.set_title(f"{name} error vs rollout time", fontsize=12)
        ax.set_xlabel("rollout time (s)"); ax.set_ylabel(f"|pred - true| ({unit})")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(f"Autoregressive drift: dynamics WM, no prior vs kinematic-prior variants "
                 f"(mean ± sd over {len(truth)} val episodes)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outp / "drift_vs_step.png", dpi=110); plt.close(fig)
    end = {lab: curves[lab][0][:, L - 1].mean() for lab in labels}
    log("    position drift @ end: " + "  ".join(f"[{lab}] {v:.3f} m" for lab, v in end.items()))


def _plot_xy(truth, preds, labels, outp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(truth); cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows), squeeze=False)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k >= n:
            ax.axis("off"); continue
        ax.plot(truth[k][:, 0], truth[k][:, 1], color="k", lw=2.5, label="true")
        for i, lab in enumerate(labels):
            p = preds[lab][k]
            ax.plot(p[:, 0], p[:, 1], color=_color(lab, i), lw=1.6, ls="--", label=lab)
        ax.scatter([truth[k][0, 0]], [truth[k][0, 1]], c="green", s=40, zorder=5, label="start")
        ax.set_title(f"val ep {k}", fontsize=10); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.axis("equal"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Top-down trajectory (x-y): true vs each model's autoregressive rollout", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outp / "trajectories_xy.png", dpi=110); plt.close(fig)


def _plot_axes(truth, preds, labels, outp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(4, len(truth))
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n), squeeze=False)
    for k in range(n):
        tt = np.arange(len(truth[k])) / 30.0
        for j, axis in enumerate("xyz"):
            ax = axes[k][j]
            ax.plot(tt, truth[k][:, j], color="k", lw=2, label="true")
            for i, lab in enumerate(labels):
                p = preds[lab][k]
                ax.plot(tt[:len(p)], p[:, j], color=_color(lab, i), ls="--", label=lab)
            ax.grid(alpha=0.3)
            if k == 0:
                ax.set_title(f"{axis}(t)")
            if k == n - 1:
                ax.set_xlabel("time (s)")
            if j == 0:
                ax.set_ylabel(f"ep {k}\n{axis} (m)")
            if k == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Per-axis position over the rollout: true vs each model", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outp / "trajectories_axes.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compare dynamics WM rollouts: no-prior vs kinematic-prior A/B.")
    ap.add_argument("--free", help="prior-FREE run dir or .ckpt")
    ap.add_argument("--A", help="variant A (owm-style observed-body-rate attitude) ckpt")
    ap.add_argument("--B", help="variant B (no attitude prior) ckpt")
    ap.add_argument("--model", nargs=2, action="append", metavar=("LABEL", "CKPT"),
                    help="extra model as LABEL CKPT (repeatable)")
    ap.add_argument("--root", required=True, help="packaged dataset dir (has val/)")
    ap.add_argument("--out", default="/app/logs/cave/prior_compare")
    ap.add_argument("--n_ep", type=int, default=6)
    ap.add_argument("--H", type=int, default=90)
    a = ap.parse_args()
    models = []
    if a.free:
        models.append(("no prior", a.free))
    if a.A:
        models.append(("A: obs body-rate", a.A))
    if a.B:
        models.append(("B: no attitude prior", a.B))
    models += [(lab, ck) for lab, ck in (a.model or [])]
    assert models, "pass at least one of --free/--A/--B/--model"
    run(models, a.root, out=a.out, n_ep=a.n_ep, H=a.H, log=lambda s: print(s, flush=True))
