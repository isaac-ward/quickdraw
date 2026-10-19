"""The MPPI cloud of possible futures, drawn in tank coordinates.

`viz._add_fan` already renders this exact structure -- {"pts": (K,H,3), "ret": (K,)}, candidates
coloured by return -- but it is pyvista against torus geometry and takes the torus radii, so it
cannot draw a 15 x 10 x 10 m box. This keeps the same contract and renders two orthographic panels
instead: TOP-DOWN (x,y) and SIDE (x,z), which between them show every axis and need no camera.

Each frame shows, at one replan: the candidate rollouts coloured by cost (dark = cheap = what MPPI
weights most), the executed trail so far, the vehicle, and the goal. What to look for is whether
the cheap candidates actually end on the goal -- if the whole fan is equally bad, the planner is
not finding the approach; if the fan is tight and on-goal but the vehicle still sits off it, the
problem is execution, not search.
"""
from __future__ import annotations

import numpy as np

from .tank import TANK_D, TANK_L, TANK_W


def _panel(ax, fan, trail, pos, goal, ij, lims, title):
    i, j = ij
    for k in range(len(fan["pts"])):
        ax.plot(fan["pts"][k][:, i], fan["pts"][k][:, j], lw=0.6, alpha=0.55,
                color=fan["_c"][k], zorder=2)
    if len(trail) > 1:
        ax.plot(trail[:, i], trail[:, j], color="#00d0ff", lw=1.6, zorder=3)
    ax.plot([goal[i]], [goal[j]], marker="*", ms=16, color="#ffd400", zorder=5)
    ax.plot([pos[i]], [pos[j]], marker="o", ms=7, color="#ffffff", mec="k", zorder=6)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.set_title(title, color="w", fontsize=9)
    ax.set_facecolor("#10151a")
    ax.tick_params(colors="#889", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#445")


def frames(res, goal, stride=1, max_frames=120, dpi=96):
    """-> (N,H,W,3) uint8. `res` is a run_episode result whose `fan` list was collected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fan, traj = res["fan"], res["traj"]
    exec_n = max(1, len(traj) // max(1, len(fan)))
    out = []
    sel = list(range(0, len(fan), stride))[:max_frames]
    for n in sel:
        f = dict(fan[n])
        r = f["ret"]
        z = (r - r.min()) / max(1e-9, np.ptp(r))         # 1 = cheapest
        f["_c"] = cm.viridis(z)
        t = min(len(traj) - 1, n * exec_n)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=dpi, facecolor="#0b0f13")
        _panel(axes[0], f, traj[:t + 1, :3], traj[t, :3], goal, (0, 1),
               ((-TANK_L / 2, TANK_L / 2), (-TANK_W / 2, TANK_W / 2)),
               f"top-down  x-y   step {t}   dist {np.linalg.norm(traj[t, :3] - goal):.3f} m")
        _panel(axes[1], f, traj[:t + 1, :3], traj[t, :3], goal, (0, 2),
               ((-TANK_L / 2, TANK_L / 2), (-TANK_D, 0.0)),
               f"side  x-z   {len(f['pts'])} candidates, cost {-r.max():.2f}..{-r.min():.2f}")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        out.append(buf)
        plt.close(fig)
    return np.stack(out)
