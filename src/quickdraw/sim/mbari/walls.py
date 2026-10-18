"""Tank walls as boxes with real window openings, and the rooms that sit behind them.

A textured plane with dark rectangles painted on it is a wall with pictures of windows. To see a room
THROUGH a window -- and to have that view shift as the vehicle moves -- the opening has to be a real
hole, so each wall is assembled from solid strips that leave gaps:

      +---------------------------------------+   top strip      (z: window row 1 top .. 0)
      |     [] pillar  []  WIN  []  WIN  []    |   row 1          (3 openings, 4 pillars)
      +---------------------------------------+   middle strip
      |     []         []  WIN  []  WIN  []    |   row 2
      +---------------------------------------+   bottom strip   (z: -D .. window row 2 bottom)

Eleven boxes per wall, 44 in all, against four planes before. The cost is worth it: all three windows
in a row look into the SAME room from three positions along its front face, which is what a row of
windows into a conference room actually looks like.

NO FIDUCIAL MARKERS ON THE WALLS. They were clutter at 144x192 and the concrete shade already
identifies which wall the camera is pointing at. The floor keeps its grid.
"""
from __future__ import annotations

import numpy as np

T = 0.18                      # wall thickness
WIN_W, WIN_H = 2.0, 1.9       # window opening
N_WIN = 3


def _spans(half, n=N_WIN, w=WIN_W):
    """Window centres along a wall of half-length `half`, and the solid pillar spans between them."""
    centres = [(-half * 0.58), 0.0, (half * 0.58)][:n] if n == 3 else list(
        np.linspace(-half * 0.6, half * 0.6, n))
    holes = [(c - w / 2, c + w / 2) for c in centres]
    pillars, edge = [], -half
    for a, b in holes:
        pillars.append((edge, a))
        edge = b
    pillars.append((edge, half))
    return centres, [p for p in pillars if p[1] - p[0] > 0.01]


def build_wall(name, axis, sign, at, half_len, tank_d, rings, mat):
    """Full wall: three horizontal strips plus four pillars per window row. Returns (geoms, centres)."""
    g = []
    centres, _ = _spans(half_len)
    rows = [(-tank_d * zf - WIN_H / 2, -tank_d * zf + WIN_H / 2) for zf in rings]
    rows.sort()                                                   # deepest first
    edges = [-tank_d] + [z for r in rows for z in r] + [0.0]

    def box(u0, u1, z0, z1, tag):
        cu, cz = (u0 + u1) / 2, (z0 + z1) / 2
        su, sz = (u1 - u0) / 2, (z1 - z0) / 2
        if su <= 0.005 or sz <= 0.005:
            return
        if axis == "y":
            pos, size = f"{cu:.3f} {sign*at:.3f} {cz:.3f}", f"{su:.3f} {T/2:.3f} {sz:.3f}"
        else:
            pos, size = f"{sign*at:.3f} {cu:.3f} {cz:.3f}", f"{T/2:.3f} {su:.3f} {sz:.3f}"
        g.append(f'<geom name="{name}_{tag}" type="box" material="{mat}" pos="{pos}" size="{size}"/>')

    # horizontal strips: below row A, between the rows, above row B
    box(-half_len, half_len, edges[0], edges[1], "s0")
    box(-half_len, half_len, edges[2], edges[3], "s1")
    box(-half_len, half_len, edges[4], edges[5], "s2")
    # pillars within each row
    for ri, (z0, z1) in enumerate(rows):
        _, pillars = _spans(half_len)
        for pi, (a, b) in enumerate(pillars):
            box(a, b, z0, z1, f"r{ri}p{pi}")
    return g, centres, [(-tank_d * zf) for zf in sorted(rings, reverse=True)]
