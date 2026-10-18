"""Tank walls as boxes with real window openings, framed in black, and the rooms behind them.

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

WINDOW FRAMES. Each opening gets a black rim flush with the inner wall face, plus two vertical
mullions dividing it into thirds. The mullions stand PROUD of the wall by `PROUD` and return to the
rim plane through 45-degree joins at top and bottom, so the frame reads as real steelwork sitting in
front of the glass rather than as lines painted on it.

NO FIDUCIAL MARKERS ON THE WALLS. They were clutter at 144x192 and the concrete shade already
identifies which wall the camera is pointing at. The floor keeps its grid.
"""
from __future__ import annotations

import numpy as np

T = 0.18                      # wall thickness
WIN_W, WIN_H = 1.6, 1.52      # window opening, 80% of the first size, shrunk about its own centre
N_WIN = 3
RIM = 0.10                    # width of the black rim around an opening
BAR = 0.055                   # half-width of a vertical mullion
PROUD = 0.16                  # how far the mullions stand out from the inner wall face
FRAME_MAT = "m_frame"
TAG = 0.40                    # half-size of a wall fiducial: 0.80 m square, matching the floor
#                               grid, whose 64 px tiles on a 1024 px texture over a 15 x 10 m plane
#                               come out 0.94 x 0.63 m
MAX_ASPECT = 1.15             # no wall box wider than this many times its own height, which
#                               bounds any residual stretch to 15% whatever MuJoCo's face UV
#                               convention turns out to be -- two attempts to out-guess it failed


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


def _place(axis, sign, at, cu, cn, cz, su, sn, sz, euler=None):
    """A box given in WALL coordinates: u along the wall, n along its outward normal, z up."""
    if axis == "y":
        pos, size = f"{cu:.4f} {sign*cn:.4f} {cz:.4f}", f"{su:.4f} {sn:.4f} {sz:.4f}"
        rot = f' euler="{euler:.4f} 0 0"' if euler is not None else ""
    else:
        pos, size = f"{sign*cn:.4f} {cu:.4f} {cz:.4f}", f"{sn:.4f} {su:.4f} {sz:.4f}"
        rot = f' euler="0 {euler:.4f} 0"' if euler is not None else ""
    return pos, size, rot


def _frame(name, axis, sign, at, cu, z0, z1, tag):
    """Black rim + two proud mullions with 45-degree returns, for one window opening."""
    g = []
    face = at - T / 2                      # inner face of the wall, where the rim sits
    hw, hh = WIN_W / 2, (z1 - z0) / 2
    cz = (z0 + z1) / 2

    def box(cu_, cn_, cz_, su_, sn_, sz_, t, euler=None):
        pos, size, rot = _place(axis, sign, at, cu_, cn_, cz_, su_, sn_, sz_, euler)
        g.append(f'<geom name="{name}_{tag}_{t}" type="box" material="{FRAME_MAT}" '
                 f'pos="{pos}" size="{size}"{rot} contype="0" conaffinity="0"/>')

    # rim: four bars flush with the inner face, hugging the outside of the opening
    box(cu, face - RIM / 2, z1 + RIM / 2, hw + RIM, RIM / 2, RIM / 2, "rt")
    box(cu, face - RIM / 2, z0 - RIM / 2, hw + RIM, RIM / 2, RIM / 2, "rb")
    box(cu - hw - RIM / 2, face - RIM / 2, cz, RIM / 2, RIM / 2, hh, "rl")
    box(cu + hw + RIM / 2, face - RIM / 2, cz, RIM / 2, RIM / 2, hh, "rr")

    # two mullions, standing PROUD, each returning to the rim plane through a 45-degree join
    diag = PROUD * np.sqrt(2) / 2
    for i, du in enumerate((-WIN_W / 6, WIN_W / 6)):
        box(cu + du, face - RIM - PROUD, cz, BAR, BAR, hh - PROUD, f"m{i}")
        for j, zend in enumerate((z1, z0)):
            s = 1 if j == 0 else -1            # +1 = the join at the top of the opening
            zc = zend - s * PROUD / 2
            nc = face - RIM / 2 - PROUD / 2
            # the join runs 45 degrees in the (normal, z) plane; which way depends on the wall's
            # outward normal and on whether it is the top or the bottom return
            e = (-0.7854 * s * sign) if axis == "y" else (0.7854 * s * sign)
            box(cu + du, nc, zc, BAR, BAR, diag + BAR, f"m{i}j{j}", euler=e)
    return g


def build_wall(name, axis, sign, at, half_len, tank_d, rings, mat_fn):
    """Full wall: three horizontal strips, four pillars per window row, and a frame per opening.

    `mat_fn(tag, u0, u1, z0, z1) -> material name` lets the caller give every box its OWN material.
    That is what kills the stretched concrete: each box gets the crop of a single wall-wide texture
    that matches its own footprint, so the grain is isotropic and continuous across the whole wall.

    Returns (geoms, window centres, ring depths, fiducial placements). The caller makes the fiducial
    materials, because it owns the texture build."""
    g, boxes, tags = [], [], []
    centres, _ = _spans(half_len)
    rows = [(-tank_d * zf - WIN_H / 2, -tank_d * zf + WIN_H / 2) for zf in rings]
    rows.sort()                                                   # deepest first
    edges = [-tank_d] + [z for r in rows for z in r] + [0.0]

    def box(u0, u1, z0, z1, tag):
        """One wall panel, SPLIT ALONG THE WALL until no piece is wider than MAX_ASPECT x its height.

        This is what finally fixed the stretched concrete, after two attempts at out-guessing
        MuJoCo's cube-face UV convention. Whatever that convention is, it stretches one image to
        fill a face -- so if the face is near-square, the result is near-isotropic, and no knowledge
        of the mapping is needed. A 15 x 1.2 m strip becomes nine 1.7 x 1.2 m panels, each carrying
        its own crop of the same wall-wide concrete field, so the surface still reads as continuous.
        """
        if u1 - u0 <= 0.01 or z1 - z0 <= 0.01:
            return
        n = max(1, int(np.ceil((u1 - u0) / ((z1 - z0) * MAX_ASPECT))))
        edges = np.linspace(u0, u1, n + 1)
        for i in range(n):
            a, b = float(edges[i]), float(edges[i + 1])
            t = tag if n == 1 else f"{tag}c{i}"
            cu, cz = (a + b) / 2, (z0 + z1) / 2
            pos, size, _ = _place(axis, sign, at, cu, at, cz, (b - a) / 2, T / 2, (z1 - z0) / 2)
            g.append(f'<geom name="{name}_{t}" type="box" material="{mat_fn(t, a, b, z0, z1)}" '
                     f'pos="{pos}" size="{size}"/>')
            boxes.append((t, a, b, z0, z1))

    box(-half_len, half_len, edges[0], edges[1], "s0")
    box(-half_len, half_len, edges[2], edges[3], "s1")
    box(-half_len, half_len, edges[4], edges[5], "s2")
    for ri, (z0, z1) in enumerate(rows):
        _, pillars = _spans(half_len)
        for pi, (a, b) in enumerate(pillars):
            box(a, b, z0, z1, f"r{ri}p{pi}")
        for ci, cu in enumerate(centres):
            g += _frame(name, axis, sign, at, cu, z0, z1, f"r{ri}w{ci}")
        # FOUR FIDUCIALS PER WALL: one centred in each gap between adjacent windows, on each of the
        # two rows, so the spacing to the window either side of a marker is equal by construction.
        # They sit on the inner wall face, and every one carries a unique ArUco id, so a frame
        # showing any marker localises the camera unambiguously.
        for si, cu in enumerate((centres[i] + centres[i + 1]) / 2 for i in range(len(centres) - 1)):
            pos, size, _ = _place(axis, sign, at, cu, at - T / 2 - 0.012, (z0 + z1) / 2,
                                  TAG, 0.012, TAG)
            tags.append((f"{name}_t{ri}{si}", pos, size))
    return g, centres, [(-tank_d * zf) for zf in sorted(rings, reverse=True)], tags
