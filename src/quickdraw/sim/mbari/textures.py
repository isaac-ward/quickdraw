"""Procedural textures for the tank: concrete, and an ArUco 5x5 dictionary.

ArUco is generated rather than shipped as PNGs because the marker set has to be large (a grid over four
walls), and a generator makes the id -> pattern mapping explicit and reproducible. This is the real
5x5_250 construction: each id is a 4x4 bit matrix from the dictionary's byte list, framed by a one-cell
black border, so a detector would read them as genuine markers rather than as marker-shaped noise.
"""
import numpy as np

def aruco_tile(idx: int, px: int = 64, border: int = 1) -> np.ndarray:
    """Marker `idx` of DICT_5X5_250 as (px,px,3) uint8, with a white quiet zone of `border` cells.

    5x5_250 rather than 4x4_50 because the tank carries ~96 markers across four walls and the floor,
    and a 50-id dictionary cannot label them uniquely -- the floor was repeating the +y wall's ids,
    which would make any pose-from-fiducials work ambiguous.

    Generated from cv2's own dictionary, not from a transcribed byte table. The first version of this
    file carried a hand-typed table and claimed it was the real construction; it was not -- id 3 came
    out as [[0,1,1,1],[1,0,1,1],[1,0,0,1],[1,0,1,0]] against cv2's actual
    [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]. Marker-shaped noise would have looked fine in a render
    and failed the moment anyone pointed a detector at it.
    """
    import cv2

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    n = 7                                                   # 5 payload cells + 1 black border each side
    cell = max(1, px // (n + 2 * border))
    m = cv2.aruco.generateImageMarker(d, int(idx) % 250, n * cell)
    img = np.full(((n + 2 * border) * cell, (n + 2 * border) * cell), 255, np.uint8)
    o = border * cell
    img[o:o + n * cell, o:o + n * cell] = m
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)
    return np.repeat(img[:, :, None], 3, 2)


def window_rows(img, cols, rooms, rings=(0.22, 0.62), h_frac=0.20, w_frac=0.15, tint=(0.62, 0.86, 0.92)):
    """Two RINGS of viewing ports, each opening onto a rendered room.

    DENSITY HALVED from the first version (7 and 5 ports per wall became 4 and 3) and both rings are
    at the same two depths on every wall, so they read as two continuous rows around the tank rather
    than as rows on two walls and columns on the other two -- which is what the first version did,
    because it indexed rows down the IMAGE and the end-wall textures are portrait while the long-wall
    textures are landscape.

    `rooms` is one rendered room image per ring. The glass is the room image under a slight blue-green
    tint rather than a dark rectangle, so a window reads as something you can see through.
    """
    H, W = img.shape[:2]
    tint = np.asarray(tint, np.float32)
    for r, zf in enumerate(rings):
        y0, y1 = int(H * zf), int(H * zf) + int(H * h_frac)
        room = rooms[r % len(rooms)]
        for c in range(cols):
            cx = W * (0.14 + c * 0.72 / max(1, cols - 1))
            x0, x1 = int(cx - W * w_frac / 2), int(cx + W * w_frac / 2)
            x0, x1 = max(0, x0), min(W, x1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            img[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6] = (206, 206, 200)     # frame
            pane = _fit(room, y1 - y0, x1 - x0).astype(np.float32)
            img[y0:y1, x0:x1] = np.clip(pane * tint[None, None, :] * 0.92, 0, 255).astype(np.uint8)
            img[y0:y0 + max(2, (y1 - y0) // 7), x0:x1] = (
                0.45 * img[y0:y0 + max(2, (y1 - y0) // 7), x0:x1] + 0.55 * np.array([150, 195, 210])
            ).astype(np.uint8)                                                       # specular band
    return img


def _fit(src, h, w):
    """Nearest-neighbour resize without pulling in cv2 for three lines."""
    yi = (np.linspace(0, src.shape[0] - 1, h)).astype(int)
    xi = (np.linspace(0, src.shape[1] - 1, w)).astype(int)
    return src[np.ix_(yi, xi)]


def _splotches(acc, rng, n, radius, strength):
    """Soft elliptical stains, added in-place to a [0,1] field."""
    h, w = acc.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry = radius * rng.uniform(0.5, 1.8)
        rx = radius * rng.uniform(0.5, 1.8)
        d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        acc += strength * rng.choice([-1.0, 1.0]) * np.exp(-d)
    return acc


def concrete(h, w, seed=0):
    """Weathered concrete: multi-octave grey plus BROWN staining and light/dark patches.

    Deterministic in `seed`, and every surface gets a fixed seed, so the tank looks the same in every
    episode of every run -- variation across the walls, none across time. A texture that resampled per
    run would be a nuisance variable sitting right in the world model's input.

    Multi-octave rather than white noise because at 144x192 the wall is seen mostly through its low
    frequencies; white noise aliases into per-frame shimmer on a moving camera, which is exactly the
    signal the rollout is supposed to carry."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    for oct_ in range(5):
        hh, ww = max(2, h >> (4 - oct_)), max(2, w >> (4 - oct_))
        n = rng.random((hh, ww), np.float32)
        yi = np.linspace(0, hh - 1, h).astype(int)
        xi = np.linspace(0, ww - 1, w).astype(int)
        acc += amp * n[np.ix_(yi, xi)]
        amp *= 0.55
    acc = (acc - acc.min()) / max(1e-6, float(np.ptp(acc)))
    acc = _splotches(acc, rng, n=14, radius=h * 0.11, strength=0.16)      # broad light/dark patches
    acc = np.clip(acc, 0, 1)

    base = 112 + 58 * (acc - 0.5)                                          # grey carrier
    img = np.repeat(np.clip(base, 0, 255).astype(np.float32)[:, :, None], 3, 2)

    # brown staining: a second, coarser field that warms and darkens where it is strong
    stain = np.zeros((h, w), np.float32)
    stain = _splotches(stain, rng, n=9, radius=h * 0.16, strength=0.5)
    stain = np.clip((stain - stain.min()) / max(1e-6, float(np.ptp(stain))), 0, 1) ** 2
    warm = np.array([1.18, 0.98, 0.74], np.float32)                        # rust/ochre tint
    img = img * (1 - 0.45 * stain[..., None]) + img * warm * (0.45 * stain[..., None])

    # NO FORM-WORK SEAMS. They were drawn as hard horizontal lines, and once the walls became BOX
    # geoms with texuniform mapping those lines tiled into what read as vertical planks. The
    # multi-octave noise and the staining already carry the concrete look.
    return np.clip(img, 0, 255).astype(np.uint8)
