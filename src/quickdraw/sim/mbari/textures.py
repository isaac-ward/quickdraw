"""Procedural textures for the tank: concrete, and an ArUco 4x4 dictionary.

ArUco is generated rather than shipped as PNGs because the marker set has to be large (a grid over four
walls), and a generator makes the id -> pattern mapping explicit and reproducible. This is the real
4x4_50 construction: each id is a 4x4 bit matrix from the dictionary's byte list, framed by a one-cell
black border, so a detector would read them as genuine markers rather than as marker-shaped noise.
"""
import numpy as np

def aruco_tile(idx: int, px: int = 64, border: int = 1) -> np.ndarray:
    """Marker `idx` of DICT_4X4_50 as (px,px,3) uint8, with a white quiet zone of `border` cells.

    Generated from cv2's own dictionary, not from a transcribed byte table. The first version of this
    file carried a hand-typed table and claimed it was the real construction; it was not -- id 3 came
    out as [[0,1,1,1],[1,0,1,1],[1,0,0,1],[1,0,1,0]] against cv2's actual
    [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]. Marker-shaped noise would have looked fine in a render
    and failed the moment anyone pointed a detector at it.
    """
    import cv2

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    n = 6                                                   # 4 payload cells + 1 black border each side
    cell = max(1, px // (n + 2 * border))
    m = cv2.aruco.generateImageMarker(d, int(idx) % 50, n * cell)
    img = np.full(((n + 2 * border) * cell, (n + 2 * border) * cell), 255, np.uint8)
    o = border * cell
    img[o:o + n * cell, o:o + n * cell] = m
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)
    return np.repeat(img[:, :, None], 3, 2)


def concrete(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Grey concrete: pink-ish multi-octave noise plus faint horizontal form-work seams. Deterministic.

    Multi-octave rather than white noise because a world model at 144x192 sees the wall mostly through
    its low frequencies -- white noise would alias into shimmer that changes every frame of a moving
    camera, which is exactly the signal the rollout is supposed to carry."""
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
    g = 118 + 46 * (acc - 0.5)                              # mid grey, modest contrast
    seam = (np.arange(h) % max(8, h // 6) < 2)[:, None]     # form-work lines
    g = np.where(seam, g * 0.88, g)
    img = np.clip(g, 0, 255).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, 2)
