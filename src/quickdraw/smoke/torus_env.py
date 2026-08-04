"""Smoke test for the torus COLOR/POSITION ground truth in environments/torus_utils.py: prove it matches
the RENDERER and behaves. Run: uv run python -m quickdraw.smoke.torus_env"""

from __future__ import annotations

import math

import numpy as np

from ..environments import torus_utils as T
from ..logging import viz


def _pt(theta, phi=0.0, R=0.75, r=0.25):
    """World xyz on the torus surface at (major theta, tube phi)."""
    return np.array([(R + r * math.cos(phi)) * math.cos(theta),
                     (R + r * math.cos(phi)) * math.sin(theta),
                     r * math.sin(phi)])


def main():
    R, r = 0.75, 0.25
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    check("N_SEG == viz.N_SEG", T.N_SEG == viz.N_SEG, f"{T.N_SEG} vs {viz.N_SEG}")

    # rgb_at EXACTLY matches the renderer's texture at every band center
    tex = viz._texture_array("hsv")
    h, w = tex.shape[:2]
    maxerr = max(float(np.abs(T.rgb_at(_pt((k + 0.5) / T.N_SEG * 2 * math.pi))
                              - tex[h // 2, int((k + 0.5) / T.N_SEG * w)] / 255.0).max())
                 for k in range(T.N_SEG))
    check("rgb_at matches renderer texture (all 16 bands)", maxerr < 1e-2, f"max abs err {maxerr:.4f}")

    check("theta=0 -> red", T.color_at(_pt(0.0)) == "red")
    check("theta=pi -> cyan", T.color_at(_pt(math.pi)) == "cyan")
    check("phi=+90 -> top", T.position_band(_pt(0.0, math.pi / 2), r) == "top")
    check("phi=-90 -> bottom", T.position_band(_pt(0.0, -math.pi / 2), r) == "bottom")
    check("phi=0 -> middle", T.position_band(_pt(0.0, 0.0), r) == "middle")

    rr, rc = float(T.color_reward(_pt(0.0), "red")), float(T.color_reward(_pt(math.pi), "red"))
    check("color_reward(red) high@red low@cyan", rr > 0.95 and rc < 0.05, f"{rr:.2f}/{rc:.2f}")
    check("position_reward(top) max@+r", float(T.position_reward(_pt(0, math.pi / 2), r, "top")) > 0.99)
    check("position_reward(top) low@-r", float(T.position_reward(_pt(0, -math.pi / 2), r, "top")) < 0.01)

    pts = np.stack([_pt(t) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    labs = T.color_at(pts)
    check("batched color_at", isinstance(labs, list) and len(labs) == 8, str(labs))

    band_names = [T.color_at(_pt((k + 0.5) / T.N_SEG * 2 * math.pi)) for k in range(T.N_SEG)]
    reachable = sorted(set(band_names))
    print("\n  16 bands -> names:", band_names)
    print("  reachable:", reachable, "| never-rendered:", [n for n in T.NAMED_COLORS if n not in reachable])
    print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
