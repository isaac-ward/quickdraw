"""Make HIGH-QUALITY GIFs of the 2-panel clips from their mp4s (no model re-run). Per-frame adaptive 256-colour
palette + Floyd-Steinberg dither (true RGB, no washout). Picks the highest-quality (downscale, fps) on a ladder
that stays <= BUDGET bytes (uses the budget for quality). Overwrites logs/ood/gallery/*_2panel.gif.
Run: docker compose exec -T app uv run --no-sync python logs/oneoffs/gifs_from_mp4.py [glob] [budget_mb]
"""
import glob, os, sys
import numpy as np, cv2
import imageio.v3 as iio
from quickdraw.logging.viz import save_gif

SRC = "logs/ood/gallery"
DITHER = False                              # dither adds visible grain/speckle on flat areas -> off; adaptive 256 is clean
# RESOLUTION-FIRST: keep frames sharp (half-res looked soft/blocky in slides), trade fps/scale down only to fit budget
LADDER = [(1.0, 10), (0.9, 10), (0.85, 10), (0.8, 10), (0.75, 8), (0.7, 8), (0.6, 8)]


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else f"{SRC}/*_2panel.mp4"
    budget = (float(sys.argv[2]) if len(sys.argv) > 2 else 40.0) * 1e6
    for mp in sorted(glob.glob(pat)):
        name = os.path.basename(mp).replace(".mp4", "")
        v = iio.imread(mp, plugin="pyav")                       # (T,H,W,3) @30fps
        out = os.path.join(SRC, f"{name}.gif")
        chosen = None
        for ds, fps in LADDER:
            step = max(1, round(30 / fps)); eff = 30 / step
            fr = [cv2.resize(f, (int(f.shape[1] * ds), int(f.shape[0] * ds)), interpolation=cv2.INTER_AREA) for f in v[::step]]
            save_gif(out, fr, eff, colors=256, dither=DITHER)
            sz = os.path.getsize(out)
            chosen = (ds, eff, sz)
            if sz <= budget:
                break
        print(f"[{name}] {chosen[0]}x {chosen[1]:.0f}fps {fr[0].shape[1]}x{fr[0].shape[0]} -> {chosen[2]/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
