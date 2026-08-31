"""Derive `data.subsample` for lego_assemblies the way record §13 derived it for robocasa.

    uv run python -m quickdraw.smoke.lego_subsample [DATA_ROOT] [CAM]

THE QUESTION. A world model can only learn motion it can SEE through its own codec. Record §13
measured robocasa at 20 Hz and found the per-step image change was 0.61x the reconstruction error of
the autoencoder being predicted through: the target sat BELOW the codec's noise floor, so predicting
zero motion was the CORRECT minimiser of the objective. Every run duly scored motion_ratio 0.13-0.17
and no amount of action conditioning moved it. Subsampling to 4-5 Hz flipped the ratio to 1.24-1.36x.

lego_assemblies is 30 Hz, half again faster, so its stride-1 ratio should be WORSE than 0.61x.

METHOD. Numerator: per-pixel RMSE between frame t and frame t+s, over a contiguous chunk of each
sampled episode. Denominator: the frozen-TAESD reconstruction RMSE at the same resolution, from
`eval_ae_floor +ae_floor.taesd=true`. Both in [-1,1] pixel-conv units and both at the SQUARE training
img_size, which is what `load_fpv_frames` feeds the model -- measuring at the staged 16:9 resolution
would report a motion signal the model never sees.

TAESD is a REFERENCE yardstick, not vl64's own codec (vl64 trains a bespoke AE from scratch, whose
floor is a training outcome, not a fixed property). It is the right denominator here for two reasons:
it is what §13 used, so the numbers are directly comparable to robocasa's; and it needs no checkpoint,
so the stride can be chosen BEFORE the first run. Re-check against vl64's trained floor once Arm A has
a few epochs -- but the stride choice is coarse (6 vs 8), so the yardstick is good enough to start.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

IMG = 128                 # SQUARE, as load_fpv_frames feeds the model
CHUNK = 1200              # contiguous frames per episode (from the middle: skips idle head/tail)
EVERY = 7                 # sample every Nth episode -> ~11 of 74, spread across sessions
STRIDES = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16)

# Frozen-TAESD floor at 128x128 on THIS dataset, from:
#   uv run python -m quickdraw.eval_ae_floor +ae_floor.taesd=true +ae_floor.sizes=[128] ...
#   -> PSNR=23.79dB SSIM=0.828 MSE=0.00418   (robocasa's was 23.92dB / RMSE 0.0637 -- near-identical)
FLOOR_MSE = 0.00418


def _load_chunk(mp4: str, n_rows: int) -> np.ndarray:
    """A centred contiguous chunk of one clip -> (T,128,128,3) float32 in [-1,1]."""
    import imageio.v2 as imageio
    import torch

    start = max(0, (n_rows - CHUNK) // 2)
    buf, out, i = [], [], 0

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(IMG, IMG), mode="area")   # square, as training sees
        out.append(x.permute(0, 2, 3, 1).numpy())
        buf.clear()

    rd = imageio.get_reader(mp4)
    for fr in rd:
        if i >= start:
            buf.append(np.asarray(fr)[..., :3])
            if len(buf) >= 256:
                flush()
            if sum(len(o) for o in out) + len(buf) >= CHUNK:
                break
        i += 1
    flush()
    rd.close()
    return np.concatenate(out, 0)[:CHUNK] / 127.5 - 1.0                        # [-1,1] pixel-conv units


def main(root: str, cam: str) -> int:
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    chunk_sz, n_eps, fps = int(info["chunks_size"]), int(info["total_episodes"]), int(info["fps"])
    lens = {int(json.loads(l)["episode_index"]): int(json.loads(l)["length"])
            for l in open(os.path.join(root, "meta", "episodes.jsonl"))}
    vkey = f"observation.images.{cam}"

    idxs = list(range(0, n_eps, EVERY))
    print(f"[subsample] {root}\n[subsample] cam={cam} fps={fps} | {len(idxs)} of {n_eps} episodes, "
          f"<= {CHUNK} frames each, at {IMG}x{IMG}\n", flush=True)

    sq = {s: 0.0 for s in STRIDES}          # accumulate SQUARED error so episodes pool correctly
    cnt = {s: 0 for s in STRIDES}
    for k, idx in enumerate(idxs, 1):
        c = idx // chunk_sz
        mp4 = os.path.join(root, "videos", f"chunk-{c:03d}", vkey, f"episode_{idx:06d}.mp4")
        if not os.path.exists(mp4):
            print(f"  ep{idx:03d}: MISSING {vkey}, skipped", flush=True)
            continue
        x = _load_chunk(mp4, lens.get(idx, CHUNK))
        for s in STRIDES:
            if len(x) <= s:
                continue
            d = x[s:] - x[:-s]
            sq[s] += float(np.sum(d.astype(np.float64) ** 2))
            cnt[s] += d.size
        print(f"  ep{idx:03d}: {len(x)} frames  ({k}/{len(idxs)})", flush=True)

    floor = float(np.sqrt(FLOOR_MSE))
    print(f"\n  frozen-TAESD floor @ {IMG}px: RMSE {floor:.4f}  (MSE {FLOOR_MSE})")
    print(f"\n  {'stride':>6} {'Hz':>6} {'seconds':>8} {'frame-d RMSE':>13} {'vs floor':>9}   verdict")
    # RULE: smallest stride that is BOTH clear of the codec floor (>=1.25x, where robocasa crossed into
    # a learnable target) AND inside the 2-5 Hz band every published long-rollout system converges on.
    # "First to clear 1.25x" alone is too weak -- it lands on the marginal edge and buys the shortest
    # horizon, which is the thing subsampling exists to fix.
    pick, first_clear = None, None
    for s in STRIDES:
        if not cnt[s]:
            continue
        rmse = float(np.sqrt(sq[s] / cnt[s]))
        ratio = rmse / floor
        hz = fps / s
        if first_clear is None and ratio >= 1.25:
            first_clear = s
        if pick is None and ratio >= 1.25 and 2.0 <= hz <= 5.0:
            pick = s
        band = 2.0 <= hz <= 5.0          # V-JEPA-2-AC 4fps, IRASim ~4fps, HMA 2Hz, robocasa chose 4Hz
        if ratio < 1.0:
            mark = "below the codec's own error"
        elif ratio < 1.25:
            mark = "marginal"
        else:
            mark = "learnable" + ("  + IN THE 2-5 Hz BAND" if band else "")
        print(f"  {s:>6} {hz:>6.1f} {s/fps:>8.3f} {rmse:>13.4f} {ratio:>8.2f}x   {mark}")

    if pick is None:
        print("\n  NOTHING is both >=1.25x and in the 2-5 Hz band -- widen STRIDES")
        return 1
    print(f"\n  clears the floor from stride {first_clear} ({fps/first_clear:.0f} Hz); "
          f"first ALSO in the 2-5 Hz band at stride {pick} ({fps/pick:.0f} Hz)")
    print(f"  F=64 at stride {pick} spans {64*pick/fps:.1f} s of robot time (vs {64/fps:.1f} s at stride 1)")
    print(f"\n  ==> data.subsample={pick}   (with data.subsample_all_phases=true to recover the "
          f"~{pick}x windows the decimation would discard)")
    return 0


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/isaac/data/lego_assemblies"
    cam = sys.argv[2] if len(sys.argv) > 2 else "head_right"
    sys.exit(main(root, cam))
