"""Tiled overview of ALL training data (like the robocasa montage): every train episode's scene cam, tiled into
a grid, played together. 43 block-stack train episodes -> 7x7 grid. Loaded at subsample=6 (~5 Hz) to bound memory;
first T frames of each (looped if shorter); played sped-up.
Out: logs/ood/train_data_tiled.mp4
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/dataset_tiles.py <ckpt>
"""
import os, sys, math, dataclasses
import numpy as np
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.logging.viz import tile_clips, save_mp4
from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root

HEAD, T, SUB, FPS, OUT = "cam_scene", 400, 6, 20, "logs/ood"


def main():
    ck = sys.argv[1]
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    dcfg = dataclasses.replace(DataConfig.from_cfg(rc), subsample=SUB, subsample_all_phases=False)
    eps = load_split_episodes_mm(resolve_data_root(rc), "train", dcfg=dcfg,
                                 img_size=image_head_sizes(rc) or 128,
                                 cam=image_head_cams(rc) or rc.data.get("cam", "fpv"),
                                 repo_id=rc.data.get("repo_id", "torus"))
    clips = []
    for o, a, fr in eps:
        f = fr[HEAD].astype(np.uint8)
        c = f[:T] if len(f) >= T else np.concatenate([f] * math.ceil(T / len(f)))[:T]   # loop short eps
        clips.append(c)
    grid = math.ceil(math.sqrt(len(clips)))
    tiled = tile_clips(clips, grid)
    os.makedirs(OUT, exist_ok=True)
    save_mp4(os.path.join(OUT, "train_data_tiled.mp4"), tiled, fps=FPS)
    print(f"{len(clips)} train episodes -> {grid}x{grid} grid | tiled {tiled.shape} @ {FPS}fps -> logs/ood/train_data_tiled.mp4", flush=True)


if __name__ == "__main__":
    main()
