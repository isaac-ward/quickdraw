"""Render every episode of the held-out EVAL splits to mp4, so what was recorded can be looked at.

starling-2 ships four splits beyond train/val, and they are the two experiments the paper still needs:

    eval_ood_noodle        12 eps   a novel object enters the room
    eval_ood_leafblower    12 eps   a novel object enters the room
    eval_memory_backwall1  10 eps   turn away from a wall and back to it
    eval_memory_backwall2  10 eps

Short clips -- ~125 frames at 15 Hz, so ~8 s each, which at the trained stride of 4 is only ~31 model
steps. That bounds any open-loop horizon measured on them and is worth knowing before designing the eval.

    python scratch/render_eval_splits.py <out_dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.logging import viz
from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root

SPLITS = {"eval_ood": ["eval_ood_noodle", "eval_ood_leafblower"],
          "eval_memory": ["eval_memory_backwall1", "eval_memory_backwall2"]}


def main(out_root: str = "logs/paper_icra_2027") -> int:
    cfg = OmegaConf.create(json.load(open(
        "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    root = resolve_data_root(cfg)
    set_subsample(1)                    # RAW rate for viewing: the recording as filmed, not as strided
    set_action_aggregate("concat")
    for group, splits in SPLITS.items():
        for sp in splits:
            eps = load_split_episodes_mm(root, sp, img_size=image_head_sizes(cfg),
                                         cam=image_head_cams(cfg), repo_id="starling-2")
            d = os.path.join(out_root, group, sp)
            os.makedirs(d, exist_ok=True)
            key = list(eps[0][2])[0]
            lens = [len(o) for o, _, _ in eps]
            print(f"  {sp:24s} {len(eps):2d} eps | frames {min(lens)}-{max(lens)} "
                  f"({min(lens) / 15:.1f}-{max(lens) / 15:.1f}s) -> {d}")
            for i, (o, a, fr) in enumerate(eps):
                viz.save_mp4(os.path.join(d, f"ep{i:02d}.mp4"), fr[key], 15)
            np.save(os.path.join(d, "_lengths.npy"), np.array(lens))
    print("\n  15 fps = real time. At the trained stride of 4 each clip is ~31 model steps, so an")
    print("  open-loop rollout on these can reach ~23 predicted steps after 8 context frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
