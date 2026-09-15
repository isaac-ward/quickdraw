"""Is prefix-conditioned RETRIEVAL viable for DataProposal, or is the bank too thin?

Prefix guidance works for the prior because a flow can be steered to start anywhere its joint supports.
A bank cannot be steered -- it can only be searched. So the question is empirical: when a plan needs the
next chunk to continue from action a_last, does the bank actually CONTAIN chunks that open near a_last?

Measured against the only meaningful yardstick: the recorded step-to-step change (0.055). If the nearest
bank opening sits within that, retrieval stitches as smoothly as a real flight. If it is far outside, the
bank is too thin in 4-D and prefix retrieval is a stretch -- mark it incompatible and move on.

    python scratch/check_bank_coverage.py <steer_run_using_the_data_proposal>
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml
from omegaconf import OmegaConf

AXES = [a["name"] for a in yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]]
NA = len(AXES)


def main(run: str, split: str = "train", stride: int = 1, k_keep: int = 64) -> int:
    from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    cfg = OmegaConf.create(json.load(open(
        "logs/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(4); set_action_aggregate("concat")
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    K = 32
    # the bank exactly as DataProposal builds it, folded to per-axis stick values
    from quickdraw.data.dataset import load_split_episodes
    stride = int(stride)
    be = load_split_episodes(resolve_data_root(cfg), split, repo_id="starling-2")
    bank = np.stack([a[i:i + K] for _, a in be for i in range(0, len(a) - K, stride)])
    print(f"  bank split {split!r} stride {stride}")
    bank = bank.reshape(len(bank), K, -1, NA).mean(axis=2)          # (n, K, NA)
    opens = bank[:, 0]                                              # (n, NA) what each chunk starts on
    print(f"\n  bank: {len(bank)} chunks of {K}, opening actions in {NA}-D")

    # the prefixes a real plan actually needed: the last committed action before each seam
    fs = sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                       "actions.npy")))
    pre = []
    for f in fs:
        a = np.load(f).reshape(-1, len(np.load(f)[0]) // NA, NA).mean(axis=1) if False else \
            np.load(f).reshape(len(np.load(f)), -1, NA).mean(axis=1)
        pre.extend(a[i - 1] for i in range(K, len(a), K))           # the action just before each seam
    pre = np.stack(pre)
    print(f"  prefixes needed: {len(pre)} seams across {len(fs)} plans")

    d = np.linalg.norm(opens[None, :, :] - pre[:, None, :], axis=-1)    # (n_pre, n_bank)
    nn = np.sort(d, axis=1)
    rec = 0.0545 * np.sqrt(NA)          # the recorded per-step change, as a 4-D distance
    print(f"\n  recorded step-to-step change as a 4-D distance: {rec:.3f}")
    for j, lab in ((0, "nearest bank chunk"), (k_keep - 1, f"{k_keep}th nearest (the whole candidate set)")):
        print(f"  {lab:38s} mean {nn[:, j].mean():.3f}   median {np.median(nn[:, j]):.3f}   "
              f"p90 {np.percentile(nn[:, j], 90):.3f}   ({nn[:, j].mean() / rec:.1f}x recorded)")
    frac = float((nn[:, 0] < rec).mean())
    print(f"\n  seams where the nearest bank chunk opens within one recorded step: {100 * frac:.0f}%")
    print(f"  seams where at least {k_keep} bank chunks do: "
          f"{100 * float((nn[:, k_keep - 1] < rec).mean()):.0f}%   <- retrieval needs a POOL, not one hit")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
