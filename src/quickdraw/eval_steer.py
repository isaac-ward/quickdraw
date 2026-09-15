"""Language-steered planning inside the world model's imagination (no environment, no true dynamics).

    python -m quickdraw.eval_steer interpret=starling checkpoint=<action-model run or ckpt> \
        steer.head=<train_reward_model run>/reward_head.pt data=starling2 environments=recorded \
        data.subsample=4 data.action_aggregate=concat

The checkpoint must carry a TRAINED ACTION HEAD (a train_action_model run), because the candidate action
chunks are drawn from it; a bare world model has no proposal to sample from.
"""
from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    run_standalone(cfg, ["steer"], label="steer")


if __name__ == "__main__":
    main()
