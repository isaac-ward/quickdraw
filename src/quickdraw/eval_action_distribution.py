"""Standalone action-distribution eval (learned action prior vs true data action dist). Requires a checkpoint
whose model has an action head. `python -m quickdraw.eval_action_distribution checkpoint=<run> data.root=<data>`"""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, ["action_distribution"], label="action_distribution")


if __name__ == "__main__":
    main()
