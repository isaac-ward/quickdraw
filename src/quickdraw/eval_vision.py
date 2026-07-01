"""Standalone vision rollout eval (multimodal models). `python -m quickdraw.eval_vision model=mm_lsar checkpoint=...`"""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, "vision")


if __name__ == "__main__":
    main()
