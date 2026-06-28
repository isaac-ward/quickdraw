"""Standalone OOD open-loop eval (visual + geometric + dynamics). `python -m quickdraw.eval_ood checkpoint=...`"""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, ["ood_visual", "ood_geometric", "ood_dynamics"], label="ood")


if __name__ == "__main__":
    main()
