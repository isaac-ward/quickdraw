"""Standalone recovered-manifold UMAP eval (any method). `python -m quickdraw.eval_manifold checkpoint=...`"""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, "manifold")


if __name__ == "__main__":
    main()
