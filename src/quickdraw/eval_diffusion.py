"""Standalone denoising eval (denoising_multistep + denoising_aggregate). `python -m quickdraw.eval_diffusion checkpoint=...`"""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, ["denoising_multistep", "denoising_aggregate"], label="denoising")


if __name__ == "__main__":
    main()
