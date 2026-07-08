"""Standalone interpretability eval (VLM-labeled latent manifolds).
`python -m quickdraw.eval_interpret checkpoint=<run_or_ckpt> [interpret=torus]`  (needs OPENAI_API_KEY)."""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, ["interpret"], label="interpret")


if __name__ == "__main__":
    main()
