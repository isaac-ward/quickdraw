"""Print the model architecture summary for a config — total params + the per-component `arch_table`
dataflow (component | shape transform | params). CPU-only, no data/env needed. This is the SAME summary
the training run prints at the top of `progress.log` (via `logging/callback.py::arch_summary_lines`), so the
wizard can report the intended shape of every component before you commit to a run.

  uv run python -m quickdraw.model_summary model=mm_flow \\
      model.d=128 model.action_dim=4 'model.modalities.0.dim=16' \\
      'model.modalities.1.img_size=[112,192]' model.modalities.1.decode_kind=mse
"""

from __future__ import annotations

import hydra

from .logging.callback import arch_summary_lines
from .training.setup import build_model


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    model = build_model(cfg)
    for line in arch_summary_lines(model):
        print(line)


if __name__ == "__main__":
    main()
