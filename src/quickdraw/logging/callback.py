"""EvalCallback: during training, run the subscribed eval routines every N epochs.

Subscription comes from cfg.eval.during_train.{every_epochs, run}; `run` is a subset of the routine
names in evaluation.routines.REGISTRY (long_horizon | ood | control). train/val happen normally;
these routines are the periodic in-loop evals.
"""

from __future__ import annotations

import lightning as L


class EvalCallback(L.Callback):
    def __init__(self, cfg, normalizer, ecfg, run_dir: str, every_epochs: int, routines: list[str]):
        self.cfg = cfg
        self.norm = normalizer
        self.ecfg = ecfg
        self.run_dir = run_dir
        self.every = every_epochs
        self.routines = routines

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self.every <= 0 or not self.routines or epoch % self.every != 0:
            return
        from ..evaluation.routines import REGISTRY

        run = getattr(getattr(trainer, "logger", None), "experiment", None)
        for name in self.routines:
            REGISTRY[name](self.cfg, pl_module.model, self.norm, self.ecfg, self.run_dir, pl_module.device, run)
