"""Entrypoint: train the base world model. `python -m quickdraw.train`"""

from __future__ import annotations

import os
import shutil

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from .logging.callback import EvalCallback
from .utils.logging import make_run_dir
from .training.lit import LitWorldModel
from .training.setup import build_model, data_exists, env_cfg, normalizer, window_loaders


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    if not data_exists(cfg):
        raise FileNotFoundError(
            "No dataset found. Run `python -m quickdraw.data_generation` first, then pass its run "
            f"dir as data.root=logs/data_generation_<ts>_<exp> (got data.root={cfg.data.root!r})."
        )

    run_dir = make_run_dir("train", cfg.experiment)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))

    norm = normalizer(cfg)
    loaders = window_loaders(cfg, norm)
    model = build_model(cfg)
    if torch.cuda.is_available():
        model = torch.compile(model, mode="max-autotune")

    e = env_cfg(cfg)
    lit = LitWorldModel(model, norm, e.R, e.r, cfg.data.P, cfg.data.F, cfg.model.p_tf,
                        cfg.optim.lr, cfg.optim.weight_decay)

    # train/val run normally; subscribe to eval routines (long_horizon | ood | control) to run
    # every cfg.eval.during_train.every_epochs (see conf/eval/default.yaml)
    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"), monitor="val/manifold_distance_error",
                        mode="min", save_top_k=cfg.trainer.save_top_k, save_last=True),
        EvalCallback(cfg, norm, e, run_dir, cfg.eval.during_train.every_epochs, list(cfg.eval.during_train.run)),
    ]
    logger = WandbLogger(project=cfg.logging.project, group=cfg.logging.group, save_dir=run_dir,
                         config=OmegaConf.to_container(cfg, resolve=True))
    trainer = L.Trainer(max_epochs=cfg.trainer.max_epochs, precision=cfg.trainer.precision,
                        gradient_clip_val=1.0, check_val_every_n_epoch=1, callbacks=callbacks, logger=logger)
    trainer.fit(lit, loaders["train"], loaders["val"])

    # stable name for the best checkpoint, used by eval_ood / eval_control
    best = callbacks[0].best_model_path
    if best and os.path.exists(best):
        shutil.copy(best, os.path.join(run_dir, "checkpoints", "best.ckpt"))
    print(f"[train] done. run_dir={run_dir} (best -> checkpoints/best.ckpt)")


if __name__ == "__main__":
    main()
