"""Entrypoint: train the action-distribution head POST-HOC on a FROZEN world model.
`python -m quickdraw.train_action_model checkpoint=<train_world_model run dir> data.root=<data> ...`

Sibling of train_world_model / train_reward_model. Given a finished world-model checkpoint, train ONLY the
action-flow head (the learned play/behavior prior used as the MPPI proposal) on the frozen WM's pooled
context features. The WM is frozen because JOINT action-head training destabilized it (killed control ~0 vs
3.88 and NaN'd/collapsed the WM — see conf/model/mm_flow.yaml action_head notes).

Mechanism: build the WM with model.action_head.enabled=true so build_model constructs a fresh
`action_flow`; load_checkpoint is strict=False, so the WM weights load and `action_flow` (absent from the
checkpoint) stays fresh-initialized. Freeze everything, unfreeze only action_flow, train its flow loss on
contexts computed under no_grad (LitActionModel). The saved checkpoints contain the FULL model (frozen WM +
trained head), so eval_action_distribution / MPPI load them like any other checkpoint."""

from __future__ import annotations

import json
import os
import shutil
import time

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from .logging.callback import LoggingCallback, ProgressPrinter
from .logging.writer import make_writer
from .train_world_model import _assert_summary_unique, _run_summary_text, _startup_log
from .training.lit import LitActionModel
from .training.setup import build_model, data_exists, env_cfg, load_checkpoint, normalizer, window_loaders
from .utils.logging import make_run_dir


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    summary_text = _run_summary_text(cfg)  # fail BEFORE any setup if the run note is missing
    _assert_summary_unique(summary_text, cfg)  # ...and fail if it merely copies a previous run's note
    assert cfg.checkpoint, ("train_action_model requires checkpoint=<frozen WM train run dir or .ckpt> "
                            "(the finished world model whose context features the action head trains on).")
    if not data_exists(cfg):
        raise FileNotFoundError(
            "No dataset found. Run `python -m quickdraw.data_generation` first, then pass its run "
            f"dir as data.root=logs/data_generation_<ts>_<exp> (got data.root={cfg.data.root!r})."
        )

    # rebuild the WM ARCHITECTURE from the checkpoint run's saved config (like evaluation/standalone.py),
    # so the frozen WM matches regardless of the CLI default model; keep THIS run's action_head knobs
    # (CLI-overridable), with enabled FORCED true so build_model constructs the fresh action_flow head.
    ck = str(cfg.checkpoint)
    run = os.path.dirname(os.path.dirname(ck)) if ck.endswith(".ckpt") else ck
    cfgj = os.path.join(run, "logs", "config.json")
    OmegaConf.set_struct(cfg, False)
    ah = cfg.model.get("action_head", None)                  # THIS run's head knobs (CLI-overridable)
    ah_shortcut = ah.get("shortcut", None) if ah is not None else None   # explicit CLI shortcut override, if any
    if os.path.exists(cfgj):
        saved = OmegaConf.create(json.load(open(cfgj)))
        cfg.model = saved.model                              # adopt the trained WM config (arch + modalities)
        if ah is not None:
            cfg.model.action_head = ah                       # ...but the head knobs belong to THIS run
    if cfg.model.get("action_head", None) is None:
        cfg.model.action_head = {}
    cfg.model.action_head.enabled = True
    # POST-HOC default: train the action head as PURE rectified flow (shortcut OFF). The shortcut
    # self-consistency loss is self-referential (target = the model's own chained steps) and, on the FROZEN
    # WM trunk with no co-adaptation to anchor it, runs away (val 1.6 -> 1e22; LR warmup only delays it to
    # ~ep15). shortcut=false is stable + gives a better action-dist match (W1 0.13 vs diverged/NaN). Override
    # with ++model.action_head.shortcut=true to force it back on.
    if ah_shortcut is None:
        cfg.model.action_head.shortcut = False
    assert str(cfg.model.name) in ("mm_flow", "flow"), (
        f"the action-flow head needs a flow world model (model.name in mm_flow/flow; got {cfg.model.name!r})")

    run_dir = make_run_dir("train_action", cfg.experiment)   # logs/train_action_<ts>_<exp>
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))

    _t = time.perf_counter()
    _startup_log(run_dir, "[startup] loading dataset (GPU-resident windows) + normalizer...")
    from .data.dataset import set_action_aggregate, set_obs_keep, set_subsample
    set_subsample(int(cfg.data.get("subsample", 1) or 1))          # match the rate + obs layout the WM trained on
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))   # beside the stride: same one-shot rule
    set_obs_keep(cfg.data.get("obs_keep", None))                   # applied inside every loader + the normalizer
    norm = normalizer(cfg)
    loaders = window_loaders(cfg, norm)
    _startup_log(run_dir, f"[startup] data ready in {time.perf_counter() - _t:.1f}s: "
                          f"{getattr(loaders['train'], 'N', '?')} train / {getattr(loaders['val'], 'N', '?')} val windows")

    model = build_model(cfg)
    # frozen-WM load: the WM checkpoint has NO action_flow weights and load_checkpoint is strict=False, so
    # the WM loads and action_flow stays fresh-initialized. Verify both (a wrong/mismatched checkpoint would
    # otherwise silently load nothing and the head would train on random features).
    before = {k: v.clone() for k, v in model.state_dict().items()}
    load_checkpoint(model, cfg.checkpoint)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    wm_keys = [k for k in before if not k.startswith("action_flow.")]
    n_wm = sum(k in changed for k in wm_keys)
    af_loaded = sorted(k for k in changed if k.startswith("action_flow."))
    assert n_wm > 0, f"checkpoint {cfg.checkpoint!r} loaded NO world-model weights (architecture mismatch?)"
    _startup_log(run_dir, f"[startup] frozen WM loaded: {n_wm}/{len(wm_keys)} WM tensors from {ck}" + (
        f"; NOTE action_flow was ALSO in the ckpt ({len(af_loaded)} tensors -> warm start)" if af_loaded
        else "; action_flow fresh-initialized"))

    # freeze EVERYTHING, then unfreeze ONLY the action-flow head; the WM stays in eval mode
    # (LitActionModel re-pins eval each epoch — Lightning flips train mode).
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.action_flow.parameters():
        p.requires_grad_(True)
    model.eval()
    n_train = sum(p.numel() for p in model.action_flow.parameters())
    _startup_log(run_dir, f"[startup] model built: {sum(p.numel() for p in model.parameters()) / 1000:.0f}K "
                          f"params total, {n_train / 1000:.1f}K trainable (action_flow ONLY; WM frozen)")

    lit = LitActionModel(model, cfg.optim.lr, cfg.optim.weight_decay,
                         lr_warmup_steps=int(cfg.optim.get("lr_warmup_steps", 0)))

    # one writer -> local run folder + wandb, identically (see logging/writer.py); Lightning's logger is OFF.
    writer = make_writer(run_dir, cfg, job_type="train_action")
    with open(os.path.join(run_dir, "auto_run_summary.txt"), "w") as f:
        f.write(summary_text + "\n")
    print(summary_text, flush=True)  # after wandb.init -> captured in the wandb console logs too

    e = env_cfg(cfg)
    # eval: ONLY the action_distribution routine (learned prior vs true data actions) — the model's own eval.
    # Cadence: the during-train cadence + at_epochs, UNIONED with the final epoch so it always runs at the end.
    at = sorted({int(x) for x in (cfg.eval.during_train.get("at_epochs", None) or [])}
                | {int(cfg.trainer.max_epochs) - 1})
    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"),
                        monitor="val/loss/total", mode="min",
                        save_top_k=cfg.trainer.save_top_k, save_last=True),
        LoggingCallback(writer, cfg, norm, e, cfg.eval.during_train.every_epochs,
                        ["action_distribution"], at_epochs=at),
        ProgressPrinter(run_dir),
    ]
    trainer = L.Trainer(max_epochs=cfg.trainer.max_epochs, precision=cfg.trainer.precision,
                        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                        gradient_clip_val=1.0, enable_progress_bar=False,
                        check_val_every_n_epoch=int(cfg.trainer.get("check_val_every_n_epoch", 1)),
                        callbacks=callbacks, logger=False,
                        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
                        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0))
    trainer.fit(lit, loaders["train"], loaders["val"])

    # stable name for the best checkpoint (FULL model: frozen WM + trained action_flow), loadable by
    # the standalone eval_action_distribution / eval_control like any other checkpoint.
    best = callbacks[0].best_model_path
    if best and os.path.exists(best):
        shutil.copy(best, os.path.join(run_dir, "checkpoints", "best.ckpt"))
    print(f"[train_action] done. run_dir={run_dir} (best -> checkpoints/best.ckpt)")


if __name__ == "__main__":
    main()
