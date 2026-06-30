"""Entrypoint: train the base world model. `python -m quickdraw.train`"""

from __future__ import annotations

import os
import shutil
import time

import hydra
import lightning as L
import torch
import torch._inductor.config  # noqa: F401  ensure the submodule is importable for compile_threads below
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from .logging.callback import LoggingCallback, ProgressPrinter
from .logging.writer import make_writer
from .utils.logging import make_run_dir
from .training.lit import LitWorldModel
from .training.setup import build_model, data_exists, env_cfg, normalizer, window_loaders


_SUMMARY_FIELDS = [("problem", "Problem we are facing"), ("tried", "What we have tried"),
                   ("trying", "What we are trying"), ("trying_detail", "In more detail"),
                   ("rationale", "Why we expect it to solve the problem")]


def _startup_log(run_dir: str, msg: str):
    """Timestamped startup line -> progress.log (+ stdout), so the pre-[ep 0] phases (data load, build,
    compile) are visible and measurable, matching ProgressPrinter's stamp format."""
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(run_dir, "progress.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _run_summary_text(cfg) -> str:
    """Validate (fail-hard) the required 5-point run_summary and format it for the file + logs."""
    rs = cfg.get("run_summary") or {}
    missing = [k for k, _ in _SUMMARY_FIELDS if not str(rs.get(k) or "").strip()]
    assert not missing, (
        "A 5-point run_summary is REQUIRED before training (saved to auto_run_summary.txt). Missing: "
        f"{missing}. Provide all 5 on the CLI, e.g. +run_summary.problem=... +run_summary.tried=... "
        "+run_summary.trying=... +run_summary.trying_detail=... +run_summary.rationale=...")
    lines = ["=== AUTO RUN SUMMARY ==="]
    lines += [f"{i}. {label}: {rs[k]}" for i, (k, label) in enumerate(_SUMMARY_FIELDS, 1)]
    return "\n".join(lines + ["========================="])


def _assert_summary_unique(summary_text, cfg, root="logs") -> None:
    """A run_summary must NEVER duplicate a prior run's. Every launch describes THIS run's current
    hypothesis + what changed since the last attempt — a reused note is a stale, meaningless note.
    Escape hatch: run_summary.allow_duplicate=true for a deliberate exact rerun."""
    import glob
    if bool((cfg.get("run_summary") or {}).get("allow_duplicate", False)):
        return
    norm = " ".join(summary_text.split())
    for f in sorted(glob.glob(os.path.join(root, "*", "auto_run_summary.txt"))):
        try:
            prev = " ".join(open(f).read().split())
        except OSError:
            continue
        if prev == norm:
            raise AssertionError(
                f"run_summary is IDENTICAL to a previous run ({f}). Every run needs a UNIQUE 5-point note "
                "describing THIS run's current hypothesis and what changed since the last attempt — never "
                "copy-paste a prior summary. Rewrite run_summary.* (set run_summary.allow_duplicate=true "
                "only for a deliberate exact rerun).")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    # Safety net for dynamo recompiles: the rollout/eval flex-attention paths can produce several mask
    # variants; a too-small cache (default 8) evicts and thrashes. The fixed-window rollout already
    # holds the attention KERNEL to one shape; this just keeps any residual mask variants cached.
    torch._dynamo.config.cache_size_limit = 256
    # Compile inductor kernels IN-PROCESS (no async subprocess worker pool). Inductor's default
    # SubprocPool forks ~32 compile workers from this parent, which already holds a live CUDA context
    # and a large thread pool — forking after CUDA-init + threads deadlocks: the run wedges at 0% GPU
    # right after the FlexAttention compile starts and never reaches epoch 0. Single-threaded compile is
    # a touch slower per kernel (~60s startup) but reliable; with 6 concurrent runs the CPU cost is fine.
    torch._inductor.config.compile_threads = 1
    summary_text = _run_summary_text(cfg)  # fail BEFORE any setup if the run note is missing
    _assert_summary_unique(summary_text, cfg)  # ...and fail if it merely copies a previous run's note
    if not data_exists(cfg):
        raise FileNotFoundError(
            "No dataset found. Run `python -m quickdraw.data_generation` first, then pass its run "
            f"dir as data.root=logs/data_generation_<ts>_<exp> (got data.root={cfg.data.root!r})."
        )

    run_dir = make_run_dir("train", cfg.experiment)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))

    _t = time.perf_counter()
    _startup_log(run_dir, "[startup] loading dataset (GPU-resident windows) + normalizer...")
    norm = normalizer(cfg)
    loaders = window_loaders(cfg, norm)
    _startup_log(run_dir, f"[startup] data ready in {time.perf_counter() - _t:.1f}s: "
                          f"{getattr(loaders['train'], 'N', '?')} train / {getattr(loaders['val'], 'N', '?')} val windows")
    model = build_model(cfg)
    _startup_log(run_dir, f"[startup] model built: {sum(p.numel() for p in model.parameters()) / 1000:.0f}K "
                          f"params (model={cfg.model.name})")
    if torch.cuda.is_available():
        # Compile the parallel forward only; the rollout stays EAGER. Compiling the transformer for the
        # rollout backfired badly: the rollout hits ~57 distinct sequence lengths, which blows past
        # torch._dynamo's recompile cache limit and thrashes (~18x slower). Eager rollout = the fast path.
        # NOTE: keep compile ON — FlexAttention needs torch.compile to build its kernel (disabling it
        # forces a slow eager-attention fallback). Use DEFAULT mode, not max-autotune: the forward is
        # used ~1 epoch under the p_tf curriculum, so max-autotune's long kernel search isn't worth the
        # multi-minute startup; default mode compiles fast and FlexAttention still gets its fused kernel.
        model = torch.compile(model)
        _startup_log(run_dir, "[startup] torch.compile wrapped (default mode, compile_threads=1). The "
                              "forward + FlexAttention JIT-compile on the first sanity/train batch — "
                              "watch for the [startup] sanity-check and [compile] lines below.")

    e = env_cfg(cfg)
    lit = LitWorldModel(model, norm, e.R, e.r, e.init_speed, cfg.data.P, cfg.data.F,
                        cfg.model.p_tf_start, cfg.model.p_tf_end, cfg.model.p_tf_warmup_epochs,
                        cfg.optim.lr, cfg.optim.weight_decay, cfg.model.detach_every,
                        variations=cfg.get("variations"), dt=e.dt)

    # one writer -> local run folder + wandb, identically (see logging/writer.py). Lightning's own
    # logger is OFF; all logging flows through the writer via LoggingCallback.
    writer = make_writer(run_dir, cfg, job_type="train")
    with open(os.path.join(run_dir, "auto_run_summary.txt"), "w") as f:
        f.write(summary_text + "\n")
    print(summary_text, flush=True)  # after wandb.init -> captured in the wandb console logs too
    # train/val run normally; subscribe to the eval routines enabled in conf/eval/default.yaml
    # (ood_horizon | ood_visual | ood_geometric | ood_dynamics | control), run every every_epochs
    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"),
                        monitor="val/metric/proprio/manifold_distance_error",
                        mode="min", save_top_k=cfg.trainer.save_top_k, save_last=True),
        LoggingCallback(writer, cfg, norm, e, cfg.eval.during_train.every_epochs,
                        [name for name, on in cfg.eval.during_train.evals.items() if on],
                        at_epochs=cfg.eval.during_train.get("at_epochs", None)),
        ProgressPrinter(run_dir),
    ]
    # single GPU: the GPU-resident loader holds the whole set on one device (no DistributedSampler),
    # so we pin devices=1 rather than let Lightning auto-pick DDP across both H100s.
    # enable_progress_bar=False: no tqdm; ProgressPrinter emits plain per-epoch lines instead.
    trainer = L.Trainer(max_epochs=cfg.trainer.max_epochs, precision=cfg.trainer.precision,
                        accelerator="gpu", devices=1, gradient_clip_val=1.0, enable_progress_bar=False,
                        check_val_every_n_epoch=1, callbacks=callbacks, logger=False,
                        inference_mode=cfg.trainer.get("inference_mode", False),  # False (val under no_grad,
                        #   not inference_mode) lets the contraction variation build its Jacobian graph on
                        #   val for val/loss/contraction. Measured to have NO speed cost vs inference_mode.
                        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
                        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0))
    trainer.fit(lit, loaders["train"], loaders["val"])

    # stable name for the best checkpoint, used by eval_ood / eval_control
    best = callbacks[0].best_model_path
    if best and os.path.exists(best):
        shutil.copy(best, os.path.join(run_dir, "checkpoints", "best.ckpt"))
    print(f"[train] done. run_dir={run_dir} (best -> checkpoints/best.ckpt)")


if __name__ == "__main__":
    main()
