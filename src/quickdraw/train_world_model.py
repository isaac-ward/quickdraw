"""Entrypoint: train the base world model. `python -m quickdraw.train_world_model`"""

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

from .environments.base import log_env_capabilities
from .environments.registry import make_env
from .logging.callback import BestCkptMirror, LoggingCallback, ProgressPrinter
from .logging.writer import make_writer
from .utils.logging import make_run_dir
from .training.lit import LitWorldModel
from .training.setup import autobatch_find, build_model, data_exists, env_cfg, normalizer, window_loaders


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


def _assert_summary_unique(summary_text, cfg, root=None) -> None:
    """A run_summary must NEVER duplicate a prior run's. Every launch describes THIS run's current
    hypothesis + what changed since the last attempt — a reused note is a stale, meaningless note.
    Escape hatch: run_summary.allow_duplicate=true for a deliberate exact rerun."""
    import glob
    if bool((cfg.get("run_summary") or {}).get("allow_duplicate", False)):
        return
    # Resolve the SAME root make_run_dir uses (QUICKDRAW_LOG_ROOT else "logs"); otherwise this globs "logs/"
    # while grouped runs land in "logs/<group>/<run>/" and the check silently never matches. Glob one AND two
    # levels deep to catch both flat (logs/<run>/) and grouped (logs/<group>/<run>/) layouts.
    root = root or os.environ.get("QUICKDRAW_LOG_ROOT", "logs")
    norm = " ".join(summary_text.split())
    prev = (glob.glob(os.path.join(root, "*", "auto_run_summary.txt"))
            + glob.glob(os.path.join(root, "*", "*", "auto_run_summary.txt")))
    for f in sorted(prev):
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
    resume = cfg.get("resume", None)   # +resume=<ckpt> -> CONTINUE that checkpoint's own run (see run_dir below)
    if resume:
        resume = os.path.expanduser(str(resume))
    summary_text = _run_summary_text(cfg)   # fail if the run note is missing (present on resume via the
    if not resume:                          # resolved config). A resume is a CONTINUATION, so skip the unique
        _assert_summary_unique(summary_text, cfg)   # run-note gate + the config.resolved.yaml re-write below.
    if not data_exists(cfg):
        raise FileNotFoundError(
            "No dataset found. Run `python -m quickdraw.data_generation` first, then pass its run "
            f"dir as data.root=logs/data_generation_<ts>_<exp> (got data.root={cfg.data.root!r})."
        )

    if resume:
        # CONTINUE the checkpoint's OWN run_dir so ModelCheckpoint's dirpath is unchanged -> Lightning reloads
        # best_model_score and best.ckpt tracks again (a fresh run_dir leaves best_model_score=NaN, which
        # freezes best.ckpt at the resumed epoch). Falls back to a new run_dir if the ckpt isn't laid out as
        # <run_dir>/checkpoints/<file>.
        ckpt_dir = os.path.dirname(resume)
        run_dir = (os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "checkpoints"
                   else make_run_dir("train_world_model", cfg.experiment))
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        print(f"[train] RESUMING from ckpt_path={resume} -> continuing run_dir={run_dir}", flush=True)
    else:
        run_dir = make_run_dir("train_world_model", cfg.experiment)   # logs/train_world_<ts>_<exp> (prefix names the entrypoint)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))

    # Auto-size the batch to fill VRAM (the AR step is dispatch-bound -> bigger batch is nearly-free
    # throughput; accelerations.md Exp 8). Fresh runs only — a resume keeps its original batch. Disable with
    # data.autobatch=false for controlled A/Bs where a FIXED batch matters.
    if not resume and bool(cfg.data.get("autobatch", True)) and torch.cuda.is_available():
        cfg.data.batch = int(autobatch_find(cfg, torch.device("cuda"), log=lambda m: _startup_log(run_dir, m)))
        OmegaConf.save(cfg, os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))  # record chosen batch

    _t = time.perf_counter()
    _startup_log(run_dir, "[startup] loading dataset (GPU-resident windows) + normalizer...")
    norm = normalizer(cfg)
    loaders = window_loaders(cfg, norm)
    _startup_log(run_dir, f"[startup] data ready in {time.perf_counter() - _t:.1f}s: "
                          f"{getattr(loaders['train'], 'N', '?')} train / {getattr(loaders['val'], 'N', '?')} val windows")
    # data inventory per split (trajectories / transitions / seconds / hours / windows) so coverage is legible
    hz = round(1.0 / cfg.environments.dt); P, Fh, strd = cfg.data.P, cfg.data.F, int(cfg.data.get("window_stride", 1)); winL = P + Fh
    _startup_log(run_dir, f"[startup] data inventory ({hz} Hz, P={P} F={Fh} L={winL} window_stride={strd}):")
    for name, s in cfg.data.splits.items():
        nt, st = int(s["n_traj"]), int(s["steps"]); frames = nt * st; secs = frames / hz
        line = (f"[startup]   {name:<18} {nt:>4} traj x {st:>5} steps = {frames:>8} frames "
                f"({nt * (st - 1):>8} transitions) = {secs:8.1f}s = {secs / 3600:5.2f}h")
        if name in ("train", "val"):
            ss = strd if name == "train" else 1                    # val stays dense (stride 1)
            per = (st - winL) // ss + 1 if st >= winL else 0
            line += f" | windows: {nt} x {per} (stride {ss}) = {nt * per}"
        else:
            line += f" | full-traj eval rollouts: {nt}"
        _startup_log(run_dir, line)
    model = build_model(cfg)
    _startup_log(run_dir, f"[startup] model built: {sum(p.numel() for p in model.parameters()) / 1000:.0f}K "
                          f"params (model={cfg.model.name})")
    if torch.cuda.is_available() and not cfg.model.get("modalities"):
        # (multimodal token-bag models skip WHOLE-MODEL compile: the per-batch image gather + ViT AE
        # complicate it.) This compile wraps the PARALLEL forward, which does get a fused FlexAttention kernel.
        # The AR rollout is DISPATCH-bound by the serial F-step Python loop (Self CPU ~3.5s >> Self CUDA ~0.68s,
        # GPU ~15-20% util; Exp 8). Two levers: (1) batch — fills the idle GPU, nearly free; (2) compile the
        # rollout STEP — MEASURED ~6x, parity-safe (opt-in model.compile_rollout, Exp 9): in the eager rollout
        # attention runs UNFUSED, and compiling the step both fuses it and collapses the ~256 dispatches.
        # Compile the parallel forward only; the rollout stays EAGER by default. (mode="reduce-overhead"/CUDA
        # graphs does NOT work for the rollout — incompatible with retained-BPTT; the old ~57-shape recompile
        # thrash that made rollout-compile look hopeless is fixed by the fixed-window pad_block_mask. Exp 9 +
        # design/rollout_throughput.md.)
        # NOTE: keep compile ON — FlexAttention needs torch.compile to build its kernel (disabling it
        # forces a slow eager-attention fallback). Use DEFAULT mode, not max-autotune: the forward is
        # used ~1 epoch under the p_tf curriculum, so max-autotune's long kernel search isn't worth the
        # multi-minute startup; default mode compiles fast and FlexAttention still gets its fused kernel.
        model = torch.compile(model)
        _startup_log(run_dir, "[startup] torch.compile wrapped (default mode, compile_threads=1). The "
                              "forward + FlexAttention JIT-compile on the first sanity/train batch — "
                              "watch for the [startup] sanity-check and [compile] lines below.")

    e = env_cfg(cfg)
    # the env supplies its OWN val rollout metrics (WorldEnv.rollout_metrics) — batch=1: metrics only, never stepped
    env_name = cfg.environments.get("name", "torus_world")
    env = make_env(env_name, cfg.environments, batch=1)
    # WorldEnv contract self-report (environments/base.py): one ✓/✗ line -> progress.log (additive, log-only)
    log_env_capabilities(env, lambda m: _startup_log(run_dir, m), name=env_name)
    lit = LitWorldModel(model, norm, e.R, e.r, e.init_speed, cfg.data.P, cfg.data.F,
                        cfg.model.p_tf_start, cfg.model.p_tf_end, cfg.model.p_tf_warmup_epochs,
                        cfg.optim.lr, cfg.optim.weight_decay, cfg.model.detach_every,
                        variations=cfg.get("variations"), dt=e.dt,
                        recon_frac=float(cfg.model.get("recon_frac", 1.0)),
                        lr_warmup_steps=int(cfg.optim.get("lr_warmup_steps", 0)), env=env)

    # one writer -> local run folder + wandb, identically (see logging/writer.py). Lightning's own
    # logger is OFF; all logging flows through the writer via LoggingCallback.
    writer = make_writer(run_dir, cfg, job_type="train")
    with open(os.path.join(run_dir, "auto_run_summary.txt"), "w") as f:
        f.write(summary_text + "\n")
    print(summary_text, flush=True)  # after wandb.init -> captured in the wandb console logs too
    # train/val run normally; subscribe to the eval routines enabled in conf/eval/default.yaml
    # (ood_horizon | ood_visual | ood_geometric | ood_dynamics | control), run every every_epochs
    # best.ckpt monitors the env's declared checkpoint metric (torus: manifold_distance_error, unchanged;
    # default: the generic pointwise_error) — must be a key of env.rollout_metrics.
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"),
                              monitor=f"val/metric/proprio/{getattr(env, 'checkpoint_metric', 'pointwise_error')}",
                              mode="min", save_top_k=cfg.trainer.save_top_k, save_last=True)
    callbacks = [
        ckpt_cb,
        LoggingCallback(writer, cfg, norm, e, cfg.eval.during_train.every_epochs,
                        [name for name, on in cfg.eval.during_train.evals.items() if on],
                        at_epochs=cfg.eval.during_train.get("at_epochs", None)),
        ProgressPrinter(run_dir),
        # keep checkpoints/best.ckpt current after EVERY val (not just at fit end) + log the best epoch to
        # progress.log — so an interrupted/collapsed run still has a correct best.ckpt. Must follow ckpt_cb.
        BestCkptMirror(ckpt_cb, run_dir),
    ]
    # single GPU: the GPU-resident loader holds the whole set on one device (no DistributedSampler),
    # so we pin devices=1 rather than let Lightning auto-pick DDP across both H100s.
    # enable_progress_bar=False: no tqdm; ProgressPrinter emits plain per-epoch lines instead.
    trainer = L.Trainer(max_epochs=cfg.trainer.max_epochs, precision=cfg.trainer.precision,
                        accelerator="gpu", devices=1, gradient_clip_val=1.0, enable_progress_bar=False,
                        accumulate_grad_batches=int(cfg.trainer.get("accumulate_grad_batches", 1)),  # effective
                        #  batch = data.batch x this; use it to keep a large effective batch when the per-step
                        #  micro-batch is memory-bound (no batchnorm here, so it's gradient-equivalent).
                        check_val_every_n_epoch=int(cfg.trainer.get("check_val_every_n_epoch", 1)),  # val is an
                        #  autoregressive rollout ~as long as the train epoch (~50% of wall time); raise this to
                        #  validate less often and train faster (e.g. 5). Eval routines have their own cadence.
                        callbacks=callbacks, logger=False,
                        inference_mode=cfg.trainer.get("inference_mode", False),  # False (val under no_grad,
                        #   not inference_mode) lets the contraction variation build its Jacobian graph on
                        #   val for val/loss/contraction. Measured to have NO speed cost vs inference_mode.
                        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
                        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
                        # +trainer.fast_dev_run=true -> 1 train + 1 val batch, no ckpt/logger: a build+forward
                        # preflight (used by the wizard) to prove data loads and the model runs before a real launch.
                        fast_dev_run=bool(cfg.trainer.get("fast_dev_run", False)))
    # `resume` (+ its run_dir continuation) was resolved at the top of main. ckpt_path restores
    # weights+optimizer+LR-scheduler+epoch — p_tf/physical warmups are current_epoch-keyed and the LR warmup
    # is a step-keyed Lightning LambdaLR, so every schedule restores correctly on resume.
    trainer.fit(lit, loaders["train"], loaders["val"], ckpt_path=resume)

    # stable name for the best checkpoint, used by eval_ood / eval_control
    best = callbacks[0].best_model_path
    if best and os.path.exists(best):
        shutil.copy(best, os.path.join(run_dir, "checkpoints", "best.ckpt"))
    print(f"[train] done. run_dir={run_dir} (best -> checkpoints/best.ckpt)")


if __name__ == "__main__":
    main()
