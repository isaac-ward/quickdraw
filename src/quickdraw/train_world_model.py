"""Entrypoint: train the base world model. `python -m quickdraw.train_world_model`"""

from __future__ import annotations

import os
import shutil
import time

# PYTORCH_CUDA_ALLOC_CONF now lives in quickdraw/__init__.py (the package chokepoint that already exists for
# the BLAS thread caps), so that ANY entrypoint -- including a bare `from quickdraw.training.setup import
# autobatch_find` -- probes the same allocator training uses. It used to be set here only, which silently made
# every probe-only measurement invalid. Kept as a comment, not a duplicate setdefault, so there is one owner.

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
from .training.setup import (apply_size_preset, autobatch_find, build_model, data_exists, env_cfg, normalizer,
                             window_loaders)


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


def _cli_overrides():
    """The hydra task overrides the caller actually typed (so a resume can tell 'unset' from 'explicitly set')."""
    try:
        from hydra.core.hydra_config import HydraConfig
        return list(HydraConfig.get().overrides.task)
    except Exception:
        return []


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    # Seed FIRST, before anything draws a random number (model init happens in build_model far below, but the
    # autobatch probe builds a throwaway model too). workers=True also seeds dataloader workers.
    L.seed_everything(int(cfg.get("seed", 0) or 0), workers=True)
    torch.set_float32_matmul_precision("high")
    # cuDNN autotuning: on the FIRST occurrence of each conv input shape, benchmark every available algorithm
    # and cache the winner, instead of picking by heuristic. Our shapes are STATIC after step 0 (fixed batch
    # from autobatch, fixed F/window/img_size) and the step is conv-heavy -- the image decoder is ~78% of
    # per-sample memory (design/decode_memory.md) and runs TWICE per step -- which is exactly the case this
    # flag exists for. Algorithm SELECTION only: the convolution computed is the same, so the only numerical
    # effect is float reassociation, on a par with set_float32_matmul_precision above. Costs a one-off probe
    # per new shape, which is why it is only safe BECAUSE the shapes are static; a shape-varying workload
    # would re-benchmark forever. See design/accelerations.md (earmarked 2026-08-26, item 6).
    torch.backends.cudnn.benchmark = True
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
    if not resume:                     # resume's config.resolved already has the preset baked (d/num_tokens/decode_base)
        apply_size_preset(cfg)         # model.size=tiny|small -> set the hidden capacity knobs (raise on knob clash)
    summary_text = _run_summary_text(cfg)   # fail if the run note is missing (present on resume via the
    if not resume:                          # resolved config). A resume is a CONTINUATION, so skip the unique
        _assert_summary_unique(summary_text, cfg)   # run-note gate + the config.resolved.yaml re-write below.
    # Frame stride, set ONCE before anything loads episodes (autobatch below loads data too). Applied inside
    # the loaders so the training windows and every eval routine cannot end up at different rates.
    from .data.dataset import (set_action_aggregate, set_action_control, set_obs_keep, set_subsample,
                               set_subsample_all_phases)
    set_subsample(int(cfg.data.get("subsample", 1) or 1))
    # Emit all `subsample` phase offsets as separate TRAIN episodes -- ~s x the windows at the SAME frame rate,
    # using the frames the decimation otherwise throws away. Off = bit-identical. See set_subsample_all_phases.
    set_subsample_all_phases(bool(cfg.data.get("subsample_all_phases", False)))
    set_action_aggregate(cfg.data.get("action_aggregate", "sum"))   # how subsample combines skipped actions
    set_action_control(cfg.data.get("action_control", "none"))      # ABLATION: destroy the action's info
    # action_aggregate=concat makes the action width a function of data.subsample, so model.action_dim must
    # NOT be a hand-kept constant. Derive it from whatever is configured through the ONE definition in
    # data/dataset.py, and say so. physical_loss consumes act_raw positionally (physics_proprio_chained),
    # so a reshaped action would be silently misread -- refuse rather than mis-train.
    if str(cfg.data.get("action_aggregate", "sum")) == "concat":
        if float(cfg.get("variations", {}).get("physical_loss", {}).get("weight", 0.0) or 0.0) > 0.0:
            raise ValueError("data.action_aggregate=concat with variations.physical_loss.weight>0: the "
                             "physics prior reads act_raw positionally and cannot interpret a concatenated "
                             "action. Set physical_loss.weight=0 or use action_aggregate=last.")
        print(f"[action_aggregate] concat: the action width is derived from data.subsample="
              f"{cfg.data.get('subsample', 1)} in training/setup.py, not from model.action_dim", flush=True)
    set_obs_keep(cfg.data.get("obs_keep", None))   # process-wide obs subset -> applied inside every loader + normalizer
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
    _ab_on = not resume and bool(cfg.data.get("autobatch", True)) and torch.cuda.is_available()
    if _ab_on:
        cfg.data.batch = int(autobatch_find(cfg, torch.device("cuda"), log=lambda m: _startup_log(run_dir, m)))
        OmegaConf.save(cfg, os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))  # record chosen batch
    if not resume:
        # ALWAYS report the batch, and say WHERE it came from. With data.autobatch=false the entire
        # [autobatch] block is skipped, so progress.log contained NO record of the batch size at all and the
        # only trace was checkpoints/config.resolved.yaml. That gap directly caused a wrong reading of two
        # arms (2026-08-30): a stale "fit chose data.batch=16" from an earlier CRASHED launch was carried
        # forward, and windows-per-epoch was computed from it, producing a claimed 30% data handicap that did
        # not exist -- both arms were actually at 26. One unconditional line prevents that class of error.
        _startup_log(run_dir, f"[batch] data.batch={int(cfg.data.batch)} "
                              f"({'autobatch' if _ab_on else 'PINNED via data.autobatch=false'}) | "
                              f"F={int(cfg.data.get('F', 0))} subsample={cfg.data.get('subsample')}")
    elif resume:
        # A RESUME KEEPS ITS ORIGINAL BATCH -- which the comment above always claimed but nothing implemented
        # (fixed 2026-08-18). autobatch is skipped on resume regardless of data.autobatch, so cfg.data.batch fell
        # through to the CONFIG DEFAULT (1024 in conf/data/torus.yaml) while the run had actually trained at e.g.
        # 8, and the resume OOM'd instantly. The chosen batch was already recorded in config.resolved.yaml by the
        # branch above ("record chosen batch") -- it was written and never read back. Read it back.
        # An explicit `data.batch=` on the resume command still wins: this only fills in when the caller did not
        # say, which is exactly when guessing 1024 was doing damage.
        _rc = os.path.join(run_dir, "checkpoints", "config.resolved.yaml")
        # lstrip("+") because `+data.batch=17` and `++data.batch=17` are both legal hydra forms for a key that
        # already exists, and a bare startswith() missed them -- silently overriding an EXPLICIT user batch with
        # the saved one. Only the plain form was ever used in production, so this was safe by luck.
        _explicit = any(str(o).lstrip("+").startswith("data.batch=") for o in _cli_overrides())
        if not os.path.exists(_rc) and not _explicit:
            # The resume path was not laid out as <run_dir>/checkpoints/<file>, so run_dir fell back to a NEW dir
            # and there is no recorded batch to restore -- meaning cfg.data.batch is still the CONFIG DEFAULT
            # (1024), which is the exact instant-OOM this branch exists to prevent. Say so instead of proceeding.
            print(f"[train] resume WARNING: no config.resolved.yaml at {_rc}, so the batch this run trained at "
                  f"could not be restored and data.batch is the config default ({cfg.data.batch}). This will "
                  f"very likely OOM. Pass data.batch=<the value the run used> explicitly.", flush=True)
        if os.path.exists(_rc) and not _explicit:
            _saved = OmegaConf.load(_rc).data.get("batch", None)
            if _saved is not None and int(_saved) != int(cfg.data.batch):
                print(f"[train] resume: restoring data.batch={int(_saved)} from config.resolved.yaml "
                      f"(was {int(cfg.data.batch)} from the config default -- autobatch does not re-probe on "
                      f"resume, so without this the run would train at the wrong batch or OOM)", flush=True)
                cfg.data.batch = int(_saved)

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
                        lr_warmup_steps=int(cfg.optim.get("lr_warmup_steps", 0)), env=env,
                        p_tf_batch_granular=bool(cfg.model.get("p_tf_batch_granular", True)),
                        # sample the expensive grad diagnostics (per-module norms, nan/inf counts) instead of
                        # running them every step; the non-finite guard and grad/norm_preclip stay per-step.
                        grad_diag_every=int(cfg.trainer.get("grad_diag_every", 25) or 25))

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
    # best.ckpt monitors THIS metric (min). Default: the env's declared checkpoint metric under
    # val/metric/proprio/ (torus: manifold_distance_error; recorded/default: pointwise_error, the black-box
    # decode error). Override with ANY fully-qualified logged metric via trainer.checkpoint_monitor — e.g.
    # "val/loss/physics/proprio" for a physics-prior WM, where pointwise_error is a black-box red herring.
    # BestCkptMirror prints the resolved rule to progress.log at fit start so the standard is never ambiguous.
    # AUTO monitor: prefer the IMAGE metric when the model has an image modality. The old fallback was always
    # the env's proprio metric, which on an image run picks best.ckpt blind to every image result (measured:
    # bott_bott16 pinned best.ckpt to e8 while its floor peaked e14 and its perceptual distance e18). mse and
    # NOT psnr because psnr = -10*log10(mse) -> minimising mse IS maximising psnr, while staying a min-metric.
    _img = next((m for m in cfg.model.get("modalities", []) or [] if str(m.get("kind", "")) == "image"), None)
    # DEFAULT IS THE VISUAL-LOSS MIX, not mse. Both are open-loop rollout metrics on val, but mse is
    # structurally blind to sharpness (record §22), so monitoring it picks the blurriest-acceptable epoch --
    # exactly the criterion the L1+LPIPS loss was adopted to replace. Measured on vl_l1x3: best val mse was
    # ep15 (open-loop LPIPS@+128 0.1455) while the best open-loop LPIPS was ep11 (0.1370). `.../visual` is
    # the mix the model is actually trained on, so best.ckpt now tracks the objective rather than a proxy
    # that contradicts it. For a pure-L2 config VisualLoss IS mse, so this changes nothing there.
    # Override with trainer.checkpoint_monitor (e.g. a specific head when there are several).
    ckpt_monitor = cfg.trainer.get("checkpoint_monitor", None) or (
        f"val/metric/{_img.get('name', 'image')}/visual" if _img is not None
        else f"val/metric/proprio/{getattr(env, 'checkpoint_metric', 'pointwise_error')}")
    # AUTO direction from the metric NAME. mode used to be hardcoded "min", so aiming checkpoint_monitor at a
    # higher-is-better metric silently selected the WORST epoch. Override with trainer.checkpoint_mode.
    _hi = ("psnr", "ssim", "acc", "accuracy", "reward", "r2", "return")
    ckpt_mode = str(cfg.trainer.get("checkpoint_mode", None)
                    or ("max" if ckpt_monitor.rsplit("/", 1)[-1].lower() in _hi else "min"))
    assert ckpt_mode in ("min", "max"), f"trainer.checkpoint_mode must be min|max, got {ckpt_mode!r}"
    ckpt_cb = ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"),
                              monitor=ckpt_monitor,
                              mode=ckpt_mode, save_top_k=cfg.trainer.save_top_k, save_last=False)
    # save_last on the monitored callback only writes last.ckpt when Lightning ALSO saves a top-k file, so
    # once the monitored metric stops improving the newest weights stop being written — a collapsed run then
    # leaves NOTHING from after the collapse and the failure cannot be inspected (observed 2026-08-09: a run
    # that collapsed at epoch 5 had best/last/epoch=4 all identical at epoch 4). This monitor-free callback
    # saves every epoch unconditionally so post-collapse state always exists. It writes the SAME last.ckpt
    # name (ckpt_cb's save_last is off), so there is exactly one "newest weights" file, not two.
    latest_cb = ModelCheckpoint(dirpath=os.path.join(run_dir, "checkpoints"), filename="last",
                                save_top_k=1, every_n_epochs=1, monitor=None, save_last=False)
    callbacks = [
        ckpt_cb,
        latest_cb,
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
                        accelerator="gpu", devices=1, enable_progress_bar=False,
                        # exposed (default unchanged) so the clip can be varied; was hardcoded at 1.0
                        gradient_clip_val=float(cfg.trainer.get("gradient_clip_val", 1.0) or 0.0) or None,
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
