"""LoggingCallback: the single bridge from Lightning to the RunWriter.

Every epoch it forwards Lightning's aggregated metrics (train/* and val/*) to the writer, and on the
eval cadence (cfg.eval.during_train) it runs the subscribed eval routines — which themselves log only
through the same writer. step = epoch, so the local mirror and wandb stay in lockstep.
"""

from __future__ import annotations

import os
import shutil
import time

import lightning as L
import torch
from omegaconf import OmegaConf


def arch_summary_lines(m, *, max_epochs=None, device=None) -> list[str]:
    """The model architecture summary — total params + the per-component `arch_table` dataflow (component |
    shape transform | params) — as a list of printable lines. SHARED by the training
    `LoggingCallback.on_fit_start` (the top of progress.log) and the standalone `quickdraw.model_summary`
    entrypoint, so the two never drift."""
    m = getattr(m, "_orig_mod", m)   # unwrap torch.compile's OptimizedModule
    n = sum(p.numel() for p in m.parameters())
    head = f"[train] {type(m).__name__} {n / 1e6:.2f}M params"
    if max_epochs is not None:
        head += f" | max_epochs={max_epochs}"
    if device is not None:
        head += f" | device={device}"
    lines = [head]
    # PRETRAINED-AE adapter mode (#12). The token bag either holds the AE's latent EXACTLY (a parameter-free
    # index rearrangement -> identity round-trip at init) or it does not, and that distinction decides whether
    # the epoch-0 ae_floor is an assertable guarantee or merely a number to watch. Say which, out loud.
    for _nm, _mod in getattr(m, "modalities", {}).items():
        info = getattr(_mod, "adapter_info", None)
        if not info:
            continue
        c, gh, gw = info["grid"]
        T, dd = info["bag"]
        detail = ("bijective reshape, no pad, no idle decode width" if info["mode"] == "EXACT" else
                  f"{info['per']} real floats/token + {info['pad_per_token']} pad "
                  f"({100 * info['per'] / dd:.0f}% of width read back at init)" if info["mode"] == "PADDED" else
                  f"learned {info['per']}->{dd} projection (dense tokens)")
        guarantee = ("identity-at-init GUARANTEED" if info["identity_at_init"]
                     else "identity NOT guaranteed — round-trip must be LEARNED")
        lines.append(f"[adapter] {_nm}: AE latent ({c},{gh},{gw})={info['L']} floats -> bag {T}x{dd}={info['M']}"
                     f" | {info['mode']}: {detail} | {guarantee}"
                     f" | roundtrip_loss w={getattr(_mod, 'latent_loss_weight', 0.0):g}")
        if info["mode"] in ("PADDED", "PROJECTED"):
            # M > L is NOT free. The bag carries M floats but only L of them are real, so every step pays
            # attention + backbone compute on num_tokens tokens while the decode path reads back only `per`
            # dims each. Say the waste out loud, with the exact resize that makes it EXACT.
            waste = 100.0 * (1.0 - info["L"] / info["M"])
            fixes = []
            if info["L"] % T == 0:
                fixes.append(f"model.d={info['L'] // T}")
            if info["L"] % dd == 0:
                fixes.append(f"num_tokens={info['L'] // dd}")
            hint = " or ".join(fixes) if fixes else f"any num_tokens*d == {info['L']}"
            lines.append(f"[adapter] {_nm}: WASTING {waste:.0f}% of the bag ({info['M'] - info['L']} of "
                         f"{info['M']} floats carry no latent) — {T} tokens are attended every step but only "
                         f"{info['per']}/{dd} dims per token feed the decoder at init. For EXACT set {hint}.")
    if hasattr(m, "arch_table"):   # token-bag dataflow (component | shape transform | params)
        lines.append(f"[arch] d={m.d} window={m.window} | per-step bag = {m.n_state} state token(s) + 1 action = {m.n_input} tokens")
        for comp, shape, params in m.arch_table():
            lines.append(f"  {comp:<34} {shape:<60} {params / 1e6:7.3f}M")
    else:
        for name, mod in m.named_children():
            sub = sum(p.numel() for p in mod.parameters())
            if sub:
                lines.append(f"  [model] {name:<14} {sub / 1000:8.1f}K params")
    return lines


@torch.no_grad()
def _bench_rollout(model, P, F, obs_dim, act_dim, device, B, warmup=1, iters=3):
    """Mean wall-clock seconds for ONE full F-step autoregressive rollout at batch B. Warmup +
    cuda-sync so the number reflects real compute, not async launch overhead or first-call compile."""
    was_training = model.training
    model.eval()
    ctx = torch.randn(B, P, obs_dim, device=device)
    act = torch.randn(B, P + F - 1, act_dim, device=device)
    cuda = device.type == "cuda"
    for _ in range(warmup):
        model.imagine_eval(ctx, act, F)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.imagine_eval(ctx, act, F)
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    if was_training:
        model.train()
    return dt


def _fmt_secs(s: float) -> str:
    s = int(s)
    h, m = s // 3600, (s % 3600) // 60
    return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m:02d}:{s % 60:02d}"


class ProgressPrinter(L.Callback):
    """Plain-text per-epoch progress (no tqdm), so it shows up in piped stdout, `> out` files and
    log tails. Prints to stdout AND appends to `<run_dir>/progress.log`."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "progress.log")
        self._t0 = None
        self._epoch_times = []
        self._t_b0 = None       # first-train-batch start (the batch that pays the torch.compile cost)
        self._compiled = False

    def _emit(self, line: str):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {line}"  # wall-clock stamp so durations are exact
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def on_fit_start(self, trainer, pl_module):
        for line in arch_summary_lines(pl_module.model, max_epochs=trainer.max_epochs, device=pl_module.device):
            self._emit(line)

    def on_sanity_check_start(self, trainer, pl_module):
        self._emit("[startup] sanity-check validation running (this JIT-compiles the val/forward path)...")

    def on_sanity_check_end(self, trainer, pl_module):
        self._emit("[startup] sanity-check done")

    def on_train_start(self, trainer, pl_module):
        self._emit("[startup] epoch 0 training started")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx == 0 and self._t_b0 is None:
            self._t_b0 = time.perf_counter()  # the first batch pays the (remaining) torch.compile cost

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._t_b0 is not None and not self._compiled:
            self._emit(f"[compile] first training batch (incl. any torch.compile) {time.perf_counter() - self._t_b0:.1f}s")
            self._compiled = True
        # live intra-epoch progress at ~quartiles: AR-rollout epochs (p_tf<1) are MUCH longer than the
        # parallel epoch 0, so without this a long epoch looks hung. The ~left is a THIS-EPOCH estimate.
        nb = trainer.num_training_batches
        if self._t0 and nb and nb != float("inf"):
            q = max(1, int(nb) // 4)
            done = batch_idx + 1
            if done % q == 0 and done < nb:
                el = time.time() - self._t0
                frac = done / nb
                eta_ep = el / frac - el   # seconds left in THIS epoch
                self._emit(f"[ep {trainer.current_epoch:>3} {100 * frac:3.0f}%] {done}/{int(nb)} batches | "
                           f"{el:.0f}s elapsed, epoch done ~{time.strftime('%H:%M:%S', time.localtime(time.time() + eta_ep))}")

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        nb = trainer.num_val_batches
        nb = int(nb[0]) if isinstance(nb, (list, tuple)) and nb else (int(nb) if nb and nb != float("inf") else None)
        # explain the post-training-epoch pause: val is AUTOREGRESSIVE (p_tf=0), ~as slow as an AR train epoch
        self._emit(f"[ep {trainer.current_epoch:>3}] validating (autoregressive"
                   + (f", {nb} batches" if nb else "") + ") — this is ~as long as the train epoch...")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        dt = time.time() - self._t0 if self._t0 else 0.0
        self._epoch_times.append(dt)
        left = max(0, trainer.max_epochs - (trainer.current_epoch + 1))
        m = trainer.callback_metrics

        def g(k):
            v = m.get(k)
            return v.item() if v is not None else float("nan")

        self._emit(
            f"[ep {trainer.current_epoch:>3}/{trainer.max_epochs}] "
            f"train_loss={g('train/loss/total'):.4f} val_loss={g('val/loss/total'):.4f} "
            f"| {dt:.1f}s/ep  run_eta {_fmt_secs(dt * left)} "  # run_eta = THIS epoch's time x epochs left
            f"(finish ~{time.strftime('%m-%d %H:%M', time.localtime(time.time() + dt * left))})"
        )


class LoggingCallback(L.Callback):
    def __init__(self, writer, cfg, normalizer, ecfg, every_epochs: int, routines: list[str],
                 at_epochs=None):
        self.writer = writer
        self.cfg = cfg
        self.norm = normalizer
        self.ecfg = ecfg
        self.every = every_epochs
        self.at_epochs = set(int(e) for e in at_epochs) if at_epochs else None  # extra epochs to ALSO eval at (unioned with the every-N cadence)
        self.routines = routines
        self._t_fit = self._t_epoch = self._t_b0 = None
        self._eval_cum = 0.0
        self._compile_s = None
        self._skipped = []   # (epoch, name) of every non-fatally-skipped eval, for the run-end summary

    def _eval_due(self, epoch: int) -> bool:
        # Cadence uses Lightning's (epoch+1)%N phase — SAME phase as validation (check_val_every_n_epoch) and
        # the checkpoint — so the eval cadence COINCIDES with a val + fresh checkpoint (e.g. every=2 -> {1,3,5}
        # 0-indexed, the epochs val runs). Eval still fires from on_train_epoch_end (every epoch, self-gated),
        # NOT the val hook, so it never gets silently dropped when the eval cadence != val cadence. Epoch 0
        # (untrained) is skipped. at_epochs is a UNION of explicit one-off epoch INDICES (literal, not phase-shifted).
        if epoch <= 0:
            return False
        cadence = self.every > 0 and (epoch + 1) % self.every == 0
        extra = self.at_epochs is not None and epoch in self.at_epochs
        return cadence or extra

    def on_fit_start(self, trainer, pl_module):
        self.writer.config(OmegaConf.to_container(self.cfg, resolve=True))
        self._t_fit = time.perf_counter()
        n_params = sum(p.numel() for p in pl_module.model.parameters())   # constant -> log ONCE, not per epoch
        self.writer.scalars({"model/params": float(n_params), "model/params_millions": n_params / 1e6}, step=0)
        # EPOCH-0 pretrained-AE gate (#12 §4), BEFORE any training. In an identity-guaranteed adapter mode the
        # round-trip must already equal the raw AE; a mismatch is an indexing/zero-init bug that would otherwise
        # masquerade as a bad model for the whole run (which is exactly what the ~10.5 dB Perceiver collapse did).
        # No-op unless a pretrained-AE trunk is present.
        try:
            from ..controller.run import _plog
            from ..evaluation.ae_floor import assert_identity_floor
            res = assert_identity_floor(self.cfg, pl_module.model, log=lambda s: _plog(self.writer, s))
            for nm, r in (res or {}).items():
                self.writer.scalars({f"eval_ae_floor/{nm}/init_adapter_psnr": r["adapter_db"],
                                     f"eval_ae_floor/{nm}/init_raw_ae_psnr": r["raw_db"],
                                     f"eval_ae_floor/{nm}/init_delta_db": r["delta_db"]}, step=0)
        except AssertionError:
            raise                                  # a broken identity is fatal — do NOT train through it
        except Exception as e:                     # a missing AE/dataset must not kill a run over a diagnostic
            from ..controller.run import _plog
            _plog(self.writer, f"[ae_floor @ep0] skipped ({type(e).__name__}: {e})")

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx == 0 and self._t_b0 is None:
            self._t_b0 = time.perf_counter()  # the very first batch pays the torch.compile cost

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._compile_s is None and self._t_b0 is not None:
            self._compile_s = time.perf_counter() - self._t_b0  # ~ one-time compile

    def on_train_epoch_end(self, trainer, pl_module):
        # Eval routines fire on the eval CADENCE, decoupled from the validation cadence. They used to live in
        # on_validation_epoch_end, which only fires on val epochs — so with check_val_every_n_epoch=2 (odd val
        # epochs) and an even eval cadence (10, 20, 40, ...) they NEVER coincided and no eval ran. Running them
        # here (every train-epoch end) means the eval cadence is honored regardless of the val cadence. The
        # model is put in eval mode for the routines, then restored (validation, which follows, sets its own).
        if trainer.current_epoch == 0 and self.routines and not trainer.sanity_checking:
            # state the epoch-0 skip in progress.log (for eval AND val): the untrained baseline has nothing
            # meaningful to evaluate/validate, so both intentionally skip epoch 0. Only logged here (once).
            from ..controller.run import _plog
            cv = int(getattr(trainer, "check_val_every_n_epoch", 1) or 1)
            who = "eval" + (" + val" if cv > 1 else "")
            _plog(self.writer, f"[{who} @ep0] SKIPPED — epoch-0 baseline not evaluated (barely trained); "
                               f"cadence begins at ep1.")
        if trainer.sanity_checking or not (self.routines and self._eval_due(trainer.current_epoch)):
            return
        from ..evaluation.routines import REGISTRY
        epoch = trainer.current_epoch
        m = pl_module.model
        was_training = m.training
        m.eval()
        t_eval = time.perf_counter()
        try:
            for name in self.routines:
                try:
                    REGISTRY[name](self.cfg, m, self.norm, self.ecfg, self.writer, pl_module.device, epoch)
                except Exception as e:   # a DIAGNOSTIC eval must NEVER kill training — a diverged model can emit
                    #                       non-finite renders (e.g. a NaN control-arrow direction -> pyvista
                    #                       "matrix must have finite values"), OOM a viz, etc. Log loudly + go on.
                    import traceback
                    from ..controller.run import _plog   # house helper -> the failure line reaches progress.log,
                    #                                       not just stdout (a docker-exec session loses stdout).
                    _plog(self.writer, f"[eval:{name} @ep{epoch}] FAILED non-fatally ({type(e).__name__}: {e}); "
                          f"skipping this routine, CONTINUING training.")
                    self._skipped.append((epoch, name))
                    self.writer.scalar(f"eval/skipped/{name}", 1.0, step=epoch)  # logged -> a skipped eval is
                    #                       now distinguishable from a metric that was never enabled.
                    traceback.print_exc()
            bench = self._bench(pl_module)
            if bench:
                self.writer.scalars(bench, step=epoch)   # inference-speed scalars (empty for multimodal)
        finally:
            m.train(was_training)
        self._eval_cum += time.perf_counter() - t_eval

    def _bench(self, pl_module):
        """time/ms/* + time/hz/* from a controlled rollout micro-benchmark (batch B and batch 1)."""
        m, dev = pl_module.model, pl_module.device
        if hasattr(getattr(m, "_orig_mod", m), "layout"):   # multimodal (dict obs) — skip the vector benchmark
            return {}
        P, F, B = self.cfg.data.P, self.cfg.data.F, int(self.cfg.data.batch)
        od, ad = m.cfg.obs_dim, m.cfg.action_dim
        roll_b = _bench_rollout(m, P, F, od, ad, dev, B)   # full rollout, whole batch
        roll_1 = _bench_rollout(m, P, F, od, ad, dev, 1)   # full rollout, single sample (true latency)
        secs = {"step_batch": roll_b / F, "step_sample_amortized": roll_b / F / B, "step_sample_true": roll_1 / F,
                "rollout_batch": roll_b, "rollout_sample_amortized": roll_b / B, "rollout_sample_true": roll_1}
        out = {}
        for k, s in secs.items():
            out[f"time/ms/{k}"] = s * 1000.0
            out[f"time/hz/{k}"] = (1.0 / s) if s > 0 else 0.0
        return out

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        # forward every aggregated scalar metric (train/* and val/*) through the one writer
        self.writer.scalars({k: v.item() for k, v in trainer.callback_metrics.items()}, step=epoch)

        metrics = {}                                  # eval routines now run in on_train_epoch_end (decoupled from val cadence)

        # wall-clock + cost, every epoch (eval time — run earlier in on_train_epoch_end — is in self._eval_cum)
        now = time.perf_counter()
        total_s = now - self._t_fit
        metrics["time/epoch_seconds"] = now - self._t_epoch
        avg_s = total_s / (epoch + 1)  # mean wall-clock per epoch so far
        for nm, s in (("total", total_s), ("train", max(0.0, total_s - self._eval_cum)),
                      ("eval", self._eval_cum), ("avg_epoch", avg_s)):
            metrics[f"time/{nm}_seconds"], metrics[f"time/{nm}_minutes"], metrics[f"time/{nm}_hours"] = s, s / 60, s / 3600
        if self._compile_s is not None:
            metrics["time/compile_seconds"] = self._compile_s
        if torch.cuda.is_available():
            metrics["mem/peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self.writer.scalars(metrics, step=epoch)

    def on_fit_end(self, trainer, pl_module):
        if self._skipped:   # run-end summary so silently-skipped diagnostics are visible in progress.log
            from ..controller.run import _plog
            items = ", ".join(f"{n}@ep{e}" for e, n in self._skipped)
            _plog(self.writer, f"[eval] {len(self._skipped)} eval(s) SKIPPED non-fatally this run: {items}")
        self.writer.finalize()


class BestCkptMirror(L.Callback):
    """Keep `checkpoints/best.ckpt` current THROUGHOUT training (not just at fit end): after every
    validation, mirror the ModelCheckpoint's running best to best.ckpt and log the decision to
    <run_dir>/progress.log. So an interrupted or collapsed run still leaves a correct best.ckpt, and the
    best-so-far epoch is visible live. Must be placed AFTER the ModelCheckpoint in the callback list so its
    best_model_path is already updated for this validation."""

    def __init__(self, ckpt_cb, run_dir: str):
        self.cb = ckpt_cb
        self.dst = os.path.join(run_dir, "checkpoints", "best.ckpt")
        self.path = os.path.join(run_dir, "progress.log")
        self._prev = None

    def _emit(self, line: str):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {line}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")

    def on_validation_end(self, trainer, pl_module):
        bp = self.cb.best_model_path
        if not bp:                                  # no monitored checkpoint yet (e.g. sanity check)
            return
        ep = trainer.current_epoch
        try:
            score = float(self.cb.best_model_score)
        except (TypeError, ValueError):
            score = float("nan")
        name = os.path.basename(bp)
        if bp != self._prev:                        # the best changed this validation -> mirror + announce
            if os.path.exists(bp):
                shutil.copy(bp, self.dst)
            self._prev = bp
            self._emit(f"[best-ckpt] NEW BEST at epoch {ep} (monitor={score:.5f}) -> best.ckpt now {name}")
        else:
            self._emit(f"[best-ckpt] best unchanged: epoch {ep} did not beat {name} (best monitor={score:.5f})")
