"""LoggingCallback: the single bridge from Lightning to the RunWriter.

Every epoch it forwards Lightning's aggregated metrics (train/* and val/*) to the writer, and on the
eval cadence (cfg.eval.during_train) it runs the subscribed eval routines — which themselves log only
through the same writer. step = epoch, so the local mirror and wandb stay in lockstep.
"""

from __future__ import annotations

import os
import time

import lightning as L
import torch
from omegaconf import OmegaConf


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
        m = getattr(pl_module.model, "_orig_mod", pl_module.model)  # unwrap torch.compile's OptimizedModule
        n = sum(p.numel() for p in m.parameters())
        self._emit(f"[train] {n/1000:.0f}K params | max_epochs={trainer.max_epochs} | device={pl_module.device}")
        for name, mod in m.named_children():  # per-submodule param summary (so progress.log shows the model)
            sub = sum(p.numel() for p in mod.parameters())
            if sub:
                self._emit(f"  [model] {name:<14} {sub/1000:8.1f}K params")

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

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        dt = time.time() - self._t0 if self._t0 else 0.0
        self._epoch_times.append(dt)
        avg = sum(self._epoch_times) / len(self._epoch_times)
        left = max(0, trainer.max_epochs - (trainer.current_epoch + 1))
        m = trainer.callback_metrics

        def g(k):
            v = m.get(k)
            return v.item() if v is not None else float("nan")

        self._emit(
            f"[ep {trainer.current_epoch:>3}/{trainer.max_epochs}] "
            f"train_loss={g('train/loss/total'):.4f} val_loss={g('val/loss/total'):.4f} "
            f"val_MDE={g('val/manifold_distance_error'):.4f} val_pw={g('val/pointwise_error'):.4f} "
            f"val_tv={g('val/tangent_velocity_error'):.4f} p_tf={g('schedules/p_tf'):.2f} "
            f"| {dt:.1f}s/ep  eta {_fmt_secs(avg * left)}"
        )


class LoggingCallback(L.Callback):
    def __init__(self, writer, cfg, normalizer, ecfg, every_epochs: int, routines: list[str],
                 at_epochs=None):
        self.writer = writer
        self.cfg = cfg
        self.norm = normalizer
        self.ecfg = ecfg
        self.every = every_epochs
        self.at_epochs = set(int(e) for e in at_epochs) if at_epochs else None  # explicit benchmark epochs
        self.routines = routines
        self._t_fit = self._t_epoch = self._t_b0 = None
        self._eval_cum = 0.0
        self._compile_s = None

    def _eval_due(self, epoch: int) -> bool:
        # explicit list (e.g. [20, 40]) takes precedence over the every-N cadence
        if self.at_epochs is not None:
            return epoch in self.at_epochs
        return self.every > 0 and epoch % self.every == 0

    def on_fit_start(self, trainer, pl_module):
        self.writer.config(OmegaConf.to_container(self.cfg, resolve=True))
        self._t_fit = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx == 0 and self._t_b0 is None:
            self._t_b0 = time.perf_counter()  # the very first batch pays the torch.compile cost

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._compile_s is None and self._t_b0 is not None:
            self._compile_s = time.perf_counter() - self._t_b0  # ~ one-time compile

    def _bench(self, pl_module):
        """time/ms/* + time/hz/* from a controlled rollout micro-benchmark (batch B and batch 1)."""
        m, dev = pl_module.model, pl_module.device
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

        metrics = {}
        if self.routines and self._eval_due(epoch):
            from ..evaluation.routines import REGISTRY

            t_eval = time.perf_counter()
            for name in self.routines:
                REGISTRY[name](self.cfg, pl_module.model, self.norm, self.ecfg, self.writer, pl_module.device, epoch)
            self._eval_cum += time.perf_counter() - t_eval
            metrics.update(self._bench(pl_module))  # inference-speed scalars, only on eval epochs

        # wall-clock + cost, every epoch (computed after the eval routine so epoch time includes it)
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
        metrics["model/params"] = float(sum(p.numel() for p in pl_module.model.parameters()))
        self.writer.scalars(metrics, step=epoch)

    def on_fit_end(self, trainer, pl_module):
        self.writer.finalize()
