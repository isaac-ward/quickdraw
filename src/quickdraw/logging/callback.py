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
        # The guarantee is a property of the ADAPTER ALONE. encode_state LayerNorms the bag when
        # model.latent_norm is on, and non-affine LN discards each token's mean+std, so the END-TO-END
        # round-trip is legitimately below the adapter-only one. Do not claim otherwise (the earlier line did).
        _ln_on = bool(getattr(m, "latent_norm", False))
        guarantee = ("exactly invertible" if info["identity_at_init"] else "LEARNED (lossy) grid<->token map")
        if _ln_on:
            guarantee += "; bag is LayerNormed before the dynamics, so the real floor is lower — see [ae_floor @ep0]"
        lines.append(f"[adapter] {_nm}: AE latent ({c},{gh},{gw})={info['L']} floats -> bag {T}x{dd}={info['M']}"
                     f" | {info['mode']}: {detail} | {guarantee}"
                     f" | roundtrip_loss w={getattr(_mod, 'latent_loss_weight', 0.0):g}")
        if info["mode"] in ("PADDED", "PROJECTED"):
            # M > L is NOT free, and it is NOT merely idle capacity. LayerNorm is PER TOKEN, so the pad floats
            # are included in the mean/std the real floats are normalized BY -- they actively distort the signal
            # rather than sitting inert. MEASURED (robocasa 128px + frozen TAESD, 2026-08-09): bag 32x128 (75%
            # pad) floors at 16.03 dB vs 20.41 dB for the EXACT 8x128 bag -- padding costs 4.4 dB, a bigger hit
            # than any num_tokens choice. Say that out loud, with the exact resize that makes it EXACT.
            waste = 100.0 * (1.0 - info["L"] / info["M"])
            fixes = []
            if info["L"] % T == 0:
                fixes.append(f"model.d={info['L'] // T}")
            if info["L"] % dd == 0:
                fixes.append(f"num_tokens={info['L'] // dd}")
            hint = " or ".join(fixes) if fixes else f"any num_tokens*d == {info['L']}"
            damage = (" and it is NOT just idle width: LayerNorm is per-token, so those floats enter the "
                      "mean/std the real ones are divided by and ACTIVELY DEGRADE reconstruction "
                      "(measured -4.4 dB on robocasa/TAESD: 16.03 dB padded vs 20.41 dB EXACT)") if _ln_on else ""
            lines.append(f"[adapter] {_nm}: WASTING {waste:.0f}% of the bag ({info['M'] - info['L']} of "
                         f"{info['M']} floats carry no latent) — {T} tokens are attended every step but only "
                         f"{info['per']}/{dd} dims per token feed the decoder at init{damage}. "
                         f"For EXACT set {hint}.")
    ae = getattr(m, "act_enc", None)                       # ACTION CONDITIONING, stated explicitly
    if ae is not None:
        nf = int(getattr(ae, "n_freq", 0) or 0)
        sq = float(getattr(ae, "squash", 4.0))
        fo = "OFF" if nf == 0 else f"{nf} bands (raw {ae.in_raw} + {2 * ae.in_raw * nf} sin/cos, |x|<={sq:g})"
        cat_on = bool(getattr(m, "concat_action_embedding", False))
        why = ("the action token's backbone output is concatenated onto every state token, so the denoiser has "
               "a dedicated action channel it cannot route around" if cat_on else
               "WARNING: readout() DISCARDS the action slot, so actions reach the prediction ONLY via attention "
               "onto 1 of the bag's slots — measured grad/norm/act_enc was 0.17% of the total gradient")
        lines.append(f"[action] fourier={fo} | concat_to_denoiser={'ON' if cat_on else 'OFF'}  <- {why}")
    for _n, _md in getattr(m, "modalities", {}).items():     # per-modality fourier (vector heads)
        _e = getattr(_md, "enc", None)
        _nf = int(getattr(_e, "n_freq", 0) or 0) if _e is not None else 0
        if _nf > 0:
            lines.append(f"[fourier] {_n}: {_nf} bands (raw {_e.in_raw} + {2 * _e.in_raw * _nf} sin/cos)")
    _nt = getattr(m, "latent_norm_type", "layernorm")
    _why = {"layernorm": "per-token non-affine LN on the bag at encode + after every dynamics step — scale-free "
                         "but NOT invertible (drops 2 scalars/token; -3.51 dB measured on robocasa/TAESD)",
            "affine": "fixed per-channel scale+shift on the AE latent, exactly inverted before decode — "
                      "invertible, 0 dB cost; the bag itself is left alone",
            "none": "no latent normalization anywhere"}.get(_nt, "")
    lines.append(f"[latent_norm] {_nt}: {_why}")
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
    EVAL_FAIL_LIMIT = 2   # consecutive failures of ONE routine before it is treated as deterministic -> fatal

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
        self._fail_streak: dict = {}   # routine -> CONSECUTIVE failures; EVAL_FAIL_LIMIT in a row is fatal

    def _eval_due(self, epoch: int) -> bool:
        # Cadence uses Lightning's (epoch+1)%N phase — SAME phase as validation (check_val_every_n_epoch) and
        # the checkpoint — so the eval cadence COINCIDES with a val + fresh checkpoint (e.g. every=2 -> {1,3,5}
        # 0-indexed, the epochs val runs). Eval still fires from on_train_epoch_end (every epoch, self-gated),
        # NOT the val hook, so it never gets silently dropped when the eval cadence != val cadence.
        # at_epochs is a UNION of explicit one-off epoch INDICES (literal, not phase-shifted).
        #
        # EPOCH 0 IS EVALUATED (user, 2026-08-10). It used to be skipped as an "untrained baseline", which was
        # simply wrong: on_train_epoch_end fires AFTER a FULL epoch of training (thousands of steps), so the
        # model is not untrained -- and epoch 0 is the p_tf=1.0 teacher-forced epoch, which makes it the single
        # most useful baseline point on the curve. It is also the cheapest epoch to evaluate. Losing it meant
        # every long-horizon series started at epoch 1 with nothing to compare against.
        cadence = self.every > 0 and (epoch + 1) % self.every == 0
        extra = self.at_epochs is not None and epoch in self.at_epochs
        return cadence or extra

    def on_fit_start(self, trainer, pl_module):
        self.writer.config(OmegaConf.to_container(self.cfg, resolve=True))
        self._t_fit = time.perf_counter()
        n_params = sum(p.numel() for p in pl_module.model.parameters())   # constant -> log ONCE, not per epoch
        self.writer.scalars({"model/params": float(n_params), "model/params_millions": n_params / 1e6}, step=0)
        # Calibration is FUNCTIONAL, not diagnostic: it must run BEFORE the gate (it changes what the floor
        # is) and it must NOT share the gate's except, which would let a latent_norm=affine run train with
        # identity stats (silently == latent_norm:none) reported only as "[ae_floor @ep0] skipped".
        self._calibrate_latent_affine(pl_module)
        # EPOCH-0 pretrained-AE gate (#12 §4), BEFORE any training. In an identity-guaranteed adapter mode the
        # round-trip must already equal the raw AE; a mismatch is an indexing/zero-init bug that would otherwise
        # masquerade as a bad model for the whole run (which is exactly what the ~10.5 dB Perceiver collapse did).
        # No-op unless a pretrained-AE trunk is present.
        try:
            from ..controller.run import _plog
            from ..evaluation.ae_floor import assert_identity_floor
            pass                                       # (calibration moved OUT of this try — see above)
            res = assert_identity_floor(self.cfg, pl_module.model, log=lambda s: _plog(self.writer, s))
            for nm, r in (res or {}).items():
                self.writer.scalars({f"eval_ae_floor/{nm}/init_adapter_psnr": r["adapter_db"],
                                     f"eval_ae_floor/{nm}/init_raw_ae_psnr": r["raw_db"],
                                     f"eval_ae_floor/{nm}/init_delta_db": r["delta_db"]}, step=0)
        except AssertionError:
            if self.cfg.get("resume"):             # on RESUME the adapter has TRAINED — its residual is no longer
                from ..controller.run import _plog  # bit-exact identity (that's expected/fine), so the init-only
                _plog(self.writer, "[ae_floor @ep0] identity-floor gate SKIPPED on resume "  # gate must not fire.
                      "(a resumed adapter's trained residual legitimately deviates from init identity).")
            else:
                raise                              # a broken identity is fatal at FRESH init — do NOT train through it
        except Exception as e:                     # a missing AE/dataset must not kill a run over a diagnostic
            from ..controller.run import _plog
            _plog(self.writer, f"[ae_floor @ep0] skipped ({type(e).__name__}: {e})")

    def _calibrate_latent_affine(self, pl_module):
        """latent_norm=affine: fit the per-channel latent stats from TRAIN frames, once, before anything runs.
        Also publishes the active normalization + its parameters under wandb `normalization/`."""
        from ..controller.run import _plog
        from ..models.modalities import calibrate_latent_affine
        m = getattr(pl_module.model, "_orig_mod", pl_module.model)
        nt = getattr(m, "latent_norm_type", "layernorm")
        self.writer.scalars({"normalization/is_layernorm": float(nt == "layernorm"),
                             "normalization/is_affine": float(nt == "affine"),
                             "normalization/is_invertible": float(nt in ("affine", "none"))}, step=0)
        if nt != "affine":
            return
        if not any(hasattr(md, "taesd") for md in m.modalities.values()):
            return                                 # no pretrained trunk -> nothing to calibrate. NOT an error:
            #                                        this used to raise StopIteration and log a scary empty
            #                                        "probe disabled ()" line on every proprio-only run.
        from ..data.dataset import load_fpv_frames
        from ..training.setup import resolve_data_root
        import numpy as _np, torch as _t
        n = int(self.cfg.get("eval", {}).get("latent_affine_frames", 256))
        sz = next(md.ae.cfg.img_size for md in m.modalities.values() if hasattr(md, "latent_affine"))
        fr = load_fpv_frames(resolve_data_root(self.cfg), "train", size=sz,
                             cam=self.cfg.data.get("cam", "fpv"), max_frames=n, cache=False)
        stats = calibrate_latent_affine(m, _t.from_numpy(_np.asarray(fr)).float().div(255.0))
        for nm, st in stats.items():
            self.writer.scalars({f"normalization/{nm}/mean_c{i}": v for i, v in enumerate(st["mean"])}, step=0)
            self.writer.scalars({f"normalization/{nm}/std_c{i}": v for i, v in enumerate(st["std"])}, step=0)
            self.writer.scalars({f"normalization/{nm}/n_frames": float(st["n_frames"])}, step=0)
            _plog(self.writer, f"[latent_norm] {nm}: affine calibrated on {st['n_frames']} train frames | "
                               f"mean={[round(v, 4) for v in st['mean']]} std={[round(v, 4) for v in st['std']]}")

    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx == 0 and self._t_b0 is None:
            self._t_b0 = time.perf_counter()  # the very first batch pays the torch.compile cost

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._compile_s is None and self._t_b0 is not None:
            self._compile_s = time.perf_counter() - self._t_b0  # ~ one-time compile

    @torch.no_grad()
    def _log_normalization(self, pl_module, step):
        """EVERY EPOCH under normalization/. The affine PARAMETERS are frozen after calibration, so what is
        worth tracking is the STATISTIC the normalizer acts on: the per-token mean/std of the encoded bag
        BEFORE normalization, measured on a FIXED set of val frames so epochs are comparable.

        Why it matters under layernorm: _ln divides by each token's own std, so a shrinking std means LN
        amplifies whatever is left, by up to 1/sqrt(eps) ~ 316x. ln_gain_{mean,max} report that gain directly.

        SCOPE -- this probes the ENCODER path only (mod.encode on clean frames). The feedback loop suspected
        behind the epoch-5 collapse lives in the ROLLOUT: _ln runs again after every predict_next, on the
        model's own drifting predictions. With a frozen TAESD and the parameter-free EXACT adapter, the
        encode-side statistic is nearly inert by construction (only the zero-init refine MLP can move it), so
        a flat series here is NOT evidence the rollout is healthy. Probing the rollout side is a TODO."""
        m = getattr(pl_module.model, "_orig_mod", pl_module.model)
        nt = getattr(m, "latent_norm_type", "layernorm")
        out = {"normalization/is_layernorm": float(nt == "layernorm"),
               "normalization/is_affine": float(nt == "affine"),
               "normalization/is_invertible": float(nt in ("affine", "none"))}
        probe = [(n, md) for n, md in m.modalities.items() if hasattr(md, "taesd")]
        if not probe:                              # no pretrained trunk -> nothing to probe. NOT an error:
            self.writer.scalars(out, step=step)    # this used to raise StopIteration and log a scary empty
            return                                 # "probe disabled ()" on every proprio-only run.
        was = m.training
        try:
            frames = self._norm_probe_frames(m)
            m.eval()
            for name, mod in probe:
                tok = mod.encode(frames.to(next(mod.parameters()).device))    # pre-bag, pre-_ln
                sd, mu = tok.std(dim=-1), tok.mean(dim=-1)                    # per token
                out[f"normalization/{name}/pre_norm_std_mean"] = float(sd.mean())
                out[f"normalization/{name}/pre_norm_std_min"] = float(sd.min())
                # The GAIN _ln actually applies, 1/sqrt(var+eps) -- directly readable against the 1/sqrt(eps)
                # ~= 316x ceiling, unlike the raw std. This is the number to alarm on.
                out[f"normalization/{name}/ln_gain_mean"] = float((1.0 / (sd ** 2 + 1e-5).sqrt()).mean())
                out[f"normalization/{name}/ln_gain_max"] = float((1.0 / (sd ** 2 + 1e-5).sqrt()).max())
                out[f"normalization/{name}/pre_norm_absmean_mean"] = float(mu.abs().mean())
                if getattr(mod, "latent_affine", False):                      # frozen, but re-logged so the
                    for i, v in enumerate(mod.lat_mean.flatten().tolist()):   # folder is self-contained per epoch
                        out[f"normalization/{name}/mean_c{i}"] = v
                    for i, v in enumerate(mod.lat_std.flatten().tolist()):
                        out[f"normalization/{name}/std_c{i}"] = v
        except Exception as e:                     # telemetry must never take a run down -- but it must not
            if not getattr(self, "_norm_warned", False):   # fail SILENTLY either (a swallowed AttributeError
                self._norm_warned = True                   # hid this diagnostic for a full cycle once)
                from ..controller.run import _plog
                _plog(self.writer, f"[latent_norm] per-epoch probe disabled ({type(e).__name__}: {e})")
        finally:
            m.train(was)                           # restore even if the probe threw (the eval-routine runner
        self.writer.scalars(out, step=step)        # already does this; this brings the probe up to it)

    def _norm_probe_frames(self, m):
        """8 FIXED val frames, loaded once and cached — the point is comparability across epochs."""
        if getattr(self, "_norm_frames", None) is None:
            import numpy as _np, torch as _t
            from ..data.dataset import load_fpv_frames
            from ..training.setup import resolve_data_root
            sz = next(md.ae.cfg.img_size for md in m.modalities.values() if hasattr(md, "taesd"))
            fr = load_fpv_frames(resolve_data_root(self.cfg), "val", size=sz,
                                 cam=self.cfg.data.get("cam", "fpv"), max_frames=8, cache=False)
            self._norm_frames = _t.from_numpy(_np.asarray(fr)).float().div(255.0)
        return self._norm_frames

    def on_train_epoch_end(self, trainer, pl_module):
        # Eval routines fire on the eval CADENCE, decoupled from the validation cadence. They used to live in
        # on_validation_epoch_end, which only fires on val epochs — so with check_val_every_n_epoch=2 (odd val
        # epochs) and an even eval cadence (10, 20, 40, ...) they NEVER coincided and no eval ran. Running them
        # here (every train-epoch end) means the eval cadence is honored regardless of the val cadence. The
        # model is put in eval mode for the routines, then restored (validation, which follows, sets its own).
        if trainer.current_epoch == 0 and self.routines and not trainer.sanity_checking:
            from ..controller.run import _plog
            cv = int(getattr(trainer, "check_val_every_n_epoch", 1) or 1)
            if cv > 1:      # VAL is on a slower cadence and will not run at ep0. Eval does (see _eval_due).
                _plog(self.writer, f"[val @ep0] not run — check_val_every_n_epoch={cv}. Eval DOES run at ep0.")
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
                except NotImplementedError as e:   # the routine genuinely CANNOT run in THIS environment (e.g.
                    #   control/interp need a live simulator; a recorded dataset has none). This is by design, not
                    #   a bug and not a regression — it "fails" every eval forever — so it must NEVER count toward
                    #   the fatal streak below (that streak is for DETERMINISTIC BUGS like a bad kwarg). Clean-skip.
                    from ..controller.run import _plog
                    _plog(self.writer, f"[eval:{name} @ep{epoch}] not supported in this env "
                          f"({type(e).__name__}: {e}); skipping. Not counted as a failure.")
                    self._skipped.append((epoch, name))
                    self.writer.scalar(f"eval/skipped/{name}", 1.0, step=epoch)
                    continue   # leave _fail_streak[name] untouched
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
                    # ESCALATE A PERSISTENT failure (user, 2026-08-10). "Never kill training" is right for a
                    # TRANSIENT failure (one bad render, one OOM viz) but catastrophic for a DETERMINISTIC one:
                    # a 30-epoch run whose whole purpose is the eval metrics lost every single eval to the same
                    # TypeError and would have burned 46 GPU-hours producing nothing, reporting it only as a
                    # non-fatal line nobody was watching. N consecutive failures of the SAME routine is not a
                    # blip -- it will not fix itself, so fail loudly NOW instead of at the end.
                    self._fail_streak[name] = self._fail_streak.get(name, 0) + 1
                    if self._fail_streak[name] >= self.EVAL_FAIL_LIMIT:
                        raise RuntimeError(
                            f"eval routine {name!r} failed {self._fail_streak[name]} times IN A ROW "
                            f"(last: {type(e).__name__}: {e}). This is deterministic, not transient -- the rest "
                            f"of this run would produce no {name} metrics at all. Fix the routine and restart "
                            f"(or disable eval.during_train.evals.{name}) rather than training on blind."
                        ) from e
                else:
                    self._fail_streak[name] = 0                     # a success clears the streak
        finally:
            m.train(was_training)
        self._eval_cum += time.perf_counter() - t_eval

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        # forward every aggregated scalar metric (train/* and val/*) through the one writer
        self.writer.scalars({k: v.item() for k, v in trainer.callback_metrics.items()}, step=epoch)
        self._log_normalization(pl_module, epoch)

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
