"""LightningModule: p_tf rollout training, MSE on delta, env-space metrics (design/training.md)."""

from __future__ import annotations

import lightning as L
import torch

from ..environments.base import default_rollout_metrics
from .schedules import linear_schedule
from .variations import VarContext, PhysicalLoss, make_variation_suite


def adamw_with_warmup(params, lr: float, weight_decay: float, warmup_steps: int = 0):
    """AdamW (+ optional linear LR warmup 0->1 over `warmup_steps` OPTIMIZER STEPS). Every flow head here
    regresses a clean target from a near-pure-noise input -> high-variance early gradients; full LR from
    step 0 lets one oversized early update blow up (bf16 overflow, or the shortcut self-consistency loss
    running away). Warmup lets them settle. `interval="step"` counts batches so it finishes early in epoch 0.
    NOT fused: Lightning's gradient_clip_val is incompatible with a fused optimizer (negligible speedup at
    this size anyway). Shared by LitFlow (WM) and LitActionModel (action head) — one warmup, no duplication."""
    params = [p for p in params if p.requires_grad]   # exclude FROZEN params (e.g. a frozen pretrained AE, #12)
    #                                                   so AdamW allocates NO moments for them. No-op when nothing
    #                                                   is frozen (all requires_grad=True) -> bit-identical.
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if warmup_steps and warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_steps))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
    return opt


class LitWorldModel(L.LightningModule):
    def __init__(self, model, normalizer, R: float, r: float, v_scale: float, P: int, F: int,
                 p_tf_start: float, p_tf_end: float, p_tf_warmup: int,
                 lr: float, weight_decay: float, detach_every: int = 8, variations=None, dt: float = 1.0 / 60.0,
                 recon_frac: float = 1.0, lr_warmup_steps: int = 0, env=None, p_tf_batch_granular: bool = True):
        super().__init__()
        self.model = model
        self.norm = normalizer
        self.env = env   # WorldEnv: val rollout_metrics + the optional physical_loss hook (None -> generic defaults)
        self.R, self.r, self.v_scale, self.P, self.F = R, r, v_scale, P, F
        self.dt = dt
        self.recon_frac = float(recon_frac)   # <1 -> supervise the decode recon on a random subset of F frames (ALL heads)
        self.p_tf_start, self.p_tf_end, self.p_tf_warmup = p_tf_start, p_tf_end, p_tf_warmup
        self.p_tf_batch_granular = bool(p_tf_batch_granular)   # True -> ramp p_tf across batches (fractional epoch)
        self.lr, self.weight_decay, self.detach_every = lr, weight_decay, detach_every
        self.lr_warmup_steps = int(lr_warmup_steps)
        # train-time shaping variations (off by default -> empty suite, zero overhead). See variations.py.
        self.variations = make_variation_suite(variations)
        # physical-loss warmup: ramp its weight 0 -> 1 over warmup_epochs (same linear schedule as p_tf;
        # logged under schedules/). Only when the physical variation is actually active. The early decode
        # is jittery, so hitting it with full physical weight at epoch 0 destabilizes -> NaN; ramp avoids it.
        self.has_physical = any(isinstance(v, PhysicalLoss) for v in self.variations.variations)
        pl = (variations or {})
        pl = (pl.get("physical_loss", {}) if hasattr(pl, "get") else getattr(pl, "physical_loss", {})) or {}
        self.physical_warmup = float((pl.get("warmup_epochs", 0) if hasattr(pl, "get")
                                      else getattr(pl, "warmup_epochs", 0)) or 0) if self.has_physical else 0.0

    def on_train_start(self):
        # TRIP-WIRE: full teacher-forcing (p_tf never drops) means NO in-rollout drift training. This produced
        # collapsed/drifting models (val rises, rollout predictions -> origin) on the flow model. Warn LOUDLY to
        # stdout->progress.log so it's caught while monitoring. If intentional (e.g. isolating a decode test),
        # ignore; otherwise set p_tf_end<1 for in-rollout (use dynamics shortcut=true to keep it affordable).
        if float(self.p_tf_end) >= 1.0 and float(self.p_tf_start) >= 1.0:
            msg = ("[WARNING] p_tf is CONSTANT 1.0 (FULL teacher-forcing, NO in-rollout drift training). "
                   "This previously produced drifting/collapsed rollouts (val rises, preds->origin). "
                   "Set p_tf_end<1 for in-rollout training (dynamics shortcut=true keeps the cost sane) "
                   "unless this is deliberate.")
            print("\n" + "!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
            import warnings
            warnings.warn(msg)

    def _cur_p_tf(self) -> float:
        # curriculum: ramp p_tf_start (e.g. 1.0, full teacher forcing) -> p_tf_end over p_tf_warmup epochs.
        # p_tf_batch_granular=True (default) uses a FRACTIONAL epoch (current_epoch + how far through this
        # epoch's batches we are) so the ramp is smooth ACROSS batches, not a per-epoch step — critical for
        # short warmups on large datasets (warmup=1 per-epoch would sit at 1.0 all epoch 0 then hard-jump to 0).
        # False -> the legacy per-epoch step (only changes at epoch boundaries).
        epoch = float(self.current_epoch)
        if self.p_tf_batch_granular:
            nb = getattr(self.trainer, "num_training_batches", 0) or 0
            if nb and nb != float("inf"):
                epoch += getattr(self, "_batch_idx", 0) / nb
        return linear_schedule(self.p_tf_start, self.p_tf_end, self.p_tf_warmup, epoch)

    def _physical_ramp(self) -> float:
        # physical-loss weight multiplier: 0 -> 1 over physical_warmup epochs (no ramp if warmup<=0)
        return linear_schedule(0.0, 1.0, self.physical_warmup, self.current_epoch)

    def _core(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _step(self, batch, tag):
        """Multimodal (token-bag) step: per-head recon losses `loss/<head>` (config-weighted) + the model
        term (pred_latent / flow) + the shaping variations (physical_loss on the proprio decode, contraction on
        the one-step token-bag map) via the unified VariationSuite. Per-stream input noise applied inline here;
        metrics grouped as `metric/<head>/*` (val-only). For EMA/JEPA heads the obs recon is a detached probe."""
        import torch.nn.functional as F
        m = self._core()
        P, L = self.P, self.P + self.F
        p_tf = self._cur_p_tf() if tag == "train" else 0.0   # val = pure autoregressive + deterministic (no teacher forcing)
        obs = {"proprio": batch["obs_seq"]}
        for name, _ in m.layout:
            if name != "proprio":
                obs[name] = batch[name]
        act = batch["act_seq"]
        # relative-position encoding: the per-window anchor = the CLEAN position at the window's first step.
        # Threaded (as an ARG, never stored) into encode/rollout/recon/decode so the codec works in the small
        # within-window frame while this step stays in ABSOLUTE obs. None (default) when the feature is off.
        anchor = m.rel_anchor(obs) if m._rel_on() else None
        # per-stream input noise (training only): perturb the model INPUTS; targets/metrics use clean obs.
        obs_in = obs
        if tag == "train":
            obs_in = {k: (v + torch.randn_like(v) * m.modalities[k].noise_std) if m.modalities[k].noise_std > 0 else v
                      for k, v in obs.items()}
        # SHARED-ENCODE fast path (design/accelerations P4): when inputs == targets (all noise_std==0) in the
        # p_tf==0 AR regime, encode the frames ONCE and slice for the rollout ctx + loss_terms + roundtrip
        # instead of re-encoding them 2-3x. Bit-identical (smoke-verified). Flow model only; encode is per-frame,
        # so z_full[:, sl] == encode(obs[:, sl]) exactly. noise_std>0 -> inputs differ from targets -> OFF.
        share = (p_tf == 0.0) and hasattr(m, "flow") and \
            (tag != "train" or all(m.modalities[k].noise_std == 0 for k in obs))
        z_full = m.encode_state(obs_in, anchor) if share else None
        if tag == "train" and not getattr(self, "_noise_share_noted", False):
            self._noise_share_noted = True
            noisy = [k for k in obs if m.modalities[k].noise_std > 0]
            if noisy:
                print(f"[encode-share] input noise on {noisy} -> shared-encode fast path OFF; frames are "
                      f"re-encoded per loss site (slower). Set modalities.<i>.noise_std=0 to enable it.", flush=True)
        if p_tf >= 1.0:                                        # parallel teacher forcing
            preds = m({k: v[:, :-1] for k, v in obs_in.items()}, act[:, :-1], anchor)[:, P - 1:]
        else:                                                 # autoregressive rollout (TF source = noised input)
            ctx = {k: v[:, :P] for k, v in obs_in.items()}
            preds = m.rollout_train(ctx, act[:, : L - 1], {k: v[:, P:] for k, v in obs_in.items()}, p_tf,
                                    self.detach_every, precomputed_ctx=(z_full[:, :P] if share else None),
                                    anchor=anchor)
        future = {k: v[:, P:] for k, v in obs.items()}         # CLEAN targets
        # EMA/JEPA heads: obs recon is a decoder-only probe (detach preds so it doesn't shape the encoder).
        recon_src = preds if getattr(m, "pred_obs_in_loss", True) else preds.detach()
        # recon on a RANDOM subset of the F rollout frames when recon_frac<1 (train only), the SAME subset across
        # ALL output modalities. The ViT-AE decode is F x per-step, so fewer frames = less compute; random (not a
        # fixed stride) -> every frame gets recon gradient over an epoch (unbiased). The DYNAMICS loss (flow /
        # pred_latent, below) stays on all F frames regardless. frac=1.0 (default) = decode all frames.
        frac = self.recon_frac if tag == "train" else 1.0
        if frac < 1.0:
            Tf = recon_src.shape[1]; k = max(1, int(round(frac * Tf)))
            idx = torch.randperm(Tf, device=recon_src.device)[:k]
            src, fut = recon_src[:, idx], {kk: v[:, idx] for kk, v in future.items()}
            z_tgt = z_full[:, P:][:, idx] if share else None   # shared-encode: the SAME encode sliced to `fut`
        else:
            src, fut = recon_src, future
            z_tgt = z_full[:, P:] if share else None
        # UNIFIED decode/physics: when proprio declares a prior, decode/proprio IS the chained prior-anchored
        # prediction (the old `physics/proprio` term). It runs on the FULL rollout (chain -> not frac-subsampled),
        # so pass it via the `prior` bundle rather than the recon_frac-subset `src`. prior=none -> legacy path below.
        _uni = getattr(m, "_proprio_prior", "none") != "none" and getattr(m, "dynamics_prior", None) is not None
        _prior = None
        if _uni:
            Pn, Fn = self.P, self.F
            _prior = dict(preds=preds, p_tf=p_tf, norm=self.norm, target=future["proprio"],
                          prev0_abs=self.norm.denorm_obs(obs["proprio"][:, Pn - 1]),         # (B,obs_dim) true ctx-last
                          true_future_abs=self.norm.denorm_obs(future["proprio"]),           # (B,F,obs_dim) true future
                          act_raw=self.norm.denorm_act(act[:, Pn - 1:Pn - 1 + Fn]))          # (B,F,act_dim) raw force (N)
        recon, rw = m.recon_losses(src, fut, pre_z_targets=z_tgt, anchor=anchor, prior=_prior)   # decode/<name>[_shortcut] + codec/roundtrip_<name>
        # NOTE: do NOT decode here (to_obs) in train — recon_losses is the decode loss, and for flow decoders
        # to_obs would SAMPLE the ViT decoder every step (with grad) for nothing -> huge wasted memory (OOM). The
        # decoded sample is only needed for val metrics; computed there under no_grad.
        lt_kw = {"anchor": anchor} if anchor is not None else {}   # only reaches the flow loss_terms; non-flow untouched
        raw, w = (m.loss_terms(preds, future, obs, p_tf, act, pre_z=z_full, **lt_kw) if share
                  else m.loss_terms(preds, future, obs, p_tf, act, **lt_kw))   # pre_z: shared-encode fast path (flow only)
        if getattr(m, "dynamics_prior", None) is not None and not _uni:   # LEGACY physics/proprio (prior=none)
            Pn, Fn = self.P, self.F
            # p_tf-respecting CHAINED physics rollout: prev0 = true ctx-last obs, then feed each step's own
            # prediction forward with prob (1-p_tf) — so the residual trains on the SAME compounding regime the
            # eval rollout uses (closes the teacher-forced-train / AR-eval exposure gap). act aligned as before.
            prev0_abs = self.norm.denorm_obs(obs["proprio"][:, Pn - 1])               # (B,obs_dim) true ctx-last
            true_future_abs = self.norm.denorm_obs(future["proprio"])                 # (B,F,obs_dim) true future
            act_raw = self.norm.denorm_act(act[:, Pn - 1:Pn - 1 + Fn])               # (B,F,act_dim) raw force (N)
            phys_abs = m.physics_proprio_chained(preds, prev0_abs, true_future_abs, act_raw, p_tf)
            raw["physics/proprio"] = F.mse_loss(self.norm.norm_obs(phys_abs), future["proprio"])
            w["physics/proprio"] = 1.0
        # recon_losses returns its OWN weights (same contract as loss_terms). It used to be reconstructed here
        # by parsing the key -- wts[k.split("/")[-1]] -- which mapped "roundtrip/image" to the IMAGE DECODE
        # weight, so ablating a head's decode recon with modalities.<i>.weight=0 silently also deleted that
        # head's adapter supervision. Weights now travel WITH the losses; nothing infers them from a name.
        loss = sum(w[k] * raw[k] for k in raw) + sum(rw[k] * recon[k] for k in recon)

        # train-time shaping variations (per-stream input noise applied inline above; here the LOSS terms:
        # physical_loss on the proprio decode, contraction on the one-step token-bag map). Routed through the
        # ONE VariationSuite so any variation applies to every model. obs is the dict-of-streams bag; enable_grad
        # lets contraction build its Jacobian graph on val (Trainer runs with inference_mode=False).
        if self.variations:
            ctx = VarContext(m, preds, future["proprio"], obs, act, self.norm,
                             tag == "train", self._physical_ramp(), env=self.env)
            with torch.enable_grad():
                extra, comps, diags = self.variations.losses(ctx)
            if extra is not None:
                loss = loss + extra
            for k, val in comps.items():                              # {tag}/loss/{physical,contraction} (weighted)
                self.log(f"{tag}/loss/{k}", val)
            if tag == "train":
                for k, val in diags.items():                          # physical_loss/{d_off,v_off,continuity}, contraction/sigma_max
                    self.log(k, val)

        self.log(f"{tag}/loss/total", loss, prog_bar=(tag == "train"))
        for k, v in {**raw, **recon}.items():                 # loss/{flow|pred_latent}, loss/proprio, loss/image
            self.log(f"{tag}/loss/{k}", v)
        if tag == "train":
            self.log("schedules/p_tf", p_tf)
            if self.has_physical:
                self.log("schedules/physical_loss_ramp", self._physical_ramp())
        if tag == "val":
            with torch.no_grad():
                dec = m.to_obs(src, anchor=anchor)            # decode (mse) / 1-step sample (flow) — val metrics only; de-relativized -> ABSOLUTE
                # UNIFIED decode/physics: in prior mode the black-box proprio decoder is UNTRAINED (decode/proprio
                # is the physics chain, no round-trip), so score the DEPLOYED prediction — the physics chain at
                # p_tf=0 (fully AR) — not dec["proprio"] (audit-F4: the monitor must track what we deploy).
                if _uni and _prior is not None:
                    phys_abs = m.physics_proprio_chained(preds, _prior["prev0_abs"], _prior["true_future_abs"],
                                                         _prior["act_raw"], 0.0)
                    proprio_hat_norm = self.norm.norm_obs(phys_abs)
                    p_hat = torch.nan_to_num(phys_abs.float(), nan=10.0, posinf=10.0, neginf=-10.0)
                else:
                    proprio_hat_norm = dec["proprio"]
                    p_hat = torch.nan_to_num(self.norm.denorm_obs(dec["proprio"]), nan=10.0, posinf=10.0, neginf=-10.0)
                p_true = self.norm.denorm_obs(future["proprio"])
                # env-polymorphic rollout metrics (WorldEnv.rollout_metrics): torus returns its three errors
                # (byte-identical tags/values to the old hardcoded calls); other envs return their own set.
                metrics_fn = self.env.rollout_metrics if self.env is not None else default_rollout_metrics
                for mk, mv in metrics_fn(p_hat, p_true).items():
                    self.log(f"val/metric/proprio/{mk}", mv.mean())
                self.log("val/metric/proprio/obs_error", F.mse_loss(proprio_hat_norm, future["proprio"]))  # decoded-proprio MSE (normalized) — comparable across decoders
                for name, _ in m.layout:
                    if name == "proprio":
                        continue
                    dclamp = dec[name].clamp(0, 1)
                    mse = F.mse_loss(dclamp, future[name])
                    self.log(f"val/metric/{name}/mse", mse)
                    self.log(f"val/metric/{name}/l1", F.l1_loss(dclamp, future[name]))
                    self.log(f"val/metric/{name}/psnr", -10.0 * torch.log10(mse.clamp_min(1e-12)))
                if hasattr(m, "collapse_diagnostics"):        # latent-collapse (esp. for EMA); on the encoded bag
                    for k, val in m.collapse_diagnostics(obs).items():
                        self.log(f"collapse/{k}", val)
                if not getattr(self, "_kv_logged", False) and hasattr(m, "kvcache_report"):
                    # ONE-TIME temporal KV-cache A/B (kvcache/*): realized wall-clock speedup of the inference
                    # rollout + the (benign) latent divergence. proprio-only decode -> cheap; runs once per run.
                    self._kv_logged = True
                    ctxk = {k: v[:, :P] for k, v in obs.items()}
                    for k, val in m.kvcache_report(ctxk, act[:, : L - 1], L - P, heads=["proprio"]).items():
                        self.log(k, val)
        return loss

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # clip (Trainer sets val=1.0) AND log the total grad norm pre- and post-clip, generically for
        # every model, so BPTT blow-ups (DSAR diverged ~ep30 even with clipping) are diagnosable.
        grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
        def _total_norm():
            gs = [g.norm() for g in grads]
            return torch.norm(torch.stack(gs)) if gs else torch.zeros((), device=self.device)
        # pre-clip inf/nan COUNTS: localize a blow-up (kind + extent) BEFORE clipping mangles it — norm-clipping
        # turns a single inf into an all-NaN grad, so these must be read here. Counts, not fractions: one inf is
        # fatal but ~2e-7 as a fraction of ~5M elements, so it would round away; a count shows it as "1".
        n_nan = sum(torch.isnan(g).sum() for g in grads) if grads else 0
        n_inf = sum(torch.isinf(g).sum() for g in grads) if grads else 0
        # per-module grad norms (pre-clip): the "WHERE" axis — localize which subnetwork blows up first. Grouped
        # so the decode head (differs between mse/flow runs) is separable from the shared trunk: encode_<mod>,
        # decode_<mod>, backbone (dynamics context), flow (latent-dynamics head).
        module_sq = {}
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            parts = name.split(".")
            if "decode_head" in parts and "modalities" in parts:
                key = "decode_" + parts[parts.index("modalities") + 1]
            elif "modalities" in parts:
                key = "encode_" + parts[parts.index("modalities") + 1]
            elif len(parts) > 1 and parts[0] == "model":
                key = parts[1]
            else:
                key = parts[0]
            module_sq[key] = module_sq.get(key, 0.0) + p.grad.detach().float().pow(2).sum()
        pre = _total_norm()
        # non-finite guard: a single inf/nan grad makes norm-clipping compute a NaN total-norm and scale EVERY
        # grad to NaN, which then poisons AdamW's state permanently (unet_flow died this way ~ep2). Skip the
        # step instead — zero the grads so optimizer.step() is a harmless no-op — and count skips so a
        # SYSTEMATIC problem (vs a rare transient batch) is visible in grad/nonfinite_skipped.
        skipped = not bool(torch.isfinite(pre))
        if skipped:
            for g in grads:
                g.zero_()
        else:
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
                                gradient_clip_algorithm=gradient_clip_algorithm)
        self.log("grad/norm_preclip", pre)
        self.log("grad/norm_postclip", _total_norm())
        # grad/clip_ratio = preclip / clip_val. ONE number: 1 means clipping never engaged, >>1 means the
        # gradient is being LAUNDERED -- clipping throws away the magnitude but keeps the DIRECTION, so the
        # optimizer takes a full-size confident step along whatever exploded. That degrades smoothly instead of
        # NaN-ing, which is exactly why a 767x blowup in the transformer denoiser (2026-08-11) read as a
        # modelling failure for two days: preclip and postclip were both logged the whole time, but the signal
        # only screams once you divide them. reduce_fx=max -> the epoch value is the WORST step.
        if gradient_clip_val:      # None/0 = clipping disabled -> the ratio is undefined, not infinite
            self.log("grad/clip_ratio", pre / max(float(gradient_clip_val), 1e-12), reduce_fx="max")
        # reduce_fx=max -> the epoch value is the WORST step (mean would dilute one spike across 1000s of clean
        # steps into ~0); nonfinite_skipped uses sum -> total # of skipped steps this epoch.
        self.log("grad/num_nans", float(n_nan), reduce_fx="max")
        self.log("grad/num_infs", float(n_inf), reduce_fx="max")
        self.log("grad/nonfinite_skipped", float(skipped), reduce_fx="sum")
        for key, sq in module_sq.items():
            self.log(f"grad/norm/{key}", sq.sqrt(), reduce_fx="max")   # worst-step per-module norm

    def training_step(self, batch, batch_idx):
        self._batch_idx = batch_idx   # for _cur_p_tf's fractional-epoch (batch-granular) teacher-forcing ramp
        return self._step(batch, "train")

    def on_train_batch_end(self, *_):
        self.model.on_optimizer_step()  # EMA target update for LSAR-EMA; no-op otherwise

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return adamw_with_warmup(self.parameters(), self.lr, self.weight_decay, self.lr_warmup_steps)


class LitActionModel(L.LightningModule):
    """POST-HOC action-prior training (train_action_model): the world model is FROZEN (eval mode,
    requires_grad False — the caller freezes it); ONLY `model.action_flow` trains. Contexts come from the
    frozen WM under no_grad; the action-flow loss uses the SAME leak-free alignment as
    MultiModalFlow.loss_terms: cond = pooled h[t-1] (never saw a[t]) -> target a[t], for t=1..L-2."""

    def __init__(self, model, lr: float, weight_decay: float, lr_warmup_steps: int = 0):
        super().__init__()
        self.model = model
        self.lr, self.weight_decay = lr, weight_decay
        self.lr_warmup_steps = int(lr_warmup_steps)

    def on_train_epoch_start(self):
        # Lightning flips the whole module to train mode each epoch; re-pin the frozen WM to eval (the
        # trainable head has no mode-dependent layers, but keep it in train mode for correctness).
        self.model.eval()
        self.model.action_flow.train()

    def _step(self, batch, tag):
        m = self.model
        obs = {"proprio": batch["obs_seq"]}
        for name, _ in m.layout:
            if name != "proprio":
                obs[name] = batch[name]
        act = batch["act_seq"]                                # (B, L, action_dim), normalized
        L_ = act.shape[1]
        # exclude the FLASH SDPA backend for the frozen-WM context pass: Lightning's val context selects it for
        # the image ViT's attention, where it aborts with CUDA 'invalid configuration argument' on this head
        # config (the WM's own training/val avoids it). mem-efficient/math handle the same shapes fine.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):  # frozen WM: contexts only
            h_ctx = m.action_context(obs, act)                # (B, L-1, d): h[k] predicts a[k+1] (leak-free)
        cond = h_ctx[:, :-1]                                  # h[t-1], aligned to predict a[t] for t=1..L-2
        a_target = act[:, 1:L_ - 1].detach()                  # a[1..L-2] — same alignment as loss_terms
        l_flow, l_cons = m.action_flow.loss(cond, a_target, time_sampling=m.time_sampling)
        loss = l_flow if l_cons is None else l_flow + l_cons
        self.log(f"{tag}/loss/action/flow", l_flow)
        if l_cons is not None:
            self.log(f"{tag}/loss/action/shortcut", l_cons)
        self.log(f"{tag}/loss/total", loss, prog_bar=(tag == "train"))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # ONLY the action-flow head trains; the WM is frozen (requires_grad=False, not passed here). Same
        # warmup as the WM flow (adamw_with_warmup) — the action head is a flow too, so it needs it just as much.
        return adamw_with_warmup(self.model.action_flow.parameters(), self.lr, self.weight_decay,
                                 self.lr_warmup_steps)
