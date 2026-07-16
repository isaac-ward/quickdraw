"""LightningModule: p_tf rollout training, MSE on delta, env-space metrics (design/training.md)."""

from __future__ import annotations

import lightning as L
import torch

from ..environments import torus as T
from .schedules import linear_schedule
from .variations import VarContext, PhysicalLoss, make_variation_suite


class LitWorldModel(L.LightningModule):
    def __init__(self, model, normalizer, R: float, r: float, v_scale: float, P: int, F: int,
                 p_tf_start: float, p_tf_end: float, p_tf_warmup: int,
                 lr: float, weight_decay: float, detach_every: int = 8, variations=None, dt: float = 1.0 / 60.0,
                 recon_frac: float = 1.0, lr_warmup_steps: int = 0):
        super().__init__()
        self.model = model
        self.norm = normalizer
        self.R, self.r, self.v_scale, self.P, self.F = R, r, v_scale, P, F
        self.dt = dt
        self.recon_frac = float(recon_frac)   # <1 -> supervise the decode recon on a random subset of F frames (ALL heads)
        self.p_tf_start, self.p_tf_end, self.p_tf_warmup = p_tf_start, p_tf_end, p_tf_warmup
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
        # curriculum: ramp from p_tf_start (e.g. 1.0, full teacher forcing) down to p_tf_end over warmup
        return linear_schedule(self.p_tf_start, self.p_tf_end, self.p_tf_warmup, self.current_epoch)

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
        # per-stream input noise (training only): perturb the model INPUTS; targets/metrics use clean obs.
        obs_in = obs
        if tag == "train":
            obs_in = {k: (v + torch.randn_like(v) * m.modalities[k].noise_std) if m.modalities[k].noise_std > 0 else v
                      for k, v in obs.items()}
        if p_tf >= 1.0:                                        # parallel teacher forcing
            preds = m({k: v[:, :-1] for k, v in obs_in.items()}, act[:, :-1])[:, P - 1:]
        else:                                                 # autoregressive rollout (TF source = noised input)
            ctx = {k: v[:, :P] for k, v in obs_in.items()}
            preds = m.rollout_train(ctx, act[:, : L - 1], {k: v[:, P:] for k, v in obs_in.items()}, p_tf, self.detach_every)
        future = {k: v[:, P:] for k, v in obs.items()}         # CLEAN targets
        # EMA/JEPA heads: obs recon is a decoder-only probe (detach preds so it doesn't shape the encoder).
        recon_src = preds if getattr(m, "pred_obs_in_loss", True) else preds.detach()
        wts = {mod.name: float(mod.weight) for mod in m.modalities.values()}
        # recon on a RANDOM subset of the F rollout frames when recon_frac<1 (train only), the SAME subset across
        # ALL output modalities. The ViT-AE decode is F x per-step, so fewer frames = less compute; random (not a
        # fixed stride) -> every frame gets recon gradient over an epoch (unbiased). The DYNAMICS loss (flow /
        # pred_latent, below) stays on all F frames regardless. frac=1.0 (default) = decode all frames.
        frac = self.recon_frac if tag == "train" else 1.0
        if frac < 1.0:
            Tf = recon_src.shape[1]; k = max(1, int(round(frac * Tf)))
            idx = torch.randperm(Tf, device=recon_src.device)[:k]
            src, fut = recon_src[:, idx], {kk: v[:, idx] for kk, v in future.items()}
        else:
            src, fut = recon_src, future
        recon = m.recon_losses(src, fut)                      # per-head decode LOSS: {name} (mse) or {flow/name,shortcut/name}
        # NOTE: do NOT decode here (to_obs) in train — recon_losses is the decode loss, and for flow decoders
        # to_obs would SAMPLE the ViT decoder every step (with grad) for nothing -> huge wasted memory (OOM). The
        # decoded sample is only needed for val metrics; computed there under no_grad.
        raw, w = m.loss_terms(preds, future, obs, p_tf, act)
        loss = sum(w[k] * raw[k] for k in raw) + sum(wts[k.split("/")[-1]] * recon[k] for k in recon)

        # train-time shaping variations (per-stream input noise applied inline above; here the LOSS terms:
        # physical_loss on the proprio decode, contraction on the one-step token-bag map). Routed through the
        # ONE VariationSuite so any variation applies to every model. obs is the dict-of-streams bag; enable_grad
        # lets contraction build its Jacobian graph on val (Trainer runs with inference_mode=False).
        if self.variations:
            ctx = VarContext(m, preds, future["proprio"], obs, act, self.norm,
                             self.R, self.r, self.v_scale, self.dt, tag == "train", self._physical_ramp())
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
                dec = m.to_obs(src)                           # decode (mse) / 1-step sample (flow) — val metrics only
                p_hat = torch.nan_to_num(self.norm.denorm_obs(dec["proprio"]), nan=10.0, posinf=10.0, neginf=-10.0)
                p_true = self.norm.denorm_obs(future["proprio"])
                self.log("val/metric/proprio/manifold_distance_error", T.manifold_distance_error(p_hat, self.R, self.r).mean())
                self.log("val/metric/proprio/pointwise_error", T.pointwise_error(p_hat, p_true).mean())
                self.log("val/metric/proprio/tangent_velocity_error", T.tangent_velocity_error(p_hat, self.R, self.v_scale).mean())
                self.log("val/metric/proprio/obs_error", F.mse_loss(dec["proprio"], future["proprio"]))  # decoded-proprio MSE (normalized) — comparable across decoders
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
        # pre-clip non-finite fractions: localize a blow-up (which/how much of the grad is bad) BEFORE clipping
        # mangles it — norm-clipping turns a single inf into an all-NaN grad, so the fractions must be read here.
        total = sum(g.numel() for g in grads) or 1
        n_nan = sum(torch.isnan(g).sum() for g in grads) if grads else 0
        n_inf = sum(torch.isinf(g).sum() for g in grads) if grads else 0
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
        self.log("grad/fraction_nans", n_nan / total)
        self.log("grad/fraction_infs", n_inf / total)
        self.log("grad/nonfinite_skipped", float(skipped))

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def on_train_batch_end(self, *_):
        self.model.on_optimizer_step()  # EMA target update for LSAR-EMA; no-op otherwise

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # not fused: Lightning's gradient_clip_val is incompatible with a fused optimizer, and at this
        # model size the fused speedup is negligible while grad clipping aids autoregressive stability.
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_warmup_steps > 0:
            # linear LR warmup 0->1 over N OPTIMIZER STEPS. The flow decode regresses a clean target from a
            # near-pure-noise input -> high-variance early gradients; full LR from step 0 let one oversized
            # early update blow activations up into a bf16 overflow. Warmup lets them settle. interval="step"
            # counts batches (not epochs) so warmup finishes early in epoch 0.
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / self.lr_warmup_steps))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt
