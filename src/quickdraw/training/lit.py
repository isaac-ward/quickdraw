"""LightningModule: p_tf rollout training, MSE on delta, env-space metrics (design/training.md)."""

from __future__ import annotations

import lightning as L
import torch

from ..environments import torus as T
from ..models.base import BaseWorldModel
from .variations import VarContext, make_variation_suite


class LitWorldModel(L.LightningModule):
    def __init__(self, model: BaseWorldModel, normalizer, R: float, r: float, v_scale: float, P: int, F: int,
                 p_tf_start: float, p_tf_end: float, p_tf_warmup: int,
                 lr: float, weight_decay: float, detach_every: int = 8, variations=None, dt: float = 1.0 / 60.0):
        super().__init__()
        self.model = model
        self.norm = normalizer
        self.R, self.r, self.v_scale, self.P, self.F = R, r, v_scale, P, F
        self.dt = dt
        self.p_tf_start, self.p_tf_end, self.p_tf_warmup = p_tf_start, p_tf_end, p_tf_warmup
        self.lr, self.weight_decay, self.detach_every = lr, weight_decay, detach_every
        # train-time shaping variations (off by default -> empty suite, zero overhead). See variations.py.
        self.variations = make_variation_suite(variations)

    def _cur_p_tf(self) -> float:
        # curriculum: ramp from p_tf_start (e.g. 1.0, full teacher forcing) down to p_tf_end over warmup
        if self.p_tf_warmup <= 0:
            return self.p_tf_end
        frac = min(1.0, self.current_epoch / self.p_tf_warmup)
        return self.p_tf_end + (self.p_tf_start - self.p_tf_end) * (1.0 - frac)

    # ---- shared: produce future predictions (normalized) for a batch window ----
    # obs_input feeds the model (may be noise-augmented); obs_target is the clean target/TF-source split.
    def _future_preds(self, obs_input, act_seq, obs_target):
        P, L = self.P, self.P + self.F
        p_tf = self._cur_p_tf()
        if p_tf >= 1.0:  # parallel teacher forcing (one causal pass)
            preds = self.model(obs_input[:, :-1], act_seq[:, :-1])  # predict obs[:,1:]
            return preds[:, P - 1 :], obs_target[:, P:]
        ctx, actions = obs_input[:, :P], act_seq[:, : L - 1]
        tf_source = obs_input[:, P:]  # teacher-forcing feed is an INPUT (noised); loss target stays clean
        preds = self.model.rollout_train(ctx, actions, tf_source, p_tf, self.detach_every)
        return preds, obs_target[:, P:]

    def _step(self, batch, tag):
        obs_seq, act_seq = batch["obs_seq"], batch["act_seq"]
        training = tag == "train"
        p_tf = self._cur_p_tf()
        # variations: perturb the obs INPUTS only (noise injection); targets/metrics use clean obs_seq.
        obs_input, t_logs = self.variations.transform_obs(obs_seq, training)
        preds, future_obs = self._future_preds(obs_input, act_seq, obs_seq)  # preds = STATES (obs for DSAR, latent for LSAR)
        # model-specific RAW terms + weights (DSAR: {}; LSAR: {pred_latent[, reg]}).
        raw, weights = self.model.loss_terms(preds, future_obs, obs_seq, p_tf)
        # unified obs-rollout error: decode the predicted rollout and compare to the true future obs.
        in_loss = getattr(self.model, "pred_obs_in_loss", True)   # True: DSAR/recon (shapes model); else probe
        lam = getattr(self.model, "lambda_pred_obs", 1.0)
        src = preds if in_loss else preds.detach()  # detached -> trains the decoder only (readout probe)
        obs_mse = torch.nn.functional.mse_loss(self.model.to_obs(src), future_obs)
        loss_total = sum(weights[k] * raw[k] for k in raw)  # actual minimized objective (scaled)
        if in_loss:                                 # DSAR / reconstruction: loss_pred_obs is a real loss
            raw["pred_obs"] = obs_mse
            loss_total = loss_total + lam * obs_mse
            objective = loss_total
        else:                                       # JEPA variants: obs term is a decoder-only readout probe
            objective = loss_total + lam * obs_mse
        # train-time shaping variations (noise already applied to inputs above; here the loss terms):
        # add their penalties to the objective and log everything under {variation_name}/.
        if training and self.variations:
            ctx = VarContext(self.model, preds, future_obs, obs_seq, act_seq,
                             self.norm, self.R, self.r, self.v_scale, self.dt, training)
            extra, v_logs = self.variations.losses(ctx)
            if extra is not None:
                objective = objective + extra
            for k, val in {**t_logs, **v_logs}.items():
                self.log(k, val)
        # logging: loss/* are RAW (pre-scaling, comparable across methods); loss/total is the actual
        # scaled objective. obs_error is the decoded-rollout obs MSE logged for ALL methods (one plot).
        self.log(f"{tag}/loss/total", loss_total, prog_bar=(tag == "train"))
        for k, v in raw.items():
            self.log(f"{tag}/loss/{k}", v)
        self.log(f"{tag}/obs_error", obs_mse.detach())
        if tag == "train":
            self.log("diag/p_tf", p_tf)
        with torch.no_grad():  # metrics in real (denormalized) obs space; LSAR decodes via to_obs
            p_hat = self.norm.denorm_obs(self.model.to_obs(preds))
            p_true = self.norm.denorm_obs(future_obs)
            # a freshly-initialized decoder (esp. the no-LN reg variants at ep0) can emit non-finite obs ->
            # the metric reduces to NaN. Clamp non-finite preds to a far-but-finite ±10 so a broken model
            # reads as a LARGE-but-plottable error, not NaN (metric path only — never the loss).
            p_hat = torch.nan_to_num(p_hat, nan=10.0, posinf=10.0, neginf=-10.0)
            self.log(f"{tag}/manifold_distance_error", T.manifold_distance_error(p_hat, self.R, self.r).mean())
            self.log(f"{tag}/pointwise_error", T.pointwise_error(p_hat, p_true).mean())
            self.log(f"{tag}/tangent_velocity_error", T.tangent_velocity_error(p_hat, self.R, self.v_scale).mean())
            # latent collapse diagnostics (LSAR only), once per validation epoch
            if tag == "val" and hasattr(self.model, "collapse_diagnostics"):
                for k, v in self.model.collapse_diagnostics(obs_seq).items():
                    self.log(f"collapse/{k}", v)
        return objective

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # clip (Trainer sets val=1.0) AND log the total grad norm pre- and post-clip, generically for
        # every model, so BPTT blow-ups (DSAR diverged ~ep30 even with clipping) are diagnosable.
        def _total_norm():
            gs = [p.grad.detach().norm() for p in self.parameters() if p.grad is not None]
            return torch.norm(torch.stack(gs)) if gs else torch.zeros((), device=self.device)
        pre = _total_norm()
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm)
        self.log("grad/norm_preclip", pre)
        self.log("grad/norm_postclip", _total_norm())

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def on_train_batch_end(self, *_):
        self.model.on_optimizer_step()  # EMA target update for LSAR-EMA; no-op otherwise

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # not fused: Lightning's gradient_clip_val is incompatible with a fused optimizer, and at this
        # model size the fused speedup is negligible while grad clipping aids autoregressive stability.
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
