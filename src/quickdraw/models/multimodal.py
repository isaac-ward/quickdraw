"""Multimodal world-model spine (design/models/vision.md).

The carried state is a per-step TOKEN BAG (B, T, n_state, d): each enabled modality contributes
fixed token(s) (proprio:1, image:num_tokens) via the registry. Action is its OWN input-only token, so the
per-step bag fed to the backbone is [state tokens ++ action token] = n_input = n_state+1. The backbone is
the factorized space-time transformer; `readout` predicts the NEXT state bag from the state-token context
(the action token is never decoded/predicted). `to_obs` decodes each modality's slice back to its obs.

Proprio-only is the degenerate case (n_state=1, n_input=2) — the same spine, no special-casing. The
prediction mechanism in token space is model-specific (`predict_next`): LSAR = per-token MLP residual;
diffusion (P4) = a DiT denoiser. Shared rollout/forward/loss live here so every model inherits them."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .flow import FlowField
from .modalities import ModalitySpec, build_modalities
from .spacetime import SpaceTimeTransformer
from .transformer import pad_block_mask


def _ln(x: Tensor) -> Tensor:
    """Per-token (last-dim) non-affine LayerNorm — scale-invariant carried-token normalization."""
    return F.layer_norm(x, (x.shape[-1],))


def _mlp(i: int, o: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MultiModalSequenceModel(nn.Module):
    """Token-bag backbone + the ONE autoregressive rollout. Subclasses implement `predict_next` (and may
    add model-specific loss terms via `loss_terms`)."""

    def __init__(self, specs: list[ModalitySpec], *, d: int, depth: int, heads: int, window: int,
                 mlp_ratio: float, rope_theta: float, action_dim: int):
        super().__init__()
        self.modalities = build_modalities(specs, d)
        self.layout = [(m.name, m.n_tokens) for m in self.modalities.values()]  # bag order + slices
        self.n_state = sum(n for _, n in self.layout)
        self.n_input = self.n_state + 1                       # + action token
        self.d, self.window = d, window
        self.act_enc = _mlp(action_dim, d, d)                 # action -> 1 token
        self.backbone = SpaceTimeTransformer(d, depth, heads, window, mlp_ratio,
                                             n_slots=self.n_input, rope_theta=rope_theta)

    # ---- modality <-> token bag ----
    def encode_state(self, obs: dict[str, Tensor]) -> Tensor:        # {name:(B,T,*)} -> (B,T,n_state,d)
        toks = [self.modalities[name].encode(obs[name]) for name, _ in self.layout]
        return _ln(torch.cat(toks, dim=-2))

    def to_obs(self, bag: Tensor) -> dict[str, Tensor]:              # (B,*,n_state,d) -> {name:(B,*,*)}
        out, off = {}, 0
        for name, n in self.layout:
            out[name] = self.modalities[name].decode(bag[..., off:off + n, :])
            off += n
        return out

    def _to_input(self, bag: Tensor, act: Tensor) -> Tensor:        # (B,T,n_state,d),(B,T,2)->(B,T,n_input,d)
        return torch.cat([bag, self.act_enc(act).unsqueeze(-2)], dim=-2)

    # ---- model-specific token-space prediction (subclass) ----
    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        """h_state: backbone context at the state-token slots (B,*,n_state,d); prev_bag: same shape.
        Returns the next state bag (B,*,n_state,d)."""
        raise NotImplementedError

    def readout(self, h_bag: Tensor, prev_bag: Tensor) -> Tensor:
        return self.predict_next(h_bag[..., : self.n_state, :], prev_bag)

    # ---- teacher-forced parallel forward ----
    def forward(self, obs: dict[str, Tensor], act: Tensor) -> Tensor:
        s = self.encode_state(obs)
        h = self.backbone(self._to_input(s, act))
        return self.readout(h, s)

    # ---- shared autoregressive rollout (token-bag analogue of SequenceWorldModel._rollout) ----
    def _rollout(self, ctx_obs: dict[str, Tensor], actions: Tensor, horizon: int, p_tf: float,
                 true_future: dict[str, Tensor] | None, detach_every: int) -> Tensor:
        W = self.window
        bag_buf = list(self.encode_state(ctx_obs).unbind(dim=1))     # P bags of (B,n_state,d)
        tf_future = self.encode_state(true_future) if true_future is not None else None
        B = bag_buf[0].shape[0]
        preds = []
        for h in range(horizon):
            Lh = len(bag_buf)
            real = min(Lh, W)
            pad = W - real
            s_win = torch.stack(bag_buf[-real:], dim=1)              # (B,real,n_state,d)
            a_win = actions[:, Lh - real:Lh]                        # (B,real,2)
            if pad:
                s_win = F.pad(s_win, (0, 0, 0, 0, pad, 0))          # pad the TIME axis at front
                a_win = F.pad(a_win, (0, 0, pad, 0))
            x = self._to_input(s_win, a_win)                        # (B,W,n_input,d)
            bm = pad_block_mask(W, pad, x.device)                   # temporal causal + drop padded steps
            h_last = self.backbone(x, temporal_block_mask=bm)[:, -1]  # (B,n_input,d)
            s_pred = self.readout(h_last, bag_buf[-1])              # (B,n_state,d)
            preds.append(s_pred)
            if tf_future is not None and p_tf > 0.0:
                tf = (torch.rand(B, 1, 1, device=s_pred.device) < p_tf).float()
                s_feed = tf * tf_future[:, h] + (1.0 - tf) * s_pred
            else:
                s_feed = s_pred
            if detach_every and ((h + 1) % detach_every == 0):
                s_feed = s_feed.detach()
            bag_buf.append(s_feed)
        return torch.stack(preds, dim=1)                            # (B,horizon,n_state,d)

    def rollout_train(self, ctx_obs, actions, true_future: dict, p_tf: float, detach_every: int = 8) -> Tensor:
        horizon = next(iter(true_future.values())).shape[1]
        return self._rollout(ctx_obs, actions, horizon, p_tf, true_future, detach_every)

    @torch.no_grad()
    def imagine_eval(self, ctx_obs: dict, actions: Tensor, horizon: int) -> dict[str, Tensor]:
        with torch.autocast(device_type=actions.device.type, dtype=torch.bfloat16, enabled=actions.is_cuda):
            bag = self._rollout(ctx_obs, actions, horizon, 0.0, None, 0)
            out = self.to_obs(bag)
        return {k: v.float() for k, v in out.items()}

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        return {}, {}


class MultiModalLSAR(MultiModalSequenceModel):
    """Latent-space AR over the token bag: predict the next bag with a per-token MLP residual (+ LN);
    pred_latent = MSE to the encoded true-next bag (Reconstruction collapse: obs heads ground the encoder)."""

    def __init__(self, specs, *, d, depth, heads, window, mlp_ratio, rope_theta, action_dim,
                 pred_hidden: int = 0, lambda_pred_latent: float = 1.0):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim)
        h = pred_hidden or d
        self.predictor = _mlp(d, d, h)                          # per-token residual predictor
        self.lambda_pred_latent = lambda_pred_latent
        self.pred_obs_in_loss = True
        self.lambda_pred_obs = 1.0

    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        return _ln(prev_bag + self.predictor(h_state))

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        target = self.encode_state({k: future_obs[k] for k, _ in self.layout}).detach()
        return {"pred_latent": F.mse_loss(pred_bag, target)}, {"pred_latent": self.lambda_pred_latent}


class MultiModalDiffusion(MultiModalSequenceModel):
    """Latent flow-matching over the token bag. `predict_next` denoises the next-bag RESIDUAL with the
    shared FlowField (rectified flow / shortcut), applied PER TOKEN (the token is a leading dim, and the
    per-token context h already carries cross-token structure from the space-time backbone). Mirrors
    models/diffusion.py's teacher-forced flow loss, generalized to the bag. `predict_next` samples
    (deterministic ε=0 at eval unless stochastic_eval — the committed prediction)."""

    def __init__(self, specs, *, d, depth, heads, window, mlp_ratio, rope_theta, action_dim,
                 sampling_steps: int = 6, shortcut: bool = False, predict: str = "residual",
                 stochastic_eval: bool = False, time_sampling: str = "uniform", flow_hidden: int = 0,
                 lambda_flow: float = 1.0, lambda_consistency: float = 1.0):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim)
        assert predict in ("residual", "absolute")
        self.predict_residual = predict == "residual"
        self.sampling_steps = int(sampling_steps)
        self.stochastic_eval = bool(stochastic_eval)
        self.time_sampling = time_sampling
        self.lambda_flow, self.lambda_consistency = lambda_flow, lambda_consistency
        self.flow = FlowField(d, h_dim=d, hidden=(flow_hidden or d), cond="concat", shortcut=shortcut)
        self.pred_obs_in_loss = True
        self.lambda_pred_obs = 1.0

    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        det = (not self.training) and (not self.stochastic_eval)
        out = self.flow.sample(h_state, steps=self.sampling_steps, deterministic=det)   # per-token over the bag
        return _ln(prev_bag + out) if self.predict_residual else _ln(out)

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        """Teacher-forced rectified-flow loss over the bag (mirrors models/diffusion.py)."""
        assert act_seq is not None
        z = self.encode_state(obs)                              # (B,L,n_state,d)
        L = z.shape[1]
        s = z[:, :-1]                                           # contexts (B,L-1,n_state,d)
        h = self.backbone(self._to_input(s, act_seq[:, :L - 1]))
        h_state = h[..., : self.n_state, :]                     # (B,L-1,n_state,d)
        target = (z[:, 1:] - s).detach() if self.predict_residual else z[:, 1:].detach()
        l_flow, l_cons = self.flow.loss(h_state, target, time_sampling=self.time_sampling)
        raw, w = {"flow": l_flow}, {"flow": self.lambda_flow}
        if l_cons is not None:
            raw["flow_consistency"], w["flow_consistency"] = l_cons, self.lambda_consistency
        return raw, w
