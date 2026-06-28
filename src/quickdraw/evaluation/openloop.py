"""Open-loop long-horizon evaluation: autoregressive rollout vs. true trajectory.

`eval_batched` rolls out the WHOLE eval split in one batched (no-grad, on-device) pass, so both the
per-episode plots and the dataset-aggregated error curves come from a single fast rollout.
"""

from __future__ import annotations

import numpy as np
import torch

from ..environments import torus as T

_METRICS = ("manifold_distance_error", "pointwise_error", "tangent_velocity_error")


@torch.no_grad()
def eval_batched(model, normalizer, R, r, v_scale, P, obs_seq, act_seq):
    """obs_seq (N,L,6), act_seq (N,L,2) normalized on device. One batched rollout over all N episodes.

    Returns per-step metrics for every episode (N,horizon), the dataset-averaged curves (mean over
    episodes per step), and denormalized positions/actions for plotting.
    """
    L = obs_seq.shape[1]
    horizon = L - P
    preds = model.imagine_eval(obs_seq[:, :P], act_seq[:, : L - 1], horizon)  # (N,horizon,6)
    p_hat = normalizer.denorm_obs(preds)
    p_true = normalizer.denorm_obs(obs_seq[:, P:])
    # clamp non-finite decoded preds to ±10 so a broken decoder reads as a large-but-finite error, not
    # NaN (metric path only; matches lit._step). obs_vector_mse below uses raw preds intentionally.
    p_hat = torch.nan_to_num(p_hat, nan=10.0, posinf=10.0, neginf=-10.0)
    per_step = {
        "obs_vector_mse": ((preds - obs_seq[:, P:]) ** 2).mean(-1),  # (N,horizon) normalized = the loss
        "manifold_distance_error": T.manifold_distance_error(p_hat, R, r),
        "pointwise_error": T.pointwise_error(p_hat, p_true),
        "tangent_velocity_error": T.tangent_velocity_error(p_hat, R, v_scale),
    }
    agg = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}  # mean over episodes -> (horizon,)
    return {
        "per_step": {k: v.cpu().numpy() for k, v in per_step.items()},  # (N,horizon) each
        "agg": agg,
        "ctx_xyz": normalizer.denorm_obs(obs_seq[:, :P])[:, :, :3].cpu().numpy(),  # (N,P,3)
        "p_hat_xyz": p_hat[:, :, :3].cpu().numpy(),   # (N,horizon,3)
        "p_true_xyz": p_true[:, :, :3].cpu().numpy(),
        "actions": normalizer.denorm_act(act_seq).cpu().numpy(),  # (N,L,2) for the action arrow
    }
