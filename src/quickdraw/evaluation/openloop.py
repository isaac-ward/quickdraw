"""Open-loop long-horizon evaluation: autoregressive rollout vs. true trajectory.

`eval_batched` rolls out the WHOLE eval split in one batched (no-grad, on-device) pass, so both the
per-episode plots and the dataset-aggregated error curves come from a single fast rollout.
"""

from __future__ import annotations

import numpy as np
import torch

def proprio_curves(preds_norm, true_norm, p_hat, p_true, env):
    """The open-loop proprio per-step metrics (each (N,H)), shared by eval_batched (vector spine) and
    eval_ood_horizon (multimodal spine) so the metric definitions live in ONE place. preds_norm/true_norm
    are normalized (obs_error == the training loss); p_hat/p_true are denormalized physical positions.
    The physical metrics are env-polymorphic (`env.rollout_metrics`, elementwise over leading dims ->
    (N,H)): torus returns its manifold/pointwise/tangent errors (same functions, same arguments as the
    old hardcoded calls — byte-identical); a generic env returns the default pointwise_error."""
    return {
        "obs_error": ((preds_norm - true_norm) ** 2).mean(-1),          # normalized 6-vec MSE (== the loss)
        **env.rollout_metrics(p_hat, p_true),
    }


@torch.no_grad()
def eval_batched(model, normalizer, env, P, obs_seq, act_seq):
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
    # NaN (metric path only; matches lit._step). obs_error below uses raw preds intentionally.
    p_hat = torch.nan_to_num(p_hat, nan=10.0, posinf=10.0, neginf=-10.0)
    per_step = proprio_curves(preds, obs_seq[:, P:], p_hat, p_true, env)
    agg = {k: v.mean(0).cpu().numpy() for k, v in per_step.items()}  # mean over episodes -> (horizon,)
    return {
        "per_step": {k: v.cpu().numpy() for k, v in per_step.items()},  # (N,horizon) each
        "agg": agg,
        "ctx_xyz": normalizer.denorm_obs(obs_seq[:, :P])[:, :, :3].cpu().numpy(),  # (N,P,3)
        "p_hat_obs": p_hat.cpu().numpy(),             # (N,horizon,obs_dim) full obs — render_obs fallback
        "p_hat_xyz": p_hat[:, :, :3].cpu().numpy(),   # (N,horizon,3)
        "p_true_xyz": p_true[:, :, :3].cpu().numpy(),
        "actions": normalizer.denorm_act(act_seq).cpu().numpy(),  # (N,L,2) for the action arrow
    }
