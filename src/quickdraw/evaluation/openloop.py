"""Open-loop long-horizon evaluation: autoregressive rollout vs. true trajectory."""

from __future__ import annotations

import numpy as np
import torch

from ..environments import torus as T


@torch.no_grad()
def eval_episode(model, normalizer, R, r, P, obs_seq, act_seq):
    """obs_seq (1,Tn,6), act_seq (1,Tn,2) normalized. Returns curves + summary + xyz for plots."""
    L = obs_seq.shape[1]
    horizon = L - P
    ctx, actions = obs_seq[:, :P], act_seq[:, : L - 1]
    preds = model.imagine_eval(ctx, actions, horizon)  # (1,horizon,6) normalized
    p_hat = normalizer.denorm_obs(preds)
    p_true = normalizer.denorm_obs(obs_seq[:, P:])
    curves = {
        "manifold_distance_error": T.manifold_distance_error(p_hat, R, r)[0].cpu().numpy(),
        "pointwise_error": T.pointwise_error(p_hat, p_true)[0].cpu().numpy(),
        "tangent_velocity_error": T.tangent_velocity_error(p_hat, R)[0].cpu().numpy(),
    }
    mde = curves["manifold_distance_error"]
    summary = {f"manifold_distance_error@{k}": float(mde[min(k, len(mde) - 1)]) for k in (500, 1000, 2000)}
    summary["manifold_distance_error_auc"] = float(mde.mean())
    true_xyz = p_true[0, :, :3].cpu().numpy()
    pred_xyz = p_hat[0, :, :3].cpu().numpy()
    return {"curves": curves, "summary": summary, "true_xyz": true_xyz, "pred_xyz": pred_xyz}
