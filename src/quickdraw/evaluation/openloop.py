"""Open-loop long-horizon evaluation: autoregressive rollout vs. true trajectory.

`eval_batched` rolls out the WHOLE eval split in one batched (no-grad, on-device) pass, so both the
per-episode plots and the dataset-aggregated error curves come from a single fast rollout.
"""

from __future__ import annotations

import numpy as np
import torch

def _ssim(a, b):
    """Windowed SSIM over (N,H,W,3) images in [0,1] (uniform 7x7 window via avg_pool — pooling, not a
    learned conv). Returns mean SSIM scalar."""
    import torch.nn.functional as F
    a, b = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.avg_pool2d(a, 7, 1), F.avg_pool2d(b, 7, 1)
    va = F.avg_pool2d(a * a, 7, 1) - mu_a ** 2
    vb = F.avg_pool2d(b * b, 7, 1) - mu_b ** 2
    cab = F.avg_pool2d(a * b, 7, 1) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


_LPIPS_CACHE: dict = {}


def _lpips_net(device):
    """Cached LPIPS (SqueezeNet backbone — the cheapest of the three; ~0.1 GFLOP/frame at 128px, negligible
    beside the rollout that produced the frames). Returns None if the weights can't be fetched, so a missing
    download degrades the metric rather than killing an eval."""
    key = str(device)
    if key not in _LPIPS_CACHE:
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            net = LearnedPerceptualImagePatchSimilarity(net_type="squeeze", normalize=True).to(device).eval()
            for prm in net.parameters():
                prm.requires_grad_(False)
            _LPIPS_CACHE[key] = net
        except Exception as e:                     # LOUD, once: a silent None disables lpips for the whole
            _LPIPS_CACHE[key] = None               # run (design/logging.md: fail-soft but NOT silent)
            print(f"[lpips] DISABLED for this process ({type(e).__name__}: {e}) — "
                  f"image_curves will omit the lpips key")
    return _LPIPS_CACHE[key]


def image_curves(pred, true):
    """Per-timestep IMAGE metrics -> {psnr, ssim, mse, l1, lpips, psnr_frozen, motion_ratio}, each a (H,) numpy
    array. pred/true: (N, H, s, s, 3) in [0,1] (caller clamps pred). SHARED by eval_ood_horizon (rollout preds)
    and eval_ae_floor (encode->decode recon) -- the arithmetic is bit-for-bit the same in both.

    psnr/ssim/mse/l1 are all PIXELWISE similarities, so a model that predicted "next frame = current frame"
    would score respectably while modelling nothing at all. The last three keys exist to catch that:
      lpips        perceptual distance (LOWER better). Rises with blur even when MSE does not, separating
                   "hedging toward the mean frame" from "confidently wrong".
      psnr_frozen  PSNR of holding frame 0 for the whole rollout -- the do-nothing baseline. psnr must stay
                   ABOVE it or no change is being predicted. Also reads as difficulty: a static scene has a
                   high psnr_frozen, so beating it is the real bar.
      motion_ratio ||pred_t - pred_{t-1}|| / ||true_t - true_{t-1}||. 1 = right amount of motion, <1 =
                   under-predicting it (drifting toward a frozen scene), >1 = jitter. This is the one metric
                   that separates "blurry but moving" from "sharp but static"; the others conflate them.
    """
    H = pred.shape[1]
    psnr_s, ssim_s, mse_s, l1_s, lp_s = [], [], [], [], []
    lp = _lpips_net(pred.device)
    for t in range(H):
        mse = float(torch.mean((pred[:, t] - true[:, t]) ** 2)); mse_s.append(mse)
        l1_s.append(float(torch.mean((pred[:, t] - true[:, t]).abs())))
        psnr_s.append(-10.0 * np.log10(max(mse, 1e-12)))
        ssim_s.append(max(0.0, min(1.0, _ssim(pred[:, t], true[:, t]))))   # clamp SSIM to [0,1]
        if lp is not None:
            with torch.no_grad():
                lp_s.append(float(lp(pred[:, t].permute(0, 3, 1, 2).clamp(0, 1).float(),
                                     true[:, t].permute(0, 3, 1, 2).clamp(0, 1).float())))
    if lp is not None:
        # LearnedPerceptualImagePatchSimilarity is a stateful Metric: EVERY __call__ appends to .all_scores, and
        # the net is cached for the process, so without this the state grows without bound on the eval device.
        lp.reset()

    frozen = true[:, :1].expand_as(true)                                   # the do-nothing prediction
    fz = [float(torch.mean((frozen[:, t] - true[:, t]) ** 2)) for t in range(H)]
    frozen_s = np.array([-10.0 * np.log10(max(m, 1e-12)) for m in fz])
    dp = [float(torch.mean((pred[:, t] - pred[:, t - 1]) ** 2)) ** 0.5 for t in range(1, H)]
    dt_ = [float(torch.mean((true[:, t] - true[:, t - 1]) ** 2)) ** 0.5 for t in range(1, H)]
    ratio = np.array([p / max(q, 1e-12) for p, q in zip(dp, dt_)])
    ratio = np.concatenate([ratio[:1], ratio]) if len(ratio) else np.ones(H)   # t=0 has no delta -> repeat t=1

    out = {"psnr": np.array(psnr_s), "ssim": np.array(ssim_s), "mse": np.array(mse_s), "l1": np.array(l1_s),
           "psnr_frozen": frozen_s, "motion_ratio": ratio}
    if lp_s:
        out["lpips"] = np.array(lp_s)          # LOWER is better (unlike psnr/ssim)
    assert all(len(v) == H for v in out.values()), \
        f"image_curves must return length-H arrays; got { {k: len(v) for k, v in out.items()} } for H={H}"
    return out


def emit_horizon_readouts(writer, routine, head, icurves, H, step):
    """Quarter-horizon `@+x` scalar readouts of a head's per-step curves (x in {q, 2q, 3q, H}, q=floor(0.25H)):
    one scalar per (stat, x) at `{routine}/{head}/{stat}/@+{x}` so the accuracy decay vs depth is trackable in
    wandb without the curve. SHARED by eval_ood_horizon + eval_ae_floor (identical readout)."""
    # Quarters of H, UNIONED with fixed early steps. Quarters alone are useless when H is large: at H=791
    # they give @+197/394/591/791 and there is NO number at step 32 or 64 -- the regime that actually matters
    # for a model trained on F=64 rollouts. (2026-08-10: every headline figure was a mean over 791 steps, ~12x
    # past the horizon we care about, which distorted the whole DF investigation.)
    q = max(1, int(0.25 * H))
    for x in sorted({1, 8, 16, 32, 64, q, 2 * q, 3 * q, H} & set(range(1, H + 1))):
        for stat, arr in icurves.items():
            writer.scalar(f"{routine}/{head}/{stat}/@+{x}", float(arr[min(x, H) - 1]), step)


def proprio_curves(preds_norm, true_norm, p_hat, p_true, env, pos_slice=None):
    """The open-loop proprio per-step metrics (each (N,H)), shared by eval_batched (vector spine) and
    eval_ood_horizon (multimodal spine) so the metric definitions live in ONE place. preds_norm/true_norm
    are normalized (obs_error == the training loss); p_hat/p_true are denormalized physical positions.
    The physical metrics are env-polymorphic (`env.rollout_metrics`, elementwise over leading dims ->
    (N,H)): torus returns its manifold/pointwise/tangent errors (same functions, same arguments as the
    old hardcoded calls — byte-identical); a generic env returns the default pointwise_error.
    `pos_slice`: None (default) -> pointwise_error is the env's FULL-observation error (unchanged). A list of
    obs dims -> pointwise_error is the POSITION L2 over just those dims (used when position_idx is EXPLICIT;
    e.g. docking cares about position, not the full state). `obs_error` stays the FULL-vector error in BOTH
    cases. NOTE: pointwise_error is the recorded env's checkpoint_metric, so an explicit position_idx makes
    best.ckpt select on POSITION error."""
    out = {
        "obs_error": ((preds_norm - true_norm) ** 2).mean(-1),          # normalized full-vec MSE (== the loss)
        **env.rollout_metrics(p_hat, p_true),
    }
    if pos_slice is not None:                                           # explicit position_idx -> position-only L2
        out["pointwise_error"] = (p_hat[..., pos_slice] - p_true[..., pos_slice]).norm(dim=-1)
    return out


@torch.no_grad()
def eval_batched(model, normalizer, env, P, obs_seq, act_seq, pos=(0, 1, 2)):
    """obs_seq (N,L,6), act_seq (N,L,2) normalized on device. One batched rollout over all N episodes.

    Returns per-step metrics for every episode (N,horizon), the dataset-averaged curves (mean over
    episodes per step), and denormalized positions/actions for plotting. `pos` = the world-xyz obs dims
    (from `_pos_idx`; torus is [0,1,2] so the default keeps torus byte-identical).
    """
    pos = list(pos)
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
        "ctx_xyz": normalizer.denorm_obs(obs_seq[:, :P])[:, :, pos].cpu().numpy(),  # (N,P,3)
        "p_hat_obs": p_hat.cpu().numpy(),             # (N,horizon,obs_dim) full obs — render_obs fallback
        "p_hat_xyz": p_hat[:, :, pos].cpu().numpy(),  # (N,horizon,3)
        "p_true_xyz": p_true[:, :, pos].cpu().numpy(),
        "actions": normalizer.denorm_act(act_seq).cpu().numpy(),  # (N,L,2) for the action arrow
    }
