"""Conditional scores for the action prior: does it know WHEN, or only HOW OFTEN?

`eval_action_distribution`'s W1 compares POOLED histograms -- the head's draws against the recorded actions,
both pooled over every context. A model that ignored its context entirely and sampled from the training
marginal scores near-perfectly on that, on held-out data, forever, because val's marginal is train's
marginal. The hole has always been there; `action_head.target_transform=pit` makes it much easier to fall
into, because it hands the head the correct marginal shape by construction.

So these three, all scored against a CONTEXT-BLIND NULL -- the recorded chunks themselves, shuffled across
contexts, which has the right marginal AND the right within-chunk dependence and no context at all:

  energy_score   ES = mean||X_i - y|| - 0.5 mean||X_i - X_j||, a strictly proper scoring rule for the full
                 joint that needs only ONE observation per context. The second term is what makes it honest:
                 hedging by spreading the draws inflates it and cancels the credit the first term gives.
                 Reported as SKILL = 1 - ES_model / ES_null, so 0 means the context bought nothing.
  rank calib.    where the observation falls among the draws. Uniform = calibrated; piled at the ends =
                 overconfident (draws too tight); piled in the middle = overdispersed. The only one of the
                 three that says WHICH WAY the head is wrong.
  rest skill     AUC and Brier of the head's implied P(at rest | context) against what actually happened.
                 AUC 0.5 = knows nothing about WHEN however perfectly it matches the overall rest fraction.

Kept out of routines.py so scratch/eval_conditional.py can score finished checkpoints with the SAME code
the training-time eval logs -- two implementations of a scoring rule is how two numbers that are supposed to
be comparable stop being comparable.
"""
from __future__ import annotations

import numpy as np


def energy_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ES per context. X: (N, M, D) draws, y: (N, D) observations. Lower is better."""
    d1 = np.linalg.norm(X - y[:, None, :], axis=-1).mean(1)
    i, j = np.triu_indices(X.shape[1], k=1)
    d2 = np.linalg.norm(X[:, i, :] - X[:, j, :], axis=-1).mean(1)
    return d1 - 0.5 * d2


def energy_skill(X: np.ndarray, y: np.ndarray, null: np.ndarray) -> float:
    """1 - ES(model) / ES(context-blind null). 0 = the context bought nothing, 1 = perfect."""
    en = float(energy_score(null, y).mean())
    return float(1.0 - float(energy_score(X, y).mean()) / en) if en > 0 else 0.0


def blind_null(y: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    """The recorded chunks, shuffled across contexts: right marginal, right dependence, wrong context."""
    rng = np.random.default_rng(seed)
    return y[rng.integers(0, y.shape[0], size=(y.shape[0], M))]


def rank_calibration(X: np.ndarray, y: np.ndarray, bins: int = 10):
    """Returns (decile mass, max deviation from uniform, verdict). X: (N, M, D), y: (N, D)."""
    r = (X < y[:, None, :]).sum(1).reshape(-1) / X.shape[1]
    hist = np.histogram(r, bins=bins, range=(0.0, 1.0))[0] / r.size
    dev = float(np.max(np.abs(np.cumsum(hist) - np.linspace(1.0 / bins, 1.0, bins))))
    ends, mid = float(hist[0] + hist[-1]), float(hist[bins // 2 - 1] + hist[bins // 2])
    verdict = ("overconfident (truth lands outside the draws)" if ends > 3.0 / bins else
               "overdispersed (draws too wide)" if mid > 3.0 / bins else "roughly calibrated")
    return hist, dev, verdict


def rest_skill(X: np.ndarray, y: np.ndarray):
    """Per dim: does the head know WHEN the stick is at its most common value, not just how often?

    X: (N, M, D) draws, y: (N, D) observations, both in ACTION units. Yields one dict per dim whose rest
    value is neither vanishing nor almost-everything (a constant dim has no WHEN to know)."""
    for d in range(y.shape[-1]):
        vals, counts = np.unique(y[:, d], return_counts=True)
        rest = vals[counts.argmax()]
        truth = y[:, d] == rest
        if not (0.05 <= truth.mean() <= 0.95):
            continue
        p = (X[:, :, d] == rest).mean(1)                    # the head's implied P(at rest | context)
        brier = float(np.mean((p - truth) ** 2))
        base = float(np.mean((truth.mean() - truth) ** 2))   # climatology: always predict the base rate
        pos, neg = p[truth], p[~truth]
        auc = (float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
               if pos.size and neg.size else 0.5)
        yield dict(dim=d, rest=float(rest), rate=float(truth.mean()), pred_rate=float(p.mean()),
                   auc=auc, brier=brier, brier_skill=float(1.0 - brier / base) if base > 0 else 0.0)
