"""Verify the loss refactor for all 6 variants: correct loss-term names per paper, probe-vs-in-loss
pred_obs, expander only on VICReg, EMA target/metric, and that a full train step backprops finitely.
Run: uv run python -m quickdraw.smoke.loss_refactor
"""
import torch
import torch.nn.functional as F

from quickdraw.models.dsar import BaseModelConfig, DataSpaceAR
from quickdraw.models.lsar import LSARConfig, LatentSpaceAR
from quickdraw.models.collapse import Naked, Reconstruction, EMA, SIGReg, VICReg

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, P, Fh = 2, 4, 6

EXPECT = {  # (loss_total term keys, pred_obs_in_loss, has_expander, has_ema)
    "dsar":           ({"pred_obs"},               True,  False, False),
    "naked":          ({"pred_latent"},            False, False, False),
    "reconstruction": ({"pred_latent", "pred_obs"}, True, False, False),
    "ema":            ({"pred_latent"},            False, False, True),
    "sigreg":         ({"pred_latent", "reg"},     False, False, False),
    "vicreg":         ({"pred_latent", "reg"},     False, False, False),  # var/cov now on the latent (no expander)
}


def build(name):
    if name == "dsar":
        return DataSpaceAR(BaseModelConfig(d=32, depth=2, heads=2, window=8))
    strat = {"naked": Naked(), "reconstruction": Reconstruction(), "ema": EMA(tau=0.99),
             "sigreg": SIGReg(n_sketches=16), "vicreg": VICReg()}[name]
    return LatentSpaceAR(LSARConfig(d=32, dz=8, depth=2, heads=2, window=8), strat)


def step(m):
    """Mimic LitWorldModel._step: raw terms + weights + unified obs term (detached if probe)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    obs = torch.randn(B, P + Fh, 6, generator=g, device=DEV)
    act = torch.randn(B, P + Fh - 1, 2, generator=g, device=DEV)
    preds = m.rollout_train(obs[:, :P], act, obs[:, P:], p_tf=0.0, detach_every=0)
    raw, weights = m.loss_terms(preds, obs[:, P:], obs, 0.0)
    in_loss = m.pred_obs_in_loss
    src = preds if in_loss else preds.detach()
    obs_mse = F.mse_loss(m.to_obs(src), obs[:, P:])
    loss_total = sum(weights[k] * raw[k] for k in raw)
    if in_loss:
        raw["pred_obs"] = obs_mse
        loss_total = loss_total + m.lambda_pred_obs * obs_mse; objective = loss_total
    else:
        objective = loss_total + m.lambda_pred_obs * obs_mse
    return raw, objective  # raw now includes pred_obs only for in_loss methods; obs_error logged for all


ok = True
for name in EXPECT:
    torch.manual_seed(0)
    m = build(name).to(DEV).train()
    want_keys, want_inloss, want_exp, want_ema = EXPECT[name]
    terms, objective = step(m)
    objective.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    has_exp = getattr(m, "expander", None) is not None
    has_ema = getattr(m, "ema_enc", None) is not None
    finite = torch.isfinite(objective).all().item()
    checks = [set(terms) == want_keys, m.pred_obs_in_loss == want_inloss, has_exp == want_exp,
              has_ema == want_ema, finite, has_grad]
    status = "OK" if all(checks) else "FAIL"
    if not all(checks):
        ok = False
    pm = getattr(getattr(m, "collapse", None), "pred_metric", "-")
    print(f"[{status}] {name:14s} keys={sorted(terms)} in_loss={m.pred_obs_in_loss} "
          f"exp={has_exp} ema={has_ema} pred_metric={pm} obj={float(objective):.4f} grad={has_grad}")

print("ALL OK" if ok else "SOME FAILED")
