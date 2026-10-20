"""Smoke for latent overshoot (design/combatting_drift.md).
  1. threaded + builds with overshoot on.
  2. BIT-IDENTICAL: overshoot_weight=0 adds NO overshoot term, and the clean anchor is unchanged vs weight>0
     (same seed) -- overshoot only ADDS a term, never perturbs the depth-1 anchor.
  3. ACTIVE: overshoot_weight>0 + feeds -> a finite `dynamics/latent_overshoot` term + gradient into self.flow.
  4. DETACH: the overshoot term does not backprop into `feeds` (pure-forward).
  5. GUARDS raise: p_tf_dynamics!=1, compile_rollout, diffusion forcing, non-flow model.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.overshoot
"""
import torch
torch.set_num_threads(1)
from hydra import compose, initialize_config_dir

from ..training.setup import build_model, effective_action_dim


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _cfg(overrides):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        return compose(config_name="config", overrides=overrides)


def main():
    over = ["model=mm_flow_proprio", "+model.overshoot_weight=0.5"]
    cfg = _cfg(over)
    m = build_model(cfg).train()
    if float(m.overshoot_weight) != 0.5:
        _fail("overshoot_weight not threaded to the model")
    print(f"builds with overshoot on; weight={m.overshoot_weight}   OK")

    # --- synthetic inputs for loss_terms ---
    B, P, F = 2, 2, 5
    L = P + F
    pdim = int(cfg.model.modalities[0].dim)
    adim = effective_action_dim(cfg)
    torch.manual_seed(0)
    obs = {"proprio": torch.randn(B, L, pdim)}
    fut = {"proprio": torch.randn(B, F, pdim)}
    act = torch.randn(B, L, adim)
    pred = torch.randn(B, F, m.n_state, m.d)
    feeds = torch.randn(B, F, m.n_state, m.d, requires_grad=True)

    def run(weight):
        m.overshoot_weight = float(weight)
        torch.manual_seed(1)                                   # SAME rng -> the anchor is computed identically
        return m.loss_terms(pred, fut, obs, 0.0, act, feeds=feeds)

    raw0, _ = run(0.0)
    if "dynamics/latent_overshoot" in raw0:
        _fail("overshoot_weight=0 still produced an overshoot term (not bit-identical)")
    raw1, w1 = run(0.5)
    if "dynamics/latent_overshoot" not in raw1:
        _fail("overshoot_weight>0 did NOT add the overshoot term")
    if not torch.allclose(raw0["dynamics/latent"], raw1["dynamics/latent"]):
        _fail("overshoot changed the clean anchor `dynamics/latent` (must only ADD a term)")
    lo = raw1["dynamics/latent_overshoot"]
    if not torch.isfinite(lo):
        _fail(f"overshoot term is not finite: {lo}")
    print(f"weight=0 -> no term; weight>0 -> anchor unchanged + finite overshoot={float(lo.detach()):.4g}   OK")

    # gradient into the flow, and NOT into feeds (detached)
    feeds.grad = None
    lo.backward()
    g = sum(float(p.grad.abs().sum()) for p in m.flow.parameters() if p.grad is not None)
    if not (g > 0):
        _fail("overshoot term produced NO gradient into self.flow")
    if feeds.grad is not None and float(feeds.grad.abs().sum()) > 0:
        _fail("overshoot backpropped into feeds -- must be detached (pure-forward)")
    print(f"overshoot grad into flow={g:.4g}; feeds grad None (detached)   OK")

    # --- guards ---
    for bad, why in [
        (over + ["+model.p_tf_dynamics=0.8"], "p_tf_dynamics!=1"),
        (over + ["variations.noise_injection.observations_encoded_pre_fusion.scale=0.3"], "diffusion forcing"),
        (["model=mm_dsar", "+model.overshoot_weight=0.5"], "non-flow model"),
    ]:
        try:
            build_model(_cfg(bad)); _fail(f"guard did NOT raise for {why}")
        except (ValueError, Exception) as e:
            if not isinstance(e, ValueError):
                raise
            print(f"guard raises: {why}   OK")

    print("\nALL OVERSHOOT SMOKE PASSED")


if __name__ == "__main__":
    main()
