"""Smoke for the latent rollout loss (the attached latent twin of the decode rollout loss; replaces the
removed detached overshoot).
  1. OFF (weight 0, default) -> latent_rollout_losses returns {} (no term, bit-identical).
  2. metric is identity+distance: on hand-made tensors, mse == F.mse_loss, cosine in [0,2].
  3. ON: dynamics/latent_rollout term present + finite on a real rollout.
  4. ATTACHED: gradient from the term flows back into the backbone (through the rollout) -- NOT detached.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.latent_rollout
"""
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

from ..training.setup import build_model


def _fail(m): print("FAIL:", m); raise SystemExit(1)


def _cfg(ov):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        # target_transform=none: mm_flow (main) now defaults the action prior's target to `pit`,
        # whose knots are read from a packed dataset this smoke neither has nor needs. The flow
        # config is longhand's, the pit default is main's; they first met in the 2026-10-20 merge.
        return compose(config_name="config",
                       overrides=list(ov) + ["model.action_head.target_transform=none"])


def _fake(m, B=2, P=8, F_=10):
    L = P + F_
    obs = {}
    for name, n in m.layout:
        mod = m.modalities[name]
        if name == "proprio":
            obs[name] = torch.randn(B, L, mod.dim)
        else:
            sh = mod.ae.cfg.img_size; sh = sh if isinstance(sh, (tuple, list)) else (sh, sh)
            obs[name] = torch.rand(B, L, sh[0], sh[1], 3)
    act = torch.randn(B, L, 5)
    return obs, act, P, F_


def main():
    torch.manual_seed(0)
    base = ["model=vl128_blockstack_2cam"]

    # 1. OFF
    off = build_model(_cfg(base))
    r, w = off.latent_rollout_losses(torch.randn(2, 4, off.n_state, off.d), torch.randn(2, 4, off.n_state, off.d))
    if r or w:
        _fail(f"weight 0 must yield no term; got {list(r)}")
    print("[1/4] OFF -> no term (bit-identical)   OK")

    # 2. metric correctness
    on = build_model(_cfg(base + ["++model.latent_rollout.weight=1.0", "++model.latent_rollout.metric=mse"]))
    a = torch.randn(2, 4, on.n_state, on.d); b = torch.randn(2, 4, on.n_state, on.d)
    r, _ = on.latent_rollout_losses(a, b)
    if not torch.allclose(r["dynamics/latent_rollout"], F.mse_loss(a, b)):
        _fail("mse metric != F.mse_loss")
    onc = build_model(_cfg(base + ["++model.latent_rollout.weight=1.0", "++model.latent_rollout.metric=cosine"]))
    rc, _ = onc.latent_rollout_losses(a, b)
    if not (0.0 <= float(rc["dynamics/latent_rollout"]) <= 2.0):
        _fail("cosine metric out of [0,2]")
    # target is detached (no grad leaks into the true latent)
    bt = b.clone().requires_grad_(True)
    rg, _ = on.latent_rollout_losses(a.requires_grad_(True), bt)
    rg["dynamics/latent_rollout"].backward()
    if bt.grad is not None and float(bt.grad.abs().sum()) != 0.0:
        _fail("target z_true must be detached (got gradient)")
    print("[2/4] metric: mse==F.mse_loss, cosine in [0,2], target detached   OK")

    # 3 & 4. ON, real rollout, term finite + ATTACHED (grad reaches the backbone)
    on.compile_rollout = False; on.train()
    obs, act, P, Fh = _fake(on)
    ctx = {k: v[:, :P] for k, v in obs.items()}; fut = {k: v[:, P:] for k, v in obs.items()}
    preds = on._rollout(ctx, act[:, :-1], Fh, p_tf=0.0, true_future=None, detach_every=8, use_cache=False)
    z_true = on.encode_state(fut)
    lr, lw = on.latent_rollout_losses(preds, z_true)
    val = lr["dynamics/latent_rollout"]
    if not torch.isfinite(val):
        _fail("latent_rollout non-finite")
    print(f"[3/4] ON -> dynamics/latent_rollout={float(val):.4f} finite (weight {lw['dynamics/latent_rollout']})   OK")
    on.zero_grad()
    (lw["dynamics/latent_rollout"] * val).backward()
    gb = sum(float(p.grad.abs().sum()) for p in on.backbone.parameters() if p.grad is not None)
    if gb == 0.0:
        _fail("no gradient reached the backbone -> preds were NOT attached (this is the whole point vs overshoot)")
    print(f"[4/4] ATTACHED: gradient reaches backbone (sum {gb:.3g}) -- scores the rolled latent, no detach   OK")

    print("\nALL LATENT-ROLLOUT SMOKE PASSED")


if __name__ == "__main__":
    main()
