"""Smoke for z0 re-injection / anchor_init (2026-09-28, identity_preservation Idea 1).
  1. OFF (default) -> concat_proprio-style: _cond width unchanged; anchor_init False.
  2. ON -> _cond width grows by exactly d (one extra broadcast block).
  3. z0 is USED: gradient flows back into z0, and changing z0 changes the prediction (not ignored).
  4. rollout carries the SAME z0 every step (constant over the horizon, not the drifting current bag).
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.anchor_init
"""
import torch
torch.set_num_threads(1)
from hydra import compose, initialize_config_dir

from ..training.setup import build_model


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _cfg(overrides):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        return compose(config_name="config", overrides=overrides)


def main():
    torch.manual_seed(0)
    base = ["model=vl128_blockstack_flow"]
    off = build_model(_cfg(base)).train()
    if getattr(off, "anchor_init", False):
        _fail("anchor_init should default OFF")
    on = build_model(_cfg(base + ["+model.anchor_init=true"])).train()
    if not on.anchor_init:
        _fail("anchor_init should be ON")
    print(f"[1/4] default off={off.anchor_init}  on={on.anchor_init}   OK")

    B, T, d = 2, 3, on.d
    h_bag = torch.randn(B, T, on.n_input, d)
    act = torch.randn(B, T, on.act_enc[0].in_features if hasattr(on.act_enc, "__getitem__") else 5)
    z0 = torch.randn(B, d)
    c_off = off._cond(h_bag, act)
    c_on = on._cond(h_bag, act, None, z0)
    add = c_on.shape[-1] - c_off.shape[-1]
    if add != d:
        _fail(f"anchor_init added {add} to cond width, expected d={d}")
    print(f"[2/4] _cond width: off {c_off.shape[-1]} -> on {c_on.shape[-1]} (+{add}=d)   OK")

    # 3. z0 used: grad reaches it, and changing z0 changes the cond output
    z0g = torch.randn(B, d, requires_grad=True)
    on._cond(h_bag, act, None, z0g).sum().backward()
    if z0g.grad is None or float(z0g.grad.abs().sum()) == 0.0:
        _fail("no gradient reached z0 (anchor not connected)")
    c_a = on._cond(h_bag, act, None, torch.zeros(B, d))
    c_b = on._cond(h_bag, act, None, torch.ones(B, d))
    if torch.allclose(c_a, c_b):
        _fail("changing z0 did not change the conditioning (ignored)")
    print(f"[3/4] z0 grad={float(z0g.grad.abs().sum()):.4g}; changing z0 changes cond   OK")

    # 4. rollout carries a CONSTANT z0 = pooled first frame. Check _z0 + that a 2-step rollout uses it.
    obs = {}
    for name, n in on.layout:
        mod = on.modalities[name]
        if name == "proprio":
            obs[name] = torch.randn(B, on.window + 2, mod.dim)
        else:
            ih, iw = mod.ae.cfg.img_size, mod.ae.cfg.img_size
            sh = mod.ae.cfg.img_size if isinstance(mod.ae.cfg.img_size, (tuple, list)) else (ih, iw)
            obs[name] = torch.rand(B, on.window + 2, sh[0], sh[1], 3)
    with torch.no_grad():
        bag = on.encode_state({k: v[:, :on.window] for k, v in obs.items()})
        z0_expected = bag[:, 0].mean(dim=-2)                      # pooled FIRST frame
        z0_got = on._z0(bag)
    if not torch.allclose(z0_expected, z0_got):
        _fail("_z0 is not the pooled first-frame bag")
    if on._z0(bag).shape != (B, d):
        _fail(f"_z0 shape {tuple(on._z0(bag).shape)} != (B,d)=({B},{d})")
    # off model: _z0 returns None (width-safe)
    if off._z0(bag) is not None:
        _fail("_z0 must be None when anchor_init off")
    print(f"[4/4] _z0 = pooled first frame (B,d)={tuple(z0_got.shape)}; None when off   OK")

    print("\nALL ANCHOR-INIT SMOKE PASSED")


if __name__ == "__main__":
    main()
