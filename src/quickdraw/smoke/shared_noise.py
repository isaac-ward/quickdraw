"""Smoke for decode_shared_noise (record §8.29 temporal-coherence lever).

The mechanism: a stochastic flow decode draws noise per (batch,time) element; sharing it across the time axis
makes the SAMPLED sparkle temporally constant. This tests the noise plumbing DIRECTLY, with an identity
velocity net (output == the noise it was handed), so it depends only on how noise is drawn, not on any trained
weights -- a randomly-initialised real decoder zero-inits its residual blocks and barely uses its noise input,
so it cannot exercise this. The leading dim is the flattened (batch*time), time-contiguous.
  - share_noise_over=1 -> iid per frame -> output DIFFERS across time (both the initial eps and every renoise).
  - share_noise_over=T -> one draw reused across the T frames -> output IDENTICAL across time.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.shared_noise
"""
import torch
torch.set_num_threads(1)

from ..models.flow import TransportHead


class _Id(TransportHead):
    """param=x0 head whose velocity RETURNS its noise input, so _sample's output is exactly the (shared or iid)
    noise it drew -- initial eps plus the k-loop renoise. Nothing trained stands between the draw and the output."""
    def __init__(self):
        super().__init__(param="x0", shortcut=False, event_dims=3)

    def velocity(self, x, temb, cond, demb=None):
        return x


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def main():
    torch.manual_seed(0)
    h = _Id().eval()
    B, T = 2, 4
    event = (3, 5, 3)                       # (H,W,C)-shaped event
    cond = torch.zeros(B * T, 1, 4)         # only device/dtype/shape[0]=M are used; velocity ignores it
    kw = dict(event_shape=event, lead=(B * T,), steps=3, deterministic=False)

    def tvar(x):                            # reshape (B*T,*event)->(B,T,*event); max variation across time
        x = x.reshape(B, T, *event)
        return float((x - x[:, :1]).abs().max())

    with torch.no_grad():
        torch.manual_seed(1); iid = h._sample(cond, share_noise_over=1, **kw)
        torch.manual_seed(1); shared = h._sample(cond, share_noise_over=T, **kw)

    print(f"time-variation  iid={tvar(iid):.4g}  shared={tvar(shared):.4g}")
    if not (tvar(iid) > 1e-3):
        _fail("iid draw did not vary across time -- the test's own noise is not reaching the output")
    if tvar(shared) > 1e-6:
        _fail(f"share_noise_over=T still varies across time ({tvar(shared):.4g}) -- noise not shared")
    print("shared -> identical across time (eps + every renoise); iid -> differs   OK")

    # deterministic (eps=0) is trivially time-constant, and share is irrelevant there.
    with torch.no_grad():
        det = h._sample(cond, share_noise_over=1, event_shape=event, lead=(B * T,), steps=3, deterministic=True)
    if tvar(det) > 1e-6:
        _fail("deterministic sample varies across time -- unexpected")
    print("deterministic is time-constant   OK")

    print("\nALL SHARED-NOISE SMOKE PASSED")


if __name__ == "__main__":
    main()
