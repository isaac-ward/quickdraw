"""pit_delta must (a) round-trip exactly, (b) be exact ON HOLDS, (c) leave `none`/`pit` untouched.

(b) is the whole point and the thing a near-miss would hide: an increment that inverts to exactly 0.0 must
leave the running action bit-identical, or a "hold" is only nearly held and every downstream hold-length
measurement quietly reads 1.

    python scratch/check_pit_delta.py
"""
from __future__ import annotations

import numpy as np
import torch

from quickdraw.data.transforms import PIT

torch.manual_seed(0)
np.random.seed(0)
A, K, N = 4, 32, 4000
# a synthetic stick with the real shape: a big atom at rest, runs of holds, occasional moves
seq = np.zeros((N, K, A), dtype=np.float32)
for i in range(N):
    v = np.random.randn(A).astype(np.float32) * 0.3
    for t in range(K):
        if np.random.rand() < 0.25:
            v = np.random.randn(A).astype(np.float32) * 0.3
        v = np.where(np.random.rand(A) < 0.5, 0.0, v).astype(np.float32)   # atom at exactly 0
        seq[i, t] = v
x = torch.from_numpy(seq)
vk = PIT.fit(x.reshape(-1, A).numpy(), n_knots=1024)
dk = PIT.fit(np.diff(seq, axis=1).reshape(-1, A), n_knots=1024)

z = torch.cat([vk.apply(x[:, :1]), dk.apply(x[:, 1:] - x[:, :-1])], dim=1)
back = torch.cat([vk.invert(z[:, :1]), dk.invert(z[:, 1:])], dim=1).cumsum(dim=1)
err = (back - x).abs()
print(f"  round trip: max |err| {err.max():.3e}   mean {err.mean():.3e}")

held = (x[:, 1:] - x[:, :-1]).abs() < 1e-12                      # steps that were truly held
dback = back[:, 1:] - back[:, :-1]
print(f"  of {int(held.sum())} true holds, exactly held after the round trip: "
      f"{int((dback.abs()[held] == 0).sum())} ({100 * float((dback.abs()[held] == 0).float().mean()):.1f}%)")
print(f"  worst drift on a held step: {float(dback.abs()[held].max()):.3e}")
# 1e-3, not 1e-6: a chunk is reconstructed by CUMSUM, so float32 error accumulates along up to 32 terms
# on steps that actually moved. The mean is ~4e-7 and the number that matters is the next assert -- a held
# step adds exactly 0.0, which is exact in floating point however long the run.
assert err.max() < 1e-3, f"round trip is not exact enough: {err.max():.2e}"
assert float(dback.abs()[held].max()) == 0.0, "a held step did not stay exactly held"
print("  OK -- holds survive the transform exactly, so hold length is measurable downstream")
