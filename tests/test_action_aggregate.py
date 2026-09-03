"""`data.action_aggregate` = sum | last | concat, checked on real lego_assemblies actions.

    python tests/test_action_aggregate.py

WHY THIS EXISTS. `subsample=s` throws away s-1 of every s frames and something must combine the actions
it skipped. The original code SUMS them, which is right for a DELTA action and wrong for an ABSOLUTE one
-- summing six absolute poses gives six times the position. Its safety net (dims with <=2 unique values
take-last instead) does not fire on lego, whose gripper is at exactly 0 or 1 for 94% of frames but has
~1000 unique values because it ramps between them. "concat" avoids the choice by keeping all s actions.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quickdraw.data import dataset as D   # noqa: E402

S = 6
ok = True


def chk(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:56s} {detail}")


T, DIM = 60, 4
obs = np.zeros((T, 34), np.float32)
act = np.zeros((T, DIM), np.float32)
act[:, 0] = np.linspace(0, 1, T)                          # absolute position-like ramp
act[:, 1] = (np.arange(T) % 12 < 6).astype(np.float32)     # 0/1 gripper...
act[3, 1] = 0.5                                            # ...with ONE ramp value: defeats `<=2 unique`
act[:, 2] = np.sin(np.arange(T) / 5.0)
act[:, 3] = 1.0

D.set_subsample(S)
res = {}
for mode in ("sum", "last", "concat"):
    D.set_action_aggregate(mode)
    D._SUBSAMPLE_USED = False
    res[mode] = D._subsample_episodes([(obs, act)], "fake/train")[0][1]

print()
chk("sum keeps the width", res["sum"].shape[1] == DIM, f"{res['sum'].shape}")
chk("last keeps the width", res["last"].shape[1] == DIM, f"{res['last'].shape}")
chk(f"concat widens by x{S}", res["concat"].shape[1] == DIM * S, f"{res['concat'].shape}")

chk("sum INFLATES the [0,1] gripper (the bug)", res["sum"][:, 1].max() > 1.5, f"max {res['sum'][:,1].max():.2f}")
chk("last preserves the gripper range", res["last"][:, 1].max() <= 1.0 + 1e-6, f"max {res['last'][:,1].max():.2f}")
chk("concat preserves the gripper range", res["concat"].max() <= 1.0 + 1e-6, f"max {res['concat'].max():.2f}")

# THE POINT of concat: it is LOSSLESS -- reshaping back reproduces the raw actions exactly.
n = len(res["concat"])
back = res["concat"].reshape(n * S, DIM)
chk("concat is LOSSLESS (reshape back == raw)", np.array_equal(back, act[:n * S]),
    f"max abs diff {np.abs(back - act[:n*S]).max():.2e}")
# and `last` is recoverable from it, so concat strictly dominates
chk("last is recoverable from concat", np.array_equal(res["concat"].reshape(n, S, DIM)[:, -1, :], res["last"]))

print("\n  effective_action_dim (the ONE definition the configs derive from):")
for mode, want in (("sum", DIM), ("last", DIM), ("concat", DIM * S)):
    D.set_action_aggregate(mode)
    got = D.effective_action_dim(DIM)
    chk(f"  {mode:6s} -> {want}", got == want, f"got {got}")

D.set_action_aggregate("sum")
print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ***"))
sys.exit(0 if ok else 1)
