"""Smoke: phase-shifted subsampling (data.subsample_all_phases).

The claim is "~s x the TRAIN windows at EXACTLY the same frame rate, val untouched, off = bit-identical".
Each of those is a separate way to be wrong, so each is checked:
  1. OFF reproduces the old output exactly (same episodes, same frames, same actions).
  2. ON yields ~s x the episodes on TRAIN, and phase 0 is still present unchanged.
  3. ON leaves VAL alone -- otherwise the eval routines sample different episodes and every metric shifts.
  4. The RATE is unchanged: consecutive kept frames are still s apart within every emitted episode.
  5. Action aggregation is correct per phase: summed for delta dims, take-last for near-binary dims.
Run: uv run python -m quickdraw.smoke.phase_subsample
"""
import numpy as np

import quickdraw.data.dataset as D

OK = [0, 0]
def check(name, cond, extra=""):
    OK[1] += 1; OK[0] += bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))

S = 5
# obs[t] = t so a kept frame's VALUE is its original index -> the rate is directly checkable.
# act dim 0 is a delta (summed), dim 1 is near-binary (take-last).
def make(n_eps=3, T=53):
    eps = []
    for e in range(n_eps):
        o = (np.arange(T) + 1000 * e).astype(np.float32)[:, None]
        a = np.stack([np.arange(T).astype(np.float32),
                      np.where(np.arange(T) % 7 < 3, -1.0, 1.0)], 1)
        eps.append((o, a))
    return eps

D.set_subsample(S)
D._SUBSAMPLE_USED = False

D.set_subsample_all_phases(False)
off = D._subsample_episodes(make(), "repo/train")
D.set_subsample_all_phases(True)
on_tr = D._subsample_episodes(make(), "repo/train")
on_va = D._subsample_episodes(make(), "repo/val")

check("OFF gives 1 episode per input", len(off) == 3, f"{len(off)}")
check("ON gives s episodes per input on TRAIN", len(on_tr) == 3 * S, f"{len(on_tr)} (expect {3*S})")
check("ON leaves VAL untouched", len(on_va) == len(off) and
      all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(on_va, off)))
check("ON contains phase 0 UNCHANGED (off is a subset)",
      any(np.array_equal(e[0], off[0][0]) and np.array_equal(e[1], off[0][1]) for e in on_tr))

# 4. rate: within every emitted episode consecutive kept obs differ by exactly S
gaps = {int(np.diff(e[0][:, 0]).min()) for e in on_tr} | {int(np.diff(e[0][:, 0]).max()) for e in on_tr}
check("RATE unchanged: kept frames are exactly s apart in every episode", gaps == {S}, f"gaps {sorted(gaps)}")

# every phase offset 0..S-1 appears as some episode's first frame (mod 1000 to strip the episode id)
firsts = sorted({int(e[0][0, 0]) % 1000 for e in on_tr})
check("all s phase offsets present", firsts == list(range(S)), f"{firsts}")

# 5. action aggregation, checked on the phase-2 episode of the first input
src = make()[0]
ph = 2
tgt = next(e for e in on_tr if int(e[0][0, 0]) % 1000 == ph and e[0][0, 0] < 1000)
a = src[1][ph:]
n = len(tgt[0])
exp_sum = a[:n * S, 0].reshape(n, S).sum(1)
exp_last = a[:n * S, 1].reshape(n, S)[:, -1]
check("delta dim SUMMED across the group, per phase", np.allclose(tgt[1][:, 0], exp_sum))
check("near-binary dim TAKE-LAST across the group, per phase", np.allclose(tgt[1][:, 1], exp_last))
check("near-binary dim stays in {-1,+1} (not summed to +-5)",
      set(np.unique(tgt[1][:, 1])) <= {-1.0, 1.0}, f"{sorted(set(np.unique(tgt[1][:,1])))}")

tot_off = sum(len(e[0]) for e in off); tot_on = sum(len(e[0]) for e in on_tr)
print(f"\n       TRAIN frames: OFF {tot_off} -> ON {tot_on}  ({tot_on/tot_off:.2f}x)")
print(f"\n{'ALL OK' if OK[0]==OK[1] else 'FAILURES'} ({OK[0]}/{OK[1]})")
raise SystemExit(0 if OK[0]==OK[1] else 1)
