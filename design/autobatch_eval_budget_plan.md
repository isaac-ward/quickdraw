# Plan: make GPU memory observable, then size the batch from it

ONE plan, four ordered steps, one approval. Written 2026-08-18 against `src/quickdraw/training/setup.py`
(`autobatch_find`, ~234–375), `src/quickdraw/logging/callback.py:475`, `conf/data/torus.yaml:33–43`.
No code changed yet.

## The thing that invalidates most of my earlier reasoning

**Nothing ever resets the CUDA peak-memory counter.** `callback.py:475` logs
`torch.cuda.max_memory_allocated()` and there is no `reset_peak_memory_stats()` anywhere in the training path
(`_probe_path` resets before each probe, but `done()` never resets after the last one). So `mem/peak_gb` is a
monotonic running max over the WHOLE PROCESS, mixing:

* autobatch's own rejected probes — for `bsp32mse` that included **82.2 GB at batch 16**
* the sanity-check pass
* training
* eval

`bsp32mse`'s 45.1 GB is therefore `max(last probe 41.2, training, eval)` and nothing finer. Consequences:

1. **The "+3.8 to +4.9 GB eval overhead" across 8 runs is NOT a measurement of eval.** It is
   `epoch_peak − probe_peak`, i.e. "how much anything exceeded the probe". If eval's true peak sat below the
   probe's, that delta came from training, not eval.
2. **We do not currently know what eval costs.** So it cannot be used to calibrate an eval probe, and it cannot
   be used to size a reserve. Any plan that starts by tuning a margin is building on sand.

Everything below is ordered so that measurement precedes tuning.

## Step 1 — make memory observable (prerequisite)

*Nothing else in this plan is trustworthy until this lands.*

* `logging/callback.py`: call `torch.cuda.reset_peak_memory_stats()` at (a) `on_train_epoch_start` and (b) the
  top of the eval block in `on_train_epoch_end`; log **`mem/peak_train_gb`**, **`mem/peak_eval_gb`** and the
  `reserved` counterpart of each (`torch.cuda.max_memory_reserved`). Keep `mem/peak_gb` as the process max for
  continuity.
* `training/setup.py` `done()`: reset peak stats after the last probe, so autobatch stops contaminating every
  downstream reading.

**Gate:** on one short run, `mem/peak_train_gb` must land near the probe value, and `mem/peak_eval_gb` must be a
real independent number. That number — currently unknown — is the input to Steps 3 and 4. If eval turns out to
be well under training, Step 4's gate is cheap insurance; if it is above, it is the binding constraint and the
whole budgeting story changes.

## Step 2 — two correctness bugs, independent of sizing

* **`confirm_compiled` can return a batch it measured as NOT fitting** (setup.py:337–339): it steps down by 8,
  and when `b - 8 < 1` it logs "over budget down to batch {b}; using {b}" and returns `b`. At b=8 — every
  bespoke run — that is the branch taken. Fix: step down by a predicted delta, and never return an over-budget
  batch (return the last known-good, else raise).
* **We budget against ALLOCATED but OOM is driven by RESERVED.** `_probe_path` returns
  `max_memory_allocated`; fragmentation lives in the gap. On record: "batch 112 probed 81 GB then OOM'd at
  93 GB". Fix: return both, budget on reserved.

**Gate:** probe-only regression (no training) on three configs whose answers we know — `bsp32mse` (chose 8),
`bsp32mse_sharp` (chose 8, probe 60.3 GB), `anch128` (chose 32). **Record the reserved/allocated ratio**; this
is what says whether the batch moves up or down, and it is currently unmeasured. Expect the batch to DROP here,
before Step 4 gives any of it back.

## Step 3 — linear-fit sizing (core, not optional)

`peak(B) = a*B + b`. Measured on `bsp32mse`: 41.2 GB @ B=8, 82.2 @ B=16 → slope **5.125 GB/sample**, intercept
**0.20 GB** (consistent with 16 B/param × 6.37M = 0.10 GB of weights+grads+Adam).

1. analytic `b` = `16 * n_params` bytes, logged beside the fitted intercept; a large disagreement means the
   linear model does not hold for this config → fall back.
2. probe `B0 = autobatch_base` and one more distinct point (halving if B0 OOMs) → solve `a`, `b_fit`.
3. `B* = clamp(floor((budget - b_fit)/a), 1, autobatch_max)`.
4. confirm at `B*`; on failure step by the **predicted** correction `ceil((peak-budget)/a)` and re-fit. Cap at
   2 corrections, then fall back to the existing bisection (kept as `_bisect_fallback()`, no new knob).

**Why this rather than bisection — the reason is safety, not speed.** Bisection's base-fits branch walks a
doubling ladder (32, 64, 128, … `autobatch_max=512`) and every rung is a real allocation: batch 16 already
reserved 82.2 GB of a 95.8 GB card here, and a proprio-only config climbs to the cap. On a shared box that
threatens the CO-RESIDENT run. A fit probes small and jumps to the answer, never allocating far above what it
will choose. It also yields a reusable memory MODEL (GB/sample) instead of a single search result — useful for
predicting a config's cost without probing at all. The ~60–100 s of startup saved is incidental (0.1% of a run).

## Step 4 — eval gate, and ONE margin instead of two

The realisation that simplifies this: **eval memory does not depend on `data.batch`.** It is fixed by the eval
config (`ood_horizon`'s hardcoded `n_ep=8` at routines.py:133, `ae_floor_episodes`, `eval.horizon`,
`closed_loop_steps`) and by the model. So eval is a **feasibility gate, not a subtraction** from the training
budget:

```
choose B  s.t.  train_peak(B) <= budget
require         eval_peak     <= budget        # checked once at startup
budget        = total - autobatch_reserve_gb
```

* `probe_eval()` inside `autobatch_find`, reusing `synth()` for shapes so it needs no dataset: encode P context
  frames for `n_ep` episodes → `_rollout(eval.horizon)` under `no_grad` → `to_obs` decode every frame → hold
  pred+true in fp32 as `image_curves` does → `max_memory_reserved`. **Calibrate against Step 1's
  `mem/peak_eval_gb`, NOT against the contaminated 45.1 GB.** If it is not within ~2 GB of the real isolated
  eval peak, it is not modelling the eval path and must not be used as a gate.
* On `eval_peak > budget`: raise loudly, naming `eval.horizon`, `eval.closed_loop_steps`, `ae_floor_episodes`,
  `manifold`.
* **Collapse the two reserves into one.** `autobatch_headroom` (fraction) and my proposed `min_spare_gb`
  (absolute) covered the same thing, and after Steps 2 and 4 neither has its original job left: fragmentation is
  now MEASURED (Step 2), and the big single eval allocation is now GATED (Step 4). What remains is one residual —
  allocator growth over a full epoch beyond a 2-iteration probe, evidenced at **+12 GB** by the batch-112
  incident. That is a property of the allocator, not of the card, so it must be **absolute**:
  **delete `autobatch_headroom`, add `autobatch_reserve_gb`** (initial 12, then set from Steps 1–2). Deleting the
  key is deliberate: the five launch scripts still passing `autobatch_headroom=0.35` will fail LOUDLY rather
  than silently doing nothing.

## Final verification, after all four steps

1. Probe-only regression on the three known configs; chosen batch and the fitted slope/intercept logged.
2. One 2-epoch live run with evals on: assert `mem/peak_train_gb` and `mem/peak_eval_gb` are both below
   `total - reserve`, and no eval OOM.
3. Only then use it for real experiments.

## Expected outcome — deliberately not promised

Step 2 pushes the batch DOWN (reserved > allocated), Step 4 pushes it UP (one honest margin instead of a 24 GB
fraction). **The net is unknown until Step 1 and 2 are measured** — somewhere between the current 8 and the 14
the res-1 bisection fix already yields. My earlier "batch 16" assumed allocated-based probes and is withdrawn.

The reliable wins are correctness and portability, not throughput: a batch that is never accepted while
over-budget, an eval OOM that surfaces at startup instead of mid-run, memory numbers that mean something, and a
reserve in absolute units that travels to a different GPU.

## Rollback

`data.autobatch=false data.batch=<n>` bypasses all of it. Steps 1 and 2 are independently revertable; Step 3
keeps the old bisection as a live fallback.

## Sequencing constraint

`autobatch_find` runs only at process start, so landing this cannot disturb a run already training — but a
launch DURING the edit would pick up half of it. Step 1 touches `logging/callback.py`, which IS imported at
startup, so it must not land while a run is between epochs. Land while nothing is starting, or after A/B finish.
