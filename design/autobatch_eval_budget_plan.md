# Plan: eval-aware memory budgeting + linear-fit autobatch

Implementation plan only — NO code changed. Written 2026-08-18 against `src/quickdraw/training/setup.py`
(`autobatch_find`, lines ~234–375) and `conf/data/torus.yaml:33–43`.

## What the code does today

```
budget      = total_VRAM * (1 - autobatch_headroom)          # headroom 0.25 (conf/data/torus.yaml:37)
probe(B)    = max(seq_path, par_path) of torch.cuda.max_memory_ALLOCATED   # setup.py:310-311
              after 2x (fwd -> full loss -> bwd -> AdamW.step) so Adam states allocate
search      = base 16; if it fits -> double to bracket, bisect (res 1 since 2026-08-18)
                       if not     -> halve until something fits, bisect upward (res 1)
confirm     = confirm_compiled(b): re-probe on the compiled step, step DOWN BY 8 on failure
eval memory = never probed. Covered implicitly by `headroom`.
```

## The five problems, each with evidence

**P1 — the search is O(log n) probes when 2 suffice.** Peak memory is near-perfectly linear in batch:
measured on `bsp32mse`, 41.2 GB @ B=8 and 82.2 GB @ B=16 → slope **5.125 GB/sample**, intercept **0.20 GB**.
The intercept is tiny because at 6.37M params the weights+grads+Adam states are ~0.10 GB, so activations
dominate entirely. Each probe costs ~10 s eager, so today's 5–13 probes are 60–130 s of startup.

**P2 — we budget against ALLOCATED but OOM is driven by RESERVED.** `_probe_path` returns
`max_memory_allocated`; the allocator's reserved pool is larger and fragments over an epoch. The record already
documents this biting once: "batch 112 probed 81 GB then OOM'd at 93 GB". Today the fractional headroom absorbs
that difference silently instead of measuring it.

**P3 — `headroom` is a FRACTION, but what it covers is an ABSOLUTE constant.** The eval spike does not scale
with the card or with the batch — it is fixed by the eval config and the model. Measured overhead of the epoch
peak above the training probe, across 8 healthy runs: **+3.8 to +4.9 GB** (anch256 +4.9, bsp32 +3.9, bsp16 +4.2,
bsp64 +3.8, ...). At 0.25 on a 95.8 GB card we reserve **24 GB** for a ~4.5 GB phenomenon — a 5.3x margin whose
size is an accident of the card, not of the workload.

**P4 — `confirm_compiled` can return an OVER-BUDGET batch.** setup.py:337-339: on failure it steps down by 8,
and `if b - 8 < 1` it logs "compiled step over budget down to batch {b}; using {b}" and **returns b anyway**. At
b=8 (every bespoke run) that is exactly the branch taken, so a compiled step that does not fit is accepted. The
step-down granularity of 8 is also inconsistent with the res-1 search above it.

**P5 — eval is never probed.** Largest single eval allocation observed in the logs: **11.87 GiB** (and manifold
alone asked for 8.07 GiB). We only find out at eval time, and the mitigation on record is "raise
data.autobatch_headroom if an eval OOMs" — i.e. hand-tune a fraction until the symptom stops.

## Plan A — linear-fit sizing (replaces the search)

Key insight to exploit: `peak(B) = a*B + b`, where `b` is persistent (weights + grads + 2 Adam moments) and `a`
is per-sample activations.

1. **Analytic `b`**: `16 bytes * n_params` (fp32 weights + grads + m + v). Sanity-check it against the fitted
   intercept and log both; a large disagreement means the linear model is wrong for this config and we should
   fall back.
2. **Two probes**: `B0 = autobatch_base`. If `probe(B0)` OOMs, halve until it returns (as today). Then probe a
   second distinct point (`B0/2` if B0 fit, else the next halving) → `a = Δpeak/ΔB`, `b_fit = peak(B0) - a*B0`.
3. **Predict**: `B* = clamp(floor((budget - b_fit) / a), 1, autobatch_max)`.
4. **Confirm and correct**: probe `B*`. If over budget, do not step by a constant — step by the *predicted*
   correction `ceil((peak - budget)/a)` and re-fit with the new point. Cap at 2 corrections.
5. **Fallback**: if 2 corrections still fail, call the existing bisection unchanged. Keep it as
   `_bisect_fallback()` so behaviour is recoverable without a config knob.

Probe count 2 + 1 confirm = **3**, vs 5–13. Expected choice for `bsp32mse` at headroom 0.25: **batch 14**
(vs the 8 it ran at).

## Plan B — eval-aware budgeting (replaces headroom-as-a-fudge)

The realisation that makes this simple: **eval memory does not depend on `data.batch`.** It is set by the eval
config (`ood_horizon` uses a hardcoded `n_ep=8` at routines.py:133, plus `ae_floor_episodes`, `eval.horizon`,
`eval.closed_loop_steps`/`closed_loop_horizon`) and by the model. Training activations are freed before eval
runs and the allocator reuses those blocks — which is exactly why the measured overlap is only ~4.5 GB and not
additive. So eval is a **feasibility constraint, not a subtraction from the training budget**:

```
choose B  s.t.  train_peak(B) <= budget          # unchanged
require        eval_peak      <= budget          # NEW: a hard gate, checked once at startup
budget    = min( total*(1-frag) , total - autobatch_min_spare_gb )
```

1. **New `probe_eval()`** inside `autobatch_find`, reusing the existing `synth()` for shapes so it needs no
   dataset: encode `P` context frames for `n_ep` synthetic episodes → `model._rollout(..., eval.horizon, 0.0,
   None, 0)` under `no_grad` → `to_obs` decode every frame → hold `pred` and `true` in fp32 as `image_curves`
   does. Measure `max_memory_reserved`. Cost ~20–40 s once.
2. If `eval_peak > budget`: **log loudly and raise**, naming the knobs that shrink it (`eval.horizon`,
   `eval.closed_loop_steps`, `ae_floor_episodes`, or disable `manifold`). This is the failure the docstring
   currently tells you to paper over with headroom.
3. With eval explicitly gated, `headroom` reverts to its real job — allocator fragmentation only — so
   **0.25 → 0.10**, and it should be measured on `max_memory_reserved` (P2) rather than assumed.
4. **New safety floor** `autobatch_min_spare_gb: 12`, so that a single large eval allocation (11.87 GiB
   observed) still fits even if the fractional term says otherwise. This is the belt to the fraction's braces,
   and it is expressed in the same units as the thing it protects against.

Expected effect for `bsp32mse`: budget 71.8 → 86.2 GB, batch 14 → **16**.

## Also fix while in here (P4)

`confirm_compiled`: step down by the predicted correction rather than 8, and **never return an over-budget
batch** — return the last known-good, or raise if none. Today at b≤8 it returns a batch it just measured as not
fitting.

## Precise edit sites

| file | site | change |
|---|---|---|
| `training/setup.py` | `_probe_path` ~310 | return reserved as well as allocated; budget against reserved |
| `training/setup.py` | new `_persistent_bytes()` | analytic 16 B/param, logged next to the fitted intercept |
| `training/setup.py` | new `probe_eval()` | eval-shaped forward, `max_memory_reserved` |
| `training/setup.py` | ~345–375 | replace both search branches with fit+predict; keep bisection as fallback |
| `training/setup.py` | `confirm_compiled` ~325 | predicted step-down; never return over-budget |
| `conf/data/torus.yaml` | :37 | `autobatch_headroom` 0.25 → 0.10, comment says fragmentation-only |
| `conf/data/torus.yaml` | new | `autobatch_min_spare_gb: 12` |
| eval routines | — | **no change** |

## Verification, in order — do not skip to live runs

1. **Offline arithmetic**: the fit must reproduce the numbers already measured — batch 12 @ headroom 0.35,
   14 @ 0.25, 16 @ 0.10 for `bsp32mse`'s slope/intercept.
2. **Probe-only regression**, no training: call `autobatch_find` for three configs whose answers we know —
   `bsp32mse` (ran at 8), `bsp32mse_sharp` (ran at 8, probe 60.3 GB), `anch128` (ran at 32) — and assert the new
   choice is ≥ the old one and the compiled confirm passes.
3. **`probe_eval` calibration**: compare `eval_peak` against the observed epoch peaks (45.1 GB for `bsp32mse`
   at batch 8). If the probe is not within ~2 GB it is not modelling the real eval path and must be fixed
   before it is trusted as a gate.
4. **One short live run**: 2 epochs with evals on at the new batch; assert `mem/peak_gb < total - min_spare`
   and no eval OOM.
5. Only then use it for real experiments.

## Risks and rollback

- **Tail risk**: `bsp32widest` overshot its probe by **+38 GB** (43.6 → 81.8). That run had diverged, so I do
  not think it generalises, but it is the one datapoint saying the overhead is not always ~4.5 GB. The
  `min_spare_gb` floor plus a reserved-based measurement are the mitigations; if it recurs, raise `min_spare_gb`
  rather than the fraction.
- **Linearity assumption** breaks if a config is dominated by fixed cost (very large model, tiny batch). The
  analytic-vs-fitted intercept comparison detects that, and the bisection fallback covers it.
- **Rollback** is `data.autobatch=false data.batch=<n>`, which bypasses everything.
- Do NOT land this while runs are starting: `autobatch_find` runs only at startup, so live runs are unaffected,
  but a launch during the edit would pick up half the change.
