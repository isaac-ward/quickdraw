# AR-rollout throughput: is the "attention thing" a real speedup? — investigation + plan

**Status:** investigation, 2026-08-05. Question raised from other-machine perf probes (robocasa/starling):
~3.9 s/batch, GPU at **0–13% util**, ~12.7 days for 30 epochs at batch 32 / F 64. Is there a large speedup
available by compiling FlexAttention, or is that the wrong bottleneck?

**Bottom line up front:** there are TWO competing diagnoses that imply DIFFERENT fixes, and one of them is
contradicted by this repo's own measured record. **Do not implement a fix until Phase 0 (profiling)
resolves which bottleneck is real.** That is this repo's own rule (`design/accelerations.md`: *"PROBE —
never infer; never infer memory/cost from the cheap epoch-0 phase"*).

> ## ✅ PHASE 0 RESULT (2026-08-05) — resolved. Diagnosis **B** is correct.
> Profiled the real AR step (`scripts/profile_rollout.py`, joint mm_flow d=128/F=64, dedicated H100). See
> `design/accelerations.md` **Experiment 8** for the full numbers.
> - **FlexAttention is ALREADY FUSED** (`FlexAttentionAutogradOp` + flash-SDPA, no `B×H×T×T`). Diagnosis A's
>   "eager reference path" premise is **false**; there is **no attention-compile speedup to get**.
> - **The step is DISPATCH-bound:** Self CPU 3.36 s ≫ Self CUDA 0.68 s (GPU ~20% busy). The cost is *calling*
>   the fused ops ~256× through the serial F-step Python loop — exactly Diagnosis B / Exp 5–7.
> - **Immediate free win: raise `data.batch`** (dispatch-bound ⇒ nearly free): 32→96 = **2.7× throughput**
>   (+10% per-step, 68 GB). The bigger lever (its own project) is killing the per-step dispatch via
>   CUDA-graph / compiled+`reduce-overhead` rollout — Phase 1B below.

---

## The two diagnoses (they conflict)

**Diagnosis A — "unfused attention / launch-bound"** (other-machine AI, 2026-08-05):
FlexAttention is meant to be fused by `torch.compile` into one block-sparse Triton kernel. Multimodal models
**skip `torch.compile`** (`train_world_model.py:139`, `not cfg.model.get("modalities")`), so attention falls
back to the eager reference path that materializes `B×H×T×T` scores + many small kernel launches → CPU is
pegged, GPU idle. Fix: compile the attention with pinned shapes. Predicts a **3–5×** win.

**Diagnosis B — "serial F-step rollout / latency-bound"** (`design/accelerations.md`, Exp 4–7, MEASURED):
The AR training step (`rollout_train`, p_tf<1) is a **sequential F-step loop** — step *t+1* depends on step
*t* — running the backbone (+ image decode/re-encode) at every step, held for BPTT. The GPU is idle because
it waits on that serial dependency chain, not because attention is unfused. Direct quotes:
- Exp 5 (`accelerations.md:180`): *"Batch barely speeds AR epochs — they're latency-bound, not
  throughput-bound … the per-batch AR rollout is a sequential F-step loop whose wall-time rises with batch."*
- Exp 7 (`accelerations.md:293`): this exact config family (mm_flow d=64, in-rollout) measured at
  **~15–20% GPU util**; raising batch *"mainly raises utilization … rather than cutting wall-clock
  proportionally."*
- Named levers: **F, max_epochs, activation-checkpointing, the contraction penalty** — NOT attention fusion.

**Unresolved sub-fact:** is attention even unfused for mm models? The comments contradict each other:
- `train_world_model.py:141`: "multimodal … skip compile … FlexAttention still runs, **just eager**."
- `launch_shootout.sh:8`: "FlexAttention … **self-compiles even in the eager rollout**."
If FlexAttention self-compiles its kernel regardless of whole-model compile, A's premise is largely false.

Both diagnoses are consistent with the observed "batch is nearly free," but for OPPOSITE reasons (A: launch
overhead amortizes over samples; B: batch fills the serial loop's idle slots). So "batch is free" does not
discriminate between them — only a profile does.

---

## History — what was already tried, and why it didn't work

1. **A compiled rollout was the original design** (`design/training.md:42`: "one `torch.compile`d rollout fn
   shared by training/eval/control").
2. **It backfired:** the AR rollout hit ~57 distinct sequence lengths → blew past dynamo's recompile cache →
   **~18× slower** (`train_world_model.py:143`).
3. **That was fixed — via exactly the "pin the shape" idea.** The training rollout now feeds a CONSTANT length
   every step via front-padding: `pad_block_mask` (`transformer.py:69`) used in `rollout_train`
   (`multimodal.py:224`), with the ~57 pad-variant masks cached under `cache_size_limit=256`
   (`train_world_model.py:79`). `launch_shootout.sh:56-57`: *"the per-shape recompile thrash … is FIXED
   (fixed-window rollout + cache_size_limit)."* **So the fixed-shape restructure a naive plan would propose
   already exists in the code.**
4. **Whole-model compile is deliberately skipped for multimodal** (`train_world_model.py:139`): "the
   per-batch image gather + ViT AE complicate it." (Avoided; not clearly tried-and-failed — Phase 0 should
   check whether a NARROW compile of just the backbone/attention sidesteps this.)
5. **Hard constraint:** FlexAttention has **no double-backward under `torch.compile`** (`variations.md:178`),
   so the contraction penalty (`variations.contraction`, off by default) requires the eager `attn_eager`
   (SDPA-MATH) path. Any attention-compile must keep an eager fallback / be gated off when contraction is on.
6. **max-autotune → default mode** (the forward is used ~1 epoch under the p_tf curriculum; the long kernel
   search isn't worth the multi-minute startup).

---

## Phase 0 — PROFILE (the gate). Resolve A vs B before touching the hot path.

Goal: attribute the per-batch AR time and settle "is attention fused?". No fix ships until this is done.

**Harness.** Run a handful of `rollout_train` steps at `p_tf=0` (the steady AR regime) on the real config
(`mm_flow`, d=128, F=64, image head) under `torch.profiler` + a manual per-section timer, plus a batch
sweep {32, 64, 128}. Keep it short (few steps) so it can co-locate on spare GPU headroom without disturbing a
live run (the runs sit at ~30 GB / 15–20% util, so there's room — note co-location inflates absolute timing;
the ATTRIBUTION and the fused-vs-eager answer are what matter and are robust to it).

**Measure / attribute:**
1. **Is `flex_attention` fused?** Profiler kernel trace: a block-sparse FlexAttention Triton kernel (fused)
   vs an explicit `bmm`+`softmax`+`bmm` with a `B×H×T×T` score allocation (eager reference). This settles the
   `train_world_model.py:141` vs `launch_shootout.sh:8` contradiction directly.
2. **Per-batch breakdown:** attention fwd/bwd, image-AE encode, image decode, token-bag assembly, MLP, and
   the Python per-step loop overhead (CPU gaps between kernels = launch-bound signature).
3. **GPU util + CPU:** is a single CPU core pegged while the GPU idles? (launch-bound signature).
4. **Batch scaling:** s/batch at 32/64/128 — ~flat ⇒ launch-bound (A); rises ~per-sample ⇒ compute/latency
   (B). (The other machine saw +5% for 2× batch = nearly free; `accelerations.md` Exp 5 saw the per-batch
   time ~double so net ~15%. This is config-dependent — MEASURE on the target config.)

**Decision rule:**
- Attention UNFUSED **and** a large share of time in attention **and** CPU-gap-dominated → **A** → Phase 1A.
- Attention FUSED (or small share) **and** time in the serial loop / image-AE / F-step serialization → **B**
  → Phase 1B.
- Realistically a mix; the profile quantifies each so the fix targets the biggest slice.

---

## Phase 1A — attention compile (ONLY if Phase 0 says attention-bound)

The fixed-window `pad_block_mask` already bounds the shapes; the missing piece would be compiling the
attention path for mm models. Do it NARROWLY — compile `SpaceTimeTransformer.forward` (or `flex_attention`
alone), **not** the whole model (that's what the image-gather/ViT-AE skip was about). Expect ~57 one-time
pad-variant compiles (cached; a slow first epoch that amortizes). Keep the eager `attn_eager` fallback and
gate the compile off when `variations.contraction` is on (double-backward constraint).

## Phase 1B — serial-loop levers (if Phase 0 says latency-bound)

- **Batch-for-utilization** — free memory-wise; measure the ACTUAL wall-clock delta on the target config (the
  A/B disagreement above means don't assume 2×).
- **Activation-checkpoint the rollout** (`torch.utils.checkpoint`, per detach-segment) — decouples F from
  memory (~1.3× compute), lets F grow (`accelerations.md` Exp 6). Helps memory/F, NOT the ∝F time.
- **F / max_epochs** — direct but modeling tradeoffs (F is the BPTT horizon).
- **Contraction penalty + decode→encode carry** — cheaper long-horizon stability than brute-force F
  (`accelerations.md` Exp 6 verdict).

---

## Testing / iterative development

- **Numerical parity is the gate.** Any compile/restructure must reproduce the eager forward AND backward
  grads within tolerance, and leave val-loss + env metrics unchanged over a few epochs. A speedup that moves
  the loss is a bug, not a speedup.
- **Benchmark protocol:** time the steady 50%→75% batch delta (exclude first-batch/compile overhead), sample
  peak/steady mem live, record GPU + CPU util, and the first-epoch compile overhead separately. Always
  baseline-vs-change on the same config/seed.
- **Iterate:** land Phase 0 first; make the SMALLEST change Phase 0 justifies; re-benchmark; report the delta;
  repeat. Don't bundle a hot-path change into a training launch.

## What to log to `progress.log` (benchmark signals)

Beyond s/batch, GPU util, and first-epoch compile overhead:
- **samples/s** (throughput) + peak & steady GPU memory.
- a one-time **`[profile]` attribution dump** (attn / image-AE / rollout-loop / launch-gap shares).
- **recompile count / dynamo cache occupancy** (regression guard against a return of the ~57-shape thrash).
- **CPU-core utilization** (the launch-bound signature).
- **val loss + env metrics vs baseline** (the parity guard — must be unchanged).
- a **batch-scaling row** (s/batch at 32/64/128) so the latency-vs-throughput signature is on record.

## Consistency cleanup (do AFTER Phase 0 establishes ground truth)

The repo currently tells two stories about attention/slowness; make them one:
- Reconcile `train_world_model.py:141` ("just eager") with `launch_shootout.sh:8` ("self-compiles even
  eager") to whatever the profile shows.
- Annotate the `train_world_model.py:143` "~18× thrash" comment as the PRE-fix state (pad_block_mask +
  cache_size_limit fixed it) so it doesn't read as a current problem.
- Append a Phase-0 experiment row to `design/accelerations.md` with the attribution + the A-vs-B resolution.
