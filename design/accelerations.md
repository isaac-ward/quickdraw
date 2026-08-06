# Accelerations: GPU footprint, throughput, and packing strategy

Empirical numbers for how the multimodal (proprio + image_fpv) world models fill an H100, how throughput
scales with batch size, and — the practical question — **what packing strategy gets the most runs done over
time**. Recorded 2026-07-01 so future launch decisions can be based on measurement, not guesswork.

> **PURPOSE — read this before choosing `batch` / `F` / model width (`d`, `num_tokens`, `patch`) for any run.**
> This is the running record of MEASURED GPU-efficiency data points on the H100s: how each config fills the
> card, where throughput saturates, and what packing/queuing finishes the most runs. Its job is to make the
> *next* tuning decision data-driven instead of a guess. **Append a row / experiment every time you measure a
> new config** (date it). When unsure, PROBE (see "How to probe" in Experiment 5) — never infer memory from the
> cheap epoch-0 phase.
>
> ⚠ **Version note:** Experiments 1–4 were recorded at **d=256**. The default `d` is now **32** (and a wide
> variant runs at **64**), so the absolute GB figures in Exp 1–4 are ~an order of magnitude high for today's
> models — trust Experiment 5 for the current d. The *scaling shapes and lessons* in Exp 1–4 still hold.

## Hardware & common setup

- **2× NVIDIA H100** (95.8 GB each, reported as ~96 GB), one Docker container (`quickdraw-app`).
- **Precision:** `bf16-mixed` (matches `conf/trainer/default.yaml`).
- **Model:** token-bag multimodal spine (`design/models/vision.md`) — proprio MLP trunk/head +
  128²-px ViT autoencoder image trunk/head (num_tokens=8, patch 16), factorized space-time backbone.
  d=256, heads=8, window=32. ~12–19M params depending on family.
- **Data:** GPU-resident frame loader (uint8 store + `index_select` gather, no host copy). `data.F=24`
  (training window), 128² frames.
- **Optimizer:** AdamW.

Experiments below: (1) single-run footprint, (2) throughput vs batch (controlled synthetic bench),
(3) the real 3-run packed layout, (4) autoregressive-rollout cost + the packing-OOM (the decisive one —
it supersedes the packing advice in Experiment 3 / the decision section).

---

## Experiment 1 — single-run GPU footprint (batch 96, training-only)

One run per GPU, sampled mid-epoch during steady training (no validation, no eval routines).

| Model | mem (GB) | util | notes |
|---|---:|---:|---|
| DSAR (proprio+image) | 24.6 | ~80–100% | plain data-space AR |
| LSAR + EMA + physical | 25.6 | ~80–100% | EMA target encoder + physical loss |
| Diffusion (proprio+image) | 33.1 | ~85–97% | per-token FlowField ≈ +8 GB / heavier step |

**Takeaway:** one run at batch 96 already pins the GPU near 100% util. Diffusion is ~8 GB heavier because
the rectified-flow FlowField + flow-consistency term roughly doubles the per-step work.

## Experiment 2 — throughput vs batch (controlled synthetic bench)

Replicates the epoch-0 step (parallel forward `m(obs,act)` + decode + backward + AdamW) under bf16 autocast.
Warmup 3 iters, timed over 8. Script: `src/quickdraw/scripts/bench_batch.py`. **Caveat:** this measures the shared
backbone+decode compute; it omits each family's extra loss terms (pred_latent, flow), so absolute
samples/s is optimistic — but the *scaling shape* is what matters and it is consistent across families.

**LSAR + EMA (19.2M params):**

| batch | s/step | samples/s | Δ vs prev | peak GB |
|---:|---:|---:|---:|---:|
| 96  | 0.1245 | 770.8 | —      | 19.3 |
| 192 | 0.2262 | 848.9 | +10.1% | 38.3 |
| 288 | 0.3293 | 874.6 | +3.0%  | 57.2 |
| 384 | 0.4344 | 884.1 | +1.1%  | 76.2 |

**Diffusion (11.9M params):**

| batch | s/step | samples/s | Δ vs prev | peak GB |
|---:|---:|---:|---:|---:|
| 96  | 0.1292 | 743.2 | —      | 19.8 |
| 192 | 0.2340 | 820.4 | +10.4% | 39.2 |
| 288 | 0.3406 | 845.6 | +3.1%  | 58.7 |

**Takeaway:** bigger batch **does** run faster — but only **~10–15% total**, and it **flattens by batch ~192**
(the "knee"). We are *near*-saturated: one run at batch 96 already extracts ~87% of the GPU's throughput
ceiling (770 / 884). Memory scales ~linearly: LSAR ≈ 25.6 GB at 96 real + ~4 GB per +32 batch.

> `nvidia-smi` "100% util" only means a kernel was running each sample — **not** peak FLOPs. That's why a
> bigger batch can still add a little throughput despite "100% util". The bench above is the real signal.

## Experiment 3 — the 3-run packed layout (batch 96)

GPU 0 packs **DSAR + LSAR** (two processes sharing the GPU); GPU 1 runs **diffusion** alone. Real full
training loop (200 epochs, eval every 10, un-trimmed control), including validation each epoch.

| GPU | runs | s/epoch | reserved mem (GB) |
|---|---|---:|---:|
| 0 | DSAR + LSAR (packed) | 139.5 / 151.1 | **64.7** (both) |
| 1 | diffusion (alone) | 163.6 | **73.1** |

- **Epoch times → ~8–9 h per 200-epoch run** (dsar eta 7:42, lsar 8:21, diffusion 9:02).
- **Reserved memory is far above the training-only profile** (64.7/73 GB vs 51/33 GB). Validation +
  checkpoint + the caching allocator's reserved (not just active) blocks inflate it. This is the number
  that matters for OOM headroom.
- **⚠ Eval headroom risk:** eval routines spike memory further (decoding the ~247-frame image rollout +
  rendering FPV in the MPPI control loop). Diffusion at 73 GB reserved has only ~23 GB of headroom going
  into the epoch-10 eval — **watch for OOM at the first eval epoch.**

---

## The decision: what gets the most runs done over time?

**Per-GPU sample-throughput has a fixed ceiling (~880 samples/s here).** You reach it *either* with one
large-batch run *or* with two packed small-batch runs that overlap and fill each other's scheduling gaps —
but you cannot *exceed* it. Packing therefore does **not** increase runs/hour; it only changes *when* runs
finish and *how much OOM risk* you take on.

Makespan for **3 runs on 2 GPUs** (T = one saturating run's wall-time):

- **1 run/GPU, queued** — GPU0: run1→run3 (2T); GPU1: run2 (T). Makespan **2T**, and **2 runs finish at T**.
- **Packed (2 on GPU0, 1 on GPU1)** — GPU0's two each run ~half-speed → both finish at ~2T; GPU1 at T then
  idle. Makespan **2T**, but only **1 run finishes early**.

Same makespan; single-per-GPU delivers more completed runs sooner and carries no OOM risk.

### Recommendation
1. **Max runs over time → ONE run per GPU at the batch "knee" (~192), run the rest as a queue.** Saturates
   each GPU, cleanest cross-run comparison, no packing overhead, no eval-OOM risk.
2. **Pack (2/GPU) only when you want every run's curves progressing simultaneously** for early comparison —
   accepting later completions and tighter memory. Keep packed runs at **batch 96** for eval headroom;
   do *not* raise batch on a packed GPU (eval will OOM).
3. **3+ runs per GPU: never.** Strictly more overhead and OOM risk with zero throughput gain.
4. **Bumping batch to "use all the memory" buys ~10–15% at most and flattens by ~192** — not worth the
   OOM risk on a packed GPU. Memory utilization is not the bottleneck; per-GPU compute throughput is.

### Rule of thumb
> The GPU's job/hour is fixed by its compute ceiling, reached at batch ≈192. To finish a *queue* of runs
> fastest, keep each GPU saturated with the **fewest** processes: one run per GPU at the knee, queue the
> rest. Pack only to watch many runs at once.

---

## Experiment 4 — autoregressive-rollout cost + the packing-OOM (the big one)

Discovered 2026-07-01 during the first real launch. The `p_tf` teacher-forcing curriculum ramps 1.0→0.0 over
`p_tf_warmup_epochs` (4). Epoch 0 (p_tf=1) is a cheap **parallel** forward; **from epoch 1 the training step
switches to sequential autoregressive rollout** (`rollout_train`) over the F future steps — running the backbone
(and for DSAR, a full ViT image decode+re-encode) at *every* step, held for BPTT.

Measured at batch 96, F=24, 1 run/GPU:

| epoch | mode | time/epoch | vs epoch 0 |
|---|---|---:|---:|
| 0 | parallel (p_tf=1) | ~110–165 s | 1× |
| 1+ | autoregressive (p_tf<1) | ~1000–1160 s (~17–19 min) | **~7–8×** |

- **This invalidated the first ETA.** `run_eta` computed from epoch 0 read ~8 h; the true figure was **~2 days
  for 200 epochs**. (Fix: `ProgressPrinter.run_eta` now uses the *current* epoch's time, so it self-corrects
  after epoch 1, and prints a projected finish clock-time.)
- **AR rollout balloons activation memory:** reserved memory jumped from ~25–33 GB (parallel) to **~65–73 GB
  per run** (AR). **vis_dsar was silently CUDA-OOM-killed** at the epoch-1 transition while packed on a shared
  GPU. So **packing is unsafe once AR rollout is active** — this supersedes Experiment 3's "pack for concurrent
  curves" note.
- **Dominant cost lever is F** (AR epoch ∝ windows/epoch × F). Cutting F is the main speed knob; we instead
  cut `max_epochs` 200→**50** (default) to keep wall-clock sane while holding F=24.

### Revised packing rule
- **One run per GPU, always, for anything using AR rollout** (all the world-model families). AR memory (~65–73 GB
  at batch 96, F=24) leaves no room for a second run.
- Packing only made sense under the (wrong) assumption that the parallel epoch-0 footprint held. It doesn't.
- For 3+ runs on 2 GPUs: **queue** them (see the queuing discussion), don't pack.

---

## Experiment 5 — larger image codec (d 64 / num_tokens 16 / patch 8) + the "probe the AR epoch" lesson (2026-07-02)

Context: the AE-only recon diagnostic (`logs/ae_recon_diag/`, encode→decode a val frame, no world model) showed
the d=32 / num_tokens=8 / patch=16 codec reconstructs FPV frames **blocky at ~26 dB** — the world model can't
beat that ceiling. A repeat run widened the codec: **d 32→64, num_tokens 8→16, patch 16→8** (head_dim 4→8).
Measured 128² frames, 1 run/GPU:

| model | d | head_dim | img_tok | patch | batch | F | epoch-0 mem | AR mem (ep≥1) | AR epoch | note |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| lsar / diff | 32 | 4 | 8 | 16 | 96 | 24 | low | fits | ~15 min | d=32 `vis_refactor` baseline |
| lsar | 64 | 8 | 16 | 8 | 48 | 16 | ~14 GB | ~19 GB (flat) | ~20 min | probe — only 21% of card, too conservative |
| diff | 64 | 8 | 16 | 8 | 48 | 16 | ~14 GB | ~20 GB (flat) | ~23 min | probe — 5× over-provisioned |
| lsar | 64 | 8 | 16 | 8 | 96 | 24 | ~30 GB | **~46 GB (flat)** | **~17 min** | ✅ CHOSEN for overnight; ~half the card, matches d=32 batch/F |
| diff | 64 | 8 | 16 | 8 | 96 | 24 | ~31 GB | **~48 GB (flat)** | **~18 min** | ✅ diffusion ~+2 GB heavier; still ~51% of card |
| lsar | 64 | 8 | 32 | 8 | 96 | 24 | ~44 GB | **~62 GB (flat)** | ~19 min | vis_refactor2 (iter 2, tokens 16→32 to de-blur); fits, ~65% card |
| diff | 64 | 8 | 32 | 8 | 96 | 24 | ~45 GB | **~66 GB (flat)** | ~21 min | vis_refactor2; ~70% card, ~30 GB headroom |

> ⚠ **Batch barely speeds AR epochs — they're latency-bound, not throughput-bound.** batch 48→96 (F 24)
> only cut the AR epoch ~20→17 min (~15%), NOT ~2×, even though it halves batches/epoch. Reason: an AR epoch's
> cost is `(batches/epoch) × (per-batch time)`, and the per-batch AR rollout is a **sequential F-step loop** whose
> wall-time rises with batch (more samples through the same serial depth) — so the two effects nearly cancel.
> Contrast Exp 2, where the *parallel* epoch-0 forward sped up with batch. **Takeaway: to speed AR epochs, cut
> `F` or `max_epochs` — raising batch mainly buys card-utilization + a bigger effective batch, not wall-clock.**
> (We still chose batch 96 for the clean A/B match to the d=32 baseline + better card use, not for speed.)

**THE LESSON — epoch-0 memory badly under-predicts the working set; always probe the AR epoch.** Two regimes:
epoch 0 (`p_tf=1`, parallel) is cheap; epoch ≥1 (`p_tf<1`, autoregressive) balloons (F sequential decode/re-encode
steps held for BPTT). But the AR balloon is a **bounded steady-state** — flat, not creeping — because
`detach_every` truncates the BPTT graph and `F` caps the rollout window. Here it settled at ~20 GB and stayed
there. Sizing off epoch-0 (~14 GB) *or* off intuition led to `batch=48/F=16` = **~21% of the 95 GB card, AR epochs
a slow ~20 min**. The fix was to relaunch at **batch 96 / F 24** (matches the d=32 baseline for a controlled A/B; measured ~46–48 GB
AR, ~half the card) to use the card properly and get a bigger effective batch. Note it only modestly sped the AR
epoch (~20→17 min, not ~2×) — see the latency-bound callout below.

### How to probe a new config (~7 min, cheap)
1. Launch the full config (self-limits to one run/GPU — see Exp 4).
2. `nvidia-smi --query-gpu=memory.used --format=csv,noheader` every ~10 s from epoch 0 **into epoch 1**. The
   epoch-1 (AR) value is the real working set; epoch-0 under-estimates it ~3–5×.
3. AR mem is a flat steady-state. Leave ~30% headroom for eval spikes + the allocator's reserved blocks.
4. AR mem ≪ 95 GB → raise `batch` (then `F`) and re-probe. OOM → drop `batch`, then `F`.

### Memory scaling rules of thumb
- AR activation ≈ **`batch × F × per_step`**. `per_step` grows with **`d`** (token width), **`num_tokens`**
  (image tokens carried through the backbone), and for the image AE the **patch-token count = `(img_size/patch)²`**
  — so `patch 16→8` is **4× more AE patch tokens**, the single biggest memory lever of the three codec changes.
- `detach_every` caps BPTT depth; `F` caps the rollout window — together they bound the balloon (why it's flat).
- Epoch time ≈ `(windows/batch) × per_batch_time`; on an under-utilized card (mem ≪ 95 GB) raising batch ~halves
  epoch time per doubling until compute saturates (the ~192 "knee" from Exp 2 was for d=256 — re-measure per `d`).

### GPU/kernel caveat — head_dim (`d / heads`)
**head_dim = 4** (d=32, heads=8) triggers a CUDA SDPA `invalid configuration argument` crash when the batch dim
is very large — hit in `control` eval, whose MPPI candidate batch is `n_episodes × num_samples × T`
(16 × 256 × ~32 ≈ 131k) inside the spatial attention. Fixed by chunking the spatial SDPA
(`spacetime.py` → `_SpatialAttention._M_CHUNK = 8192`, exact, no-op for training). **head_dim ≥ 8 (d ≥ 64) avoids
the fragility** — a secondary reason to prefer d=64.

### Eval-phase wall-clock — budget it, it dominates at checkpoints
- `control` MPPI: **~59 min/checkpoint at `n_episodes=16`** (~24 min at 4); scales with `n_episodes × num_samples`.
- `denoising_multistep` render: **~24–40 min** at full length (VTK torus renders, ~0.77 s/frame).
  `denoising_aggregate` ~90 s; `manifold` UMAP ~40 s (2k pts) / ~2–3 min (8k pts).
- At `every_epochs=20` over 100 epochs = ~5 checkpoints × the above **per run** → budget several hours of eval on
  top of training. Cap `eval.denoising_max_steps` or `control.n_episodes` if you need faster checkpoints.

---

## Experiment 6 — F (training rollout horizon): memory + time budget (2026-07-06)

Motivation: eval rolls **~247 steps open-loop**, but training uses **F=24**, so the model never learns long-horizon
stability → open-loop drift (in the FPV rollout, color goes wrong by ~+106 steps, then structure). Can we raise F?

**Memory scales with F, NOT `detach_every`.** In `MultiModalSequence._rollout_from` (`models/multimodal.py`) every
one of the F step-predictions is appended to `preds` and held for the *single* backward over the summed loss;
`detach_every` only `.detach()`s the CARRIED state (truncates gradient DEPTH for stability), it does **not** free the
resident forward activations. Empirically (confirms the linear fit in Exp 2/5):

    AR memory ≈ base + c · (batch · F)          # linear in batch·F; detach_every does not reduce it

So **F and batch trade linearly.** At the iter-2 codec (d=64, tokens=32, patch=8), measured AR = ~64 GB at
batch 96 / F 24. To hold ~64 GB while raising F, drop batch to keep `batch·F ≈ 2304`:

| target F | batch (≈const ~64 GB) |
|--:|--:|
| 24 (now) | 96 |
| 48 | 48 |
| 96 | 24 |
| 247 | ~9 (impractical — tiny batch hurts optimization) |

**The binding constraint is TIME, not memory.** AR epochs are latency-bound (time ∝ F; batch barely helps — see the
Exp 5 latency callout). At F=24 an epoch is ~19 min:

| F | ~epoch | ~100 epochs |
|--:|--:|--:|
| 24 | 19 min | ~1.3 day |
| 48 | ~38 min | ~2.6 days |
| 96 | ~76 min | ~5 days |
| 247 (eval horizon) | ~3.2 h | **~13 days (infeasible)** |

**Options to raise F:**
- **Activation-checkpoint the rollout** (`torch.utils.checkpoint`, per step or per detach-segment): recompute the step
  forward during backward instead of storing it → AR memory drops to ∝ `detach_every` (not F), so F=96 fits at batch 96.
  ~1.3× compute. **Decouples F from MEMORY; does NOT help the ∝F TIME.**
- **Trade batch for F** (batch 48/F 48, …) — free, but small batch hurts optimization.
- **F-curriculum** (ramp F up during training like `p_tf`): cheap short-F early, long-F late. Not currently supported.
- **Cheaper stability levers instead of brute-force F:** the contraction penalty (`variations.contraction`, weight 0)
  + a decode→encode carry (re-project onto the manifold each step). Likely higher ROI than large F.

**Verdict:** F=247 is time-infeasible. Practical path = **F≈48 + checkpointing** (keeps batch 96, ~2.6 days) **paired
with the contraction penalty** — "sees its own multi-step drift" + "dynamics that pull back" is the actual cure for the
color→structure drift, at a fraction of the cost of training at the full horizon.

## Experiment 7 — `expandable_segments` allocator mode (2026-07-20)

**Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ON for all runs.** It is a PyTorch CUDA caching-allocator
mode (NOT a numerical change — zero effect on results), that lets memory segments grow/shrink so freed blocks of
differing sizes are reusable instead of leaving fragmentation gaps ("reserved but unallocated" memory that OOMs
even with free VRAM). Measured on mm_flow d=128, conv encoder, num_tokens=16, in-rollout, batch 16:

| allocator | peak (GB) |
|---|--:|
| default | 86.0 |
| expandable_segments:True | **62.2** |

~28% less peak from fragmentation alone — it raised the real-batch ceiling from ~16 to ~20 at this config. Set it
in the launch env (`docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Legit acceleration, always on.

> Note (2026-07-20): conv/deconv `base` channel width is a **weak** batch lever — halving base 32→16 only moved the
> ceiling batch 20→~30 (and hurts decode). Memory is dominated by the ∝`batch·F` rollout (Exp 6), not conv channels.

---

## Experiment 8 — profiling the AR step: attention is FUSED, the loop is DISPATCH-bound (2026-08-05)

Motivation: an off-machine perf probe (robocasa, batch 32 / F 64: **3.9 s/batch, GPU 0–13% util, ~12.7 days / 30 ep**)
raised "is compiling FlexAttention a big speedup?". A candidate diagnosis said FlexAttention runs on the eager
`B×H×T×T` reference path (mm models skip `torch.compile`). **Measured — it's wrong.** Harness:
`scripts/profile_rollout.py` (the real `rollout_train` at p_tf=0, joint mm_flow d=128/F=64), dedicated H100.

**Finding 1 — FlexAttention is already FUSED.** The profiler shows `FlexAttentionAutogradOp` + a `Torch-Compiled
Region` (the temporal sliding-window attn *self-compiles per call*) and `_flash_attention` (spatial + image-AE attn,
flash-fused) — **no T×T score materialization**.

> ⚠️ **CORRECTED 2026-08-05 (later same day) — see Experiment 9.** This finding's *conclusion* ("attention is
> already fused → compiling the rollout buys ~0") is **wrong**. A dedicated-GPU prototype compiled the rollout
> step and measured **~6× faster, parity-safe** — so compiling emphatically *does* help. On the eager path the
> step even emits `flex_attention called without torch.compile()… materializes the full scores matrix` (i.e.
> UNFUSED), contradicting the "no T×T materialization" reading above. So the contradiction resolves the OPPOSITE
> way from what this said: `train_world_model.py`'s "just eager" (= unfused in the serial rollout) was RIGHT;
> `launch_shootout.sh:8`'s "self-compiles even eager" was WRONG (now fixed). Findings 2 & 3 below
> (dispatch-bound, batch-nearly-free) stand — they're independently confirmed.

**Finding 2 — the step is DISPATCH-bound, not compute-bound.** At batch 32: **Self CPU 3.36 s ≫ Self CUDA 0.68 s**
→ GPU **~20% busy, ~80% idle**. The top CPU costs are the `Torch-Compiled Region` FlexAttention dispatch (0.78 s) +
`FlexAttentionAutogradOpBackward` (0.62 s) — i.e. the cost of *calling* the op ~256× (F·depth) through the
**serial F-step Python loop**, not the kernels. This is exactly the latency-bound thesis of Exp 5–7, confirmed at the
kernel level. (The original conclusion here — "compiling attention would not help — it's already fused" — was
**wrong**; compiling the whole step *does* collapse those dispatches: ~6×, see Exp 9.)

**Finding 3 — batch is nearly FREE** (the dispatch-bound corollary: extra samples ride the idle GPU). Clean sweep:

| batch | s/batch | peak GB | throughput vs 32 |
|--:|--:|--:|--:|
| 32 | 2.752 | 22.9 | 1.00× |
| 64 | 2.803 (+1.9%) | 45.7 | **1.96×** |
| 96 | 3.040 (+10%) | 68.5 | **2.72×** |
| 128 | 3.356 (+22%) | 91.3 | **3.29×** |

### Recommendation
1. **Immediate, free, modeling-neutral: raise `data.batch`.** 32→**96** = **2.7× throughput** for +10% per-step, at
   68 GB (leaves ~27 GB eval headroom — respect the eval-OOM caution in Exp 3/5). 128 = 3.3× but 91 GB is too tight for
   the eval spikes. This alone ~halves any AR-training wall-clock. (Do NOT use `accumulate_grad_batches`/`window_stride`
   — forbidden; you don't need them, real batch is the lever.)
2. **Bigger, its own parity-gated project (NOT a launch bundle): kill the per-step dispatch.** The overhead is the
   per-step op *call* ×256/step. **DONE — see Experiment 9:** `torch.compile(step, mode="default")` collapses the
   dispatches (parity-safe, ~6×). Note the originally-proposed `reduce-overhead` (CUDA graphs) does **NOT** work here
   — it's fundamentally incompatible with the retained-BPTT rollout (Exp 9). Behind the opt-in `model.compile_rollout`.
   See `design/rollout_throughput.md` for the plan + parity gate.

### Measured headroom: mm_flow d=64 in-rollout (2026-07-15)
The flow-x0 / mse-control pair (d=64, batch 128, in-rollout, dynamics shortcut, flow-x0 decode, recon_frac 0.25,
detach_every 16) sits at **~63.8 GB / 93 GB** and **~15–20% GPU util** at ~30 min/epoch — latency-bound exactly as the
callout above predicts. So **batch 256 fits with room to spare for this setup** (memory ~doubles from ~64 GB, still
< 93 GB; detach_every caps the in-rollout graph). It mainly raises utilization (fills the idle sequential steps) rather
than cutting wall-clock proportionally — but it's free memory-wise and the right default next time we launch this config.

---

## Why the "compile FlexAttention for a 3–5× win" idea was a red herring (2026-08-05)

> ⚠️ **PARTIALLY SUPERSEDED 2026-08-05 (later same day) — see Experiment 9.** The reasoning below (the *fix* the
> candidate proposed — swapping in a fused attention kernel — buys nothing, because the bottleneck is dispatch,
> not kernel efficiency) is correct **about dispatch**. But its headline "there is no compile speedup to get" is
> **wrong**: compiling the *whole step* (`mode="default"`) — which is a different lever than "fuse the attention
> kernel" — collapses the per-step dispatch and is **~6× faster, parity-safe**. It also assumed the eager rollout
> ran attention *fused*; the prototype found it runs **UNFUSED** (emits the without-`torch.compile` warning). Keep
> this section for the dispatch-vs-compute method; take the ~6× result from Exp 9.

The candidate diagnosis reasoned: *FlexAttention needs `torch.compile` to be fast → multimodal models skip
`torch.compile` (`train_world_model.py`, `not cfg.model.get("modalities")`) → so attention runs the eager
`B×H×T×T` reference path → that's why the GPU idles.* **The middle link is false.** `flex_attention`
**self-compiles its own kernel on every call**, independent of the outer model's compile status — so the
attention was already fused. Two things caused the wrong read: (1) the repo's own comment said "FlexAttention
still runs, **just eager**," where "eager" meant "not inside the outer compiled graph," NOT "unfused reference
kernel" (now corrected); and (2) it conflated **kernel efficiency** (fused ✓) with **dispatch/launch
overhead** (the real bottleneck). The GPU idles because the AR rollout is a serial F-step Python loop that
*calls* the per-step ops ~256× (F·depth), each with CPU dispatch cost. Swapping in a fused *attention kernel*
does nothing for the *number of dispatches* — so *that* fix would have bought ~0. (The lever that DID work is
different: compile the whole step so inductor collapses the ~256 dispatches — ~6×, Exp 9.)

**How this was established (method — reproducible via `scripts/profile_rollout.py`):**
1. Ran the real `rollout_train` step at `p_tf=0` under `torch.profiler` (CPU+CUDA), a few steps post-warmup,
   on a dedicated H100.
2. **Fusion check** — the CUDA kernel table showed `FlexAttentionAutogradOp` + `_flash_attention_*` kernels
   and **no** `B×H×T×T` score tensor (an eager reference would show explicit `bmm → softmax → bmm` over a
   materialized `T×T`). ⇒ read as fused. **(This read was wrong for the rollout — Exp 9: the eager rollout
   step emits the without-`torch.compile` warning and compiling it is ~6×. The fused kernels this step saw
   were the compiled parallel forward, not the serial eager rollout.)**
3. **Bottleneck check** — **Self CPU 3.36 s vs Self CUDA 0.68 s** (GPU ~20% busy) ⇒ dispatch/launch-bound,
   not compute-bound. The top CPU lines were the `Torch-Compiled Region` (FlexAttention) dispatch (0.78 s) +
   its autograd (0.62 s) — the *calls*, not the kernels.
4. **Batch-scaling check** — clean sweep 32→64 = +1.9% wall / 32→96 = +10% confirms it: extra samples ride
   the idle GPU nearly free (a compute-bound step would scale ~per-sample). This is what motivates the
   `autobatch` finder (fill VRAM ≈ free) and the compiled rollout (kill the dispatch) in
   `design/rollout_throughput.md`.

---

## Experiment 9 — compiling the AR rollout step: ~6× (mode=default), and why CUDA graphs can't (2026-08-05)

Exp 8 concluded the dispatch-bound serial loop needed a **CUDA-graph** (`torch.compile(step, mode="reduce-overhead")`)
rollout to kill the per-step dispatch, and that fusing attention would buy ~0. A dedicated-GPU prototype tested
both claims on the joint `mm_flow` config (d=128 / F=64 / W=32 / P=8, image head). **Both were partly wrong: the
CUDA-graph mechanism can't work here, but compiling the step *does* — ~6×.** Behind the opt-in flag
`model.compile_rollout` (default **off**); implemented as `torch.compile` of the per-step backbone+flow-readout
unit (`multimodal._rollout_step`), applied only in the steady `p_tf==0` regime (teacher-forcing warmup stays eager).

**Gate 1 — parity: PASS.** One `rollout_train` step, compiled vs eager, same weights/batch/seed, flow sampler
forced deterministic (ε=0) to isolate compile fidelity from the sampler RNG:
- forward latent bag: max abs diff **5.5e-3**, max rel diff **1.6e-3**
- all **163** parameter grads: max abs diff **5.0e-8** (mean 1.2e-9), **0** None-mismatches
- threshold bf16 rollout 3e-2 rel — comfortably inside.

**Gate 2 — speed: the proposed mechanism fails; a variant beats the target.**

- **`mode="reduce-overhead"` (CUDA graphs — what Exp 8 / Phase 1B.1 asked for) does NOT work for training, and
  it's not fixable.** It never hits the cudagraph fast-path (`"Unable to hit fast path of CUDAGraphs … pending,
  uninvoked backwards"`) and hard-crashes with `"accessing tensor output of CUDAGraphs that has been overwritten
  by a subsequent run"`. **Fundamental:** CUDA graphs reuse ONE static memory pool per replay, but the retained
  BPTT graph needs *every* F-step's saved-for-backward activations alive until the single end-of-rollout
  `backward()`; the next step's replay clobbers them. CUDA graphs assume fwd→bwd per capture; a retained-graph AR
  rollout violates that. Output-cloning doesn't help (tried). So the "big dispatch-killer" of Exp 8 rec #2 is a
  dead end for the training rollout.

- **`mode="default"` (inductor fusion, no CUDA graphs, BPTT-safe) WORKS: ~6.1×.** Dedicated H100:

  | | eager rollout | compiled step (default) |
  |---|--:|--:|
  | s/batch | 2.76 | **0.45** (~6.1×) |
  | samples/s | 11.6 | **70.6** |
  | GPU util | 27% | **82%** |
  | peak mem | 22.9 GB | 22.9 GB |
  | compile | — | one-time ~96 s |

  The win has **two sources**, and the second one corrects Exp 8: (1) inductor collapses the ~256 tiny per-step
  dispatches (the dispatch-bound bottleneck Exp 8 correctly identified); (2) **the eager rollout runs FlexAttention
  UNFUSED** — it emits `flex_attention called without torch.compile() … materializes the full scores matrix` —
  so compiling the step *also* fuses attention. Exp 8's "already fused" observation was of the **compiled parallel
  forward**, not the serial eager rollout; this resolves the repo contradiction the OPPOSITE way from Exp 8's call
  (`train_world_model.py`'s "just eager" = unfused was right; `launch_shootout.sh:8`'s "self-compiles even eager"
  was wrong — both now fixed in-repo).

**Constraints (both fail-fast in `setup.build_model`, no silent disable):**
- **`head_dim = d/heads` must be ≥ 16.** The compiled FlexAttention Triton kernel raises `NYI: embedding dimension
  … must be at least 16` mid-compile below that. The **eager** rollout has no such floor (unfused fallback), so
  this only blocks `compile_rollout`. The default `mm_flow` (d=32/heads=8 → head_dim=**4**) is *ineligible*; the
  joint config (d=128/heads=8 → head_dim=**16**) is exactly at the floor. Discovered the hard way: the first
  end-to-end run used the d=32 default and crashed 12 min in at the epoch-4 compile — hence the build-time guard.
- **Mutually exclusive with `variations.contraction` (weight>0)** — FlexAttention has no double-backward under
  `torch.compile`, which the contraction Jacobian power-iteration needs.

**Ship-gate (end-to-end validation) — DONE 2026-08-05.** The ~6.1× was measured on the rollout step in isolation
(`profile_rollout.py`). Ran a real side-by-side through the full Lightning loop: identical joint `mm_flow` config
(**d=128**, image head, F=64, batch 32, detach_every 16), 60 train batches/epoch, 8 epochs, eval off, one run per
dedicated H100 — only `compile_rollout` differs.

- **Speed — CONFIRMED ~6.2×.** Steady AR train epoch (p_tf=0, no val): **eager ~228 s/ep (~3.8 s/batch)** vs
  **compiled ~37 s/ep (~0.6 s/batch)** (epochs 5–6 on both). The epoch-4 transition ate the one-time compile
  (first 15 batches 89 s) then fell to ~0.6 s/batch — amortized *within* that epoch. Matches the isolation number.
- **Stability — CONFIRMED.** The compiled run completed all 8 epochs, no NaN/crash, loss descending
  (train 0.81→0.73, val 2.34→0.95 as p_tf ramped to 0). The eager→compiled transition at epoch 4 is safe.
- **Parity — NOT testable from this A/B, and that's expected.** No global seed is pinned (`seed_everything` absent),
  so the two runs diverge from init: their epoch-3 losses already differ (train 0.77 vs 0.81, val 2.00 vs 2.34)
  *before* compile engages (epochs 0–3 are identical code). Bitwise parity is the isolation test's job, already
  passed (fwd rel 1.6e-3, grad abs 5e-8). This run corroborates behaviorally-equivalent, healthy training + the win.
- **head_dim trap found here.** The first attempt used the `mm_flow` **default d=32** (head_dim=4) and crashed 12 min
  in at the epoch-4 compile with the inductor NYI. Fixed by the build-time guard above; rerun at d=128 (head_dim=16)
  passed. Net: the isolation ~6.1× **does** survive the full training loop, on any config with head_dim ≥ 16.
- **Full-run confirmation (2026-08-06).** A real 50-epoch no-head run (d=128, batch 64, full eval, torus image
  data) crossed the eager→compiled boundary cleanly: **eager warmup epochs ~53 min (~4.3 s/batch) → compiled
  steady epochs (p_tf=0) ~10 min (~0.81 s/batch) = ~5.3× per-epoch end-to-end** (and ≥6× vs a *true* eager p_tf=0,
  which is slower than the p_tf=0.25 warmup this is measured against). No OOM, val descending. So the win holds on
  the real training loop at scale. The only slow part is the eager warmup (~4 epochs, p_tf>0) — inherent, since
  compile applies only to the steady p_tf=0 rollout. A second dataset (robocasa recorded, d=128/F=64/batch=32)
  independently measured **3.90 → 0.67 s/batch (5.8×)**.
