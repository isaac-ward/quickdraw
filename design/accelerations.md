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
