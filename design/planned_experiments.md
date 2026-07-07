# Planned experiments / tasks (not yet run)

Staging area for work that's designed + set up but deliberately not executed. Pick up later.

---

## 1. Long-horizon training: F=64, detach_every=32 (reduce open-loop drift)

**Why.** Eval rolls ~247 steps open-loop but training uses F=24, so the model never learns long-horizon
stability → open-loop drift (in the FPV rollout: color wrong by ~+106 steps, then structure). Diffusion does
NOT self-correct this — the flow is a per-step *conditional* predictor conditioned on the carried (drifting)
latent, with no re-projection to the manifold. Raising F gives the model exposure to its own multi-step drift;
raising detach_every gives it the gradient *credit assignment* to learn to correct it. (See accelerations.md
Experiment 6 for the memory/time analysis.)

**Config.** F=64 (forward rollout exposure) + detach_every=32 (BPTT credit-assignment window; ~free on memory,
caps at F). NO contraction penalty (explicitly out of scope for this run). Codec = the vis_refactor2 iter-2
codec (d=64, num_tokens=32, patch=8) — run this only AFTER vis_refactor2 validates that codec.

**Launcher is set up** (`src/quickdraw/scripts/launch_vision_large.sh`, `DETACH`/`F` are env knobs; experiment
names + pkill derive from `GROUP`). Ready-to-run invocation (supply fresh RS_* — train.py rejects dupes):

    F=64 DETACH=32 GROUP=vis_refactor3 BATCH=48 RS_PROBLEM=.. RS_TRIED=.. RS_DETAIL=.. \
      RS_RATIONALE=.. RS_TRYING_lsar=.. RS_TRYING_diff=.. bash src/quickdraw/scripts/launch_vision_large.sh

**Budget / caveats.**
- Memory: F=64 at batch 96 OOMs. batch·F must stay ~2304 for the ~64 GB envelope → BATCH=48 (~3072, likely
  ~60–80 GB, RE-PROBE epoch-1 AR) or BATCH=36 (~2304 ≈ 64 GB, safe). detach_every is ~free on memory.
- Time: AR epoch ∝ F → ~2.7× the F=24 rate → ~50 min/epoch → **~3.5 days** for 100 epochs.
- Stability: deeper BPTT (32) raises gradient-blow-up risk (clipping is on at 1.0, but historically imperfect).
  Watch the DIFFUSION head ep5–30; if it diverges and doesn't recover, drop DETACH=24.
- F=64 < eval's 247, so it *reduces* drift, not eliminates. Contraction penalty is the complementary lever
  (deferred here by request).

**Timing decision (open):** queue after vis_refactor2 finishes (uses the validated codec) vs kill vis_refactor2
and start now.

---

## 2. Bigger AE + backbone (image fidelity — reduce blur + seam visibility, ≤5M params)

**Why.** At d=64 / tokens=32 / patch=8 the *entire* world model is only **~0.8M** trainable params (image AE
0.49M) — massively under-parameterized. Sizing up adds reconstruction capacity; it's the cheapest fidelity lever.
NO perceptual/adversarial loss (out of scope by decision). Measured param sweep:

| d | depth | ae_depth | tokens | total | backbone | imgAE |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 4 | 4 | 32 | 0.80M | 0.27 | 0.49 | (current) |
| 128 | 4 | 4 | 48 | 3.01M | 1.06 | 1.84 | |
| 128 | 6 | 6 | 48 | 4.33M | 1.59 | 2.63 | |
| **128** | **6** | **6** | **64** | **4.34M** | 1.60 | 2.64 | **← CHOSEN (~4.3M, under 5M cap)** |
| 160 | 6 | 6 | 48 | 6.71M | — | — | (over 5M) |

**Chosen: `d=128, depth=6, ae_depth=6, num_tokens=64, patch=8` → ~4.3M** (5.4× current). num_tokens 48→64 is
~free on params (latent queries are tiny) but doubles token/patch density (64 tokens / 256 patches = 0.25, 2× the
iter-0 density) — best shot at killing seams + blur via capacity alone.

Overrides: `model.d=128 model.depth=6 model.modalities.1.ae_depth=6 model.modalities.1.num_tokens=64`
(launch_vision_large.sh needs `D`/`DEPTH`/`AEDEPTH` knobs added, or pass as extra args).

**Memory is the real cost — NOT params (~4.3M ≈ 17 MB, negligible).** AR activation memory scales ~ `d × tokens`,
so d 64→128 + tokens 32→64 ≈ **~4× the per-(batch·F) AR memory**. Current ~64 GB at batch 96/F 24 → expect OOM at
batch 96; RE-PROBE, likely need **batch ~24** (or F ~12) to hold ~64 GB. Time: d=128 also raises per-step compute →
slower epochs.

**Seams:** expect REDUCED, not eliminated — the linear non-overlapping per-patch decode is mechanistic. Only add a
seam-specific fix (overlap-add / conv or pixel-shuffle decode head / refiner) if the grid is still objectionable
after this. (Deferred by decision — size up and look first.)
