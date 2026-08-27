# The flow model's dynamics loss ignores the teacher-forcing schedule

Status: **DIAGNOSIS + IMPLEMENTATION PLAN. No code changed yet.** Written 2026-08-25.

## 0. The trick error, in one paragraph

`MultiModalFlow` computes two losses per step. One of them honours the `p_tf` teacher-forcing schedule and
one silently does not, and the one that does not is **the loss that learns the transition function**.

* `decode/<name>` is computed on `preds` — the output of `rollout_train`, which consumes its own predictions
  once `p_tf` reaches 0. It is autoregressive.
* `dynamics/latent` is computed on `z[:, :-1]` — the **encoder's** latents for the true frames. It never
  looks at `preds`. `loss_terms` receives `pred_bag` as its first argument and **never references it**
  (`models/multimodal.py:957-994`; the second pass proper is `:963-977`).

So `p_tf` ramps 1 → 0 over `p_tf_warmup_epochs`, the rollout duly becomes autoregressive, and the dynamics
loss carries on as if nothing happened. **For the transition function, the model is effectively pinned at
p_tf = 1 for the entire run.** It is trained to take one good step *from a correct latent* and is never once
asked to take a step from a latent it produced itself.

**The defect is the second pass's INPUTS, not its existence** — the pass must survive, because at
`p_tf >= 1` there is no rollout for a dynamics loss to reuse (`lit.py:127`). Any fix keeps the pass and swaps
what it conditions on.

**It was, however, noticed.** `MultiModalDistribution.loss_terms` (`multimodal.py:1057-1062`) documents the
same behaviour for itself and adds "*the p_tf ramp affects only the rolled decode losses (recon_losses), **as
for Flow***". So this was a considered position, not an oversight. It is still wrong for our goal — the
transition function never learns to recover from its own drift — but nobody missed it, and this document
should not pretend otherwise.

`MultiModalLSAR` — the older class — does the opposite, and
correctly (`multimodal.py:829-834`): its dynamics loss reads `pred_bag`, so it inherits the p_tf ramp for
free. The behaviour diverged when the flow model replaced LSAR's latent-regression objective with
flow matching and dropped the `pred_bag` argument on the way.

## 1. What the measurements look like

The symptom is a rollout whose latent rotates away from the truth and never recovers, because nothing ever
taught it to recover. Measured on `bott_recon1` (`_oneoff_latent_drift.py`, and now logged every epoch as
`latent_cos` / `latent_motion_ratio`):

| horizon h | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| `cos(pred, true)` | 0.98 | 0.71 | 0.48 | 0.26 | **0.14** | 0.06 |
| `latent_motion_ratio` | 0.95 | 0.83 | 0.75 | 0.80 | **0.56** | — |

Cosine decays roughly as a random walk (per-step angle ~11°, accumulating as √h toward the 90° ceiling), and
the step size shrinks — the rollout **under-rotates and drifts**. Flow refinement measured by
`eval_denoising_filmstrip` contributes +1.2 to +2.0 dB at h ≤ 32 and **nothing at all at h = 64**.

Ruled out as causes, each by measurement rather than argument:

| candidate | verdict |
|---|---|
| train/eval path divergence (KV cache vs windowed recompute) | **null** — A/B at every horizon: ±0.01 dB, ±0.004 LPIPS |
| action off-by-one between train and eval indexing | **clean** — both pair action `i` with state `i`; audited |
| `stochastic_eval` injecting noise each step | **refuted** — deterministic is *worse* (cos 0.037 vs 0.128 @h64) |
| `sampling_steps` overshoot | **null** — everything K≥2 within noise; K=1 badly broken |
| `latent_norm` | measured: `layernorm` beats `none` (OL@+128 0.374 vs 0.426) |

## 2. Where the gradient goes today

Components in forward order. "AR" = the gradient travels back through the autoregressive chain.

| component | `dynamics/latent` | `decode/<name>` | `codec/roundtrip` |
|---|---|---|---|
| image / proprio encoder | ✓ | ✓ (chain's first bag) | ✓ |
| action encoder | ✓ | ✓ every step | ✗ |
| backbone | ✓ | ✓ every step | ✗ |
| flow velocity net | ✓ | ✓ every step | ✗ |
| image decoder | ✗ | ✓ | ✓ |
| **input it is fed** | **clean `z`** (DF-noised if on) | the rollout `preds` | clean `z` |
| **carries AR gradient?** | **NO — one step** | **YES — the only one** | **NO** |
| **reach** | n/a | ≤ `detach_every` = 32 steps | n/a |
| **density** | all positions | `recon_frac` × F | n/a |

The rollout's only gradient arrives **through the image decoder**, so the dynamics is corrected by a
pixel-space error pushed back through the decoder's Jacobian. An MSE-trained decoder is smooth — many
latents decode to similar images — so a large latent error yields a small pixel error and a weak gradient.
That is exactly the observed state: `cos = 0.14` while the frame still looks plausible.

It also explains the single largest measured win in this project. `recon_frac` 0.25 → 1.0 (OL LPIPS@+128
0.374 → 0.278) is a 4× increase in the **density of the only autoregressive gradient in the model**. Not a
coincidence — the mechanism.

## 3. The fix, and why the target has to follow the context

The flow predicts a **residual**: `predict_next` does `next = (what I am standing on) + d`. So the target
delta is defined *relative to the latent it will be added to*. Feeding the rollout in without moving the
target is a trap:

| | **A. today** | **B. rollout context, old target** | **C. rollout context, moved target** |
|---|---|---|---|
| context fed in | `z[t]` | `preds[t]` | `preds[t]` |
| target for `d` | `z[t+1] − z[t]` | `z[t+1] − z[t]` | `z[t+1] − preds[t]` |
| standing on | `z[t]` | `preds[t]` | `preds[t]` |
| lands on | `z[t+1]` ✓ | `z[t+1] + drift` ✗ | `z[t+1]` ✓ |
| meaning | right step, right place | right step, **wrong place** — drift carried forward untouched | the **correction** back onto the truth |

`drift = preds[t] − z[t]`. Case B teaches a step measured from somewhere the model is not standing, so it
lands short by exactly the drift and never learns to remove it.

**In absolute mode there is no wrinkle at all** — `target = z[t+1]`, which never referenced the context, so
swapping `preds` in "just works" and still teaches recovery:

| | clean context | rollout context |
|---|---|---|
| target | `z[t+1]` | `z[t+1]` |
| lands on | `z[t+1]` ✓ | `z[t+1]` ✓ |

One line covers all four cases, because the existing expression is already correct once written relative to
whatever context was fed:

```python
target = (z[:, 1:] - s).detach() if self.predict_residual else z[:, 1:].detach()
```

with `s = z[:, :-1]` this is **byte-for-byte today's behaviour**; with `s = preds` it is case C.

## 4. No new schedule is needed

`p_tf` already decides how autoregressive `preds` is: at 1 every rollout step consumes the true previous
latent, at 0 its own. So a dynamics loss that reads `preds` inherits the ramp — teacher-forced early,
autoregressive later — with **no new knob, no new warmup, and `p_tf` itself untouched**. That is precisely
how LSAR gets its schedule today. An earlier draft of this plan proposed a separate `frac` mixing float; it
is redundant with `p_tf` and is dropped.

## 5. Which classes this applies to

Audited across all four. The flag is only meaningful where the dynamics loss **conditions on a context**:

| class | dynamics loss shape | flag meaningful? |
|---|---|---|
| `MultiModalFlow` (`:957`) | condition on context → predict one step | **yes** — swap the context |
| `MultiModalDistribution` (`:1057`) | condition on context → prior scored vs posterior | **yes** — swap the context |
| `MultiModalLSAR` (`:829`) | regress the rollout's **output** vs encoded truth | **no** — no context slot; `false` would compare `encode(fut)` to itself, identically **zero** |
| `MultiModalDSAR` (`:864`) | `return {}, {}` — no dynamics loss | inert |

Two corrections to earlier drafts of this file:

* `MultiModalDistribution` **can** condition on the rollout — nothing structural prevents it. Its context is
  `z = posterior samples of the TRUE frames` (`:1066-1071`), which is teacher forcing in exactly the same
  sense as Flow's `z[:, :-1]`. Teacher forcing there is **DreamerV3's published design**, i.e. a defensible
  default, NOT a constraint.
* `MultiModalLSAR` was described as "the class that does it right, inheriting the p_tf ramp". True in spirit,
  wrong in mechanism: LSAR reads `pred_bag` on the **output** side, this plan reads it on the **input** side.
  Both inherit p_tf; they are different mechanisms.

## 6. Implementation plan — condition on the rollout's FEEDS, not its predictions

### The principle, in plain language

The dynamics network learns "given where things are now, plus an action, where will they be next?". To train
it you hand it a **starting point**: either the TRUE one from the data (easy mode) or the one it just
predicted itself (hard mode — and what actually happens at test time). `p_tf` picks which.

**The image loss and the dynamics loss should ALWAYS get the same starting point.** There is no principled
reason for them to differ. They diverged for an efficiency reason: a teacher-forced dynamics loss can be one
parallel backbone pass over all positions, whereas using the rollout's starting points means plumbing them
out of the rollout. The cheap path was taken and nobody re-checked.

So the rule is: **ONE `p_tf`, and every loss follows it.** Not per-submodule teacher-forcing knobs —
submodules silently disagreeing is exactly what produced this defect. `dynamics_follows_p_tf` exists for one
reason only: to express DreamerV3's deliberate exception in `MultiModalDistribution`. Everywhere else it is
`true` and nobody touches it.

Renamed from `dynamics_on_rollout` (2026-08-25): the flag decides whether the p_tf schedule REACHES this
loss, and the name should say so.

### The 2x2, per class — the flag bites at every `p_tf < 1`

At `p_tf >= 1` no rollout runs, so there are no feeds and both settings are identical. At ANY p_tf below 1
the rollout runs, feeds exist, and every coin that picked "own" makes them differ from clean `z` — so the
tables' `p_tf = 0` row is really "p_tf < 1".

**`MultiModalFlow`**

| | `follows_p_tf = false` | `follows_p_tf = true` |
|---|---|---|
| p_tf = 1 | dynamics <- truth, decode <- truth | **identical** (no rollout, no feeds) |
| p_tf = 0 | dynamics <- **truth (THE BUG)**, decode <- own | dynamics <- own, decode <- own **(THE FIX)** |

**`MultiModalDistribution`**

| | `false` | `true` |
|---|---|---|
| p_tf = 1 | dynamics <- truth, decode <- truth | **identical** |
| p_tf = 0 | dynamics <- truth, decode <- own — **DreamerV3's convention, deliberate** | dynamics <- own, decode <- own (imagination-trained prior) |

This class is the flag's whole justification: "dynamics on truth, decode on own" is unreachable from p_tf
alone, because `p_tf = 1` un-rolls the decode loss too.

**`MultiModalLSAR`** — flag N/A. Its loss scores the rollout's OUTPUT ("the rollout produced these latents,
are they right?"), so there is no starting point to choose; p_tf already decided that when the rollout ran.
LSAR already follows p_tf — it is **permanently `true` by construction**, and the model doing this
correctly; Flow is being brought into line with it. Setting `true` is a redundant no-op; setting `false`
RAISES (see Validation below). **`MultiModalDSAR`** — no dynamics loss at all; setting the flag either way
RAISES.


**Superseded design (2026-08-25, second audit).** The plan below replaces the earlier "substitute
`pred_bag`" version. The dynamics loss should condition on **what each rollout step actually STOOD ON** —
the p_tf mix `s_feed` (`multimodal.py:598-601`) — not on what it predicted.

**Precedent: this pattern is already in the codebase.** `physics_proprio_chained` (`:415-440`) implements
exactly it for the proprio decode — per-(sample, step) Bernoulli coin at `:437`, feed truth with prob p_tf
else the model's own — and its docstring calls it "the scheduled-sampling GENERALIZATION ... closes the
teacher-forced-train / AR-eval exposure gap" (`:419-421`). The latent dynamics loss simply never got it.

### Why feeds beat pred_bag

| | `pred_bag` plan | **`s_feed` plan** |
|---|---|---|
| p_tf = 1 | model's own (stochastic, untrained at ep0) → needed a hand-rolled `p_tf < 1.0` gate | parallel `forward()` path runs, **no feeds exist**, context falls back to clean — *the gate is the absence of feeds* |
| p_tf = 0 | own latents | identical (`carry_transform` is identity for Flow, `:503-506`) |
| intermediate | conditions on states the rollout **did not stand on** whenever the coin picked truth | conditions on exactly the visited states |
| alignment | inferred from shapes (`P_ = L − F_`), needed guards | **by construction** — `feeds[k]` sits at position `P+k`, which predicts frame `k+1`, the step that stood on it |
| trap 3 (warmup junk) | live | **dissolved** |
| trap 4 (full-L smoke caller) | live, needed `P_ >= 1` guard | **dissolved** — feeds only ever come from a rollout |

Two hazards **stand unchanged** from the first audit and are restated here so this document is
self-contained:

* **Hazard A — in-place view corruption.** `s = z[:, :-1]` is a **view of `z`**. Writing into it
  (`s[:, P:] = ...`) overwrites `z` itself, wrecking the target at `:976` and breaking autograd. Use
  `torch.cat`, never in-place.
* **Hazard B — DF ordering.** The target is computed at `:976` *after* the diffusion-forcing block
  (`:966-973`) has already overwritten `s`, including its `_ln` renorm. Writing `target = (z[:,1:] - s)` at
  that location makes the target absorb the DF noise — a silent change to diffusion forcing. Hence the
  `s_ref` snapshot taken **before** the DF block.

(The first audit's other two hazards — warmup junk contexts and a full-L smoke caller — **dissolve
structurally** under feeds: feeds only ever come from a rollout, and at `p_tf >= 1` there is no rollout.)

### Cost: one honest plumbing change

`s_feed` is **not** recoverable at the loss site — the coins are drawn inside the loop and discarded, and
`_rollout_from` returns only `torch.stack(preds)` (`:607`). But `bag_buf` already holds it, so collection is
free; `return_feeds` has to be threaded through `_rollout_from` (`:558`) → `_rollout` (`:547`) →
`rollout_train` (`:634`) and accepted at `lit.py:131`.

```python
# multimodal.py:566 / :607  — collect what each step stood on
P0 = len(bag_buf)
...
if return_feeds:      # bag_buf[P0+k] fed the step predicting frame k+1
    return preds, torch.stack(bag_buf[P0:], dim=1)   # ATTACHED (no .detach) -- see Decisions below.
    #                                                  Collection is at LOOP level, outside the compiled
    #                                                  _rollout_step, so all three loop branches support it.
# lit.py:127-131 — parallel TF path passes feeds=None, which IS the correct p_tf=1 semantics
# lit.py:165 — feeds ride as an optional kwarg, same pattern as `anchor` (reaches Flow/Dist only)

# multimodal.py:963 (Flow.loss_terms)
if self.dynamics_follows_p_tf and feeds is not None:
    F_ = feeds.shape[1]
    s = torch.cat([z[:, :L - F_], feeds[:, :F_ - 1]], dim=1)   # cat, NEVER in-place (trap 1)
else:
    s = z[:, :-1]
if self.dynamics_detach_encoder: s = s.detach()
s_ref = s                                                      # snapshot BEFORE the DF block (trap 2)
...
target = (z[:, 1:] - s_ref).detach() if self.predict_residual else z[:, 1:].detach()
```

Class `__init__` default `False` (`:875`), `setup.py:325` fallback `False`, `true` in `mm_flow.yaml`.

The `setup.py` fallback is **behavioural** compat, not shape compat — the flag changes zero parameters, so
checkpoints load either way; the reason it must be config for Flow is that the two live runs have to resume
under the semantics they trained with, plus the `false` + `dynamics_detach_encoder=true` ablation quadrant.

**`MultiModalDistribution`: config, default `false` — NOT a class constant** (reversed 2026-08-25, user).
DreamerV3's world-model loss being on observed sequences is a *default convention*, not a law — imagination
training is part of the method — so hard-locking it would be deciding for a future user. Both values valid.

### Validation: raise on combinations that cannot be honoured

Follows the file's own rule at `setup.py:244`: *"Fail fast on config knobs that only one model reads
(otherwise silently ignored)"*, and the existing precedent two lines above it (`:241-243` raises when
diffusion forcing is set on a non-flow model).

| model | `true` | `false` | validate |
|---|---|---|---|
| `mm_flow` | ✓ the fix, yaml default | ✓ repro shim + IWS ablation quadrant | — |
| `mm_categorical` / `mm_gaussian` (Distribution) | ✓ imagination-trained prior | ✓ default, Dreamer convention | — |
| `mm_lsar` | ✓ **redundant no-op** — already true by construction | ✗ **RAISE** | its dynamics loss scores the rollout's OUTPUT; there is no context to pin to clean, and `false` would compare `encode(fut)` to itself ≡ 0 |
| `mm_dsar` | ✗ **RAISE** | ✗ **RAISE** | no dynamics loss exists (`:864`, `return {}, {}`) — either value is a misconception |

```python
# setup.py, next to the df_scale gate at :241
dfp = cfg.model.get("dynamics_follows_p_tf", None)
if dfp is not None:
    if name in ("mm_dsar", "dsar", "base"):
        raise ValueError("model.dynamics_follows_p_tf has no meaning for mm_dsar: it has no dynamics "
                         "loss (loss_terms returns {}). Remove the key.")
    if name in ("mm_lsar", "lsar") and not bool(dfp):
        raise ValueError("model.dynamics_follows_p_tf=false is not implementable for mm_lsar: its dynamics "
                         "loss scores the ROLLOUT'S OUTPUT against the encoded true future, so there is no "
                         "context to pin to clean latents -- 'false' would compare encode(fut) to itself "
                         "(identically zero). mm_lsar is always true by construction; remove the key.")
```

Raising beats silently ignoring: a user who sets this on LSAR believes they changed the training regime.

### Rejected alternative: compute the dynamics loss INSIDE the rollout (one pass, like LSAR)

The obvious question this framing raises: why keep a second pass at all — reuse the per-step conditioning
`h` that `_rollout_step` already computes (`:509-515`), so Flow becomes structurally identical to LSAR?
Feasible, and **strictly worse**:

1. **It still needs the feeds.** The residual target must be relative to what each step stood on, and
   `s_feed` is loop-local either way — so it is a SUPERSET of this plan's plumbing, plus a changed
   `_rollout_step` return contract (the compiled unit at `:521-539` and all three loop branches). The feeds
   plan needs **zero** change to `_rollout_step`.
2. **The second pass survives regardless.** At `p_tf >= 1` there is no rollout (`lit.py:127`), so
   `loss_terms` must keep a parallel-pass implementation anyway → **two dynamics-loss code paths to keep in
   sync**, the exact failure the file's own INVARIANT comment warns about (`:659-668`).
3. **The saving is ~nothing.** The parallel pass is one backbone forward over L-1 = 71 positions against the
   rollout's 64 forwards over W = 32 windows — **~3.5%** of the step's backbone FLOPs, and the
   well-parallelised part, while the AR loop is dispatch-bound. Wall-clock saving ≈ 0.
4. **It kills two capabilities.** Diffusion forcing (context noising + level conditioning, `:966-973`) exists
   only in the parallel pass, and `dynamics_detach_encoder` cannot be applied to a rollout-internal `h`.
5. **Coverage loss.** It would supervise only the F rollout positions; the parallel pass also covers the
   P-1 context transitions.

Also worth recording: the parallel pass is **not** a different attention regime — `forward` builds the same
sliding-causal window mask (`transformer.py:172-173`) and RoPE is relative, so a feeds-conditioned second
pass reproduces the rollout's `h` up to padding arithmetic.

### The flag survives — and Distribution is why

`null → class convention` is **dropped**: configs are already per-class yamls, so each states its own value.
More importantly the flag expresses something **p_tf cannot**: *decoupling the dynamics-loss context regime
from the decode-loss rollout regime.* DreamerV3 wants its world-model loss teacher-forced while this
codebase deliberately rolls out the decode loss — and you cannot get "dynamics TF, decode AR" from p_tf,
because `p_tf ≥ 1` flips `lit.py:127` to the parallel path and un-rolls the decode loss too. So for
`MultiModalDistribution` it is a permanent per-class semantic; for Flow it is a repro shim plus a legitimate
ablation (`dynamics_follows_p_tf=false` + `dynamics_detach_encoder=true` is the IWS quadrant).

### Two corrections to earlier claims in this file

* **"p_tf controls the context continuously"** — wrong. `s_feed` is a per-(sample, step) **hard Bernoulli
  choice** (`:600-601`), not a blend. Continuous in *distribution*, not per tensor.
* **"with `p_tf_batch_granular: false`, p_tf is only ever 1.0 or 0.0"** — true for the CURRENT defaults
  (`p_tf_warmup_epochs: 1` at `mm_flow.yaml:62`, `p_tf_batch_granular: false` **LOCKED** at `:112`), so
  under today's config intermediate p_tf never occurs. An earlier revision of this file cited
  `mm_flow.yaml:14` as "the yaml's own recommended F=64 config" with warmup=4 — **wrong**: line 14 is the
  *2026-07-28 historical* recipe (the yaml says so at `:15-19`), and it also carries `recon_frac=0.25`,
  which contradicts this project's own locked lever. Intermediate p_tf is reachable
  (`linear_schedule`, `schedules.py:10-16`, warmup=4 → 1.0/0.75/0.5/0.25) but is **not** a current default.

**Honest scope.** In the *locked default* (warmup=1, batch_granular=false) the feeds scheme is
arithmetically identical to the gated pred_bag plan. Its advantage is correctness at warmup>1 and
batch-granular ramps, plus structural removal of two traps — same size change at the loss site, one small
plumbing change instead of a special case.

### Decisions (all three now settled)

* **DECIDED: attached** (reversed 2026-08-25). Detached would teach only "from this drifted point, here is
  the one-step correction" — it still would not train COMPOUNDING, which is the entire diagnosis in section
  1. Attached lets an error at step 64 adjust the flow's behaviour at steps 1..63, and matches LSAR, which
  keeps `online = pred_bag` attached (`:832`). The graph is already retained for the decode loss, so the marginal
  cost is **≈ zero on both axes**: autograd accumulates into nodes the decode loss's backward already
  traverses and runs each once, and the memory delta is just the stacked feeds tensor (a few MB bf16).
  `detach_every=32` bounds the reach identically to the decode loss.
  **Guard:** `dynamics_detach_encoder=true` would silently sever the attachment at `:964`, demoting the fix
  to detached one-step corrections — assert or document that the two are incompatible. Drop the `.detach()`
  on the feeds collection. (Note `follows_p_tf=false` + `detach_encoder=true` is the IWS quadrant; `true` +
  `true` is just broken.)
* **DECIDED: accept the val change.** Val is hard-wired to `p_tf = 0.0` (`lit.py:99`) and train ramps to 0
  after warmup, so with the flag on val's dynamics loss matches train's steady state AND matches how we
  evaluate — MORE consistent than today, not less. The only cost is that `val/loss/dynamics/latent` and
  `val/loss/total` stop being comparable to historical runs; a bookkeeping note for the record, not a
  behaviour split. Checkpoint monitor unaffected (it tracks an image metric).
* **DECIDED: accept the action-prior change.** The prior conditions on the same substituted `h`
  (`:985-994`) so it trains on imagined latents, while `action_context` (`:996-1003`) stays clean at eval.
  Inside MPPI the prior is fed IMAGINED latents anyway, so training on imagined makes train match its actual
  use — the asymmetry is with the clean-context eval path, and that is the less important one. Recorded, not
  blocking.

## 7. Verification

* `dynamics_follows_p_tf=false` must be **bit-identical** to current `main` — with the flag off `s_ref`
  aliases `z[:, :-1]` and the target expression reduces to line `:976` exactly — same param count, same first-step
  loss on a fixed seed.
* With it on, the metric to watch is `latent_cos` at `@+32/@+64` (now logged every epoch), which is the
  quantity the change is designed to move. Then `OL LPIPS@+128` against `recon_frac=1.0`'s 0.278.
* `latent_motion_ratio` should rise toward 1 if the fix works — the model currently under-rotates (0.56 at
  h = 64), and a loss that penalises being off-course should push it to commit.
