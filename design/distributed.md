# Multi-GPU: what it would take, and why we are single-GPU today

Status: **DESIGN ONLY. No code changed.** Written 2026-08-21 at the user's request
("are we set up to do multiple GPU runs? and how would that work with autobatch?" → "how would we go
about this? can we start with a design/multigpu.md?").

Machine: 2 x H100 80GB (`nvidia-smi` reports 95830 MiB each).

---

## 0. TL;DR

We are **not** set up for multi-GPU, and for the work we are actually doing that is the right call.
The two cards are currently spent on **two independent single-GPU arms of an A/B**, which is a better
use of them than one 2x-faster run: our bottleneck is *number of hypotheses tested*, not epoch time.

If we do want data-parallel DDP, the work is roughly **1 day** and the dangerous part is not DDP
itself — it is that **three separate things in this repo would silently produce a wrong-but-plausible
run instead of an error.** Those three are listed in §2. Read them before writing any code.

Recommendation, in order:
1. **Keep 2 independent arms** for hypothesis testing (status quo). No code.
2. If we need a single big run (bigger decoder, 256px, longer F), do **§3 Phase 1 only** — DDP with a
   real distributed sampler and rank-gated logging. Skip everything else.
3. Never FSDP/model-parallel. The model is 6.4M parameters. It fits ~14,000x over in one card.

---

## 1. What is pinned today, and where

`src/quickdraw/train_world_model.py:270`:

```python
# single GPU: the GPU-resident loader holds the whole set on one device (no DistributedSampler),
# so we pin devices=1 rather than let Lightning auto-pick DDP across both H100s.
trainer = L.Trainer(..., accelerator="gpu", devices=1, ...)
```

That comment is the whole story and it is accurate. `devices=1` is a **guard**, not an oversight:
Lightning's default `devices="auto"` would pick up both H100s and spawn DDP, and every failure in §2
would fire at once. Removing the pin without doing §3 gives a run that trains, logs, and checkpoints
happily while being **wrong**.

Also relevant: `seed_everything` is called first in `main`, with the same seed on every process. Under
DDP that is *correct* for parameter init (all ranks must start identical) and *fatal* for data order
(all ranks would draw the same permutation — see §2.1).

---

## 2. Measured first: where the wall-clock actually goes

Every claim below about what DDP can buy rests on this decomposition. Measured on
`logs/holiday/train_world_model_2026_08_18_09_16_26_bsp32mse_long`, epochs 4-20 (17 consecutive epochs,
spread under 0.5%):

| phase | per epoch | share | shardable by DDP? |
|---|---|---|---|
| **train loop** (4386 batches) | 1.343 h | **84.6%** | yes |
| **val loop** (autoregressive rollout on 4004 val windows) | ~13.7 min | **14.4%** | yes — it is a *second* `MMWindowLoader` |
| **eval routines** (`ood_horizon` 42 s + `ae_floor` 17 s) | 59 s | **1.0%** | no — rank-0 only |
| epoch period | 1.588 h | 100% | |

**This corrects a claim that was repeated several times in this project, including in the first draft of
this document: "eval is ~50% of wall time".** That number came from the comment on
`check_val_every_n_epoch` in `train_world_model.py`, which is about the **val rollout**, not the eval
routines — and even that is 14.4%, not 50%. The comment is stale, probably from a proprio-only era when
train epochs were far cheaper.

Two consequences, both of which change the plan:

1. **The 2-GPU ceiling is ~1.98x, not ~1.33x.** With train and val both sharded and only the eval
   routines serialised on rank 0: `(0.846 + 0.144)/2 + 0.010 = 0.505`. So a `< 0.65x` wall-clock gate is
   demanding but achievable; realistically expect 0.55-0.60x once NCCL and the straggler tax are paid.
2. **Phase 2 ("shard the eval routines") is demoted to not-worth-doing.** It was justified on eval being
   half the wall time. It is 1.0%. Amdahl caps the entire prize at 0.5% of an epoch, against
   variable-length `all_gather` of per-episode results — the highest bug-density-per-benefit code in the
   whole plan. **Do not do it.** What *does* need doing is sharding the **val** loader, which is the same
   fix as the train loader (§3.1) applied to a second call site, and is 14.4%.

---

## 3. The three silent failures, in detail

### 3.0 THE HALF THIS DOCUMENT ORIGINALLY MISSED: rank 1 re-runs `main()`

Added 2026-08-21 after a full read-only audit of the package. **This section outranks the three below.**

Under Lightning's default DDP launcher (`SubprocessScriptLauncher`, which is what you get from
`strategy="ddp"` in a script), rank 1 is **a subprocess that re-executes the entire script from the top**.
Every line of `train_world_model.py:98-292` runs twice, in two processes, *before* `trainer.fit` and
therefore before any `trainer.is_global_zero` exists to consult. The first draft of this document put all
three failures in and around `fit`. Roughly half the implementation work is actually here.

**(a) The run-summary uniqueness gate kills rank 1, and rank 0 then hangs.** `_assert_summary_unique`
(`train_world_model.py:61-85`, called at `:120`) globs `logs/*/auto_run_summary.txt` and raises if this
run's note matches an existing one. Sequence: rank 0 runs `main()`, writes its own
`auto_run_summary.txt` at `:240`, reaches `fit` at `:292`; the launcher spawns rank 1; rank 1 re-runs
`main()` from the top, finds **rank 0's freshly written identical summary**, and dies at `:81`. Rank 0
then sits in process-group init until the store timeout (~30 min), GPU idle, no error. And because the
gate is skipped on resume (`:119`), **fresh DDP runs never start while resumes work** — an incoherent
pattern nobody would guess from the symptom. Fix: skip the gate off rank 0, keyed on the launcher's
`LOCAL_RANK` env var, because there is no trainer yet to ask.

**(b) Rank 1's autobatch probe allocates on rank 0's card.** `train_world_model.py:150` passes
`torch.device("cuda")` — i.e. **device 0** — and runs before Lightning assigns per-rank devices. So rank
1's re-executed `main()` tries to allocate up to ~87 GB of probe tensors (`training/setup.py:373-399`) on
`cuda:0` while rank 0 is mid-startup on that same card. Spurious OOM on either side. Fix: probe on the
rank's own device (`LOCAL_RANK`), which is a prerequisite for anything else in §3.3.

**(c) Each rank computes its own timestamped run_dir, giving a split-brain layout.**
`utils/logging.py:17-23` builds `logs/train_world_model_<%Y_%m_%d_%H_%M_%S>_<exp>` at
second resolution, independently per process. Rank 0 spends minutes in autobatch + data load before
`fit`, so rank 1 spawns later and its timestamp **differs by construction**. Rank 1 then gets a ghost run
dir with its own `checkpoints/`, `config.resolved.yaml`, wandb run and `progress.log` — while Lightning
*broadcasts* `ModelCheckpoint.dirpath` from rank 0, so rank 1's checkpoint bookkeeping points at rank 0's
directory and its writer points at the ghost. Fix: make run_dir rank-consistent (compute on rank 0 and
pass via env before spawn, or derive it deterministically). **This must be fixed first**, because until
it is, the section-A file races below are not even races — they are two separate directories, and every
gate you add is untested.

**(d) `wandb.init` runs on both ranks** (`logging/writer.py:176`), producing two wandb runs with the same
name per launch and double-uploading all media.

**(e) Duplicate cold-cache work.** Both ranks download the HF snapshot (`setup.py:231`) and decode every
mp4 into the frame cache (`data/dataset.py:181-183`). The cache write is already atomic (per-pid tmp file
+ `os.replace`), so there is no corruption — just ~12 GB and several minutes of duplicated work on a cold
cache.

**Note on the resume path**, which is the safest corner of all of this: on `+resume=`, both ranks derive
the *same* run_dir (`:136-139`), skip the uniqueness gate (`:119`), and skip autobatch (`:152-178`, both
reading the same `config.resolved.yaml`, hence the same batch). But because the directory is genuinely
shared there, resume is precisely where the file races in §3.2 would fire **today**, ungated.

### 3.1 The loader cannot be sharded, and would not complain

`MMWindowLoader` (`src/quickdraw/data/dataset.py:206`) is a **hand-rolled iterator**, not a
`torch.utils.data.DataLoader`:

```python
def __iter__(self):
    order = torch.randperm(self.N, device=self.device) if self.shuffle else torch.arange(self.N, ...)
    for i in range(0, self.N, self.batch):
        j = order[i: i + self.batch]
```

**Why Lightning cannot rescue this.** Lightning injects a `DistributedSampler` in
`_update_dataloader`, which reconstructs the dataloader from its `__init__` args — it requires an actual
`DataLoader` instance to read `.dataset`, `.sampler`, `.batch_size` off. Our object is *duck-typed*: it
has `__iter__` and `__len__` and nothing else. Lightning accepts it as an iterable, wraps nothing,
injects nothing, and **emits no warning**. There is no `isinstance` failure to trip over.

**Why the result is silent rather than merely slow.** Three facts compose:

* `L.seed_everything(seed, workers=True)` (`train_world_model.py:101`) sets the *same* global seed on
  every rank. The `workers=True` part seeds *DataLoader worker* processes with a rank-derived offset —
  but this loader has no workers (the `fast_gpu` path is GPU-resident by design), and `torch.randperm`
  runs in the **main** process off the plain global seed.
* So rank 0 and rank 1 generate a **bit-identical permutation** and therefore identical batches.
* DDP's gradient hook **averages** across ranks (`all_reduce(SUM)` then divide by `world_size`).
  Averaging a tensor with an identical copy of itself returns that tensor.

The run is therefore *mathematically identical* to a 1-GPU run at the same per-rank batch. Loss curves,
metrics, checkpoints — all indistinguishable. **There is no signal anywhere in the metrics.** The only
observable tell is that wall-clock per epoch barely moves, which is exactly the thing a person adding
multi-GPU support is least likely to treat as a bug.

**The fix is not just "stripe the windows" — that has two traps of its own.**

*Trap A: unequal shards deadlock.* The obvious `torch.arange(rank, N, world_size)` gives rank 0 one more
window than rank 1 whenever `N % world_size != 0`. Our train split is **35085 windows**, so 2 ranks get
17543 and 17542. Whether that matters depends on the batch: at batch 17 both `ceil` to 1032 steps and it
works *by luck*; at some other batch they differ by one step, and the rank with fewer steps enters the
next epoch's first `all_reduce` while the other is still in the last backward — a **hang with no error
message**. Never rely on the ceiling arithmetic. Truncate to a common length first:

```python
N_common = (self.N // world_size) * world_size     # drop <= world_size-1 windows of 35085
```

*Trap B: a fixed stripe is a fixed partition.* If the stripe is computed once in `__init__`, rank 0 sees
the **same half of the dataset for all 40 epochs**. Per-rank shuffling reshuffles *within* the shard, so
the gradient stays unbiased in expectation, but any systematic difference between the halves (and there
is one — episodes are concatenated in order, so a stripe by index correlates with episode identity) never
washes out across epochs. `DistributedSampler` avoids this by shuffling **globally with an epoch-keyed
seed and then sharding**, so the assignment changes every epoch. Same shape here, and it is one line:

```python
def __iter__(self):
    g = torch.Generator(device=self.device).manual_seed(self.epoch)   # SAME seed on every rank
    order = torch.randperm(self.N, generator=g, device=self.device) if self.shuffle else torch.arange(...)
    order = order[:(self.N // self.world) * self.world][self.rank::self.world]   # global shuffle THEN shard
    for i in range(0, order.numel(), self.batch):
        ...
```

Note the inversion: the seed must be **shared** (so every rank permutes identically) and the *shard*
provides the difference — the exact opposite of the usual instinct to give each rank a different seed.
Giving ranks different seeds re-creates the overlap problem, just non-deterministically.

*What NOT to shard: the frame store.* `self.frames` is one concatenated `(N_total, H, W, 3)` uint8 tensor
and `win_idx` holds **global** frame indices into it. Sharding it would require remapping every index per
rank and would cut across episode boundaries. It is ~2.8 GB at 128px against 80 GB of card, so replicate
it: shard **windows**, keep **frames** whole. This trade inverts at 256px (~11 GB) combined with long
episodes, and that is the point at which this design needs revisiting.

*It applies to val too, and val needs a DIFFERENT scheme.* Both loaders come from a single construction
site — `window_loaders()` (`training/setup.py:659-679`) loops `for split, shuffle in (("train", True),
("val", False))` and both branches hit the same `MMWindowLoader(...)` call at `:675`/`:678` — so sharding
inside `__init__`/`__iter__` covers train and val with no second call site to forget. **But val is
`shuffle=False` by design**, so the epoch-keyed global shuffle above does not apply to it: a naive stripe
of val is a *permanently* fixed half, and since index correlates with episode identity, rank 0's val
metrics become a mean over a fixed, episode-correlated half of val for the entire run. That is tolerable
for a curve read for trend and **not** tolerable for the checkpoint monitor (§3.2). Use a deterministic
round-robin stripe for val and fix the monitor with `sync_dist=True`.

*Do NOT shard the eval path.* `eval_episodes()` (`setup.py:681`) is a different class
(`TrajectoryDataset`) feeding the rank-0-only eval routines, which need the full episode set to compute a
true full-split number. Shard `MMWindowLoader`; leave `TrajectoryDataset` alone. They are already separate
classes, so this is a rule about which constructor to touch, not a runtime branch.

### 3.2 Nothing in `src/quickdraw` knows what rank it is

Verified — this returns only unrelated hits (the word "strategy" in `models/collapse.py`):

```
grep -rn "global_rank\|is_global_zero\|world_size\|sync_dist\|all_gather" src/quickdraw/
```

So under DDP *both* ranks run every callback. What that actually does, in order of how hard it is to
notice:

* **`metrics.jsonl` is corrupted, not just duplicated.** `RunWriter` (`logging/writer.py:124`) appends
  through Python's buffered writer. `O_APPEND` guarantees atomicity only for a single `write(2)` under
  `PIPE_BUF`; an 8 KB buffered flush can split a JSON record across two syscalls, and the other rank's
  flush can land in the gap. The result is unparseable lines *in the middle* of the file — which every
  downstream reader (the watchdog's `last_epoch`, every analysis script) handles by `except: continue`,
  so the corruption presents as **silently missing epochs**, not as an error.
* **Every eval routine runs twice**, wasting the second card's 59 s. Cheap, but see the timeout trap
  below — this is the section that becomes dangerous once you *stop* running it twice.
* **Two writers race on the same video and figure paths.** Non-atomic, so a half-written mp4 is possible.
* **`ModelCheckpoint` is internally rank-gated by Lightning; `BestCkptMirror` is ours and is not**
  (`logging/callback.py:524`). It copies `best_model_path` → `best.ckpt` and appends to `progress.log`
  from both ranks. Two concurrent copies of a multi-hundred-MB checkpoint to one destination path can
  interleave and produce a **corrupt `best.ckpt`** — the one artifact whose loss costs the most.

**The fix is NOT a blanket `if not trainer.is_global_zero: return` on every hook.** The first draft of
this document said exactly that, and the audit shows it would corrupt the model in four places. Some hooks
in this package are *functional*, not telemetry, and gating them silently desynchronises the ranks — which
is the same failure class as §3.1, and worse, because DDP synchronises **gradients** and never
**parameters**, so once two ranks take different optimizer steps they never re-converge.

**MUST run on ALL ranks — never gate these:**

| site | what it does | what gating it does |
|---|---|---|
| `training/lit.py:212-269` `configure_gradient_clipping` | *performs* the clip (`:251`) and zeroes non-finite grads (`:247-249`) | rank 1 takes an **unclipped or NaN** optimizer step while rank 0 takes a clipped one → parameters diverge permanently |
| `training/lit.py:275-276` `on_train_batch_end` → `on_optimizer_step()` | EMA target update, mutating `ema_modalities` in place (`models/multimodal.py:652-657`) | rank 1's EMA encoder **freezes at init** → it computes a different loss and gradient every step, averaged into rank 0's |
| `logging/callback.py:250` `_calibrate_latent_affine` (inside `on_fit_start`) | writes model buffers `lat_mean/lat_std/lat_calibrated` (`models/modalities.py:303-305`) | rank 1 keeps identity stats — i.e. **silently `latent_norm: none`**, the exact failure the comment at `callback.py:247-249` warns about |
| `logging/callback.py:255-273` `assert_identity_floor` (inside `on_fit_start`) | a deliberate fatal `raise` at `:270` on a broken fresh init | rank 0 dies alone, rank 1 blocks in its first collective → hang instead of a clean error |

So `on_fit_start` in particular must be split: **run the computation and the assertion on every rank**
(both are deterministic, so both ranks reach the same answer and die together if it is bad), and gate only
the `writer.*` / `_plog` calls inside it. Do not rely on DDP's `broadcast_buffers=True` to paper over the
calibration case — it probably would, and depending on an accident is how the latent-norm bug returns.

**Safe to gate (pure telemetry):** all nine `ProgressPrinter` hooks (`callback.py:141-204`);
`LoggingCallback.on_train_epoch_start` (`:306-314`), `on_train_batch_start/end` (`:316-322`),
`on_train_epoch_end` (`:387-486` — the eval-routine runner), `on_validation_epoch_end` (`:488-514`),
`on_fit_end` (`:516-521`); `BestCkptMirror.on_validation_end` (`:543-559`); and `RunWriter` as a no-op off
rank 0 — including `wandb.init` at `writer.py:176`.

**One thing that needs REDUCTION, not gating — and it is functional.** `val/metric/proprio/<metric>`
(`training/lit.py:190`) is the `ModelCheckpoint` monitor (`train_world_model.py:248`). Under sharding it
is a per-rank number, so the two ranks **disagree about what "new best" means**: desynced
`best_model_path` and top-k state, and `save_checkpoint` ends in a strategy barrier, so depending on the
Lightning path this is an intermittent hang at validation end rather than merely a wrong `best.ckpt`.
It *does* go through `self.log`, so the objection below about `RunWriter` not using `self.log` does not
apply to it: give this one `sync_dist=True`. Also note "`ModelCheckpoint` is internally rank-gated by
Lightning" is only half true — the **file write** is; the **top-k decision** runs on every rank against
per-rank metrics.

**The nuance is what that silently changes about the numbers**, and the first draft under-counted it.
It is not just `train/loss` and `val/loss`: **everything** logged via `self.log` in `_step` goes
shard-local — all `*/loss/{flow,pred_latent,proprio,image,roundtrip_*}` (`lit.py:176`),
`*/loss/{physical,contraction}` (`:169`), `val/metric/proprio/obs_error` (`:191`), every
`val/metric/<img>/{mse,l1,psnr}` (`:196-199`), and `collapse/*` (`:202`) — plus the checkpoint monitor
called out above. Two corrections in the other direction: `grad/*` (`:253-269`) is computed on
**post-all-reduce** gradients and is therefore already global, needing nothing; and `schedules/*`
(`:178-180`) is identical by construction. For the eval routines that is exactly
right (they are full-split autoregressive rollouts driven by their own episode lists, not per-batch
reductions, so rank 0 computes the true value). For `train/loss` and `val/loss` it means a
half-sample-size estimate — noisier, and *biased* if the shard is not representative, which is precisely
what §3.1's epoch-keyed global shuffle exists to guarantee. The alternative is plumbing `sync_dist=True`
through a writer that does not use `self.log` at all, which is real work for a cosmetic gain on a curve
we read for trend, not for absolute value. **Take the rank-0 estimate, and note it in the record so
nobody later compares a DDP `train/loss` against a single-GPU one and reads the extra noise as a result.**

**The timeout trap, which only appears after you fix this.** Once eval is rank-0-only, rank 1 finishes
the epoch and blocks in the first collective of the next one while rank 0 spends 59 s in eval. That is
correct behaviour — but NCCL has a **collective timeout** (`ProcessGroupNCCL`, 10 min in the version
Lightning configures by default), and when it expires it does not warn, it **aborts the job**. 59 s is
comfortable; it will not stay 59 s. `eval.during_train.evals.control` and `action_distribution` are off
on this dataset, `eval.horizon` is 128 of a possible 1024, and `ood_horizon` already scales with episode
count. Any of those growing past 10 minutes turns a working run into a mysterious mid-training abort.
**So raising `timeout=` on the process group is part of this fix, not an optimisation** — set it to
something like 60 min, and treat the longest rank-0-only section as a budget you are spending.

The audit found the budget drivers are much bigger than the 59 s measured today, and two of them have no
upper bound at all: `eval_interpret` makes **network calls to OpenAI** with a 120 s timeout per clip
(`evaluation/interpret.py:60-65`, threaded at `routines.py:800-804`), `eval_manifold` runs t-SNE/UMAP on
8000 points on CPU (`routines.py:396-407`), and `eval_control` runs full MPPI with EGL rendering. All
three are currently **off** on this dataset. They should stay off on a DDP cadence, or the process-group
timeout becomes a bet on a third-party API's latency.

### 3.3 Autobatch would hang the run rather than OOM it

`autobatch_find` (`src/quickdraw/training/setup.py`) currently runs **before** the Trainer exists — so
before the process group is initialised — probes on one device, budgets on `max_memory_reserved` minus
`autobatch_reserve_gb` minus the measured resident frame store, and writes a single `cfg.data.batch`.

Three problems, increasing in severity:

1. **The DDP reducer is not a budgeted term.** DDP allocates gradient buckets (and, transiently, a second
   copy of the bucketed gradients during `all_reduce`). At 6.4M parameters in fp32 that is ~25 MB per
   bucket set — genuinely negligible here, and currently absorbed by the 4 GB `autobatch_reserve_gb` *by
   luck rather than by design*. It stops being negligible if the dynamics rebalance (`d` 128→192, deeper
   backbone) lands, so it belongs in `_resident_frame_bytes`-style explicit accounting rather than in slop.
2. **The resident subtraction changes sign-of-error.** Sharding windows (§3.1) halves the per-rank window
   tensors, so the *real* budget grows. If we shard and do not re-measure, we systematically under-fill
   the card — the precise failure the autobatch rewrite spent a week removing (45% → 90% utilisation).
   The frame store, being deliberately unsharded, does **not** shrink, so this is not a simple halving.
3. **THE BLOCKER: ranks must agree on the step count.** Nothing forces two ranks to probe the same batch.
   They can legitimately differ — a stray process on one card, different fragmentation history, ECC-retired
   pages, or simply the linear fit landing either side of a boundary (our two live arms just probed 7 and
   17 for configs differing only in `recon_frac`, which shows how sharp that boundary is). Different
   per-rank batch → different number of steps per epoch → the shorter rank enters a collective the longer
   rank never reaches → **hang, no error, no OOM, no traceback**. The worst failure mode in this document,
   because it presents identically to a slow epoch.

   Fix — and note the first draft's prescription ("keep the probe exactly where it is and reduce it") was
   **self-contradictory**: this same section says the probe runs before the Trainer exists, so there is no
   process group at `train_world_model.py:149-151` and `all_reduce` is simply unavailable there. Three
   implementable options, and the loaders are built from `cfg.data.batch` at `:183`, so whichever you pick
   must land before `window_loaders`:

   * **(a) Phase-1 answer — refuse to guess.** For `world_size > 1`, raise unless
     `data.autobatch=false data.batch=N` is passed explicitly. Deletes the failure mode for one manual
     number, and is honest about what is implemented. Ship this first.
   * **(b) Init the process group yourself** before the probe (`dist.init_process_group(backend="nccl",
     timeout=timedelta(minutes=60))` reads the same env vars; Lightning's `DDPStrategy` checks
     `is_initialized()` and reuses it), then `all_reduce(b, op=ReduceOp.MIN)`. `MIN` matters — max or mean
     could pick a batch that OOMs the tighter rank. This is also where §3.2's timeout gets set, so the two
     fixes share a line.
   * **(c) Probe on rank 0 only and pass the result out-of-band** (env var / file). Natural under the
     subprocess launcher specifically, because rank 0 finishes autobatch *before* rank 1 is spawned.

   Whichever is chosen, `torch.device("cuda")` at `:150` must become the rank's own device first (§3.0b).

**And one thing that is not a bug but will confound every future A/B.** Under DDP, `data.batch` is a
**per-rank micro-batch**: effective batch is `data.batch x world_size`. With
`accumulate_grad_batches=1` and `data.window_stride=1` both LOCKED, DDP is the *only* remaining knob that
changes effective batch — and it changes it **implicitly, as a side effect of a device count**. So
`config.resolved.yaml` must record `effective_batch` explicitly. Without that, six months from now a
2-GPU run gets compared to a 1-GPU baseline at "the same batch 17" when one of them was really 34, and the
learning-rate coupling shows up as an architecture result. This project has already retracted claims for
smaller reasons.

## 4. If we do it: the phases

### Phase 1 — correct DDP (the only phase worth doing)
**Order matters — 1 and 2 are prerequisites, not conveniences.** Until run_dir is shared, the file races
in §3.2 are not races at all (they are two separate directories), so every gate you add is untested.

1. **Make the pre-fit path rank-aware** (§3.0), keyed on `LOCAL_RANK` because there is no trainer yet:
   share one run_dir, skip the uniqueness gate off rank 0, probe on the rank's own device, and init wandb
   only on rank 0.
2. **Require an explicit batch for `world_size > 1`** (§3.3 option a). Defers the whole autobatch/DDP
   interaction rather than shipping an unexercised min-reduce.
3. Thread `rank`/`world_size` into `MMWindowLoader`: shared epoch-keyed shuffle THEN shard, truncated to a
   common length, frame store left replicated. Covers train and val from one constructor; val needs a
   deterministic round-robin because it is `shuffle=False`. Leave `TrajectoryDataset` unsharded (§3.1).
4. Rank-gate telemetry only — `RunWriter`, `ProgressPrinter`, `BestCkptMirror`, and the listed
   `LoggingCallback` hooks — **splitting `on_fit_start`** so latent-affine calibration and the ae_floor
   assertion still run on every rank. Add `sync_dist=True` to the checkpoint monitor at `lit.py:190`.
   Raise the process-group `timeout` to ~60 min (§3.2).
5. Log `effective_batch = batch x world_size` into `config.resolved.yaml` (§3.3).
4. Replace `devices=1` with `devices=cfg.trainer.get("devices", 1)` and
   `strategy="ddp_find_unused_parameters_false"` when `devices > 1`. Default stays 1, so every
   existing recipe and the whole watchdog/resume path are bit-identical.

**Verification gate — do not trust it without this.** Run the same config at `devices=1` and
`devices=2` with `trainer.limit_train_batches` set so both see the same number of *windows*, and
require:
* the two `train/loss` curves agree to within run-to-run noise (our measured floor is +-0.8 dB on
  OL@+64, +-0.010 on LPIPS);
* `logs/.../metrics.jsonl` from the 2-GPU run has **no duplicated steps** (the §2.2 tell);
* `mem/peak_trainval_reserved_gb` on each rank is within ~1 GB of the 1-GPU run at the same
  per-rank batch (the §2.3 tell);
* wall-clock per epoch is **< 0.65x** the 1-GPU run. The measured floor is 0.505x (§2), so 0.55-0.60x is
  a pass and anything near 1.0x means §3.1 is still broken and both ranks are chewing the same data.
  0.75x specifically means the **val** loader was left unsharded.

### Phase 2 — DO NOT DO (withdrawn)
The plan was to shard the eval routines by episode and gather. It was justified on eval being ~50% of
wall time. **Measured, it is 1.0%** (§2), so Amdahl caps the whole prize at half a percent of an epoch —
against variable-length `all_gather` of per-episode results, the highest bug-density-per-benefit code in
this document. Withdrawn. The 14.4% that *is* worth having is the **val** loader, which is Phase 1
item 1 applied to a second call site, not a new phase.

### Phase 3 — never (for this model)
FSDP, tensor/pipeline parallelism, ZeRO. 6.4M parameters. The activation memory that actually binds
our batch is the F=64 BPTT rollout, which sharding parameters does nothing for.

---

## 5. Why two independent arms is currently better

DDP buys **wall-clock on one hypothesis**. Two cards buy **two hypotheses in the same wall-clock**.
Every result in `wizard/records/robocasa-scene4-4h.md` came from the second mode, and the binding
constraint on this project has consistently been that effects are small relative to the +-0.8 dB
run-to-run noise floor — which means we need *more replicates and more arms*, not faster arms.

DDP becomes the right answer when a single configuration stops fitting or stops finishing:
* a decoder large enough that batch drops below ~8 (gradient noise starts to matter),
* 256px (currently out of the running per the user),
* F much longer than 64,
* or a real >=100-epoch schedule on a config we have already committed to.

None of those is true today.
