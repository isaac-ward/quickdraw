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

## 2. The three silent failures

### 2.1 The loader cannot be sharded, and would not complain

`MMWindowLoader` (`src/quickdraw/data/dataset.py:206`) is a **hand-rolled iterator**, not a
`torch.utils.data.DataLoader`:

```python
def __iter__(self):
    order = torch.randperm(self.N, device=self.device) if self.shuffle else torch.arange(self.N, ...)
    for i in range(0, self.N, self.batch):
        j = order[i: i + self.batch]
```

Consequences under DDP:

* Lightning injects a `DistributedSampler` only into real `DataLoader`s. It cannot touch this. So
  `self.N` is the **full** window count on **every** rank.
* `torch.randperm` is seeded identically on every rank (§1), so rank 0 and rank 1 draw the **same
  permutation** and therefore the **same batches**.
* DDP then all-reduces (averages) two **identical** gradients. Averaging a value with itself is that
  value. So the run is mathematically identical to a 1-GPU run at the same batch, at **2x the
  electricity and 1x the throughput**.
* Nothing raises. Loss curves look normal. This is the single most likely way to "add multi-GPU" and
  believe it worked.

Fix: shard by rank inside the loader. Because the store is GPU-resident and indexed by a device
tensor, this is cheap and local — take the rank's stripe of the window index at construction:

```python
# in MMWindowLoader.__init__, after self.N is known
if world_size > 1:
    keep = torch.arange(rank, self.N, world_size, device=device)   # contiguous-free, no padding
    self.obs, self.act = self.obs[keep], self.act[keep]
    if self.frames is not None: self.win_idx = self.win_idx[keep]
    self.N = self.obs.shape[0]
```

Note this shards the **windows**, not the frame store — the frames stay fully replicated on both
cards. That is deliberate: at 128px the store is ~2.8 GB, replicating it costs 2.8 GB per card
(we have 80), and sharding it would break the global frame indexing that `win_idx` depends on.
Sharding windows-only is the right trade here; it stops being right at 256px + long episodes.

### 2.2 Every callback and every eval routine would run on all ranks

There is **zero rank-awareness anywhere in `src/quickdraw`**. Verified:

```
grep -rn "global_rank\|is_global_zero\|world_size\|sync_dist\|all_gather" src/quickdraw/
```

returns only unrelated hits (the word "strategy" in `models/collapse.py`). Nothing is gated.

So under DDP, both ranks would run:

* `RunWriter` (`logging/writer.py:124`) — two processes appending to the **same**
  `logs/.../metrics.jsonl`. Interleaved partial lines, i.e. a corrupt file, and every metric
  duplicated at each step.
* `LoggingCallback` — the eval block (ae_floor, ood_horizon, manifold, denoising_*) runs **twice**,
  both times on the full eval split. Eval is already ~50% of wall time, so this alone cancels most of
  the DDP speedup.
* Video/figure encoding — two writers racing on the same output paths.
* `ModelCheckpoint` x2 + `BestCkptMirror` — two processes writing `last.ckpt` / `best.ckpt` in the
  same dir. Lightning normally rank-gates its own checkpoint writes; `BestCkptMirror` is **ours** and
  is not gated.
* `ProgressPrinter` — doubled `progress.log`.

Fix: gate on `trainer.is_global_zero` at the top of each `on_*` hook in `LoggingCallback`,
`ProgressPrinter` and `BestCkptMirror`, and make `RunWriter` a no-op on non-zero ranks. The metrics
this produces are then **rank-0-only**, which for our eval routines is fine (they are full-split
autoregressive rollouts, not per-batch averages) but for `train/loss` means we log rank 0's shard
rather than the true mean — acceptable, and cheaper than plumbing `sync_dist=True` through a writer
that does not use `self.log`.

### 2.3 Autobatch would deadlock the run, not OOM it

`autobatch_find` (`src/quickdraw/training/setup.py`) currently:

* runs **before** the Trainer exists, so before DDP process-group init;
* probes on one device and budgets on `max_memory_reserved` minus `autobatch_reserve_gb` minus the
  measured resident frame store;
* sets a single `cfg.data.batch`.

Three problems, in increasing severity:

1. **The DDP reducer is not in the budget.** DDP holds gradient buckets plus its own view of the
   parameters. At 6.4M params that is ~50 MB — negligible here, but it is a real term that the
   4 GB `autobatch_reserve_gb` currently absorbs by luck rather than by design.
2. **The resident subtraction changes.** With windows sharded (§2.1) the per-rank window tensors
   halve, so the budget genuinely grows. If we shard and *don't* re-measure, we leave VRAM on the
   table — exactly the failure the autobatch work spent a week removing.
3. **THE BLOCKER: ranks must agree on the number of steps.** If rank 0 probes batch 32 and rank 1
   probes 30 (different fragmentation, a stray process, ECC-retired pages), the two ranks run a
   different number of batches per epoch. The rank that finishes first waits forever in the next
   all-reduce. That is a **hang with no error message**, the worst failure mode in this document.

Fix: keep the probe where it is (one rank, before init) and then **broadcast/min-reduce the result**.
Concretely — probe on every rank independently, then `torch.distributed.all_reduce(b, op=MIN)` and use
the min everywhere. Also, per-rank `data.batch` is a **micro**-batch under DDP: effective batch becomes
`data.batch x world_size`. Since `accumulate_grad_batches=1` and `data.window_stride=1` are both
LOCKED, DDP is the only knob we have that changes effective batch, and it changes it **silently** —
so the resolved config must record `effective_batch` explicitly or every future A/B against a
single-GPU baseline is confounded.

---

## 3. If we do it: the phases

### Phase 1 — correct DDP (the only phase worth doing soon)
1. Thread `rank`/`world_size` into `MMWindowLoader` and stripe the windows (§2.1).
2. Rank-gate `RunWriter`, `LoggingCallback`, `ProgressPrinter`, `BestCkptMirror` (§2.2).
3. Min-reduce the autobatch result across ranks; log `effective_batch = batch x world_size` into
   `config.resolved.yaml` (§2.3).
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
* wall-clock per epoch is **< 0.65x** the 1-GPU run. Anything near 1.0x means §2.1 is still broken
  and both ranks are chewing the same data.

### Phase 2 — only if eval becomes the bottleneck
Shard the eval routines by episode across ranks and gather. Worth it only because eval is ~50% of
wall time; not worth it before Phase 1 is proven, and it needs `all_gather` of variable-length
per-episode results, which is where most distributed bugs live.

### Phase 3 — never (for this model)
FSDP, tensor/pipeline parallelism, ZeRO. 6.4M parameters. The activation memory that actually binds
our batch is the F=64 BPTT rollout, which sharding parameters does nothing for.

---

## 4. Why two independent arms is currently better

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
