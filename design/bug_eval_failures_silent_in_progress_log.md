# BUG: non-fatal eval failures never reach `progress.log` — and `ood_horizon` OOMs under the U-Net decoder

**Status:** reported, NOT fixed. Filed from a live run so the reproduction is exact.
**Found:** 2026-08-07, on `robocasa-scene4-4h` (recorded, no-simulator) at commit `5f28c0d`.
**Reported by:** an agent session driving `wizard/prompt.md`; the fixes below are deliberately left to the
implementer. Two independent bugs, in priority order.

---

## Bug 1 (the important one) — a skipped eval is invisible in `progress.log`

`logging/callback.py:220` reports non-fatal eval failures with a **bare `print()`**:

```python
except Exception as e:   # a DIAGNOSTIC eval must NEVER kill training ...
    import traceback
    print(f"\n[eval:{name} @ep{epoch}] FAILED non-fatally ({type(e).__name__}: {e}); "
          f"skipping this routine, CONTINUING training.", flush=True)
    traceback.print_exc()
```

`print()` goes to **stdout only**. Everything else in a run — the startup banner, the `[env-contract]`
line, per-epoch progress, every eval's `start`/`done`, the loss summaries — goes to
`<run_dir>/progress.log` via `_plog()` (`controller/run.py:24`). So the one line that says *"a
diagnostic you asked for did not run"* is the single line that the run's own log file does not contain.

Measured on the live run:

```
grep -c 'FAILED non-fatally' <run_dir>/progress.log   ->  0
grep -c 'FAILED non-fatally' <captured stdout>        ->  1
```

**Why this is worse than a normal missing log line.** `progress.log` is the artifact that survives: it
lives in the run dir, it is what you open when reviewing a run days later, and it is what tooling tails.
Stdout is only retained if the launcher happened to redirect it. This run did (`> out/mini_ah_on.out`),
but the in-repo launch scripts do not universally, and `docker compose exec` sessions lose it entirely.
The failure mode is therefore: **the eval silently does not exist, and `progress.log` reads as a clean
run.** What you see is a `start:` line with no matching `done` line, ~150 lines apart, and nothing else:

```
[08-07 04:31:20] [eval_ood_horizon @ep5]   0% — start: 8 eps, H=791, heads=['proprio', 'image']
[08-07 04:31:31] [eval_action_distribution @ep5]   0% — start: 26 val episodes, ...
```

That is the *entire* trace in `progress.log` of a routine that failed. It took grepping a separately
captured stdout file to discover that ~4 epochs of a diagnostic had been silently discarded on both arms
of an A/B.

### Suggested fix

`_plog` is already the house helper and the callback already holds `self.writer`:

```python
from ..controller.run import _plog          # already imported by evaluation/routines.py
...
except Exception as e:
    import traceback
    _plog(self.writer, f"[eval:{name} @ep{epoch}] FAILED non-fatally ({type(e).__name__}: {e}); "
                       f"skipping this routine, CONTINUING training.")
    traceback.print_exc()                   # full traceback can stay on stdout
```

Worth considering alongside it:

- **A run-end summary of skipped evals.** A run that skipped `ood_horizon` at epochs 5/10/15/20 should say
  so once at the end, not only at each occurrence, e.g. `[eval] SKIPPED this run: ood_horizon x4 (OOM)`.
- **A wandb scalar** (`eval/skipped/<name>` = 1) so a skipped diagnostic is visible on the dashboard rather
  than only in a text file. Currently a missing metric is indistinguishable from a metric that was never
  enabled.
- The comment says *"Log loudly + go on"* — the intent is right, the mechanism just doesn't reach the log
  that matters.

---

## Bug 2 — `eval_ood_horizon` OOMs with the U-Net image decoder, and `autobatch_headroom=0.25` doesn't cover it

Both arms of a 2-GPU A/B lost **every** `ood_horizon` eval (0 products in either run dir):

```
[eval:ood_horizon @ep5] FAILED non-fatally (OutOfMemoryError: CUDA out of memory.
Tried to allocate 18.54 GiB. GPU 0 has a total capacity of 93.10 GiB of which 12.54 GiB is free.
Including non-PyTorch memory, this process has 80.54 GiB memory in use. Of the allocated memory
72.56 GiB is allocated by PyTorch, and 7.23 GiB is reserved by PyTorch but unallocated.)
```

The traceback lands in the U-Net decoder's upsampling path:

```
  h = up(torch.cat([h, skip], dim=1), g)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.54 GiB
```

### Root cause: two independently-reasonable defaults interacting

1. `data.autobatch=true` sizes `data.batch` to fill `VRAM * (1 - autobatch_headroom)` by probing the
   **training** step. It chose `batch=48`, confirmed at 65.8/75 GB, leaving ~12.5 GB free.
2. `eval_ood_horizon` then does the largest allocation in the whole run — 8 episodes x H=791 with image
   decode — and wants **18.54 GiB in one block**. More than the slack.
3. `autobatch_headroom: 0.25` was raised from 0.15 on 2026-08-06 *for this class of problem* — its comment
   reads *"reserve fraction of VRAM for eval-phase spikes ... RAISE if an eval OOMs (control/denoising
   spike above the train step)"*. But it was calibrated while the image decode was **ViT**. The default
   image decode changed to `decode_arch: unet` in the same era, and the U-Net's skip-connection
   concatenations at progressively full resolution make decode peak memory much larger than the ViT's
   fixed 8-token patch decode. 0.25 is no longer enough.

Evidence that the decoder is the variable: the *previous* A/B on this same dataset, same `d=128`, same
`F=64`, with **ViT-MSE** decode, ran `ood_horizon` fine at epochs 5/10/15/20 with total process memory
~27-45 GB. Nothing else about the eval changed.

### Why it matters more than "a diagnostic is missing"

On the earlier ViT A/B, `eval_ood_horizon` was the **earliest** detector of an action-head-induced
divergence — it showed a 49% degradation in `pointwise_error_mean` at epoch 10 while `val_loss` still
looked healthy (0.4601) and only blew up at epoch 15. The current A/B is testing *that same
action-head question*, and it is running with its early-warning instrument silently disabled.

### Suggested fixes (any one unblocks; the first is probably correct)

- **Make autobatch account for the eval peak, not just the train step.** Either probe a representative
  eval allocation, or scale headroom by decoder type. A budget that a known-enabled eval cannot fit in is
  the actual defect — the current design guarantees an OOM whenever image evals are on and autobatch is on.
- **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — the error itself notes 7.23 GiB reserved-but-unallocated,
  i.e. fragmentation. This alone may be sufficient and costs nothing.
- **Cap the eval** for image heads: `eval.n_episodes` is already reduced to 8 when image heads are present
  (`routines.py`), but 8 x H=791 with a U-Net is still too big at an autobatch-sized training footprint.
- **Chunk the U-Net decode** over the horizon inside the eval instead of decoding the full rollout at once.

Note the two bugs compound: fix 2 without fix 1 and the next incompatible-eval regression is again
invisible in `progress.log`.

---

## Reproduction

```bash
# commit 5f28c0d, 1x H100 (93 GiB)
uv run python -m quickdraw.train_world_model \
  model=mm_flow model.size=mini model.recon_frac=0.25 \
  model.compile_rollout=true data.autobatch=true data.F=64 \
  data.root=<robocasa-scene4-4h snapshot> data.repo_id=robocasa-scene4-4h \
  data.cam=robot0_agentview_left \
  environments=recorded environments.obs_dim=16 environments.action_dim=12 \
  model.action_dim=12 'model.modalities.0.dim=16' 'model.modalities.1.img_size=128' \
  model.p_tf_warmup_epochs=1 trainer.max_epochs=30 \
  eval.during_train.evals.control=false eval.during_train.evals.manifold=false \
  eval.during_train.evals.denoising_multistep=false eval.during_train.evals.denoising_aggregate=false \
  +run_summary.problem=... # (all 5)
```

At epoch 5, `ood_horizon` OOMs and is skipped. Confirm both bugs:

```bash
grep -c 'FAILED non-fatally' <run_dir>/progress.log     # 0  <- bug 1
find <run_dir>/logs -path '*ood_horizon*' -type f | wc -l   # 0  <- bug 2
```

Observed on both arms (`model.action_head.enabled` true and false), so it is independent of the action head.
`eval_action_distribution` is unaffected and completes normally (`w1=0.006, w1_mean=0.005`), which is why the
run otherwise looks healthy.
