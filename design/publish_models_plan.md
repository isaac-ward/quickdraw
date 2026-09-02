# Publishing world models to the Hub so someone else can load them and imagine

Status: **BOTH MODELS PUBLISHED AND PUBLIC 2026-09-02** — see "What was published" at the bottom.
Requested by the user; every claim below was checked against the source or measured on a real checkpoint.

## The one thing that makes this non-trivial

**A checkpoint is not a model.** `best.ckpt` contains `state_dict` and nothing else usable —
`hyper_parameters` is EMPTY (verified: `sorted(d['hyper_parameters'])` -> `[]`). So the weights carry no
record of the architecture that produced them. Four separate things are needed to imagine, and they
currently live in three different places:

| what | where it lives now | needed for |
|---|---|---|
| weights (`state_dict`, prefix `model.`) | `<run>/checkpoints/best.ckpt` | the network |
| architecture config | `<run>/checkpoints/config.resolved.yaml` | `build_model(cfg)` — without it the weights are unloadable |
| **normalisation stats** | **the DATASET** (`normalization_stats.json`) | obs/action scaling. WRONG stats = silently wrong rollouts |
| context frames + actions | the DATASET | you cannot imagine from nothing; the model needs P=8 real steps |

The normaliser is the trap. `training/setup.py:796` reads it from the *data root*, not the run —
`Normalizer.from_file(resolve_data_root(cfg))`. **The robocasa dataset is PRIVATE.** So a public model
repo whose stats live only in a private dataset is unusable by anyone without dataset access, and
"unusable" would present as plausible-but-wrong numbers rather than an error.

**=> every model repo must carry a COPY of `normalization_stats.json` and a small context sample.**

## Measured facts that shape the plan

* `best.ckpt` is **140.2 MB**, of which **85.9 MB is weights** (305 tensors, 21.5 M params) and
  **54.2 MB is `optimizer_states`** — dead weight for inference. Stripping it is a 39% cut for free.
  `lr_schedulers`, `loops`, `callbacks` are ~0 MB.
* `build_model(cfg)` (`training/setup.py:188`) takes the whole Hydra cfg and dispatches on
  `cfg.model.name` + `cfg.model.modalities`. It does NOT need an env, so `environments=recorded` (which
  cannot step) is fine for imagination.
* The imagination API is `imagine_eval(ctx_obs: dict, actions, horizon, heads=None, decode_chunk=None)`
  (`models/multimodal.py:746`), returning `{head: tensor}`. `heads=['proprio']` gives cheap long
  rollouts; `decode_chunk` bounds image-decoder peak memory.
* Loading is already done correctly in nine one-off scripts, e.g. `_oneoff_visual_terms.py:34-40`:
  `build_model(cfg)` then `load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})`
  — the `model.` prefix strip is the non-obvious bit.
* `push_to_hub.py` is **dataset-only**: `repo_type="dataset"` is hardcoded at lines 150 and 153. There is
  no model-publishing path anywhere in the repo (`grep -rl HfApi` returns that one file).
* `quickdraw` IS pip-installable (`pyproject.toml` has name/version/deps), so
  `pip install git+https://github.com/isaac-ward/quickdraw` should work for a consumer. UNVERIFIED — the
  repo has only ever been installed in-place via `uv` inside its own container. **Must be tested.**

## What goes in a model repo

```
<namespace>/quickdraw-wm-<dataset>-<recipe>/
  README.md                  the model card: what it is, the numbers, and a RUNNABLE example
  config.resolved.yaml       verbatim from the run -> build_model(cfg)
  weights.safetensors        state_dict with the `model.` prefix ALREADY stripped, no optimizer states
  normalization_stats.json   COPIED FROM THE DATASET. the thing that is otherwise unobtainable
  example_context.npz        a handful of (obs, act, frames) windows so the example runs with no dataset
  metrics.json              the run's headline numbers, so the card cannot drift from reality
```

Two deliberate choices:

**safetensors, not `.ckpt`.** No pickle (a `.ckpt` is arbitrary code execution on load), 86 MB instead of
140, and the Hub renders the tensor list. The `model.` prefix gets stripped at publish time so the
consumer never has to know about it.

**`example_context.npz`.** Without it the quickstart cannot run for anyone who lacks the private dataset,
and a card whose example does not run is worse than no card. A few P=8 windows plus the matching action
sequence is a few MB.

## Implementation checklist

### Phase 1 — the publisher

- [x] **1. `src/quickdraw/push_model.py`** — mirrors `push_to_hub.py` but `repo_type="model"`.
      `+run_dir=... +hub.name=... [+hub.private=true] [+hub.dry_run=true] [+hub.ckpt=<path>]`.
- [x] **2. Refuse to publish an incomplete repo.** Missing `normalization_stats.json` is a hard assert,
      not a warning: a model nobody can normalise for produces plausible-but-wrong rollouts.
- [x] **3. `metrics.json` generated, never hand-written** — best-so-far per key WITH the eval index it
      came from, read from the run's own `metrics.jsonl`.

### Phase 2 — the load path

- [x] **4. `quickdraw.load_pretrained(repo_or_path, device=...) -> (model, norm, cfg)`** —
      `src/quickdraw/pretrained.py`, exported from the package, accepts a Hub id or a local dir.
      Plus `load_example_context()`.
- [x] **5. Roundtrip verified** — staged with `+hub.dry_run=true`, loaded back, imagined 32 steps:
      MSE 0.05744, **PSNR 12.41 dB** vs ground truth, per-step decay 16.17 -> 12.70 -> 12.13 dB.
      Verified LIVE, not yet as a committed smoke — that is still owed.
- [x] **6. Clean-venv install verified, with a caveat.** A DIRECTORY install into a fresh 3.11 venv works
      (torch 2.14, safetensors, huggingface_hub; `from quickdraw import load_pretrained` imports).
      But `git+https://github.com/isaac-ward/quickdraw` FAILS with `could not read Username` — **the code
      repo is private.** Consumers need repo access plus a token, or a clone. Documented in docs §1.

### Phase 3 — the cards and docs

- [x] **7. Card generated by `push_model._card`** — what it predicts, the headline numbers with eval
      indices, a runnable quickstart, the three traps, and the fine-tuning notes.
- [x] **8. Caveats stated on the card** — squeeze-vs-vgg self-reference, the rollout being the weaker
      half, and the two settings not to touch when fine-tuning (`flow_hidden` 128, `latent_loss_weight` 10).
- [x] **9. `docs/using_pretrained_models.md`** — install, load, imagine, the traps, how to read a card,
      which checkpoint you get and why, and a troubleshooting table.
- [x] **10. `docs/using_pretrained_models.ipynb`** — 14 cells, executes clean: load, imagine, score
      against ground truth, render a pred-vs-truth strip, cheap proprio-only long rollout, read
      `metrics.json`.

## The quickstart the card must contain

```python
from quickdraw import load_pretrained          # phase 2 item 4
import numpy as np, torch

model, norm, cfg = load_pretrained("isaac-ronald-ward/quickdraw-wm-robocasa-vl128", device="cuda")

ex = np.load("example_context.npz")            # shipped in the repo; no dataset needed
P, H = int(cfg.data.P), 64
ctx = {"proprio": norm.norm_obs(torch.from_numpy(ex["obs"][:, :P])).float().cuda(),
       "image":   torch.from_numpy(ex["frames"][:, :P]).float().div(255).cuda()}
acts = norm.norm_act(torch.from_numpy(ex["act"][:, :P + H - 1])).float().cuda()

out = model.imagine_eval(ctx, acts, horizon=H, decode_chunk=16)
frames = out["image"].clamp(0, 1)              # (B, H, 96, 96, 3) imagined
proprio = norm.denorm_obs(out["proprio"])      # (B, H, 16) back in physical units
```

Three things this has to get right, each of which is a real trap:
* **normalise the inputs, de-normalise the proprio output** — the images are `[0,1]` and NOT normalised,
  the vectors are. Mixing these up produces plausible garbage.
* **`acts` needs `P + H - 1` steps**, not `H` — the context steps consume actions too.
* **`decode_chunk`** or a long horizon will OOM the image decoder (~78% of per-sample memory).

## What to publish, and when

Two models, and the timing matters because both are still training:

| repo | source run | status |
|---|---|---|
| `quickdraw-wm-robocasa-vl128` | `vl_l1x3` (GPU 0) | **published**, epoch 11, OL LPIPS@+128 0.13697. Run since stopped, so this is final unless `twocam_full` beats it |
| `quickdraw-wm-torus-vl128` | `torus_vl128b` (GPU 1) | **published**, epoch 0, OL LPIPS@+128 0.33974 — an in-progress placeholder, auto-updated by the watcher as better epochs land |

Publishing mid-run was the user's call ("would be good to even push the models in training just so the
pipeline is complete, and we can update later"), so the risk that `best.ckpt` moves is handled by a
watcher rather than by waiting: `preserve_ol.py` copies each new best-open-loop checkpoint out of
`save_top_k`'s way and `push_torus_watch.py` re-publishes when that copy changes. The card is regenerated
from the run's own `metrics.jsonl` every push, so its numbers cannot drift from the shipped weights.

## Open decisions for the user

1. **Public or private?** The robocasa *dataset* is private, so a public model trained on it leaks nothing
   directly but does publish learned representations of it. Default in the plan is private-first.
2. **Which robocasa checkpoint** — `vl_l1x3` (best floor 0.0855, best cos 0.562, matched saturation) or
   `st_fh128` (best @+128 0.1247, stopped after plateauing)? They win on different axes. Publishing both
   is defensible; publishing one requires saying which axis matters.
3. **Does the card promise fine-tuning?** If yes it needs the optimizer states, and the 140 MB checkpoint
   has to ship alongside the 86 MB weights.


---

## Implemented 2026-09-02 — what was actually built, and what it turned up

`src/quickdraw/pretrained.py` (`load_pretrained`, `load_example_context`, exported from `quickdraw`) and
`src/quickdraw/push_model.py` (`+run_dir=... +hub.name=... [+hub.dry_run=true]`), plus
`docs/using_pretrained_models.md` and an executable `docs/using_pretrained_models.ipynb`.

**THE CHECKPOINT WAS 68% FROZEN VGG.** The dry run refused to write safetensors because torchmetrics'
LPIPS aliases `lins.N` onto `linN` (shared storage). That refusal was doing us a favour: `VisualLoss._net`
is a *loss* network, not part of the world model, and it was **14.7 M of the 21.5 M parameters (59 MB)**.
Excluding it, the actual model is **6.8 M params / 27 MB**. It is rebuilt lazily on demand, so a freshly
built model does not even have it in its `state_dict` — loading reports zero missing keys.

Staged artifact for `vl_l1x3`: `weights.safetensors` 27.0 MB, `training_state.ckpt` 140.2 MB (fine-tuning),
`example_context.npz` 4.6 MB, and the config/stats/metrics/README at well under 1 MB.

**Roundtrip verified end to end**: stage locally, `load_pretrained` it, imagine 32 steps —
MSE 0.05744, **PSNR 12.41 dB** against ground truth, with per-step decay 16.17 -> 12.70 -> 12.13 dB.
`heads=['proprio']` correctly skips image decode (87-step rollout, no image output).

**The install story is worse than assumed**: the code repo is private, so `pip install git+https://...`
fails on authentication. A directory install works. Consumers need repo access; documented.

## Still owed before publishing

- [ ] A COMMITTED roundtrip smoke (`smoke/pretrained_roundtrip.py`). Verified live, not yet automated.
- [ ] Re-execute the notebook against the real Hub repo and commit its outputs. It currently ships without
      outputs, because the executed run pointed at a local staging dir and committing those outputs beside
      a Hub-id `MODEL` line would be misleading.
- [ ] Torus to actually train. Epoch 0 is published so the pipeline is complete end to end, but 0.33974
      is an epoch-0 number and will improve a lot; the watcher republishes automatically.

## What was published (2026-09-02)

Both repos are **public**, 13 files each, and both were verified by downloading from the Hub, loading with
`load_pretrained`, and imagining 24 steps:

| repo | epoch | OL LPIPS@+128 | AE floor LPIPS | imagine check |
|---|---|---|---|---|
| `isaac-ronald-ward/quickdraw-wm-robocasa-vl128` | 11 | **0.13697** | 0.0855 | `(1,24,96,96,3)`, 10.38 dB |
| `isaac-ronald-ward/quickdraw-wm-torus-vl128` | 0 | 0.33974 | 0.11964 | `(1,24,128,128,3)`, 16.42 dB |

Two things the publish path got wrong first, both now guarded in code: the card's headline numbers were
each metric's own best epoch rather than the published one (`_metrics_at`), and `create_repo(exist_ok=True)`
does not change visibility, so a push that *reported* PUBLIC left the repo private (`update_repo_settings`).

One process trap that cost real time: `preserve_ol.py` was still watching the dead run name `torus_vl128`
after the relaunch as `torus_vl128b`, so nothing was protecting the live run's checkpoints. Rename a run,
retarget the watchers.
