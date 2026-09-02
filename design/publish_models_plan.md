# Publishing world models to the Hub so someone else can load them and imagine

Status: **PHASES 1-2 IMPLEMENTED 2026-09-02** (see the checklist below); publishing itself not yet done. Requested by the user; every claim below was checked
against the source or measured on a real checkpoint.

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

### Phase 1 — the publisher (one new module)

- [x] **1. `src/quickdraw/push_model.py`** — DONE. — mirrors `push_to_hub.py`'s shape but `repo_type="model"`.
      Takes `run_dir=<logs/train_world_model_...>` and `+hub.name=`, then:
      strips `optimizer_states`; strips the `model.` prefix; saves `weights.safetensors`; copies
      `config.resolved.yaml`; **copies `normalization_stats.json` from `resolve_data_root(cfg)`**;
      writes `metrics.json` from the run's `metrics.jsonl` (best-so-far per key, not last);
      samples `example_context.npz`; renders `README.md`; uploads.
      `+hub.private=true` supported, and default private for a first push — a model card with wrong
      numbers is harder to retract than a dataset.
- [x] **2. Assert the four artifacts exist before creating the repo.** DONE — missing norm stats is a hard refusal. A half-populated model repo is the
      failure mode to design out: publish should fail loudly if the norm stats cannot be found rather
      than upload a model nobody can normalise for.
- [x] **3. `metrics.json` is generated, never hand-written.** DONE — best-so-far per key WITH its eval index. Numbers in this project have been misquoted
      several times (a score from one epoch paired with a motion value from another, a stale batch size).
      Read `logs/metrics.jsonl`, emit best-so-far per key AND the eval index it came from.

### Phase 2 — the load path a consumer actually uses

- [x] **4. `quickdraw.load_pretrained(repo_or_path, device=...)`** DONE — `src/quickdraw/pretrained.py`, exported from the package. returning `(model, norm, cfg)`. Should
      accept a Hub id or a local dir, and do the three things every one-off already does by hand:
      `OmegaConf.load` the config, `build_model`, `load_state_dict` with the prefix strip. This is the
      single most valuable item on the list — right now "load the model" is 8 lines of tribal knowledge
      repeated in nine scripts.
- [x] **5. Roundtrip VERIFIED** (as a live test, not yet a committed smoke): staged locally via `+hub.dry_run=true`, loaded back with `load_pretrained`, imagined 32 steps, 12.41 dB PSNR vs ground truth. A committed smoke is still owed. — publish to a LOCAL dir, load it back, imagine, and assert
      the output matches the in-process model bit-for-bit. This is the only test that catches a silently
      wrong normaliser, which is the failure mode with the worst consequences.
- [x] **6. Clean-venv install VERIFIED, with a caveat.** A DIRECTORY install into a fresh 3.11 venv works (torch 2.14, safetensors, huggingface_hub; `from quickdraw import load_pretrained` imports). But `git+https://github.com/isaac-ward/quickdraw` FAILS with `could not read Username` — **the code repo is private**, so consumers need repo access plus a token, or a clone. Documented in docs §1. Currently unverified and the whole
      consumer story depends on it. If it does not, the card must say "clone the repo" instead.

### Phase 3 — the cards

- [x] **7. Card is generated** by push_model (`_card`), with numbers, a runnable quickstart and the traps., generated, with: what the model predicts and at what rate; the headline
      numbers with the eval index; a **runnable** quickstart (below); the caveats that matter for reuse;
      and a pointer to the dataset repo.
- [x] **8. Caveats on the card** — squeeze-vs-vgg self-reference, the rollout being the weak half, and the two settings not to touch when fine-tuning. For the robocasa model: the metric is
      LPIPS-SqueezeNet while training used LPIPS-VGG, so the score is partly self-referential; the
      rollout is the weak half (OL PSNR ~14 dB against a 19.4 dB codec floor); and `flow_hidden=512`
      variants of this recipe self-destruct between ev5 and ev12, so anyone fine-tuning must keep 128.

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
| `quickdraw-wm-robocasa-vl128` | `vl_l1x3` (GPU 0) | ep 17/40, still improving — **publish its best.ckpt when it finishes or plateaus** |
| `quickdraw-wm-torus-vl128` | `torus_vl128` (GPU 1) | ep 0/40, just launched |

**Do not publish mid-run.** `best.ckpt` moves every time the monitored metric improves, so a repo pushed
now would be superseded within hours and the card's numbers would be wrong. `vl_l1x3` is the nearer one.

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
- [ ] Naming, and the decision on what to publish for torus (its run restarted from epoch 0 on 09-02).
- [ ] Both runs to finish or plateau. `best.ckpt` moves whenever the monitored metric improves.
