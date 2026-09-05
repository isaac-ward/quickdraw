---
license: mit
library_name: quickdraw
tags:
- world-models
- robotics
- video-prediction
---

# {name}

A latent world model: it takes {P} steps of context (proprioceptive vector + camera frame{plural}) plus a
sequence of actions, and rolls forward **open-loop** — predicting the frames and states that follow with no
further observations.

Trained with `quickdraw` (`model={model_name}`, recipe `{recipe}`) on `{dataset}`.
Image head{plural}: {head_list} at {img_size}px, {num_tokens} latent tokens each.

## What it looks like

{products_md}
## Headline numbers

All from **epoch {epoch}**, the published checkpoint — not each metric's own best epoch.

| what | value |
|---|---|
{rows}

`metrics.json` carries every logged metric across the whole run, each with the eval index it came from.
Lower is better for `lpips`, higher for `psnr`. `@+128` means 128 prediction steps with no re-grounding.

Checkpoint selection: {why_ckpt}

## Version this was trained with

```bash
pip install "git+https://github.com/isaac-ward/quickdraw@{git_commit}"
```

`quickdraw {pkg_version}` · torch `{torch_version}`. The package version is not bumped per change, so the
commit is the real pin.

## Quickstart

```python
from quickdraw import load_pretrained, load_example_context
import torch

model, norm, cfg = load_pretrained("{name}", device="cuda")
ex = load_example_context("{name}")          # real context windows, shipped with the model

P, H = int(cfg.data.P), 64
head = "{head0}"

ctx = {{"proprio": norm.norm_obs(torch.from_numpy(ex["obs"][:, :P])).float().cuda(),
       head:      torch.from_numpy(ex[f"frames__{{head}}"][:, :P]).float().div(255).cuda()}}
acts = norm.norm_act(torch.from_numpy(ex["act"][:, :P + H - 1])).float().cuda()

out = model.imagine_eval(ctx, acts, horizon=H, decode_chunk=16)
frames  = out[head].clamp(0, 1)              # (B, H, {img_size}, {img_size}, 3) imagined frames
proprio = norm.denorm_obs(out["proprio"])    # (B, H, {obs_dim}) in physical units
```

Full walkthrough: [`docs/using_pretrained_models.md`](https://github.com/isaac-ward/quickdraw/blob/main/docs/using_pretrained_models.md)
and its companion notebook.

## Common pitfalls

**Vectors are normalised, images are not.** `proprio` in through `norm_obs`, out through `denorm_obs`;
frames are plain `[0, 1]` floats. Getting this backwards produces plausible garbage, not an error.

**`acts` needs `P + H - 1` steps, not `H`** — the context steps consume actions too.

**Pass `decode_chunk`.** The image decoder is ~78% of per-sample memory; without a chunk a long horizon
will OOM. 16 is a safe default.

**`N missing keys` on load is expected** when the names contain `visual._net` — the frozen LPIPS loss
network, excluded from the published weights and rebuilt on demand.

## What ships here

| file | what |
|---|---|
| `weights.safetensors` | the model, `model.` prefix stripped, no optimizer state |
| `training_state.ckpt` | full Lightning checkpoint, for resuming training |
| `config.resolved.yaml` | the resolved training config → `build_model(cfg)` |
| `normalization_stats.json` | the training statistics, so the model loads standalone |
| `example_context.npz` | real context windows plus their ground-truth continuation |
| `metrics.json` | every logged metric with its eval index |
| `*rollout.mp4`, `*filmstrip*.png`, `*error_vs_step.png` | the run's own logged eval artifacts at the published epoch, one set per image head |
| `versions.json` | quickdraw / torch / commit the weights came from |
