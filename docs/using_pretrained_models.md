# Using a pretrained quickdraw world model

Give it a few steps of context — a proprioceptive vector and a camera frame per step — plus a sequence of
actions. It **imagines** the frames and states that follow, open-loop, with no further observations.

Runnable companion: **[`using_pretrained_models.ipynb`](using_pretrained_models.ipynb)** — loads a model,
imagines, and renders predicted-vs-truth strips.

## 1. Install

```bash
git clone https://github.com/isaac-ward/quickdraw && pip install ./quickdraw
```

Python 3.11+. A GPU is not needed to load a model, but you will want one to imagine at any length.

## 2. Load

```python
from quickdraw import load_pretrained, load_example_context

MODEL = "isaac-ronald-ward/quickdraw-wm-robocasa-vl128"
model, norm, cfg = load_pretrained(MODEL, device="cuda")
ex = load_example_context(MODEL)     # real context windows shipped in the repo
```

| | what it is |
|---|---|
| `model` | the world model, in `eval()` on your device |
| `norm` | the **training** normalisation statistics — vectors must be scaled the way training scaled them |
| `cfg` | the resolved training config; tells you `P` (context length) and the head names |

The normaliser ships inside the model repo rather than being read from the dataset, so a model is loadable
on its own. If it is missing, `load_pretrained` raises instead of guessing — wrong statistics make every
rollout silently wrong rather than visibly broken.

`load_example_context` returns a few real `(obs, act, frames)` windows bundled with the model, so
everything below runs without the training dataset.

## 3. Imagine

```python
import torch

P, H = int(cfg.data.P), 64           # context steps the model expects; how far to imagine
head = "image"                        # image head name; see cfg.model.modalities

ctx = {"proprio": norm.norm_obs(torch.from_numpy(ex["obs"][:, :P])).float().cuda(),
       head:      torch.from_numpy(ex[f"frames__{head}"][:, :P]).float().div(255).cuda()}
acts = norm.norm_act(torch.from_numpy(ex["act"][:, :P + H - 1])).float().cuda()

with torch.no_grad():
    out = model.imagine_eval(ctx, acts, horizon=H, decode_chunk=16)

frames  = out[head].clamp(0, 1)            # (B, H, 96, 96, 3) imagined frames in [0,1]
proprio = norm.denorm_obs(out["proprio"])  # (B, H, obs_dim) in physical units
```

Image decoding dominates the cost. For long rollouts where you only need states, skip it — the latent
rollout still runs, you are only choosing what gets rendered:

```python
out = model.imagine_eval(ctx, acts_long, horizon=512, heads=["proprio"])
```

## 4. Common pitfalls

**Vectors are normalised. Images are not.** `proprio` in through `norm_obs`, out through `denorm_obs`;
frames are plain `[0, 1]` floats (divide `uint8` by 255). Getting this backwards produces plausible
garbage, not an error.

**`acts` needs `P + H - 1` steps, not `H`** — the context steps consume actions too.

**Pass `decode_chunk`.** The image decoder is ~78% of per-sample memory; without a chunk a long horizon
or large batch will OOM. 16 is a safe default.

**`N missing keys` on load is expected** if the names contain `visual._net` — that is the frozen LPIPS
loss network, excluded from published weights and rebuilt on demand.

## 5. Reading the numbers on a model card

`eval_ood_horizon/open_loop/image/lpips/@+128` is the headline: **raw open-loop perceptual distance 128
prediction steps out, no re-grounding.** Lower is better. `eval_ae_floor/.../lpips_mean` is the
autoencoder's own reconstruction floor — what the model would score with *perfect* dynamics — so it bounds
the headline from below, and on these models it accounts for most of the total error.

`metrics.json` in each repo carries every logged metric with the eval index it came from, generated from
the run's own log.

Two caveats worth carrying:

* The reported `lpips` uses a **SqueezeNet** backbone while training used **VGG**. Both are ImageNet
  feature stacks and therefore correlated, so the score is partly self-referential — read `psnr`/`ssim`
  and look at the frames too.
* **The rollout is the weaker half.** The autoencoder reconstructs much better than the dynamics predicts.
  The visible failure at long horizon is large viewpoint change: when the robot base drives, the
  prediction tends to hold the original view rather than follow.
