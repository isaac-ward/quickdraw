# Using a pretrained quickdraw world model

You give it a few steps of context — a proprioceptive vector and a camera frame per step — plus a sequence
of actions. It **imagines** the frames and states that follow, open-loop, with no further observations.

There is a runnable companion notebook: **[`using_pretrained_models.ipynb`](using_pretrained_models.ipynb)**.
It loads a model from the Hub, imagines, and renders predicted-vs-truth strips. Its outputs are committed,
so you can see it worked before running anything.

---

## 1. Install

`quickdraw` is not on PyPI, and **the code repo is private**, so a bare `pip install git+https://...`
will fail with `could not read Username for 'https://github.com'`. Two options:

```bash
# A. you have repo access — clone, then install the checkout
git clone https://github.com/isaac-ward/quickdraw && pip install ./quickdraw

# B. install straight from the URL with a token
pip install "git+https://<YOUR_GITHUB_TOKEN>@github.com/isaac-ward/quickdraw@main"
```

Verified working: a directory install into a clean Python 3.11 venv pulls torch 2.14, safetensors and
`huggingface_hub`, and `from quickdraw import load_pretrained` imports. Python **3.11 or newer** is
required (`pyproject.toml`).

A GPU is not required to load a model, but you will want one to imagine at any length.

## 2. Load

```python
from quickdraw import load_pretrained, load_example_context

model, norm, cfg = load_pretrained("isaac-ronald-ward/quickdraw-wm-robocasa-vl128", device="cuda")
ex    = load_example_context("isaac-ronald-ward/quickdraw-wm-robocasa-vl128")
```

Three objects, and you need all three:

| | what it is | why you cannot skip it |
|---|---|---|
| `model` | the world model, in `eval()` on your device | — |
| `norm` | the **training** normalisation statistics | vectors must be scaled the way training scaled them, or every rollout is silently wrong |
| `cfg` | the resolved training config | tells you `P` (context length) and the head names |

**Why the normaliser ships with the model.** A checkpoint holds weights and nothing else usable — its
`hyper_parameters` is empty. The statistics normally live in the *dataset*, and the robocasa dataset is
private, so a model repo bundles its own copy. `load_pretrained` reads it from the repo, never from a
dataset you may not have. If it is missing, loading raises rather than guessing.

`load_example_context` returns a handful of real `(obs, act, frames)` windows shipped in the repo, so
everything below runs with no dataset access at all.

## 3. Imagine

```python
import torch

P    = int(cfg.data.P)                # context steps the model expects (8)
H    = 64                             # how far to imagine
head = "image"                        # image head name; see cfg.model.modalities

ctx = {"proprio": norm.norm_obs(torch.from_numpy(ex["obs"][:, :P])).float().cuda(),
       head:      torch.from_numpy(ex[f"frames__{head}"][:, :P]).float().div(255).cuda()}
acts = norm.norm_act(torch.from_numpy(ex["act"][:, :P + H - 1])).float().cuda()

with torch.no_grad():
    out = model.imagine_eval(ctx, acts, horizon=H, decode_chunk=16)

frames  = out[head].clamp(0, 1)          # (B, H, 96, 96, 3) imagined frames, [0,1]
proprio = norm.denorm_obs(out["proprio"])  # (B, H, obs_dim) back in physical units
```

## 4. The three things that will bite you

**Vectors are normalised. Images are not.** `proprio` goes in through `norm_obs` and comes out through
`denorm_obs`; frames are plain `[0, 1]` floats (divide your `uint8` by 255). Getting this backwards
produces plausible-looking garbage, not an error.

**`acts` needs `P + H - 1` steps, not `H`.** The context steps consume actions too. Passing `H` silently
gives you a shorter rollout than you asked for.

**Pass `decode_chunk`.** The image decoder is roughly 78% of per-sample memory. Without a chunk, a long
horizon or a large batch will exhaust the GPU. 16 is a safe default.

## 5. Cheaper and longer rollouts

Image decoding dominates the cost. If you only need states, skip it:

```python
out = model.imagine_eval(ctx, acts, horizon=512, heads=["proprio"])   # no image decode at all
```

`heads=` selects which modalities decode. The latent rollout runs regardless — you are only choosing what
gets rendered.

## 6. Fine-tuning

`weights.safetensors` is inference-only. `training_state.ckpt` is the full Lightning checkpoint including
AdamW's moment buffers, so training can *resume* rather than restart the optimizer (without them the
first steps lose all adaptive scaling).

Two settings not to touch, both measured the hard way:

* **`diffusion.flow_hidden` stays 128.** Every run of this recipe at 512 destroyed itself between epochs
  5 and 12 under the perceptual loss; at 128 the same config ran past epoch 21 healthy. Under a plain-MSE
  loss 512 was marginally *better*, so this is a loss-dependent interaction, not "512 is bad".
* **`latent_loss_weight` stays 10.** It is bracketed on both sides: 0.4 erased the codec by 5.4 dB in one
  epoch, 25 froze motion to 0.02 with negative latent tracking and infinite gradients.

## 7. How to read the numbers on a model card

`eval_ood_horizon/open_loop/image/lpips/@+128` is the headline: **raw open-loop perceptual distance 128
prediction steps out, no re-grounding.** Lower is better. `eval_ae_floor/.../lpips_mean` is the
autoencoder's own reconstruction floor — the score the model would get if its dynamics were *perfect* —
so it is a lower bound on the first number, and on these models it accounts for most of the total error.

`metrics.json` in each repo carries every logged metric **with the eval index it came from**, generated
from the run's own log. Numbers in this project have been misquoted before by pairing a score from one
epoch with a different quantity from another; the index makes that checkable.

Two caveats worth carrying:

* The reported `lpips` uses a **SqueezeNet** backbone while training used **VGG**. Different networks, but
  both are ImageNet feature stacks and therefore correlated, so the score is partly self-referential.
  Read `psnr`/`ssim` and look at the frames as well.
* **The rollout is the weaker half.** The autoencoder reconstructs much better than the dynamics predicts.

## 8. Which checkpoint you are getting

`best.ckpt` inside a training run tracks `val/metric/<head>/mse`. Both that and the headline metric are
open-loop rollout scores, but MSE is structurally blind to sharpness — so selecting on it picks the
blurriest acceptable epoch, which is the criterion the perceptual loss was adopted to replace. Measured on
one run: the best-MSE epoch scored open-loop LPIPS 0.1455 while the best-LPIPS epoch scored 0.1370.

Published repos therefore ship the **best open-loop-LPIPS** epoch, and `metrics.json` records
`checkpoint_selected_by` so you can see which rule was applied. Runs started after 2026-09-02 monitor the
full visual loss directly, so the two agree from then on.

## 9. Troubleshooting

| symptom | cause |
|---|---|
| `could not read Username for 'https://github.com'` | the code repo is private — see §1 |
| `FileNotFoundError: ... normalization_stats.json` | the model repo is incomplete; it cannot be loaded safely and this is deliberately fatal |
| `N missing / M unexpected keys` on load | expected and harmless if the names contain `visual._net` — that is the frozen LPIPS loss network, excluded from published weights and rebuilt on demand |
| CUDA OOM during `imagine_eval` | pass `decode_chunk=16`, lower the batch, or use `heads=["proprio"]` |
| rollout looks plausible but drifts immediately | you probably skipped `norm_obs` on the context, or passed `H` actions instead of `P + H - 1` |
