# `bsp32mse` — the recipe, and how to port it to a new dataset

Self-contained spec for the best-performing world-model configuration measured on
`isaac-ronald-ward/robocasa-scene4-4h` as of 2026-08-18. Written so another session can reproduce it on a
DIFFERENT dataset without reading the whole run log. Full history: `wizard/records/robocasa-scene4-4h.md`
§13–§16.

## What it is, in one sentence

A **fully trained-from-scratch (non-pretrained) image codec** — conv encoder + U-Net decoder — feeding an
unchanged rectified-flow latent dynamics model, on **temporally subsampled** data with the codec held in
place by a round-trip anchor.

## Why this one won

Measured against 13 other configurations on the same dataset, including frozen-TAESD arms at a 7 dB better
reconstruction ceiling:

| | this recipe | best pretrained arm |
|---|---|---|
| open-loop image PSNR (mean over horizon) | **14.25** | 14.05 |
| open-loop image **LPIPS** | **0.390** | 0.534 |
| `motion_ratio@+64` | **0.331**, and RISING with training | 0.156, and collapsing |
| 1-step PSNR | 16.66 | 18.61 |
| codec reconstruction floor | 20.03 dB | 27.23 dB |

The counter-intuitive part: it wins on perceptual quality and motion while LOSING on one-step PSNR and on
codec fidelity. One-step PSNR is decoupled from long-horizon quality on this problem (swapping in a pretrained
tokenizer once bought +6.6 dB one-step and +0.1 dB at horizon), and the reconstruction ceiling was never the
binding constraint — every model sits 4+ dB BELOW its own ceiling.

**The mechanism is trainability, not generativeness.** A decoder fit to one scene is sharp; a frozen generic
decoder is not. Ablation on the same dataset: frozen+MSE decode → LPIPS 0.525 · trainable+MSE → 0.390 ·
trainable+flow → 0.385. Making the decode *generative* (`decode_kind=flow`) bought ~nothing on sharpness,
cost 1.1 dB of PSNR, and added motion that is probably partly sampling jitter (more motion, worse PSNR, same
LPIPS). Hence `decode_kind=mse` — which is NOT a plain pixel regression, it is the same U-Net run once
deterministically. `p(image | latent)` on a single scene is nearly deterministic, so the conditional mean is
already sharp; the multimodality that MSE would blur lives in the DYNAMICS, and that path is already a flow.

## The exact overrides

```
model=mm_flow
model.d=128 model.heads=8
model.diffusion.flow_arch=transformer model.diffusion.flow_arch_depth=2 model.diffusion.flow_arch_heads=4
model.diffusion.concat_action_embedding=true
model.action_fourier_freqs=0            # OFF: measured to add gradient but no motion, and OOD-brittle
model.action_squash=none                # OFF: symlog is for observations/rewards, not actions
model.action_head.enabled=false         # collapsed 2/2 on this dataset
model.recon_frac=0.25
model.p_tf_warmup_epochs=1              # p_tf 1 -> 0 after one epoch (in-rollout training)
model.compile_rollout=true
model.latent_norm=layernorm             # REQUIRED: `affine` RAISES without a frozen pretrained latent.
                                        # Also measured better: latent_norm=none gave 0.197 motion vs 0.497
model.modalities.1.pretrained=false     # <-- the whole point: no pretrained tokenizer
model.modalities.1.encode_arch=conv     # ConvImageEncoder, mirrors the U-Net down-path
model.modalities.1.decode_kind=mse      # deterministic single pass of the U-Net (see above)
model.modalities.1.decode_arch=unet     # ViT decoder measured much worse: 0.141 motion, collapsed
model.modalities.1.num_tokens=32
model.modalities.1.img_size=128
model.modalities.1.encode_base=32 model.modalities.1.decode_base=32
model.modalities.1.latent_loss_weight=10   # round-trip anchor; see below
data.F=64
data.subsample=5                        # <-- MUST be re-derived per dataset, see below
data.autobatch=false data.batch=8       # <-- PIN IT. See "reproducibility across machines" below.
trainer.max_epochs=50 trainer.check_val_every_n_epoch=1
```

6.37M params at these settings. `num_tokens x d = 32 x 128 = 4096` latent floats = 12x compression at 128px.
With `pretrained=false` there is NO adapter and NO `num_tokens*d == latent_size` constraint — the compression
ratio is a free design choice. (That constraint only exists for a pretrained trunk whose latent size is fixed.)

## Architecture

```
ENCODER  ConvImageEncoder (models/vision.py)          trained from scratch
  image (H,W,3) -> stride-2 conv stem -> residual conv blocks pooling to an ~8x8 map
                -> 1x1 conv to width d -> num_tokens learned queries CROSS-ATTEND that map
                -> (num_tokens, d)
DECODER  ImageUNetFlowHead (models/flow.py) with no_noise=True    trained from scratch
  latent tokens --conditioning--> ConditionalUNet, run ONCE at x=0, tau=1 -> image
DYNAMICS unchanged: rectified flow on latent DELTAS, transformer denoiser (depth 2, 4 heads),
  6 sampling steps, sliding window 32, backbone depth 4
```

## `latent_loss_weight` — do not leave this at 1.0

Weight on `codec/roundtrip_<head>` = `MSE(to_obs(encode_state(x)), x)`: encode a real frame, decode it
straight back. It is the only term that asks the codec to stay faithful, and it competes with the dynamics
loss for the SAME encoder/decoder parameters. At weight 1.0 it was **1% of the objective against a dynamics
term 80x larger**, and the codec eroded 3.55 dB in 4 epochs, which cancelled the whole benefit of
subsampling. At 10 the erosion was 0.40 dB over 9 epochs. **Rule: raise it whenever the dynamics loss grows
or `codec/roundtrip_*` climbs.**

## Reproducibility across machines: PIN THE BATCH

`data.autobatch=true` sizes the batch to fill *this* GPU, which makes the effective config **hardware
dependent** -- a different card picks a different batch, and batch size is a training variable, so the run
will not reproduce. The measured runs used **batch 8** on a 95.8 GB card. Pin `data.autobatch=false
data.batch=8` to reproduce them, and only raise it deliberately.

If you do let autobatch choose, note it under-fills: its below-base branch used to halve from `autobatch_base`
and return the first batch that fit, with no upward search, so a base of 16 that did not fit landed on 8 and
never tried 9..15. Measured on this config: 82.2 GB at batch 16, 41.2 GB at batch 8 -- so it ran at 41 of a
65 GB budget, 43% of the card, at 61% GPU utilisation (the AR step is dispatch-bound, so a small batch wastes
compute as well as memory). Fixed 2026-08-18 to bisect upward, which picks batch 12 (~62 GB) for this config.
The two runs in flight deliberately keep batch 8 for comparability with the original measurement.

## Porting to a new dataset: what you MUST re-derive

**1. `data.subsample` — the single most important number, and it is dataset-specific.**

The failure this fixes: at the source frame rate the per-step image change was SMALLER than the codec's own
reconstruction error, so predicting no motion at all was the correct solution to the objective. Measure both
and pick a stride where the signal clears the noise:

```python
# per-step image change at stride k, on real frames in [0,1]
delta_k = sqrt(mean((frames[k:] - frames[:-k])**2))
# codec error: encode->decode real frames, same metric.  Then pick k so delta_k / codec_rmse > ~1
```

On robocasa (20 Hz source, codec RMSE 0.0997): stride 1 → ratio 0.87, stride 5 → **1.03**, stride 8 → 1.03.
We used 5, giving **4 Hz**, so `F=64` spans **16 seconds** of real time rather than 3.2. A dataset that is
already low-rate, or whose subject moves faster, needs a smaller stride — possibly 1.

**2. `num_tokens` / `img_size`.** Compression is free to choose here. Note that on robocasa, sweeping
8/16/32/64 tokens (48x down to 6x compression) moved the reconstruction floor **not at all** (18.7–20.4 dB
across an 8x range of latent floats) — the from-scratch codec was not rate-limited. So do not expect more
tokens to buy fidelity; 32 was chosen for the best LPIPS/PSNR balance.

**3. `model.action_dim`, `model.modalities.0.dim`, `environments.*`** to match the new dataset's action and
observation dimensions.

## What subsampling does to the ACTIONS

Keep every k-th frame, and **aggregate the actions that drive each kept transition** — `act[t]` drives
`t -> t+1`, so the action for kept step `i` is the aggregate of `act[i*k : (i+1)*k]`:

- **SUM** for delta-like dims (EEF/rotation deltas compose additively over the skipped frames)
- **TAKE-LAST** for near-binary dims — auto-detected as ≤2 unique values across the split, and logged at load.
  On robocasa those are dims 3 (constant), 4 (a flag) and 11 (the gripper). Summing would turn ±1 into ±5.

Implemented in `_subsample_episodes` (`data/dataset.py`) and applied INSIDE both episode loaders, so the
training windows and every eval routine are guaranteed to see the same rate. That placement is deliberate: a
missed call site would leave eval at the source rate while training ran subsampled, and the metrics would
look like the model failing.

## Known open items

- `recon_frac` is 0.25, i.e. the decode reconstruction is supervised on a random 25% of the F rollout frames
  (config default is 1.0). Raising it is the most obvious untested lever on sharpness.
- `decode_base` 32 -> 64 (decoder capacity) is under test as of 2026-08-18.
- LPIPS was still falling and the codec floor still climbing when the first run was stopped at epoch 15, so
  50 epochs is a floor on the useful training length, not a converged number.
