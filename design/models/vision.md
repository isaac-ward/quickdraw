# Vision era — multimodal (kinematic + image) world models

> **Status: initial thoughts.** Captures a design discussion, not a committed spec. The diffusion /
> LSAR / DSAR specs (`diffusion.md`, `latent_space_autoregressor.md`, `data_space_autoregressor.md`)
> remain the authoritative model docs; this is where the *vision extension* of all of them gets
> sketched before we commit. Nothing here is implemented yet.

## The shape: two inputs, two outputs, fused

Today the world model is single-modality: obs is a kinematic 6-vector `[p; ṗ]`. The vision era makes
it **dual-modality** — the model consumes **and produces both**:

- **Inputs:** the kinematic 6-vector **and** an image (the FPV / egocentric frame).
- **Outputs:** the kinematic 6-vector **and** an image.

The two streams are **fused** into the model's working representation and **both decoded back out**.
This is *not* "images replace the vector" — the kinematic stream stays a first-class citizen on both
ends, which (see below) is what lets all the existing physics/control/metrics/viz machinery keep
working untouched.

Encoder/decoder per stream:

| stream    | encoder | decoder      |
|-----------|---------|--------------|
| kinematic | **MLP** | **MLP**      |
| image     | **ViT** | **ViT / transposed-conv** |

The fuse step is the **same `TokenStreamFuser` pattern that already fuses state+action** — it just
gains another stream. Concretely the per-step pipeline becomes:

```
kin  6-vec ──MLP──▶ e_kin ┐
                          ├──fuse (TokenStreamFuser)──▶ working repr ──▶ backbone
img  HxWx3 ──ViT──▶ e_img ┘
```

and on the way out, two heads off the prediction:

```
prediction ──MLP──▶ kin 6-vec   (kinematic head)
           └─ViT/deconv──▶ image (image head)
```

So `encode_state` takes `(kin, image)`; `to_obs` returns `(kin, image)`. The **grounding /
reconstruction loss is the sum of both**: kinematic MSE on the 6-vec **+** image reconstruction
(MSE / LPIPS) on the frame. With both streams reconstructed, the representation is forced to carry
both — a stronger anti-collapse signal than the vector alone.

## The clean part: kinematic stays a real output

Because we **still decode the 6-vector**, position/velocity `[p; ṗ]` is a genuine model output — not
a probe we bolt on. This means **no privileged physical-state head is needed**, and the existing
machinery reads the kinematic decode directly:

- **physical loss** → on the kinematic decode (`physical_state` = the MLP head's 6-vec).
- **MPPI control** → reads planned position from the kinematic decode.
- **manifold / pointwise / tangent metrics** → on the kinematic decode, exactly as today.
- **TorusRenderer viz** → places the agent on the torus from the kinematic decode, as today.

The image is an *added* modality, not a substitution — so none of the above needs special-casing.

## Two ways to be autoregressive — and **DSAR does extend**

The DSAR-vs-latent distinction is *what gets carried across timesteps and what the feedback loop is*.
That distinction survives the jump to images for **both** classes:

- **DSAR (data-space).** The carried state **is the observation** — here the **decoded** image **and**
  kinematic 6-vector. Each step: predict the next obs, **decode it to a real image + real vector**,
  then **feed those decoded outputs back in** as next-step inputs (re-encoding them). It is still
  autoregressive *in data space*: the thing that persists across the rollout is the data
  (image + vec), and the latent is transient — re-derived from the decoded obs every step, exactly as
  the vector DSAR does today (`feeds back the observation and re-encodes`). The ViT encoder + image
  decoder are needed per-step, but they're the per-step obs↔token map, **not** a persistent latent.
  Loss is on the decoded obs (pixel + vector).

  *(This corrects an earlier claim that "DSAR can't carry a raw image." It can — you carry the decoded
  image+kinematic forward and re-encode, which is precisely data-space AR. The image just makes the
  per-step encode/decode heavier, and pixel-space feedback compounds image error directly rather than
  through an abstraction — that's the cost, not an impossibility.)*

- **LSAR / diffusion (latent-space).** The carried state is the **fused latent `z`** — never decoded
  during the rollout, fed back as latent. `z` encodes *both* "where I am and what I see." Diffusion
  still denoises this **fused latent** (latent diffusion — never touches pixels).

So the data-space/latent-space axis stays exactly the honest comparison it is today, now with a
richer observation. The trade also sharpens: DSAR's pixel-space feedback compounds image error
step-to-step; latent models compound abstracted error and decode only at the end (or for viz).

## What's reused (the payoff)

The backbone (transformer→`h`), `to_token`, `readout`, the carried representation, and the single AR
rollout (`p_tf` curriculum + `detach_every` BPTT) all operate on the fused representation — **untouched**.

- **LSAR** collapse mechanisms carry over verbatim (now grounded by *both* recon losses).
- **Diffusion** carries over verbatim; it denoises the **fused latent**. Only nudge: `dz` grows (the
  latent now carries an image too), which is where a **unified transformer-denoiser** (vs. the small
  MLP flow field for the 16-d vector latent) starts to earn its keep — see `diffusion.md`'s
  "block-by-block" table; the image-era latent is the case where transformer-as-denoiser pays off.
- **Contraction** is still the `dz×dz` latent Jacobian via the eager `sdpa(MATH)` double-vjp path —
  modality-blind, unchanged.

## Multi-feed: design the seam now, build one feed

We start with the single egocentric FPV feed, but the design must extend to **N visual feeds** (e.g.
FPV + chase + top-down) without surgery. The discipline:

- **One shared ViT encoder** applied to every feed (weights tied), so adding a feed adds no parameters.
- **Per-feed identity embedding** added to that feed's tokens (like a positional embedding, but for
  "which camera") so the model can tell feeds apart while sharing the encoder.
- Every feed's tokens flow into the **same list-based `TokenStreamFuser`** — N image streams + the
  kinematic stream + action. The fuser already takes an arbitrary list of streams, so the only
  discipline is **never hard-coding "one image stream"** anywhere downstream.
- The carried latent stays **feed-count-agnostic** (a token list with feed-identity embeddings), so
  feed #2 is a config change (`feeds: [fpv, chase]`), not an architecture change.

## What's genuinely new to build

- **ViT image encoder** + **image decoder** (ViT / transposed-conv) heads.
- **Fuser gains the image stream** (same `TokenStreamFuser` pattern).
- **Image reconstruction loss** (MSE / LPIPS) added to the grounding objective.
- **Variations / noise**: per-stream σ — noise the kinematic input and the image input separately,
  before the fusion encoder (the "per-stream σ" the variations design already anticipated, now two
  concrete knobs). physical/contraction unchanged.
- **Metrics**: keep the physical metrics (kinematic decode) **and add image metrics** (reconstruction
  PSNR / LPIPS on the predicted frame).
- **Viz**: torus viz (kinematic decode) unchanged; **add** a predicted-vs-true **image** panel (the
  FPV the model imagines next). The diffusion flow-field viz can decode each ODE step to a **position**
  (the torus-arc funnel, via the kinematic head) *and/or* to an **image** (watch the predicted FPV
  frame denoise from noise) — both from real decode heads.

## Open questions / to resolve before committing

- Image source & size: FPV egocentric frame at what resolution? (drives ViT patch size + decoder cost)
- Fusion mechanism detail: concat+project vs. cross-attention vs. token-per-stream into the fuser.
- `dz` for the fused latent: bigger than the current 16; pick once we know image latent dim.
- Image recon loss: plain MSE vs. perceptual (LPIPS) — and whether the image head is a separate
  network or shares trunk with the kinematic head.
- Whether DSAR-with-images is worth running (heavy per-step ViT encode/decode, pixel-space error
  compounding) or kept as a documented "we can, but latent is the point" comparison.

---

# Committed design (vision v1)

## Decisions locked

- **Naming.** Streams are `[proprio, image, action]`. `proprio` = the kinematic 6-vec `[p; ṗ]` (stored
  lerobot key stays `observation_vector`, aliased to `proprio` at load — no data regen). Image =
  `observation.images.fpv`.
- **FPV resolution.** Source frames are **256×256×3**; we **downsample to 128×128 once at load** (cached,
  not per-epoch).
- **Image AE.** 100% ViT, trained **end-to-end** with the world model (no pretraining, no checkpoints),
  **plain MSE** recon. Architecture: `Linear-patchify → ViT encoder → num_tokens learned queries
  cross-attend (perceiver bottleneck) → ViT decoder → Linear-unpatchify`. **`num_tokens = 8`** to start.
- **Latent = a LIST of tokens per step** (never a grid): `[proprio_token] ++ [num_tokens image tokens]`,
  each with a learned type/feed + positional embedding. Carried state is `(1+num_tokens, d)`.
- **Backbone = factorized space-time attention** (ViViT/TimeSformer/Genie-style), built **now** (not
  deferred) because num_tokens grows soon: per block, **spatial** attention within a step (bidirectional,
  no mask) then **temporal** attention across steps (causal). Cost `O(W·N² + N·W²)` vs joint `O(W²·N²)`.
  Each sub-attention is one of our existing `Transformer` blocks + FlexAttention (spatial: no mask;
  temporal: plain 1-D causal). This **supersedes** the earlier "backbone untouched" sketch.
- **Diffusion denoiser = DiT** over the token list (adaLN on flow-time τ and shortcut step dd, conditioned
  on backbone context h). Rectified-flow / shortcut math unchanged.

## Losses (training objective)

Total = weighted sum of a **world-model** term (model-specific) + **per-head reconstruction** terms:

| term                    | what                                                   | models |
|-------------------------|--------------------------------------------------------|--------|
| `world/flow`            | rectified flow-matching velocity loss                  | diffusion |
| `world/flow_consistency`| shortcut self-consistency                              | diffusion (shortcut) |
| `world/pred_latent`     | next-latent prediction loss                            | LSAR |
| `recon/proprio`         | **MSE** on decoded 6-vec (kinematic grounding)         | all latent models |
| `recon/image`           | **MSE** on decoded 128² frame (image grounding)        | all (vision) |

- Both recon terms are computed on the model's **predicted next-states** (the rollout), exactly as the
  kinematic grounding works today — this is the anti-collapse signal, now from *both* modalities.
- **Balancing matters:** `recon/image` (mean over ~49k pixels in [0,1]) and `recon/proprio` (mean over 6
  normalized dims) and `world/*` live on different scales → each gets a tunable weight
  (`lambda_recon_image`, `lambda_recon_proprio`, `lambda_flow`, …) via the existing variations weighting.
- **No LPIPS / perceptual loss** — it requires pretrained VGG/AlexNet weights, which violates the
  no-pretrained constraint. Plain MSE only.

## Image metrics (eval — not losses)

Closed-form, no pretrained nets (so LPIPS is excluded):
- `image/psnr` — peak signal-to-noise ratio, predicted vs. true frame.
- `image/ssim` — structural similarity.
- `image/recon_mse` — the recon MSE surfaced as a metric.

## wandb structure (organized by head/trunk)

```
train/                                  (every epoch — scalars)
  loss                                  total
  loss/world/{flow, flow_consistency | pred_latent}
  loss/recon/{proprio, image}
val/                                    (every epoch — scalars)
  loss + same loss/* subtree
  metric/proprio/pointwise_error        (shared rollout metric, kinematic)
  metric/image/{psnr, ssim, recon_mse}
eval_manifold/                          (benchmark epochs — any method)
  umap_{data,latent}_space_to_{2,3}d    (latent = concat of all tokens, flattened)
eval_diffusion/                         (benchmark epochs — diffusion only)
  denoising_multistep, denoising_aggregate, std_of_samples, time/*
eval_rollout/                           (benchmark epochs — NEW, vision)
  fpv_pred_vs_true                      video: PREDICTED frames (top) | GROUND-TRUTH frames (bottom)
ood_horizon/, control/                  (unchanged)
```

(Adopting `loss/world/*` + `loss/recon/*` + `metric/*` is a small logging refactor in the LightningModule;
backport to DSAR/LSAR for consistency is optional.)

## Implementation plan (phases — each ends at a verify gate)

**Phase 0 — data plumbing & naming.**
- 0.1 Alias `observation_vector → proprio`; loader returns aligned `(proprio[6], image[128,128,3], action[2])`.
- 0.2 Downsample 256→128 **once at load**, cached.
- 0.3 *Verify:* a batch yields aligned tensors; downsample is cached (not per-epoch).

**Phase 1 — ViT autoencoder (image only, standalone).**
- 1.1 Linear patchify (patch 16 → 64 patches) + posemb → ViT encoder (our block + FlexAttention, no mask).
- 1.2 Perceiver bottleneck: 8 learned queries cross-attend patches → 8 latent tokens.
- 1.3 ViT decoder: 8 tokens → per-patch query tokens → transformer → Linear-unpatchify → 128².
- 1.4 *Verify:* `smoke/vision_ae.py` + short fit; held-out recon PSNR passes threshold.

**Phase 2 — factorized space-time backbone + token-bag carried state (on LSAR first).**
- 2.1 Fuser gains image stream; per-step latent = `[proprio] ++ [8 image]`; decide action conditioning.
- 2.2 Carried state `(9, d)` token bag; per-token `_ln`; `encode_state`/`to_obs`/`readout` handle the bag.
- 2.3 **Factorized space-time block**: spatial attn (within step, unmasked) + temporal attn (across steps,
  causal) + MLP; space/time positional embeddings; FlexAttention for both.
- 2.4 Two decode heads (proprio MLP, image ViT decoder); `recon/{proprio,image}` losses.
- 2.5 *Verify:* smoke + short LSAR run; both recon losses fall; AR rollout runs; eff_rank > 1 (no collapse).

**Phase 3 — diffusion DiT denoiser.**
- 3.1 Swap MLP flow field → DiT over the 9 tokens (adaLN on τ + shortcut dd; condition on h).
- 3.2 Rectified-flow + shortcut unchanged.
- 3.3 *Verify:* adapt `smoke/diffusion.py` A–H; ε=0 readout byte-stable; decode → image + vec.

**Phase 4 — loss weighting & balancing.**
- 4.1 Wire `lambda_recon_{image,proprio}` weights; tune so no term dominates.
- 4.2 *Verify:* a run where both modalities improve together (neither recon term flatlines).

**Phase 5 — metrics + wandb restructure.**
- 5.1 Add `metric/image/{psnr,ssim,recon_mse}` (closed-form).
- 5.2 Adopt the `loss/world`, `loss/recon`, `metric/*` namespaces above.
- 5.3 *Verify:* train/val dashboards show the grouped tree; image metrics populate.

**Phase 6 — viz.**
- 6.1 `eval_rollout/fpv_pred_vs_true`: AR rollout video, **predicted on top, ground truth on bottom**.
- 6.2 `eval_manifold` latent UMAP on concat-of-all-tokens (data-space UMAP stays proprio 6-vec).
- 6.3 `denoising_*` can decode each ODE step to an image (watch the FPV denoise from noise).
- 6.4 *Verify:* eval logs the stacked rollout video + image panels cleanly.

**Phase 7 — variations + multi-feed seam.**
- 7.1 Per-stream noise σ (proprio vs image).
- 7.2 Shared ViT + per-feed identity embedding, wired feed-agnostically (one feed today).
- 7.3 *Verify:* config can declare a 2nd feed without code surgery.
