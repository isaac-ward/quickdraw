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

- **Config-driven modalities (trunks + heads).** Each modality is a `{name, kind (vector|image), encoder
  (trunk), decoder (head), loss_weight, enabled}` entry in a **config list**. The model builds
  encoders/decoders, the fuser's streams, the per-head losses, the metrics, and the viz **by iterating that
  list** — so the modality NAME flows straight through to the logs (`loss/<name>`, `metric/<name>/*`), and
  enabling/disabling a trunk+head for an ablation is a config edit that "just works". Today: `proprio` +
  `image`. Later: `proprio` + `image1` + `image2` + … with zero code change.
- **Naming.** `proprio` = kinematic 6-vec `[p; ṗ]` (stored lerobot key stays `observation_vector`, aliased
  at load — no data regen). Images = `observation.images.<name>`.
- **FPV resolution.** Source frames **256×256×3**; **downsample to 128×128 once at load** (cached).
- **Image AE.** 100% ViT, **end-to-end** (no pretraining, no checkpoints), **plain MSE** recon.
  `Linear-patchify → ViT encoder → num_tokens learned queries cross-attend (perceiver bottleneck) → ViT
  decoder → Linear-unpatchify`. **`num_tokens = 8`** to start.
- **Action is its OWN token (not folded).** The per-step token bag fed to the backbone is
  `[proprio, action, image_1..image_num_tokens]`. The action is encoded to one token (action MLP) and
  marked with a type embedding; conditioning then happens through **attention** (every state token attends
  the action token in the spatial pass) rather than pre-fusing it into another token. The action token is
  **input-only** — injected fresh each step from the given action sequence, never decoded or predicted. So
  the **carried / predicted** state is `[proprio] ++ [image_1..num_tokens]` (`1+num_tokens` tokens); the
  action token is added on input each step.
- **Latent = a LIST of tokens per step** (never a grid), each with a learned type + positional embedding.
- **Backbone = factorized space-time attention** (ViViT/TimeSformer/Genie-style), built **now** (not
  deferred) because num_tokens grows soon: per block, **spatial** attention within a step (bidirectional,
  no mask) then **temporal** attention across steps (causal). Cost `O(W·N² + N·W²)` vs joint `O(W²·N²)`.
  Each sub-attention reuses the (generalized, mask-agnostic) `Transformer` block + FlexAttention — spatial:
  no mask; temporal: plain 1-D causal. This **supersedes** the earlier "backbone untouched" sketch.
- **Diffusion denoiser = DiT** over the token list (adaLN on flow-time τ and shortcut step dd, conditioned
  on backbone context h). Rectified-flow / shortcut math unchanged.

## Losses (training objective)

No `world/`/`recon/` prefixes. Two kinds of term: a **prediction** term (model-specific, named by what it
is) and **per-head reconstruction** terms (named by the head — so they follow the config modality names):

| term                | what                                            | tag             | models |
|---------------------|-------------------------------------------------|-----------------|--------|
| flow                | rectified flow-matching velocity loss           | `loss/flow`             | diffusion |
| flow_consistency    | shortcut self-consistency (enables K=1)         | `loss/flow_consistency` | diffusion (shortcut) |
| pred_latent         | next-latent prediction loss                     | `loss/pred_latent`      | LSAR |
| `<head>` recon      | **MSE** on that head's decode (grounding)       | `loss/<head>` e.g. `loss/proprio`, `loss/image` | all latent models |

- Reconstruction terms are **head-named** (`loss/proprio`, `loss/image`, later `loss/image1` …), computed
  on the model's **predicted next-states** — the anti-collapse grounding, now from every enabled modality.
- **Per-head weights live in config** (`weight:` on each modality entry) — needed because `loss/image`
  (mean over ~49k pixels in [0,1]) and `loss/proprio` (6 normalized dims) and the prediction term live on
  very different scales. (Phase 4 tunes these.)
- **No LPIPS / perceptual loss** — needs pretrained VGG/AlexNet weights → excluded by the no-pretrained
  rule. Plain MSE only.

> *Glossary.* **flow** = the diffusion model's core objective: regress the flow field's velocity toward the
> true noise→data transport (learn the denoising vector field). **flow_consistency** = the shortcut-model
> extra term enforcing "one 2d-step == two chained d-steps," which is what lets it sample in K=1.
> **pred_latent** = LSAR's deterministic analogue: MSE predicting next latent `z_{t+1}`. These are the
> "predict the dynamics" terms; the head terms are the "reconstruct what you saw" grounding.

## Image metrics (eval — not losses)

Closed-form, no pretrained nets (so LPIPS is excluded):
- `psnr` — peak signal-to-noise ratio, predicted vs. true frame.
- `ssim` — structural similarity.
- `mse` — pixel MSE surfaced as a metric.

## wandb structure (organized by head/trunk)

```
train/                                  (every epoch — scalars)
  loss                                  total
  loss/{flow, flow_consistency | pred_latent}     prediction term(s)
  loss/<head>                           per-head recon, e.g. loss/proprio, loss/image  (config-named)
val/                                    (every epoch — scalars)
  loss + same loss/* subtree
  metric/proprio/pointwise_error        (shared rollout metric, kinematic)
  metric/<image-head>/{psnr, ssim, mse}           e.g. metric/image/psnr
eval_manifold/                          (benchmark epochs — any method)
  umap_{data,latent}_space_to_{2,3}d    (latent = concat of all carried tokens, flattened)
eval_diffusion/                         (benchmark epochs — diffusion only)
  denoising_multistep, denoising_aggregate, std_of_samples, time/*
eval_ood_horizon/                       (benchmark epochs)
  …existing proprio rollout plots + error-vs-step curve…
  <image-head>/{psnr, ssim, mse}_vs_step  CURVE: image metric over the rollout horizon (mirrors proprio error-vs-step)
  <image-head>/filmstrip                STILL: 8 steps across the horizon, pred (top) | GT (bottom)
  <image-head>/rollout                  VIDEO: pred (top, black until context plays out) | GT (bottom), synced
control/                                (unchanged — see note)
```

- The predicted-vs-true image artifacts go under **`eval_ood_horizon`** (the existing open-loop rollout
  eval), **not** a new `eval_rollout`. Two artifacts per image head, via the `viz.fig_image_filmstrip`
  (still) and `viz.image_rollout_video` (synced video) utilities.
- **`eval_control` is unchanged.** MPPI plans and scores on the **proprio decode** (position) exactly as
  today, and the control video stays the torus viz from that kinematic decode. Vision rides along (the
  latent now also carries image), but control needs no FPV to function; an FPV-during-control panel is a
  later optional add, not required.
- This naming (`loss/*`, `metric/*`, head-named) is **adopted for ALL models going forward** (DSAR/LSAR/
  diffusion), so dashboards are consistent — a small logging refactor in the LightningModule.

## Implementation plan (phases — each ends at a verify gate)

**Phase 0 — data plumbing, naming & the modality registry.**
- 0.1 Alias `observation_vector → proprio`; loader returns aligned `(proprio[6], image[128,128,3], action[2])`.
- 0.2 Downsample 256→128 **once at load**, cached.
- 0.3 **Modality registry**: config lists `{name, kind, encoder, decoder, weight, enabled}`; the model
  builds trunks/heads/fuser-streams/losses/metrics by iterating it. Enable/disable = config edit.
- 0.4 *Verify:* a batch yields aligned tensors (cached downsample); toggling a modality off in config drops
  its trunk/head/loss/metric with no code change.

**Phase 1 — logging contract / wandb restructure (ALL models, before any vision).**  *(moved up, per request)*
- 1.1 Adopt the head-named `loss/*` (`loss/flow|flow_consistency|pred_latent`, `loss/<head>`) and
  `metric/*` (`metric/proprio/pointwise_error`) namespaces in the LightningModule — **for DSAR/LSAR/
  diffusion right now**, no vision needed. Modality-specific keys (`loss/image`, `metric/image/*`) are
  *declared by the registry* and simply stay empty until the image head exists.
- 1.2 *Verify:* a current (non-vision) DSAR/LSAR/diffusion run logs the new grouped tree; nothing dropped.

**Phase 2 — ViT autoencoder (image only, standalone).**
- 2.1 Linear patchify (patch 16 → 64 patches) + posemb → ViT encoder (generalized block + FlexAttention, no mask).
- 2.2 Perceiver bottleneck: 8 learned queries cross-attend patches → 8 latent tokens.
- 2.3 ViT decoder: 8 tokens → per-patch query tokens → transformer → Linear-unpatchify → 128².
- 2.4 *Verify:* `smoke/vision_ae.py` + short fit; held-out recon PSNR passes threshold.

**Phase 3 — factorized space-time backbone + token-bag carried state (on LSAR first).**
- 3.1 Fuser gains the image + **action** streams; per-step bag = `[proprio, action, image_1..8]` (10 tokens),
  action a **separate input-only token** (type-embedded), conditioning via attention — not folded in.
- 3.2 Carried/predicted state = `[proprio] ++ [8 image]` (9 tokens); per-token `_ln`; action injected each
  step; `encode_state`/`to_obs`/`readout` handle the bag.
- 3.3 Generalize the `Transformer` block to be **mask-agnostic** (takes a `mask_mod`), then build the
  **factorized space-time block**: spatial attn (within step, unmasked) + temporal attn (across steps,
  causal) + MLP; space/time positional embeddings; FlexAttention for both.
- 3.4 Decode heads from the registry (proprio MLP, image ViT decoder); `loss/<head>` recon terms +
  `metric/image/{psnr,ssim,mse}` now populate.
- 3.5 *Verify:* smoke + short LSAR run; both recon losses fall; AR rollout runs; eff_rank > 1 (no collapse).

**Phase 4 — diffusion DiT denoiser.**
- 4.1 Swap MLP flow field → DiT over the 9 carried tokens (adaLN on τ + shortcut dd; condition on h).
- 4.2 Rectified-flow + shortcut unchanged.
- 4.3 *Verify:* adapt `smoke/diffusion.py` A–H; ε=0 readout byte-stable; decode → image + vec.

**Phase 5 — per-head loss weights & balancing.**
- 5.1 Wire the per-modality `weight:` from the registry into the objective; tune so no term dominates.
- 5.2 *Verify:* a run where every modality improves together (no recon term flatlines).

**Phase 6 — viz (under eval_ood_horizon).**
- 6.1 `eval_ood_horizon/<image-head>/filmstrip` — `viz.fig_image_filmstrip`: 8 steps across the horizon,
  pred (top) | GT (bottom), minimal text.
- 6.2 `eval_ood_horizon/<image-head>/rollout` — `viz.image_rollout_video`: pred (top, black until context
  plays out) | GT (bottom), synced, no text.
- 6.3 `eval_ood_horizon/<image-head>/{psnr,ssim,mse}_vs_step` — image metric over the rollout horizon,
  mirroring the existing proprio error-vs-step curve.
- 6.4 `eval_manifold` latent UMAP on concat-of-all-carried-tokens (data-space UMAP stays proprio 6-vec).
- 6.5 `denoising_*` can decode each ODE step to an image (watch the FPV denoise from noise).
- 6.6 *Verify:* eval logs the filmstrip + synced rollout video + metric-vs-step curves cleanly.

**Phase 7 — variations + multi-feed seam.**
- 7.1 Per-stream noise σ (proprio vs image).
- 7.2 Shared ViT + per-feed identity embedding, wired feed-agnostically (one feed today).
- 7.3 *Verify:* config can declare a 2nd feed without code surgery.
