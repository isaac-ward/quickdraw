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
