# Discrete / tokenized latents (VQ) — design note

**Status: planned, not implemented.** A *prediction mechanism* option: quantize the latent into discrete
codes and predict the next code by classification, instead of regressing a continuous latent. This is the
IRIS / Genie / VideoGPT lineage (and the discrete side of DreamerV3), distinct from what we do today
(continuous-latent regression via a direct readout or diffusion).

## What vector quantization (VQ) is

VQ turns a continuous vector into a discrete symbol by snapping it to the nearest entry of a learned
**codebook**:

- Keep a codebook of `K` learnable `d`-dim vectors — a fixed vocabulary of "prototype latents."
- To quantize a latent `z`, find the nearest codebook vector `e_k` (by L2) and replace `z` with `e_k`;
  the output is that vector *and* its integer index `k ∈ {1…K}`.
- So a continuous `d`-dim vector becomes **one discrete token** — like mapping a point to its closest
  cluster center and recording the cluster ID.

Consequence for us: our AE currently emits **continuous** tokens, so next-latent prediction is a
**regression**. Insert a VQ quantizer after the encoder and each token becomes one of `K` codes → next-latent
prediction becomes a **classification** (softmax over the codebook). Origin: VQ-VAE (arXiv 1711.00937).

## How the K codebook vectors are learned

Initialize randomly, then move each code toward the encoder outputs that select it — essentially online
k-means. Two standard update rules:

- **Codebook loss (gradient):** `‖sg[z] − e_k‖²` pulls the chosen code `e_k` toward the encoder output `z`
  (`sg` = stop-gradient), and a **commitment loss** `β‖z − sg[e_k]‖²` pulls `z` toward `e_k` (updates the
  encoder).
- **EMA update (more stable, common):** don't gradient the codebook — set each `e_k` to an exponential
  moving average of the encoder outputs assigned to it (the running cluster mean).
- **Dead codes:** codes never selected must be reinitialized (e.g. to a random recent encoder output),
  or capacity is lost.

Because "nearest neighbour" (argmin) is non-differentiable, reconstruction gradients reach the encoder via
the **straight-through estimator**: forward pass uses `e_k`, backward pass copies the gradient straight to
`z` as if quantization were identity.

## VQ vs Dreamer — two different routes to a discrete latent

Both are discrete and use straight-through gradients, but the mechanism differs:

- **VQ (this doc / IRIS / Genie):** encoder → continuous vector → snap to the nearest codebook vector
  (pick by **distance**). The codebook vectors carry the meaning.
- **Dreamer (categorical latent):** the spine emits **logits** → sample a one-hot class (pick by
  **probability**), no codebook, no nearest-neighbour. Trained by KL to a learned prior. See the parametric
  -distribution prediction mechanism.

So they are independent options, not the same thing.

## How training would work here

Two objectives on top of the existing space-time spine:

1. **Tokenizer (AE + VQ):** encode obs → continuous tokens → VQ-quantize → decode → **reconstruction MSE +
   codebook loss + commitment loss** (straight-through through the quantizer). Makes codes meaningful and
   decodable.
2. **Dynamics (next-code prediction):** feed the sequence of code embeddings + actions to the transformer,
   predict the **next token's categorical over the K codebook**, trained with **cross-entropy** against the
   true next code index. This *replaces* the current continuous latent-regression / diffusion step.

Rollout / imagination: predict the next categorical → sample (or argmax) a code → embed it via the codebook
→ feed back autoregressively.

Contrast of dynamics losses across mechanisms:

| mechanism | latent | dynamics loss |
|---|---|---|
| ours today (direct / diffusion) | continuous | MSE to true-next (or flow/denoising) |
| Dreamer (parametric categorical) | discrete (logits→one-hot) | KL(posterior‖prior) |
| VQ-tokenized (this doc) | discrete (codebook index) | cross-entropy over the codebook |

## Fit in this codebase

- Add a VQ module after the modality encoder(s) (continuous token → code index + embedding), with either
  gradient or EMA codebook updates and dead-code resets.
- Swap the prediction head from regressing a `d`-vector to classifying over the `K`-entry codebook
  (cross-entropy), and embed sampled codes for autoregressive feedback.
- The decoder / reconstruction path is unchanged except it now decodes quantized tokens.
- Everything else (the space-time spine, action conditioning, the eval routines) is untouched — VQ is a
  swap of the latent representation + prediction head, not a spine change.
