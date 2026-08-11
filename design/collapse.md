# Trainable-AE collapse in the joint world model

Why a from-scratch (bespoke) autoencoder co-trained with the dynamics **collapses**, why frozen
TAESD does not, and the loss-term structure behind it. Evidence: the owm-iss `tokenizer_ab` runs
(2026-08-10/11), `tf-bespoke` (trainable conv-enc/unet-dec) vs `tf-big` (frozen TAESD).

## The observation

`ae_floor` is a **dynamics-free** probe: `Dec(Enc(x)) → x`. So it measures the autoencoder alone.
The bespoke arm's ae_floor rose for three epochs, then fell off a cliff:

| tf-bespoke (per epoch)            | ep0  | ep1  | ep2  | **ep3** |
|-----------------------------------|------|------|------|---------|
| ae_floor PSNR (encode→decode)     | 30.7 | 30.8 | 31.8 | **12.8** |
| ae_floor LPIPS                    | 0.18 | 0.19 | 0.18 | **0.48** |
| val img PSNR (1-step)             | 18.2 | 28.1 | 30.2 | **13.6** |
| val decode/image                  | .016 | .006 | .005 | **.047** |
| val dynamics/latent               | .075 | .024 | .023 | **.060** |

The ~19 dB ae_floor cliff is a **representational collapse of the trainable AE**, not a dynamics
failure. Critically, the early *falling* `dynamics/latent` (0.075→0.023) was **the collapse in
progress, not learning** — the latent was becoming trivially predictable. Frozen TAESD (`tf-big`)
never does this: its ae_floor is flat (~33 dB) because the latent is fixed.

## The loss terms (one line each)

Let `z_t = Enc(x_t)`, `h_t = Backbone(z_≤t, a_≤t)`, `ẑ_{t+1} = z_t + Flow.sample(h_t)`, `sg` = stop-grad.

```
dynamics/latent   L_dyn = ‖ Flow_θ(h_t) − sg(z_{t+1} − z_t) ‖²     # predict the (stop-grad) latent STEP from context
                          h_t = Backbone( sg?(z_≤t) )              # z_≤t is sg iff dynamics_detach_encoder=True
decode/image      L_dec = ‖ Dec(ẑ_{t+1}) − x_{t+1} ‖²             # decode the PREDICTED latent  (recon_frac of frames)
decode/proprio    L_pro = ‖ Dec_p(ŝ_{t+1}) − s_{t+1} ‖²           # same, proprio
codec/roundtrip   L_rt  = ‖ Dec(Enc(x)) − x ‖²                    # decode the ENCODER'S OWN latent  ← ABSENT for bespoke
```

Bespoke total = `λ·L_dyn + w·L_dec + w·L_pro`. **No `L_rt`.** (`L_rt` is gated in
`multimodal.roundtrip_losses` to `hasattr(mod,"taesd")` — pretrained trunks only.)

## Why `L_dyn` has a degenerate (collapse) minimum

Set `z_t ≡ c` (constant over time). Then `z_{t+1} − z_t = 0`, `Flow` outputs 0, and **`L_dyn = 0`
exactly.** You "predict the future perfectly" by making the future have nothing to predict. The loss
cannot tell *smart-and-predictable* from *empty-and-constant* — both score 0. Stop-grad on the target
is what *allows* this (the same self-prediction collapse as BYOL/SimSiam; stop-grad is necessary but
not sufficient to prevent it).

**Refinement — it's PARTIAL, not full, collapse.** A *fully* constant `z` would also wreck `L_dec`
(constant latent → constant frame). So full collapse is not a joint optimum, which is why it took to
ep3 and looked like a cliff. The realized failure sheds exactly the information that is *hard to
predict* (high-frequency detail) while keeping low-frequency structure: that **lowers `L_dyn`** at
**near-zero `L_dec` cost**.

## Why `decode/image` does not charge for it

From `L_dec = ‖Dec(ẑ_{t+1}) − x‖²`:
1. **MSE loves blur.** Dropping detail moves the prediction toward a smooth mean; MSE barely
   penalizes that. The detail partial-collapse throws away is nearly free. (LPIPS was 0.18 the whole
   time — the blur was there from the start; MSE never saw it.)
2. **It decodes `ẑ` (the prediction), not `Enc(x)`.** It pins the *rollout output* to be decodable,
   never asserting the `Enc→Dec` identity that ae_floor measures.
3. **`recon_frac=0.25` + p_tf=0.** Quarter of frames, and under self-rollout `ẑ` drifts, so `Dec`
   chases the drift while the clean `Enc(x)` path is unconstrained — it compounds until it tips.

`L_rt = ‖Dec(Enc(x)) − x‖²` fixes (2) directly: a near-constant `Enc(x)` cannot reconstruct diverse
`x` → large loss. It forbids the catastrophic collapse. (It won't fully fix *blur* — that needs a
perceptual term or bigger decoder — but it stops the cliff.)

**Does a trivial latent optimize reconstruction too? No — the opposite.** A collapsed latent gives
the *same* `Dec(Enc(x))` for every `x` → cannot match diverse `x` → *high* recon loss. Reconstruction
is a genuine information bottleneck that *forbids* collapse. The trivial-latent minimum belongs to
`L_dyn`, not to `L_rt`.

## `dynamics_detach_encoder`: True vs False

The encoder appears in `L_dyn` in two roles, detached independently:

| role                | expression      | detached?                              |
|---------------------|-----------------|----------------------------------------|
| **target**          | `z[:,1:]`        | always (`.detach()` hardcoded)         |
| **context / cond.** | `z[:,:-1] → h_t` | only if `dynamics_detach_encoder=True`  |

- **False** (our run): `L_dyn` back-props into the encoder through the conditioning → the dynamics
  *shapes* the latent to be predictable (good) — but with no anchor, toward the collapse basin (bad).
- **True**: `s = s.detach()` cuts the context path too → `L_dyn` gives the encoder **zero** gradient.
  The encoder is trained *only* by reconstruction → cannot collapse, but is **not shaped to be
  predictable** either (reconstructable-but-dynamics-hostile — same downside as frozen TAESD).

## Design conclusion

- Frozen TAESD: reconstructable, **not predictable** — its latent was optimized for natural-image
  recon with zero temporal-smoothness pressure. That is *why* tf-big's `dynamics/latent` is high
  (0.235) and it learns slowly. Safe (can't collapse) but dynamics-hostile.
- `detach=True`: safe-but-lazy — kills the collapse pull *and* the predictability shaping.
- **The design we want:** keep `dynamics_detach_encoder=False` (dynamics shapes a *predictable*
  latent) **and add `L_rt`** (so it *cannot* collapse). The combined optimum is the compromise — a
  latent that is both smooth-for-dynamics and invertible-for-reconstruction. This is the one config we
  have not tried, and it is the only one that can give predictability AND reconstructability.

### Enabling `L_rt` for a bespoke (non-TAESD) trunk

Two-line unlock; the roundtrip computation (`Dec(Enc(x))` via `encode_state`/`to_obs`) is already
modality-agnostic — it was only *gated off*, not unimplemented:
1. `multimodal.roundtrip_losses`: drop the `hasattr(mod,"taesd")` condition (replace with "has
   encode+decode").
2. Give the bespoke image modality `latent_loss_weight > 0` (currently `None`→0).
Consider a **perceptual** roundtrip (LPIPS/feature loss, not just MSE) so it charges for *detail*,
not only for existence of structure.

## Experiments

Prereq code change (both): enable `L_rt` for a bespoke trunk (relax the `hasattr(...,"taesd")` gate +
set `latent_loss_weight>0`; consider a perceptual variant). Batch **24** for the trainable-AE arms —
the p_tf=0 autoregressive regime OOMs at 32 (see the tf-big ep2 OOM). Eval + val every epoch, run to
**≥6 epochs** (past the ep3 cliff). Reference points already in hand: `tf-big` (frozen TAESD, d4h8)
and the collapsed `tf-bespoke` (no anchor, d2h4).

### Experiment 1 — Anchor test: does the roundtrip loss stop the collapse?
Isolates the anchor. Both arms bespoke (conv-enc/unet-dec), `dynamics_detach_encoder=False`, identical
except `L_rt`:
- **1a (control):** no roundtrip → should reproduce the ep3 ae_floor cliff.
- **1b (test):** `+ codec/roundtrip_image` (MSE, `latent_loss_weight=1.0`).
- **PASS (1b):** ae_floor flat/rising through ep6+ (no cliff), LPIPS bounded, val recon stable — while
  `dynamics/latent` stays low *without* ae_floor crashing.
- **FAIL:** 1b still cliffs → MSE anchor insufficient; escalate to a perceptual roundtrip or
  `dynamics_detach_encoder=True`.
- Cheap variant: skip 1a (reuse the existing collapsed run) and run only 1b, at the cost of a
  non-identical control.

### Experiment 2 — Predictability payoff: is a co-trained *predictable* latent better than frozen TAESD?
Runs only if Exp 1's 1b survives. Removes the flow-size + batch confounds of the current A/B:
- **2a:** bespoke `+ roundtrip`, `detach=False` (= 1b).
- **2b:** frozen TAESD.
- **Matched:** same `flow_arch`/depth/heads and same batch for both.
- **Metric:** `dynamics/latent` (predictability) + open-loop image & proprio accuracy vs step, at
  comparable `ae_floor`.
- **PASS (2a):** lower `dynamics/latent` and better open-loop than 2b at similar reconstruction → a
  predictable latent is worth co-training an anchored AE.
- **FAIL:** frozen 2b matches/beats → predictability shaping isn't worth the fragility; freeze the AE
  and instead invest in the decoder (fine-tune/enlarge, encoder frozen — collapse-safe).

Optional 3rd arm for full mechanism isolation: bespoke `detach=True`, no roundtrip (reconstructable
but not predictability-shaped) — completes the {detach} × {anchor} 2×2.
