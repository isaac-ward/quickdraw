# PLAN UNDER AUDIT — matched pair, new codec + generative decoder + anti-collapse

## Standing goal (user, emphatic, in memory)
Sharpness in LONG OPEN-LOOP predictions. The AE floor is explicitly NOT to be optimised — it is the
CEILING, not the deliverable. Judge on the DYNAMICS PENALTY = `OL LPIPS@+64/@+128` minus `ae_floor LPIPS`.
A config with a worse floor and a smaller penalty is PREFERRED.

## What just happened (the evidence this plan reacts to)
Two independent changes produced the SAME collapse:
- §18 `dynamics_follows_p_tf=true` (full substitution of the dynamics loss context): latent_cos@+32 2.1x
  the control for 5 epochs, then grad/norm/flow 0.42 -> 1.3e7, ae_floor 19.1 -> 11.4, latent_cos NEGATIVE.
- `predict: absolute` (this week): epoch-matched WINS at e0/e1/e2 on every OL metric (penalty ~3x smaller
  than the control), then at e3: grad/norm/flow 0.341 -> 108.9, floor PSNR 18.35 -> 8.94, latent_cos
  0.261 -> -0.056, dynamics loss 0.151 -> 6.84, `grad/nonfinite_skipped=1`, `norm_postclip=0`.

FINGERPRINT, both times: everything INSIDE the recurrent path goes inf (flow, backbone, encoders, act_enc);
the decoders, which sit OUTSIDE it, stay finite (decode_image 5.75, decode_proprio 4.00 at the blow-up).
Diagnosis: exploded ROLLOUT JACOBIAN, not a loss-target problem. `predict: absolute` removes the identity
skip (step Jacobian `J` instead of `I + J`), which makes that product strictly less contractive.

Control (`residual`, detach_every=32, bottleneck 8, mse decode) is stable and plateaus: latent_cos@+64
~0.137 from e3 to e12, OL LPIPS@+128 ~0.34, penalty ~0.10.

## The plan
Kill BOTH runs. Launch a matched pair, ONE variable between arms.

SHARED:
  data.img_size            96   (from 128)            <-- speed
  model.ae_bottleneck      16   (from 8)              <-- §17: the binding codec constraint
  model.detach_every       16   (from 32)             <-- anti-collapse: bounds the BPTT Jacobian product
  image decode_kind        flow (from mse)
  image decode_arch        unet (unchanged)
  image decode_param       x0   (unchanged)
  image decode_stochastic  true (NEW FLAG, committed 21786e2)
  image decode_steps       6
  image decode_base        32   (unchanged, user: keep)
  model.recon_frac         1.0  (unchanged; largest measured lever)
  model.p_tf_dynamics      1.0  (default = always-clean = historical behaviour)
  latent_norm layernorm, flow_arch transformer, sampling_steps 6, stochastic_eval true
ARM A (GPU 0): model.predict = residual   <-- new control
ARM B (GPU 1): model.predict = absolute   <-- re-tests the parameterisation under detach_every=16

## Facts established by reading the code (verify these)
- `use_action_slot` is ALREADY True for MultiModalFlow (multimodal.py:925, class at :887). Conditioning is
  already h_state ++ action_slot ++ raw act embedding, `_hd = 3d`. NOTHING TO CHANGE.
- `ImageUNetFlowHead` (flow.py:336) already exists, already takes `chunk` (modalities.py:192), and
  `decode_arch` is documented ORTHOGONAL to `decode_kind` (all 4 combos). Conv generative decoder with the
  same checkpoint-chunking as the mse one is CONFIG-ONLY.
- `decode()` hardcoded deterministic=True and steps=1 for x0. Traced: deterministic -> eps=0 -> the x0
  branch's first iteration is `velocity(zeros, tau=1, cond)` == the no_noise/mse branch's one line.
  MEASURED max|diff| == 0.0. So decode_kind=flow was a TRAINING-ONLY change and bought NO sharpness.
  `decode_stochastic` (default False, mse unaffected) fixes this. Verified: mse repeatable; flow+x0
  deterministic repeatable and == velocity(zeros,tau=1) to 0.0; stochastic DIFFERS across calls (std 0.317).

## A PROBLEM I FOUND WITH 96 — confirm or refute
`n_levels = max(1, int(math.log2(max(bott, min(h,w)) // bott)))` (vision.py:346 encoder after a stride-2
stem, :385 decoder from full res). int() TRUNCATES.
    img=128 bott=16 -> enc log2(64//16)=2 -> 64->16 EXACT ; dec log2(128//16)=3 -> 128->16 EXACT
    img=96  bott=16 -> enc log2(48//16)=log2(3)=1.58 -> 1 -> 48->24 ; dec log2(96//16)=2.58 -> 2 -> 96->24
So at 96 "bottleneck=16" SILENTLY MEANS 24x24 -- a 2.25x larger spatial grid than asked for, different
memory, and "bottleneck 16" would mean two different things at the two resolutions.
    96 with bott=12 -> enc log2(48//12)=2 -> 48->12 EXACT ; dec log2(96//12)=3 -> 96->12 EXACT
and (96/12)^2 = 64x spatial reduction == (128/16)^2, i.e. the apples-to-apples match to 128+bott16.

## Questions for the auditor (be adversarial)
1. Is the 96/bottleneck truncation real? Is bott=12 at 96 the right match, or does 96 break something else
   (patch=16 -> 96/16=6 for any ViT path; dataset downsample; flex-attention shapes; decode_chunk)?
2. Does 96 invalidate the comparison to every recorded result at 128? Is a resolution change worth it at
   all given the decision window for a collapse test is e0-e4?
3. THREE simultaneous changes (codec, decoder, detach) + a 4th (resolution). If both arms underperform the
   old control we cannot attribute. Is the matched pair still worth it, or should one axis be isolated?
4. `detach_every` 32 -> 16: does this actually bound the Jacobian product, or does it just halve the
   window while the per-step Jacobian norm is what exploded? What would 8 buy? Is there a cheaper direct
   test of the exploded-Jacobian hypothesis?
5. `decode_stochastic=true` + `recon_frac=1.0`: the decode loss is the ONLY autoregressive gradient in the
   model. Does making the decode a SAMPLE inject variance into that gradient path, and could that itself
   destabilise the rollout? Note decode_loss uses `loss()` (random tau), NOT `sample()`, so training may be
   unaffected -- CHECK whether decode_stochastic touches the training graph at all or only eval/products.
6. Does anything read `ae_bottleneck` or `img_size` that would break at these values (eval routines,
   filmstrip floor decode, autobatch probe, checkpoint compat, decode_chunk_train=64)?
7. Is `p_tf_dynamics=1.0` right here, or should the pair carry 0.9?
8. Anything else that would waste 10 hours.
