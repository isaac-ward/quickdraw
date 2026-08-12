Ideas of the user, do not incorporate into designs. Ignore. Just for the user to think about stuff

can we add another thing to the
  manfiold/ logs? its that horizon thing but as an animation. so it starts at h01, then interpolate to h02,
  h04, 8, 16, 32, 64 to show how accurate the denoising is over longer predictions. just in the dataspace over
  the position coordinates. hold on each timestep poitn cloud for 2 second, then tween (slow, fast, slow)
  between the next point cloud in 1 second. if a frmae is gonna eb identical don't rerender it. just shows how
  good the predictions are over long horizons. actually let me get some feedback. should this be in the
  eval_ood_horizon? and then we can keep doubling until we're quite far? we already would have generated the
  data right? thoughts? how many long horizon eval trajectories are there?



   also where are the image head renders in eval ood horizon? and why did you remove the progress of the
  eval prcedures in progress.log? why is there ood_horizon-mm and not the old eval_ood_horizon in the
  loggingg? it seems like you tried to reimplement mms tuff from scratch rather than generalize and extend
  the existing logging. THis was a huge mistake (progress.log updates should be saying '10% done', '20%
  done', etc. for each of the eval procedures (eval_control, eval diffusiin,eval_ood_horizon) and for each
  of the main output logging products WITHIN in eval procedure (i.e. the predicteed image/gt video, and the
  scene animation in eval_horizon). also whre is the image head stuff that we discussed in the
  eval_ood_horizon. I think eval_control is hung also - look at progress.log and get back to me. i think we
  should kill everything and launch a smokle run when ready that has an eval at epoch 1 (not zero) so we
  can confirm everything works and is not missing. the smoke would be a non image diffusion run on gpu 0,
  and an image diffusion run on gpu 1. ALSO in general there SHOULDN'T BE A VECTOR PATH AND AN MM PATH.
  Just one general path that can be aprameterized to do image only, proprio only, or both. Does that make
  sense? So for example you started this fix by making mm open loop reporting report the same 4 m,etrics as
  proprio only open loop. But there should not be thse two things in the codebase - just ONE! gett it? as

  glad you're checking out the error that caused the crash, but the refactor checklist should be top of
  mind before launching another test. look at my last prompt and build that tcheclikst and start addressing
  the requests one by one so thatyoudon't lose track. be very granular when you turn what i requested into
  checklist items



  gifs



  oth paths. ✓
- Effective receptive field — stack L layers each with a W-window and the reach compounds: layer-2 at position j attends layer-1 at [j−W+1, j], and each of those attended back another W. So a depth-L windowed transformer sees back ≈ L·(W−1), not W. This is standard sliding-window attention (Mistral, Longformer): window per layer, receptive field grows with depth.

The legacy per-step rollout re-feeds only the last W raw bags each step → deep layers physically can't reach past W → receptive field capped at W. That's "sliding window over raw inputs," which is more restrictive than standard windowed attention. The cache (== full-mask forward) keeps each attention at W but lets depth compound → the standard semantics.

The reassuring part — what's actually different between train and eval:

┌─────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
│                    Path                     │                            Window semantics                             │
├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Teacher-forced training (forward(), p_tf≥1) │ builds _block_mask(W,T) over the full sequence → true window (== cache) │
├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ In-rollout training (rollout_train, p_tf<1) │ per-step truncated → capped at W                                        │
├─────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Eval (now, cached)                          │ true window (== teacher-forced training)                                │
└─────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

So the cache exactly matches the teacher-forced path. The only mismatch is with in-rollout training, and only for horizon > window (32). Within 32 steps: bit-identical.

Is it an odd thing that causes errors? No — mild, and benign-directional:
- It's a ~0.02 (≈2%) shift on decoded obs, only past step 32, and only because our A′ schedule ends in-rollout.
- Direction is safe: eval gives strictly more, correctly-computed context with the same weights — not garbage. And because the p_tf schedule spends its warmup epochs teacher-forced, the model has already seen true-window features, so they aren't foreign at eval.
- It won't produce qualitative failures (nothing like the 0,0,0 collapse) — it's a quantitative long-horizon nudge.

If you ever want zero mismatch: either train teacher-forced (matches the cache, but we left TF deliberately for drift-robustness), or apply the O(T²) in-rollout fix from before. My call: accept it, note it (done, in config + memory).

## LPIPS alongside PSNR for long-horizon (deferred 2026-08-10)

PSNR cannot distinguish "blurry hedge" from "sharp but wrong", and those need different fixes. The long-horizon
plateau on robocasa (single-step 18.39 dB, 64+ step rollout stuck at ~12.3 dB, rollouts looking like a mean
frame) is currently unattributable between the two.

NOTE this dataset is DETERMINISTIC (replayed trajectories, actions given), so unlike a stochastic env the
blur cannot be excused as correct hedging over possible futures — PSNR is a legitimate target here and the
plateau is a real failure. LPIPS is wanted as a DISCRIMINATOR, not as a replacement metric: add
eval_ood_horizon/<mod>/lpips_mean beside psnr_mean over the same rollouts. Divergence between the two curves
(PSNR flat, LPIPS rising) says blur; both falling together says wrong-but-sharp.

Cheap: one perceptual net over frames already rendered by the existing eval. Deferred only for focus.

## Position is never injected anywhere (noted 2026-08-10)

Perceiver-IO's latent array is structure-free, but position is injected at BOTH boundaries: Fourier features
(sinusoids of the (x,y) coordinates, concatenated to each input element) going IN, and positional QUERIES
(one query per output location, cross-attending into the latent array) coming OUT. TiTok does the output
half differently but does it: its decoder rebuilds an H/f x W/f grid of mask tokens with positional
embeddings.

We do NEITHER:
  - IN:  the backbone adds a LEARNED PER-SLOT embedding — one vector per slot index. That is a learned
         absolute PE over ~10 slots, not sinusoidal features over 2D coordinates. The model is never told
         where in the image a token's contents came from, only which slot it is. (Temporally we DO use RoPE,
         which is sinusoidal + relative, so we are close to Perceiver in time and not at all in space.)
  - OUT: to_obs is a fixed index rearrangement back to the latent grid, then TAESD's conv decoder. No
         cross-attention, no positional queries. We get a WEAK version of "the decoder re-imposes 2D" for
         free — once tokens are back on the grid, TAESD's convolutions supply locality — but nothing in our
         own code re-injects position.

Cheap thing to try: 2D Fourier features of each latent cell's (h,w), folded into the adapter's residual
branch. Costs no tokens and no capacity, keeps EXACT, and would tell us whether "the model does not know
where anything is" is a real handicap or a non-issue at 8 tokens.


## RETRACTED (2026-08-12): action dropout + CFG is not the next move

Measured on the §11 ep3 checkpoints (record §12). CFG amplifies `v_action - v_null` -- the true-vs-zeroed-action
axis. The model already responds strongly there and it buys nothing: `tfz_act_fourier` has **7.6x** the
response of `tfz_act` on that axis (pixel/pert 9.81 vs 1.29 at +64, -3.96 dB vs -0.48 dB when actions are
zeroed) with **identical** order-insensitivity (0.20-0.26 vs 0.12-0.28) and no better motion (0.154 vs
0.168). The dead axis is action ORDER, which CFG does not touch: at +64 the action sequence can be REVERSED
with zero accuracy cost (17.11 vs 17.03 true, reseed band 16.96-17.11). Fourier's large zeroed-action
response is most likely OOD brittleness from 384 sin/cos bands, not comprehension.

Keep the design below for reference, but do not implement it on this evidence.

## Action dropout + classifier-free guidance on the action (designed 2026-08-12, NOT implemented)

The consensus fix in the literature for a world model that under-uses its actions. Four independent systems use
it: Vid2World (arXiv:2505.14357, which fixes exactly "lacks counterfactual reasoning" this way), UniSim
(2310.06114), GAIA-2 (2503.20523) and Genie 2. It slots directly into a rectified-flow denoiser.

**Training — no extra loss term.** With probability p (~0.1, per-timestep in Vid2World) replace the action
embedding with a LEARNED NULL vector. The flow-matching loss is unchanged; because the model sometimes sees the
null, it learns both v(.|a) and v(.|null) as a byproduct. It is a data augmentation on the conditioning, not a
new objective. Free.

**Sampling — CFG.** Integrate a field that exaggerates the action-attributable component:

    v = v(null) + w * ( v(a) - v(null) )        w > 1;  w = 1 is exactly off

This is a decomposition, not a hack: v(a) - v(null) IS "the part of the predicted change attributable to the
action". It CANNOT live in the loss -- the loss must fit the true conditional, whereas CFG deliberately samples
from a SHARPENED, non-data distribution ~ p(x|a) * [p(x|a)/p(x)]^(w-1), trading diversity for conditioning
fidelity. Hence sampling-time only.

**Knobs it would need:** `model.diffusion.action_dropout_p` (training) and `model.diffusion.action_cfg_weight`
(sampling, 1.0 = off). NOT an eval routine -- CFG changes how the model samples, like stochastic_eval.

**Cost:** dropout free; CFG DOUBLES the denoiser (12 velocity evaluations per rollout step instead of 6, the
backbone still running once) -> guess +25-35% of rollout time, unmeasured. Apply at eval/inference only, as the
papers do; using it inside the training rollout would double training cost and is not standard.

**TWO TRAPS.**
1. CFG AMPLIFIES action-dependence, it does not create it. If the model truly ignored actions then
   v(a) ~ v(null), the difference is ~0 and CFG is a no-op. It cannot rescue an action-blind model; it makes a
   weak-but-real dependence visible. If the dependence is noise, it amplifies noise. So measure first (see the
   shuffled-action mode below) -- know there is something to amplify.
2. THE NULL TOKEN IS NOT A ZERO ACTION. `null` = "no conditioning given"; `a = 0` = the real command "hold
   still". v(a=0) should predict nothing moves; v(null) should predict the AVERAGE over plausible actions. They
   differ, so CFG correctly pushes commanded stillness to be STILLER. But if null were implemented as ZEROS, and
   a zero action also encodes to ~zeros, the two collapse and CFG becomes a no-op exactly when the action is "do
   nothing". null MUST be a distinct learned parameter.

## Shuffled-action rollout mode + action_delta (designed 2026-08-12, NOT implemented)

The field's minimum bar for CLAIMING action-conditioning; Genie's DeltaPSNR (arXiv:2402.15391) is the canonical
form. GameNGen is the cautionary tale -- it shipped fidelity numbers and human raters with NO action-following
metric at all.

**shuffled_action is a MODE (a rollout).** Same context, but the model is fed SOMEONE ELSE'S action sequence,
permuted across the BATCH -- so every action stays a real action from the dataset, just paired with the wrong
episode. A difference then means "the model reads WHICH action", not "the model reacts to garbage". Scored
against the SAME ground-truth future: with the true actions you should match it, with someone else's you should
not.

It folds into eval_ood_horizon as a 4th mode. The descriptor there is (name, every, horizon) where `every` is
the re-grounding interval; shuffled_action varies a different axis, so it grows a flag:

    modes = [("open_loop", H, H, shuffle=False)]
          + [(f"closed_loop_{x}_steps", x, cl_h, shuffle=False) for x in cl_steps]
          + [("shuffled_action", H, H, shuffle=True)]

Every metric comes free (psnr/ssim/mse/l1/lpips/psnr_frozen/motion_ratio, the @+N readouts, the figures). Cost:
one extra rollout, ~+15 s on top of the current 45.5 s for three modes.

**The delta is a DERIVED CURVE (subtraction), not a rollout -- and it lives INSIDE the shuffled_action
namespace** (user, 2026-08-12), not as a sibling of the modes. It only exists because that mode was run, so
attaching it anywhere else orphans it from its cause:

    eval_ood_horizon/shuffled_action/image/psnr                 the mode's own curve  (+ _mean, + @+N)
    eval_ood_horizon/shuffled_action/image/delta_psnr           open_loop - shuffled  (+ _mean, + @+N)
    eval_ood_horizon/shuffled_action/image/delta_motion_ratio
    eval_ood_horizon/shuffled_action/image/delta_lpips

Implementation shape: add `delta_<metric>` as extra KEYS in the shuffled_action mode's icurves dict --
the same trick that got lpips for free. Every consumer already iterates that dict, so the curves, the `_mean`
scalars, the `@+{1,8,16,32,64}` readouts and the figures all come along with no call-site changes.

ORDERING CONSTRAINT: the delta needs open_loop's curves, so shuffled_action must be scored AFTER open_loop in
the mode loop (or the deltas computed once all modes are done).

Both absolutes must be kept alongside the delta to interpret it -- a 0.5 dB delta means something different at
13.9/13.4 than at 5.0/4.5.

**The two deciding numbers:**

    eval_ood_horizon/shuffled_action/image/delta_psnr/@+64          ~ 0  =>  action-blind
    eval_ood_horizon/shuffled_action/image/delta_motion_ratio/@+64  ~ 0  =>  the motion produced is
                                                                            UNRELATED to the command

The second is the sharper statement: PSNR-delta says the prediction changes with the action; motion_ratio-delta
says whether the MOVEMENT is action-driven.

Related, more work: Vista (2405.17398) infers the trajectory back out of generated video with an IDM and reports
L2 to the commanded one (3.785 -> 0.832 with conditioning).
