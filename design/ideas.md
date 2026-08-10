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
