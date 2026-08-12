#!/usr/bin/env bash
# =============================================================================================
# robocasa-scene4-4h — ACTION CONDITIONING on the best-performing backbone  (user, 2026-08-12)
#
# BACKBONE = the tfz_affine config, which leads on everything except motion:
#   ae_floor 24.51 (HOLDING, vs mlp eroding to 19.88) | 1-step 18.29 | psnr@+64 13.64 (bar 10.40)
#   lpips@+1 0.166 | val 0.254 | grad/norm/flow 0.571        <- all ep1, logs/robocasa-tfaff/*_tfz_affine
# Its ONE weakness is motion: motion_ratio@+64 0.700 -> 0.126, i.e. sharper AND more static with training.
# So: add the action fix to the arm that is otherwise winning, rather than to the bespoke arm whose advantage
# rests on a single teacher-forced epoch of exploding-gradient code (see record section 9).
#
# WHY ACTION CONDITIONING. grad/norm/act_enc was 0.0010 against the flow's 0.5932 -- 0.17% of the total
# gradient. readout() sliced the action token's output off entirely, so actions reached a prediction ONLY via
# attention onto 1 of the bag's 10 slots, and a complete input->output path never touched them. A world model
# that barely reads its actions has no reason to move the arm.
#
#   GPU 0  tfz_act          concat_action_embedding=true, action_fourier_freqs=0
#   GPU 1  tfz_act_fourier  concat_action_embedding=true, action_fourier_freqs=16
# TWO clean single-variable reads:
#   A vs the LIVE tfz_affine  -> the raw action channel alone (that run has concat OFF; same everything else)
#   A vs B                    -> Fourier bands ON TOP of the channel. Fourier alone would be easy to ignore --
#                                it only makes small action differences separable, which is worthless if
#                                nothing reads them; on a dedicated channel it has something to feed.
#
# NOTE the config defaults now supply pretrained=true + latent_norm=affine + d=128 (they are COUPLED: affine
# needs a frozen trunk, and the EXACT adapter forces num_tokens*d == 1024). So this script only names what
# differs from the default.
#
# ALSO NEW SINCE THE LIVE RUNS: P1 skips encode_state(true_future) when p_tf==0 (11 of our 12 epochs) and P4
# encodes once and reuses it for context/targets/roundtrip. Both delete duplicate TAESD encodes. Before-numbers
# on identical settings for comparison: mlp 9138 s/ep, tfz 11659 s/ep.
#
# READ IT ON, in order:
#   1. grad/norm/act_enc      MUST rise off 0.17% of total if the channel is actually used. Free, already logged.
#   2. motion_ratio@+64       the goal. 0.126 is what tfz_affine decayed to. WHOLE-FRAME, so a rise could be
#                             background flicker -- the masked version is not built.
#   3. psnr@+64 - psnr_frozen@+64    tfz_affine is +3.24 dB; the bar is 10.40 dB.
#   4. ae_floor               must keep HOLDING (~24.5), not erode like the mlp arm did.
#   5. grad/norm/flow         O(1) through ep2 (the pre-zero-init transformer hit 766 exactly there).
# =============================================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export QUICKDRAW_LOG_ROOT=logs/robocasa-act
OUT=wizard/scripts/out; mkdir -p "$OUT"
DATA=/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71eb0b7d9ecbf1a7ada843da026d4bc0785

COMMON=(
  model=mm_flow                                  # defaults now: pretrained TAESD + affine + d=128 (coupled)
  model.heads=8                                  # head_dim = 128/8 = 16 (compiled FlexAttention minimum)
  model.modalities.1.num_tokens=8                # 8*128 = 1024 = the TAESD latent EXACTLY
  model.diffusion.flow_arch=transformer          # zero-init residual branches, unconditional
  model.diffusion.flow_arch_depth=2 model.diffusion.flow_arch_heads=4
  model.diffusion.concat_action_embedding=true   # BOTH arms: the RAW pre-backbone action channel
  model.recon_frac=0.25
  model.compile_rollout=true
  model.action_head.enabled=false
  model.p_tf_warmup_epochs=1
  model.action_dim=12
  model.modalities.0.dim=16
  model.modalities.1.img_size=128
  data.root="$DATA" data.repo_id=robocasa-scene4-4h data.cam=robot0_agentview_left
  data.F=64
  data.autobatch=true data.autobatch_headroom=0.35
  environments=recorded environments.obs_dim=16 environments.action_dim=12
  'environments.position_idx=[7,8,9]'
  trainer.max_epochs=12
  trainer.check_val_every_n_epoch=1
  eval.during_train.every_epochs=1
  'eval.during_train.at_epochs=[]'
  eval.horizon=128
  'eval.closed_loop_steps=[1,16]'
  eval.during_train.evals.ae_floor=true
  eval.during_train.evals.ood_horizon=true
  eval.during_train.evals.manifold=true
  eval.during_train.evals.denoising_filmstrip=true
  eval.during_train.evals.denoising_multistep=true
  eval.during_train.evals.denoising_aggregate=true
  eval.during_train.evals.control=false          # RecordedEnv cannot step
  eval.during_train.evals.action_distribution=false
)

launch(){ # gpu name action_fourier_freqs
  docker compose exec -T -e QUICKDRAW_LOG_ROOT="$QUICKDRAW_LOG_ROOT" -e CUDA_VISIBLE_DEVICES=$1 \
    app uv run python -m quickdraw.train_world_model "${COMMON[@]}" experiment=$2 model.action_fourier_freqs=$3 \
    "+run_summary.problem=\"Second restart. The previous pair completed epochs zero and one and was then killed by our own eval escalation guard, because the denoising filmstrip routine hand built the flow conditioning at the old width and failed twice in a row once the conditioning gained the action slot and raw action channels. Both are fixed: the filmstrip now routes through the single conditioning function, and a failing diagnostic disables that routine rather than killing the run. Underneath, motion is still the deficit: motion ratio at sixty four steps sat at zero point one six.\"" \
    "+run_summary.tried=\"The salvaged two epochs show the action conditioning works mechanically: the action encoder gradient norm went from zero point zero zero one and zero point zero zero eight nine without conditioning, to zero point zero one two and zero point zero five with the slot and raw channels, and to zero point zero two one and zero point zero six six with sixteen fourier bands on top, so five to seven times more gradient reaches the action pathway. Motion ratio at sixty four steps improved from zero point one two six to about zero point one six two, with one step PSNR, long horizon PSNR and validation loss all unchanged.\"" \
    "+run_summary.trying=\"Arm $2 with the corrected conditioning: the action token's backbone output is now ALWAYS part of the per token conditioning rather than sliced away, and the raw pre backbone embedding is concatenated as a third channel, so the denoiser sees the state, the action contextualised by the state, and the raw action, at each of six ODE steps. Action fourier bands are set to $3, and the two arms differ only in that.\"" \
    "+run_summary.trying_detail=\"Frozen TAESD at 128 pixels with the exact eight by one hundred and twenty eight adapter, affine latent normalization, transformer denoiser at depth two and four heads with unconditional zero initialised residual branches, recon_frac 0.25, F 64, autobatch at headroom 0.35, p_tf warmup one epoch to zero, twelve epochs, all compatible evals every epoch including epoch zero. Also first runs to include the shared encode and teacher forcing skip speedups.\"" \
    "+run_summary.rationale=\"The mechanism is confirmed to engage but the effect on motion is small so far, and only two epochs were observed before the run died on a visualisation bug. Repeating it to twelve epochs answers whether the extra action gradient compounds into real movement or plateaus, and whether the fourier bands justify themselves beyond the extra gradient they demonstrably attract.\"" \
    > "$OUT/$2.out" 2>&1 &
}

launch 0 tfz_act         0
launch 1 tfz_act_fourier 16
echo "[launched] tfz_act fourier=0 (GPU 0) | tfz_act_fourier fourier=16 (GPU 1)"
echo "[logs] $OUT/tfz_act.out | $OUT/tfz_act_fourier.out"
wait
