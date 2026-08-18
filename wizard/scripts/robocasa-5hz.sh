#!/usr/bin/env bash
# =============================================================================================
# robocasa-scene4-4h — SUBSAMPLE TO 4 Hz (data.subsample=5), TWO SEEDS  (user, 2026-08-12)
#
# THE FINDING THIS TESTS (record §13). robocasa is 20 Hz. At 20 Hz the true per-step image change is
# 0.0389 RMSE against the frozen TAESD's OWN 0.0637 reconstruction RMSE -- the motion we ask the dynamics
# to predict is 0.61x the NOISE FLOOR of the codec it predicts through, and only 3.18% of pixels move more
# than that error. Predicting ~zero motion is therefore the CORRECT solution to the objective, not a bug.
# That is what every run did (motion_ratio 0.13-0.17) and why 5.8-16x more action gradient changed nothing.
#
#   stride 1 (20 Hz): per-step delta 0.0389 = 0.61x floor | action linear R2 0.0062, correct TIMING worth 0.0002
#   stride 5  (4 Hz): per-step delta 0.0863 = 1.35x floor | action linear R2 0.0293, correct TIMING worth 0.0078
#
# So stride 5 flips SNR above 1, raises the action's explanatory power 4.7x and the value of correct action
# ORDER 39x (which is the exact axis measured dead in §12), and makes F=64 span 16 s instead of 3.2 s.
# It is also ~7x CHEAPER per epoch (35,085 windows vs 259,299 frames -> ~1,100 batches vs 7,582).
# Convergent literature: V-JEPA-2-AC 4 fps with integrated EEF deltas (2506.09985), HMA 2 Hz (2502.04296),
# IRASim ~4 fps (2406.14540); FAST (2501.09747) names high-frequency action correlation as the cause.
#
# WHY TWO SEEDS AND NOT AN A/B. §11 measured the run-to-run noise floor at +-0.8 dB on open_loop@+64 and
# +-0.03 on motion_ratio. EVERY A/B this week (df, tf_ln/affine, mlp/tfz, act/act_fourier) had an effect
# inside that floor, so none of them were resolvable -- four non-answers and one retracted claim. There is
# no seed knob in this repo (no seed_everything anywhere), so two runs of the SAME config ARE a 2-seed
# replicate: it measures the effect against the well-established 20 Hz baseline AND establishes the noise
# floor at the new rate, which every future claim needs.
#
# BASELINE TO BEAT (20 Hz, 4 runs, consistent): 1-step peaks ~18.8 dB, open_loop@+64 ~13.9 dB,
# motion_ratio@+64 0.13-0.17, and BOTH arms peaked at ep2 then COLLAPSED at ep4 (1-step 18.8 -> 9.1).
#
# READ IT ON: motion_ratio@+64 (target >0.4; the codec ceiling is ~0.92) and the action-order gap from the
# §12 probe. NOT on motion_ratio alone -- it is a direction-blind magnitude ratio and a COLLAPSED model
# scores HIGHER on it (the ep4 collapse read 0.17-0.33). Judge on PSNR + order gap together.
#
# max_epochs 40: ~7x cheaper epochs, and the small-data literature regime is 30-200 epochs (DreamGen,
# WorldEval, HMA post-train). ~25 min/epoch -> ~17 h. Collapse is expected at some point; best.ckpt keeps
# the peak and evals run every epoch so the curve is captured either way.
# =============================================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

export QUICKDRAW_LOG_ROOT=logs/robocasa-5hz
OUT=wizard/scripts/out; mkdir -p "$OUT"
REPO=isaac-ronald-ward/robocasa-scene4-4h
CAM=robot0_agentview_left

DATA=$(docker compose exec -T app uv run python -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='$REPO', repo_type='dataset'))" | tail -1 | tr -d '\r')
echo "[data] $DATA"

COMMON=(
  model=mm_flow
  model.d=128 model.heads=8
  model.modalities.1.num_tokens=8
  model.diffusion.flow_arch=transformer
  model.diffusion.flow_arch_depth=2
  model.diffusion.flow_arch_heads=4
  model.diffusion.concat_action_embedding=true
  model.action_fourier_freqs=0        # OFF: §12 showed fourier attracts 2.8x the action gradient, has the
  #                                     SAME order-insensitivity, no better motion, and 7.6x the response to
  #                                     zeroed actions -- most likely OOD brittleness from 384 sin/cos bands.
  model.action_squash=none            # symlog OFF (user, 2026-08-12). DreamerV3 symlogs obs/rewards/values,
  #                                     NOT actions; the literature standard for actions is quantile norm.
  model.recon_frac=0.25
  model.compile_rollout=true
  model.action_head.enabled=false
  model.p_tf_warmup_epochs=1
  model.action_dim=12
  model.modalities.0.dim=16
  model.modalities.1.img_size=128
  data.root="$DATA" data.repo_id=robocasa-scene4-4h data.cam="$CAM"
  data.F=64
  data.subsample=5                    # <-- THE CHANGE. 20 Hz -> 4 Hz, actions aggregated across skipped frames
  data.autobatch=true
  environments=recorded environments.obs_dim=16 environments.action_dim=12
  'environments.position_idx=[7,8,9]'
  trainer.max_epochs=40
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
  eval.during_train.evals.control=false
  eval.during_train.evals.action_distribution=false
)

PROBLEM='At twenty hertz the motion this model is asked to predict is smaller than the reconstruction error of the autoencoder it predicts through. The measured per step image change is zero point zero three eight nine root mean square against the frozen autoencoder own zero point zero six three seven, so the signal is zero point six one times the codec noise floor, and only three point one eight percent of pixels move more than that error. Predicting no motion is therefore the correct solution to the objective we wrote down, which is exactly what four runs did, and it explains why five to sixteen times more action gradient changed nothing and why the action ordering could be reversed at sixty four steps at zero cost.'
TRIED='Stronger action conditioning was tried twice and reproduced only a gradient increase, never an outcome change, and the apparent motion gain did not survive a replicate. Both arms peaked at epoch two, regressed at epoch three and collapsed at epoch four from eighteen point eight to nine point one decibels one step. A literature review found that every published robot world model that controls long rollouts subsamples to between two and five hertz and integrates the actions across the skipped frames, and that our own step count of ninety one thousand is already normal, so more epochs at twenty hertz is not the answer.'
DETAIL='Frozen TAESD at one hundred and twenty eight pixels with the exact eight by one hundred and twenty eight adapter, affine latent normalization, transformer denoiser at depth two with four heads and zero initialised residual branches, recon fraction zero point two five, F sixty four, autobatch at headroom zero point three five, teacher forcing warmup of one epoch to zero, forty epochs, all compatible evals every epoch. Action fourier bands are off and action squashing is off. Frames are kept every fifth step and the actions driving each kept transition are summed, except the three near binary dimensions which take the last value.'

launch() {   # $1=gpu  $2=experiment  $3=trying  $4=rationale
  docker compose exec -T -e QUICKDRAW_LOG_ROOT="$QUICKDRAW_LOG_ROOT" -e CUDA_VISIBLE_DEVICES="$1" \
    app uv run python -m quickdraw.train_world_model "${COMMON[@]}" experiment="$2" \
    "+run_summary.problem=\"$PROBLEM\"" \
    "+run_summary.tried=\"$TRIED\"" \
    "+run_summary.trying=\"$3\"" \
    "+run_summary.trying_detail=\"$DETAIL\"" \
    "+run_summary.rationale=\"$4\"" \
    > "$OUT/$2.out" 2>&1 &
  echo "$!"
}

P0=$(launch 0 hz4_seedA \
 'Subsample the twenty hertz data to four hertz, keeping every fifth frame and aggregating the actions that drive each kept transition, so the per step image change rises to one point three five times the codec noise floor instead of zero point six one times it. This is the first seed of two identical runs.' \
 'At four hertz the per step target finally exceeds the precision of the codec, the action linear explanatory power on the observed state change rises four point seven times and the value of correct action timing rises thirty nine times, and a sixty four step rollout spans sixteen seconds of real time instead of three point two. Two identical seeds are run because every architectural comparison this week produced an effect smaller than the measured run to run noise of zero point eight decibels, so the noise floor at the new rate must be established alongside the effect.')

P1=$(launch 1 hz4_seedB \
 'Second of two identical four hertz runs. Same configuration as the first seed with no differences at all, since this repository sets no training seed, so two runs differ only by initialisation and data order.' \
 'This arm exists to measure the run to run spread at the new sampling rate rather than to test a different hypothesis. Four underpowered A and B comparisons were run this week whose effects all fell inside the noise, so the floor is now measured first and every later claim is judged against it.')

echo "[launched] hz4_seedA pid=$P0 (GPU 0) | hz4_seedB pid=$P1 (GPU 1)"
echo "[logs] $OUT/hz4_seedA.out | $OUT/hz4_seedB.out"
wait $P0 $P1
