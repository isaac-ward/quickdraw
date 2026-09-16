#!/usr/bin/env bash
# block-stack, first pair: the robocasa record recipe at TWO temporal strides.
#
# Both arms are `vl128_blockstack_2cam` (= the robocasa record-holding `vl128_2cam`, record §24,
# with only the dims / cameras / resolution changed). The ONE variable is `data.subsample`.
#
# WHY STRIDE IS THE FIRST THING TESTED. Record §13: at 20 Hz the robocasa per-step motion was 0.61x
# its codec's own error floor, so predicting zero motion was the CORRECT solution to the objective
# and every early run did exactly that -- 5.8-16x more action gradient changed nothing. Measured on
# block-stack, native 30 Hz gives 0.24x on scene_left and 0.30x on scene_right: WORSE, because these
# scene cameras are bolted down while the robocasa move. Any camera or loss comparison run at the
# wrong rate returns a null result that means nothing. Fix the rate, then use the winner as the
# platform for the 4-camera question (robocasa queue item 2, which block-stack can uniquely answer).
#
# STRIDES 10 AND 15, and not lower: block-stack's own codec floor is unknown until ep0 reports
# `eval_ae_floor`. Across the plausible range (0.045-0.0637) these two are the pair that stay at or
# above 1.0x throughout; stride 6 would be 0.82-0.92x if the floor is robocasa-like. Landing BELOW
# the floor is the degenerate failure; landing above it merely makes the task harder. Err high.
#
# READ ONLY cam_scene's EVAL METRICS. Eval is not multi-head yet (see the model header): it loads
# one camera and feeds those frames to every head, so cam_wrist's eval numbers are scored against
# cam_scene's frames. cam_wrist is here to improve cam_scene's DYNAMICS (§24.2), not to be scored.
# For the wrist head read `train/loss/decode/cam_wrist`.
set -euo pipefail
cd "$(dirname "$0")/../.."

# WANDB_API_KEY and HF_TOKEN live in ~/.env, OUTSIDE the repo. Without this the run prints
# "wandb disabled (No API key configured)" and logs locally only -- which is not noticed until
# you go looking for the run on the dashboard hours later.
set -a; . "$HOME/.env"; set +a

DATA=logs/recording_2026_09_10_10_04_45_longhand
OUT=wizard/scripts/out; mkdir -p "$OUT"

COMMON=(
  model=vl128_blockstack_2cam
  environments=recorded
  environments.obs_dim=17
  environments.action_dim=5
  data.root="$DATA"
  data.repo_id=block_stack
  data.F=64
  data.autobatch=true
  data.action_aggregate=mean            # NOT the default `sum`. block-stack actions are Xbox STICK
  #   POSITIONS, the same class as starling's joy_axis_*, not the EEF DELTAS that summing was
  #   written for. Lag-1 autocorrelation is 0.96-0.99 on every axis, so summing s of them scales
  #   the std by ~s instead of cancelling, and normalization_stats.json is computed on the RAW
  #   stride-1 actions and never sees the aggregation. Measured on this train split, normalized
  #   action |z| std and worst excursion:
  #        stride 10  sum   7.6-9.6     max 30.2        mean  0.76-0.96   max 3.0
  #        stride 15  sum  11.0-14.2    max 45.3        mean  0.74-0.95   max 3.0
  #   Under `sum` the two arms would differ in action INPUT SCALE by ~1.5x, confounding the very
  #   stride comparison they exist to make. `mean` is exactly sum/s, so for these rate-commanded
  #   axes (the loop integrates target += stick * rate * dt) it stays a constant multiple of the
  #   physically correct total displacement, while holding z std ~1 at both strides.
  #   `concat` is the lossless option but makes action_dim 50 vs 75 across the arms -- not for a
  #   bracket whose whole point is one variable.
  trainer.check_val_every_n_epoch=2     # user, 2026-09-10. THIS IS VAL LOSS ONLY.
  eval.during_train.every_epochs=2      # ...and THIS is the eval SUITE (ae_floor, ood_horizon).
  #   They are separate cadences and they were conflated on the first launch: check_val=2 alone left
  #   the suite on its default every_epochs=10 + at_epochs=[5,15], i.e. evals at {5,9,15,19,29,39,49}
  #   -- 7 over a 50-epoch run, and nothing at all for the first five epochs.
  #   The cadence fires on Lightning's (epoch+1)%every phase, so every_epochs=2 -> evals at
  #   {1,3,5,7,...}, coinciding with validation and the checkpoint. 25 evals over the run.
  #   conf/eval/default.yaml warns that per-epoch early points "clogged training"; every-2 is the
  #   user's call (2026-09-10), and the cost is measurable from the first eval -- if it is
  #   disproportionate, trim the ROUTINE SET (control/manifold/denoising_*) before the cadence.
  trainer.checkpoint_monitor=val/metric/cam_scene/mse   # else best.ckpt watches the wrong head
)

PROBLEM="block-stack is a brand new 2.06 hour xArm7 teleop corpus and no world model has been \
trained on it. At its native 30 Hz the per-step image change on scene_right is 0.0193 RMSE which is \
only 0.30x a robocasa-like codec error floor of 0.0637 -- below the floor where record 13 showed \
that predicting zero motion is the correct solution to the objective."
TRIED="On robocasa every early run at 20 Hz sat at 0.61x the floor and hedged to zero motion. More \
action gradient (5.8x to 16x) changed nothing and Fourier action features changed nothing. \
Subsampling to 4-5 Hz was the fix there and every published robot world model that controls long \
rollouts subsamples to 2-5 Hz."
TRYING="Transplant the robocasa record-holding two-camera recipe onto block-stack and vary ONLY the \
temporal stride to find the operating point on this data."
RATIONALE="Any camera or loss comparison run below the codec floor returns a null result that means \
nothing. Fix the rate first then use the winner as the platform for the four-camera question. \
Strides 10 and 15 are the pair that stay at or above 1.0x across the whole plausible floor range \
0.045 to 0.0637 -- stride 6 would be 0.82x to 0.92x if the floor is robocasa-like. The error is \
asymmetric because below the floor is degenerate while above it is merely harder."

launch () {  # $1 = arm name, $2 = subsample, $3 = gpu, $4 = trying_detail
  echo "[launch] $1  subsample=$2  gpu=$3"
  CUDA_VISIBLE_DEVICES=$3 nohup .venv/bin/python -m quickdraw.train_world_model \
      "${COMMON[@]}" data.subsample="$2" "experiment=$1" \
      "+run_summary.problem='$PROBLEM'" \
      "+run_summary.tried='$TRIED'" \
      "+run_summary.trying='$TRYING'" \
      "+run_summary.trying_detail='$4'" \
      "+run_summary.rationale='$RATIONALE'" \
      > "$OUT/$1.out" 2>&1 &
  echo "[launch] $1 pid $!"
}

A="${1:?usage: $0 <strideA> <strideB>   # e.g. 1 5}"
B="${2:?usage: $0 <strideA> <strideB>   # e.g. 1 5}"
if pgrep -f "quickdr[a]w.train_world_model" > /dev/null; then
  echo "REFUSING: training is already running -- both gpus are busy." >&2; exit 1
fi

# Shared detail for the second bracket (strides 1 and 5). The FIRST bracket measured the thing the
# first bracket could not know:
DETAIL="The codec floor measured on this data is 0.0350 rmse from eval_ae_floor on cam_scene at \
epoch 9 of the first bracket, LOWER than the 0.045 to 0.0637 range that bracket was chosen to span. \
So strides 10 and 15 landed at 1.90x and 2.16x, both ABOVE the robocasa productive 1.24 to 1.36x zone \
rather than below it. The lower-SNR arm won: stride 10 beat stride 15 by 9.4 percent on open-loop \
l1_mean at matched epoch 9, which is 6.3 times the settled epoch-to-epoch jitter of 0.00054, while \
stride 15 had been regressing since epoch 5 by plus 3.6 percent on a median-of-halves basis. At the \
measured floor the robocasa-equivalent operating point is stride 5 at 1.36x, and stride 1 at 0.44x \
sits BELOW the floor where record 13 predicts the degenerate zero-motion solution. Together with \
the retained logs of the first bracket this completes a 1 / 5 / 10 / 15 stride ablation. This arm \
is stride"

# A single quote anywhere in a run_summary value breaks the hydra-level quoting the overrides use
# ("+run_summary.x='$VAL'") and fails with a bare grammar error naming one character. Guard it.
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE" "$DETAIL"; do
  case "$v" in *\'*) echo "REFUSING: a run_summary value contains an apostrophe, which breaks hydra quoting" >&2; exit 1;; esac
done

launch "bs_stride$A" "$A" 0 "$DETAIL $A."
sleep 90                     # stagger: both arms build the same frame cache, let one win the race
launch "bs_stride$B" "$B" 1 "$DETAIL $B."
wait
