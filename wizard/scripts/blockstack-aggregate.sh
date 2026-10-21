#!/usr/bin/env bash
# NOT QUEUED. Run this BY HAND once the stride bracket has reported, passing the winning stride:
#
#     ./wizard/scripts/blockstack-aggregate.sh <subsample>
#
# It deliberately does NOT auto-fire. An earlier version polled for the bracket to exit and then
# launched at a hardcoded subsample=10, which presupposes the answer the bracket exists to give --
# if stride 15 wins, an aggregation test at stride 10 is measuring the wrong operating point, and
# the within-window variance that separates mean from concat is itself stride-dependent (14.7% of
# move_x at stride 10, 22.7% at stride 15). Read the bracket first, then pass the stride.
#
# One `sum` and one `concat` on the two gpus. The ONE variable is data.action_aggregate. The bracket
# already provides the `mean` leg at whichever stride won, completing a three-way sum/mean/concat
# sweep -- as a CROSS-LAUNCH comparison, see the note below.
#
# WHY THIS TEST. block-stack actions are Xbox STICK POSITIONS with lag-1 autocorrelation 0.96-0.99,
# not the EEF deltas that summing was written for, so the stride bracket runs `mean`. But mean keeps
# only the window average: measured on this train split, the WITHIN-WINDOW variance that mean throws
# away and concat keeps is
#        stride 10 (0.33 s)   move_x 14.7%  move_y 12.5%  height 14.7%  yaw 12.9%  gripper 4.0%
#        stride 15 (0.50 s)   move_x 22.7%  move_y 19.4%  height 22.1%  yaw 20.6%  gripper 6.0%
# So mean keeps ~85% of the action signal at stride 10. concat is lossless.
#
# WHAT IT DECIDES. robocasa record §12 measured that the model responds to action DISTRIBUTION and
# not action ORDER -- order-only perturbations were free at +64. Within-window ordering is exactly
# concat's advantage, so if concat wins here, §12 does NOT hold on this dataset and within-window
# timing is a real lever. If it ties, mean is the cheaper representation and §12 generalises. Either
# outcome is worth knowing; a tie is a real result, not a null.
#
# WHAT THE `sum` ARM DOES AND DOES NOT MEASURE, because it is narrower than it looks. sum is
# EXACTLY 10 x mean at subsample 10, so the two carry IDENTICAL information -- this is not a test of
# what the aggregation preserves. It is a test of whether INPUT SCALE alone hurts: normalized action
# |z| std 7.6-9.6 with 30.2 sigma excursions, against mean's 0.76-0.96 and 3.0. There is no clamp in
# the way (`action_fourier_freqs=0` and `action_squash=none` in this recipe, so actions go through a
# plain linear act_enc), so a linear layer could in principle absorb the factor of 10 with 1/10 the
# weights. What is left is optimization: gradient scale, weight decay against the weight magnitude
# the layer now needs, and init scale. Commit ba8ff09 asserted sum is wrong for POSITION-valued
# actions on the strength of the z-std alone; this is the first actual training evidence either way.
#
# The mean leg comes from the bracket's `bs_stride10` rather than being re-run, at the user's
# direction. Note that makes it a CROSS-LAUNCH comparison and this repo sets no training seed
# (record §13), so treat a small sum-vs-mean gap as noise; sum-vs-concat is the matched pair.
#
# action_dim is DERIVED, never hand-set: concat at subsample 10 gives the model 5 x 10 = 50, and
# Normalizer.tile_act repeats the stats to match, so concat has no normalization mismatch (unlike
# sum). Verified before queueing: mean -> 5, concat -> 50.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a

SUB="${1:-}"
if [ -z "$SUB" ]; then
  echo "usage: $0 <subsample>   # the stride the bracket picked, e.g. 10 or 15" >&2
  exit 2
fi
if pgrep -f "quickdr[a]w.train_world_model" > /dev/null; then
  echo "REFUSING: training is already running. Both gpus are busy; stop the bracket first." >&2
  exit 1
fi

OUT=wizard/scripts/out; mkdir -p "$OUT"
DATA=logs/recording_2026_09_10_10_04_45_longhand
echo "[agg] sum vs concat at subsample=$SUB"

COMMON=(
  model=vl128_blockstack_2cam
  environments=recorded environments.obs_dim=17 environments.action_dim=5
  data.root="$DATA" data.repo_id=block_stack
  data.subsample="$SUB" data.F=64 data.autobatch=true
  trainer.check_val_every_n_epoch=2
  eval.during_train.every_epochs=2
  trainer.checkpoint_monitor=val/metric/cam_scene/mse
)

PROBLEM="On block-stack the actions are Xbox stick positions with lag-1 autocorrelation 0.96 to 0.99 \
so the default sum aggregation scales the action input by the subsample stride. mean fixes the scale \
but keeps only the window average and throws away the within-window variance which is 14.7 percent of \
move_x at stride 10. It is not known whether that discarded signal matters."
TRIED="The stride bracket at subsample 10 and 15 both ran mean because sum gave 30 to 45 sigma \
excursions and a 1.5x action scale difference between arms. concat was deliberately not used there \
because it makes action_dim 50 versus 75 across the two arms which would confound the stride test."
TRYING="Hold subsample at 10 and vary ONLY data.action_aggregate between sum and concat, against \
the stride bracket run bs_stride10 which is mean at the same stride, for a three-way sweep."
RATIONALE="robocasa record section 12 measured that the model responds to action distribution and not \
action order and that order-only perturbations were free at plus 64. Within-window ordering is exactly \
what concat adds over mean so this run tests whether that finding holds on block-stack. If concat wins \
then within-window timing is a real lever here. If it ties then mean is the cheaper representation and \
the robocasa finding generalises."

launch () {   # $1 name  $2 aggregate  $3 gpu  $4 trying_detail
  echo "[launch] $1  action_aggregate=$2  gpu=$3"
  CUDA_VISIBLE_DEVICES=$3 nohup .venv/bin/python -m quickdraw.train_world_model \
      "${COMMON[@]}" data.action_aggregate="$2" "experiment=$1" \
      "+run_summary.problem='$PROBLEM'" \
      "+run_summary.tried='$TRIED'" \
      "+run_summary.trying='$TRYING'" \
      "+run_summary.trying_detail='$4'" \
      "+run_summary.rationale='$RATIONALE'" \
      > "$OUT/$1.out" 2>&1 &
  echo "[launch] $1 pid $!"
}

launch "bs_agg_sub${SUB}_sum"    sum    0 "sum at the bracket-winning subsample. The model sees action_dim 5 which is the total of \
the subsampled raw stick samples. Normalized action z std at stride 10 is 7.6 to 9.6 with worst excursion 30.2 sigma because the \
stats are computed on raw stride-1 actions and never see the aggregation. sum is exactly ten times mean \
so it carries identical information and this arm isolates whether input scale alone hurts. There is no \
clamp in the path since action_fourier_freqs is 0 and action_squash is none."
sleep 90
launch "bs_agg_sub${SUB}_concat" concat 1 "concat at the bracket-winning subsample. The model sees action_dim 5 x subsample which is all the \
raw stick samples laid out time-major and derived automatically by effective_action_dim. Normalizer \
tile_act repeats the action stats to match so each slot carries the raw distribution the stats were \
computed on and there is no normalization mismatch. Lossless."
wait
