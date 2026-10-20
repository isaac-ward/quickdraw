#!/usr/bin/env bash
# The DERIVATIVE-LOSS arm at subsample 10, against the retained bs_stride10 as the baseline.
#
#   derivative_weight = 10 on ALL THREE heads (proprio, cam_scene, cam_wrist), and proprio moved from
#   decode_kind=flow to mse so it is eligible for the term at all.
#
# WHY THE TERM. The training objective is sum_t d(x_hat_t, x_t), SEPARABLE over t, so no term's value
# depends on the PAIR (t, t+1) and temporal incoherence is free. The derivative term is the only thing
# in the loss that can tell wrong-but-coherent from wrong-and-incoherent.
#
# THE BASELINE DIAGNOSIS SAYS IT IS AIMED AT THE RIGHT FAILURE. design/derivative_loss.md 0 asks for
# ||dpred||/||dtrue|| against cos(dpred, dtrue) on the baseline before spending a run. Computed
# post-hoc from the retained runs' raw_filmstrip npz:
#     stride10  ep1 -> ep9   ratio 0.253 -> 0.872   cos +0.066 -> +0.053
#     stride15  ep1 -> ep17  ratio 0.197 -> 0.921   cos +0.103 -> +0.074
# Motion of roughly the RIGHT MAGNITUDE in essentially RANDOM DIRECTIONS. That is the doc's
# "incoherent -- this IS flicker -- the target case". The ep1 reading looks like the wrong row purely
# from undertraining, so do not diagnose off an early eval.
#
# THE WEIGHT IS MEASURED, not the doc's figure. The doc's 25%-of-decode guidance came from RANDOM
# 112x192 data (derivative 1.03 vs decode 1.66, ratio 0.62). On REAL block-stack frames, three sample
# offsets, untrained:
#     proprio    decode 0.62-0.85   derivative 0.008-0.014   ratio 0.013-0.017
#     cam_scene  decode 2.31-2.37   derivative 0.048-0.072   ratio 0.021-0.031
#     cam_wrist  decode 2.79-2.81   derivative 0.084-0.124   ratio 0.030-0.044
# Twenty to fifty times smaller than the doc's, because real consecutive frames barely differ -- which
# is the low-motion problem itself. Anyone copying 0.25 would run the term at ~2% of intended strength.
# A flat 10 lands proprio at 13-17% of decode, cam_scene at 21-31%, cam_wrist at 30-44%.
#
# WATCH cam_wrist. It runs hottest at a flat weight AND is the head the doc argues the term suits least
# (a gripper camera's frame difference is dominated by egomotion, not object motion). The term is
# MEAN-SEEKING and the mean of "the block might go left or right" is NO MOTION, so an over-weighted
# term FREEZES the prediction. motion_ratio is the tripwire; if cam_wrist falls below the baseline's
# 0.810 the weight is too high there and 7 is the fallback.
#
# READING IT. Per-head eval metrics ARE valid now -- eval passes image_head_cams(cfg) to the loader, so
# each head is scored against its own camera. (The vl128_2cam header's "read only the first head"
# warning is stale.) Confirmed on bs_stride10 ep9: cam_wrist OL l1 0.1041 vs cam_scene 0.0327, the same
# reconstructs-best / predicts-worst split robocasa 24.2 found.
#
#   DECIDER   eval_ood_horizon/open_loop/cam_scene/lpips at a LONG horizon (@+824 = 275 s, @+1236 =
#             412 s) plus lpips_mean, at matched epochs against bs_stride10. There is no @+128 on this
#             dataset -- horizons derive from episode length.
#   TRIPWIRE  motion_ratio_mean, both heads, against bs_stride10's 0.858 / 0.810.
#   NEVER     derivative/<head> as a result. A mechanism metric improving while the decider does not is
#             exactly what latent_cos did for the depth-4 arm (moved 30% every epoch, won @+128 at 2/7).
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a
# Inspect the ACTUAL training processes rather than pattern-matching every command line. `pgrep -f`
# matches the CALLER's own argv too, so a guard written as pgrep -f "subsample=10" fires on the shell
# that merely mentions the string -- which is exactly how this guard refused its own first launch.
# Guard on the WEIGHT, not the stride. A weight sweep deliberately runs several subsample=10 arms at
# once, so refusing on the stride alone would block exactly the experiment this script exists for.
for _pid in $(pgrep -f "quickdr[a]w.train_world_model" 2>/dev/null || true); do
  if tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null | grep -q "derivative_weight=${1:-10}\b"; then
    echo "REFUSING: pid $_pid is already training derivative_weight=${1:-10}." >&2; exit 1
  fi
done

OUT=wizard/scripts/out; mkdir -p "$OUT"
W="${1:-10}"
GPU="${2:-1}"

PROBLEM="On block-stack the open-loop rollout FLICKERS. Measured on the retained bs_stride10 baseline \
from its own filmstrip frames, the predicted frame-to-frame change reaches 0.87 of the true change in \
MAGNITUDE by epoch 9 while its cosine with the true change stays at 0.05, so the model moves about the \
right amount in essentially random directions. The training objective cannot see this: it is a sum over \
t of a per-frame distance, separable in t, so no term depends on the pair t and t+1 and temporal \
incoherence costs nothing."
TRIED="Four temporal strides were bracketed at 1, 5, 10 and 15. Stride 10 wins on every basis available, \
both epoch-matched at epoch 1 and compute-matched at about 11000 steps, and its codec floor has \
plateaued at 0.0347 rmse since epoch 7 while the rollout is still the weak half. Nothing in the loss so \
far penalises incoherence rather than per-frame error."
TRYING="Add the first-order derivative term at weight 10 on all three heads, holding subsample at 10 and \
everything else identical to bs_stride10, with proprio moved from decode_kind flow to mse so it is \
eligible for the term."
RATIONALE="The derivative term is the only quantity in the loss whose value depends on a PAIR of \
consecutive steps, so it is the only thing that can distinguish wrong-but-coherent from \
wrong-and-incoherent. A stationary object the prediction loses for one frame costs one frame of error \
under the per-frame loss and costs twice under this one, at the vanish and at the reappear. The \
baseline diagnosis the design doc asks for puts this dataset squarely in the flicker regime the term \
targets, and the weight is measured on real frames rather than copied from the doc figure, which came \
from random data and is twenty to fifty times off."
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE"; do
  case "$v" in *\'*) echo "REFUSING: apostrophe in a run_summary value breaks hydra quoting" >&2; exit 1;; esac
done
DETAIL="Weight $W on proprio, cam_scene and cam_wrist alike. Flat rather than per-head tuned: measured \
25 percent targets were 14.7 to 19.9 for proprio, 8.0 to 12.1 for cam_scene and 5.6 to 8.3 for \
cam_wrist, so a flat 10 puts cam_scene dead centre, proprio slightly cool at 13 to 17 percent of decode \
and cam_wrist hottest at 30 to 44 percent. cam_wrist is also the head the design doc argues the term \
suits least because a gripper camera frame difference is dominated by egomotion, so it is the one to \
watch on the motion_ratio tripwire against the baseline value of 0.810."

echo "[launch] bs_deriv_w$W  subsample=10  weight=$W  gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -m quickdraw.train_world_model \
    model=vl128_blockstack_2cam environments=recorded \
    environments.obs_dim=17 environments.action_dim=5 \
    data.root=logs/recording_2026_09_10_10_04_45_longhand data.repo_id=block_stack \
    data.subsample=10 data.F=64 data.autobatch=true data.action_aggregate=mean \
    trainer.check_val_every_n_epoch=2 eval.during_train.every_epochs=2 \
    trainer.checkpoint_monitor=val/metric/cam_scene/mse \
    model.modalities.0.decode_kind=mse \
    +model.modalities.0.derivative_weight=$W \
    +model.modalities.1.derivative_weight=$W \
    +model.modalities.2.derivative_weight=$W \
    "experiment=bs_deriv_w$W" \
    "+run_summary.problem='$PROBLEM'" "+run_summary.tried='$TRIED'" \
    "+run_summary.trying='$TRYING'" "+run_summary.trying_detail='$DETAIL'" \
    "+run_summary.rationale='$RATIONALE'" \
    > "$OUT/bs_deriv_w$W.out" 2>&1 &
echo "[launch] pid $!"
