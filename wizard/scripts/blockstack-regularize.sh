#!/usr/bin/env bash
# REGULARISATION. The diagnosis changed: this is an OVERFITTING problem, not a capacity problem.
#
#   ./wizard/scripts/blockstack-regularize.sh wd     <value> <gpu>   optim.weight_decay
#   ./wizard/scripts/blockstack-regularize.sh tokens <n>     <gpu>   FEWER tokens (shrink the model)
#   ./wizard/scripts/blockstack-regularize.sh noise  <std>   <gpu>   noise on encoded obs pre-fusion
#
# WHAT THE BASELINE LOSS CURVE SAYS. bs_stride10, train vs val vs ratio:
#     ep1 5.71 / 6.24 / 1.09      ep5 3.03 / 4.50 / 1.48      ep9 2.52 / 4.26 / 1.69
#     ep3 3.78 / 4.98 / 1.32      ep7 2.53 / 4.23 / 1.67
# The gap widens monotonically and val TURNS UP at ep9 (4.2276 -> 4.2575) while train is flat
# (2.5279 -> 2.5179). That is overfitting, on 56 episodes / 2.06 h / about 22k training timesteps
# at subsample 10, against a 12M parameter model.
#
# IT RETRO-EXPLAINS EVERY RESULT. More capacity hurt on BOTH axes, measured at matched gradient
# steps against a batch-8 control:
#     num_tokens 64  ep1 0.1627 (41% worse)   ep3 0.1637 (66% worse)      -- monotone, clean
#     model d 256    ep1 0.1460 (26% worse)   ep5 0.1519 (62% worse)      -- worse post-blowup too
#     batch-8 CONTROL, baseline otherwise     ep1 0.1167 (1.0% worse)     -- so batch 8 is NOT the cause
# A longer attention window also hurt, and hurt MORE the stronger the manipulation (win128 broke
# earliest). Predictions are sharp and confidently wrong rather than blurred. Best-of-8 buys 2.5%
# and shrinks with horizon, so it is not a diversity problem. Every one of those is what an
# overfit dynamics model looks like: it memorises plausible scenes instead of generalising
# transitions, and extra capacity buys more memorisation.
#
# WHY NOT JUST ADD DATA. There is none to add. campaign1-tests is EMPTY (0 episodes, only
# campaign.json) and campaign2-play is 3.1 MINUTES. Campaigns 8-9 are the held-out eval set and
# spending them destroys the measurement. More data means new recording, not reprocessing.
#
# WHAT IS ACTUALLY AVAILABLE. The world model has NO dropout knob at all. optim.weight_decay is
# 1e-4, which is very low for this gap. variations.noise_injection.observations_encoded_pre_fusion
# exists and is OFF (scale 0.0). So: weight decay, shrinking the model, or latent noise.
#
# ON THE NOISE MODE, HONESTLY: df_scale=0.1 (isotropic context noise) already lost by 6-8%. This
# knob injects at a different point (encoded observations, pre-fusion) so it is not the same
# experiment, but the prior is not good. Prefer wd and tokens first.
#
# DECIDER: OL LPIPS on cam_scene at matched GRADIENT STEPS -- use wizard/scripts/olcmp.py, which
# knows that metrics.jsonl's `step` field is the EPOCH INDEX, not the step. Secondary read: does
# val/train stop widening, and does val stop turning up.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a

MODE="${1:?usage: $0 <wd|tokens|noise> <value> <gpu>}"
VAL="${2:?usage: $0 <wd|tokens|noise> <value> <gpu>}"
GPU="${3:?usage: $0 <wd|tokens|noise> <value> <gpu>}"
case "$MODE" in
  wd)     OV=(optim.weight_decay=$VAL);                     NAME="bs_wd${VAL#0.}" ;;
  tokens) OV=(model.modalities.1.num_tokens=$VAL model.modalities.2.num_tokens=$VAL); NAME="bs_tok$VAL" ;;
  noise)  OV=(variations.noise_injection.observations_encoded_pre_fusion.scale=$VAL); NAME="bs_noise${VAL#0.}" ;;
  *) echo "mode must be wd, tokens or noise" >&2; exit 2 ;;
esac
# exe must be python: a bash shell whose own cmdline contains experiment=<name> must never match
for _pid in $(pgrep -f "quickdr[a]w.train_world_model" 2>/dev/null || true); do
  case "$(readlink /proc/$_pid/exe 2>/dev/null)" in *python*) : ;; *) continue ;; esac
  tr '\0' '\n' < "/proc/$_pid/cmdline" 2>/dev/null | grep -qx "experiment=$NAME" && {
    echo "REFUSING: pid $_pid already runs $NAME." >&2; exit 1; }
done
OUT=wizard/scripts/out; mkdir -p "$OUT"

PROBLEM="On block-stack the blocks change colour and merge over an open loop rollout, sharp and \
confidently wrong rather than blurred. Identity is present at the start and lost in transit: a per \
frame encode decode of the true frames keeps colours correct for a whole clip, the rollout matches \
that codec ceiling for the first four seconds, then it compounds away by about eight."
TRIED="Nine hypotheses measured and killed, then capacity on two axes with a control. At matched \
gradient steps num_tokens 64 was 41 and 66 percent worse over two evals, model d 256 was 26, 60 and \
62 percent worse over three, and a batch 8 control with the baseline otherwise unchanged came in 1.0 \
percent off the batch 13 baseline, which eliminates small batch as the explanation. A longer \
attention window also lost, and lost more the stronger the manipulation. Best of eight sampling \
gains 2.5 percent and the gain shrinks with horizon."
TRYING="Regularisation, because the baseline loss curve says this is overfitting rather than a \
capacity or architecture problem."
RATIONALE="On bs_stride10 the ratio of val loss to train loss widens monotonically from 1.09 at \
epoch 1 to 1.69 at epoch 9, and val turns upward at epoch 9 from 4.2276 to 4.2575 while train is \
flat at about 2.52. That is overfitting on 56 episodes, 2.06 hours, roughly 22 thousand training \
timesteps at subsample 10, against a 12 million parameter model. It also retro explains every \
result: an overfit dynamics model memorises plausible scenes instead of generalising transitions, so \
extra capacity buys more memorisation, a longer attention window buys more, predictions come out \
sharp and confidently wrong rather than blurred, and best of k cannot help because the failure is \
not a lack of diversity. Adding data is not an option, since campaign1-tests is empty and \
campaign2-play is three minutes, and campaigns 8 and 9 are the held out eval set."
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE"; do
  case "$v" in *\'*) echo "REFUSING: apostrophe breaks hydra quoting" >&2; exit 1;; esac
done
case "$MODE" in
  wd) DETAIL="optim.weight_decay $VAL against the default 1e-4, a $(.venv/bin/python -c "print(int(float('$VAL')/1e-4))")x increase, \
with every other setting identical to bs_stride10 and to the batch 8 control: d 128, 32 tokens per \
camera, subsample 10, F 64, action_aggregate mean, proprio decode_kind flow, detach_every 32, batch \
pinned to 8 so it is directly comparable to the control at matched gradient steps. Weight decay is \
the canonical response to a 1.69 generalisation gap, it is a single variable, and unlike shrinking \
the model it leaves the codec representational budget untouched, so a change in open loop LPIPS \
cannot be blamed on a worse codec floor." ;;
  tokens) DETAIL="num_tokens $VAL on both image heads, FEWER than the baseline 32. This is the \
mirror image of the capacity sweep: two arms already went up that axis and both got worse, so if \
overfitting is the story then going down it should help. The risk to watch is the codec floor, since \
fewer tokens also means less room to reconstruct a single frame, so read eval_ae_floor before \
reading the rollout." ;;
  noise) DETAIL="variations.noise_injection.observations_encoded_pre_fusion.scale $VAL, up from 0.0, \
with everything else identical to the batch 8 control. This perturbs encoded observations before \
fusion during training so the dynamics sees inputs off its own data manifold, which is the standard \
remedy for compounding rollout error. Note the prior is poor: df_scale 0.1, isotropic context noise, \
already lost by 6 to 8 percent. The injection point differs so it is not the same experiment, but \
treat a null result as expected." ;;
esac
case "$DETAIL" in *\'*) echo "REFUSING: apostrophe in DETAIL" >&2; exit 1;; esac

echo "[launch] $NAME  gpu=$GPU  (${OV[*]})"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -m quickdraw.train_world_model \
    model=vl128_blockstack_2cam environments=recorded \
    environments.obs_dim=17 environments.action_dim=5 \
    data.root=logs/recording_2026_09_10_10_04_45_longhand data.repo_id=block_stack \
    data.subsample=10 data.F=64 data.autobatch=false data.batch=8 data.action_aggregate=mean \
    trainer.check_val_every_n_epoch=2 eval.during_train.every_epochs=2 \
    trainer.checkpoint_monitor=val/metric/cam_scene/mse \
    "${OV[@]}" "experiment=$NAME" \
    "+run_summary.problem='$PROBLEM'" "+run_summary.tried='$TRIED'" \
    "+run_summary.trying='$TRYING'" "+run_summary.trying_detail='$DETAIL'" \
    "+run_summary.rationale='$RATIONALE'" \
    > "$OUT/$NAME.out" 2>&1 &
echo "[launch] pid $!"
