#!/usr/bin/env bash
# WINDOW SWEEP. How long can the model see back, and does object identity survive exactly that long?
#
#   ./wizard/scripts/blockstack-window.sh <window> <gpu>
#
# THE OBSERVATION (operator, watching rollouts): blocks CHANGE COLOUR and merge over a rollout. They
# do not fade or blur -- the arm tracks well, everything stays sharp, and a block is drawn crisply in
# roughly the right place. It is the wrong block. The failure is object IDENTITY, not permanence.
#
# WHAT IS ALREADY RULED OUT, each measured on a checkpoint (see _oneoff_rollout_diagnosis.py):
#   loss reweighting (LPIPS / DINOv3 / derivative terms)  -- moved little; high doses actively hurt
#   mode collapse            -- 8 draws spread to 2.8x their own step-motion
#   scoring away diversity   -- best-of-8 gains 2.5% at 21 s, and the gain SHRINKS with horizon
#   codec capacity           -- per-frame encode->decode of the TRUE frames keeps blue blue and green
#                               green for the whole clip. The 32-token latent CAN hold identity.
#   identity is low-variance -- FALSE: a recolour moves the latent 2.3x MORE per pixel than a
#                               reposition, so the L2 flow loss is not blind to it
#   the flow never learned it -- FALSE: at steps 0-12 (0-4 s) the rollout matches the codec ceiling,
#                               colours correct. It knows the mapping. IT COMPOUNDS.
#   p_tf_dynamics / df_scale -- the two compounding knobs, run to ep13: both 6-8% WORSE on OL LPIPS
#
# THE HYPOTHESIS THIS TESTS. Temporal attention is a SLIDING WINDOW of `window` steps
# (spacetime.py:6, per token-slot, causal + RoPE). With P=8 context frames, at rollout step t the
# window holds 8 context + t predicted -- so THE LAST TRUE FRAME SCROLLS OUT AT t = window - P.
#
#     window 32 (current)  anchor gone at step 24 =  8 s
#     window 64            anchor gone at step 56 = 19 s
#     window 128           never, within P+F = 72
#
# Observed colour swaps begin at steps 24-32, i.e. 8-10 s. That is where the anchor leaves at
# window=32. No reweighting of any objective can fix a model that has structurally forgotten what the
# scene contained -- which would explain why every loss-side intervention did nothing.
#
# FALSIFIABLE: if the break point MOVES with the window, confirmed. If it stays at ~8 s, the timing
# match was coincidence and the hypothesis is wrong. Read WHEN identity breaks, from filmstrips over
# 3-4 clips -- NOT aggregate LPIPS, which barely moves for a green->red swap on a small block (it
# spatially averages over a frame that is 84.6% static, diluting the moving region 5x).
#
# THE RISK IN window=128, stated before the run. It is larger than the whole 72-step training
# sequence, so the true context is NEVER out of view during training -- the model never practises
# running unanchored, while at eval it rolls 1651 steps and always loses the anchor. That WIDENS the
# train/eval gap. If 128 is good in-horizon and collapses past ~19 s, that is the exposure-bias
# signature, and df_scale=0.1 on top becomes the principled follow-up rather than a guess.
#
# EVERYTHING ELSE IS bs_stride10. In particular proprio stays decode_kind=FLOW (the recipe default) --
# proprio=mse existed only in the derivative arms, where the eligibility guard forced it.
# p_tf_dynamics stays 1.0 and df_scale stays 0.0: one variable.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a

W="${1:?usage: $0 <window> <gpu>}"
GPU="${2:?usage: $0 <window> <gpu>}"
for _pid in $(pgrep -f "quickdr[a]w.train_world_model" 2>/dev/null || true); do
  tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null | grep -q "experiment=bs_win$W\b" && {
    echo "REFUSING: pid $_pid already runs bs_win$W." >&2; exit 1; }
done
OUT=wizard/scripts/out; mkdir -p "$OUT"

PROBLEM="On block-stack the blocks CHANGE COLOUR and merge over an open-loop rollout. They do not \
fade or blur: the arm tracks well, frames stay sharp at 0.90 to 1.01 times the gradient energy of \
real frames, and a block is drawn crisply in roughly the right place. It is the wrong block. The \
failure is object IDENTITY, not permanence, and aggregate LPIPS barely registers it because a green \
to red swap on a small block is diluted fivefold by spatial averaging over a frame that is 84.6 \
percent static."
TRIED="Seven explanations were measured and killed on checkpoints. Loss reweighting between LPIPS, a \
DINOv3 per patch cosine and a first order derivative term moved little and high doses hurt. Mode \
collapse is ruled out, eight draws spread to 2.8 times their own step motion. Scoring away diversity \
is ruled out, best of eight gains 2.5 percent at twenty one seconds and the gain shrinks with \
horizon. Codec capacity is ruled out, a per frame encode decode of the true frames keeps blue blue \
and green green for the whole clip. Identity being a low variance latent direction is ruled out, a \
recolour moves the latent 2.3 times more per pixel than a reposition. The flow never having learned \
identity is ruled out, at zero to four seconds the rollout matches the codec ceiling with correct \
colours. And the two compounding knobs, p_tf_dynamics and df_scale, both ran to epoch thirteen at six \
to eight percent worse on open loop LPIPS."
TRYING="Widen the temporal attention window from 32 so the last true context frame stays in view for \
longer, and read WHEN identity breaks rather than the aggregate."
RATIONALE="Temporal attention is a sliding window of window steps, per token slot, causal with rotary \
embeddings. With eight context frames the last TRUE frame scrolls out of the window at rollout step \
window minus eight. At the current window of 32 that is step 24, which is eight seconds. Observed \
colour swaps begin at steps 24 to 32, which is eight to ten seconds. From that point the model \
attends only to its own predictions and has nothing real to anchor identity against. No reweighting \
of any objective can repair a model that has structurally forgotten what the scene contained, which \
would explain why every loss side intervention did nothing. This is falsifiable: if the break point \
moves with the window it is confirmed, and if it stays at eight seconds the timing match was \
coincidence."
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE"; do
  case "$v" in *\'*) echo "REFUSING: apostrophe breaks hydra quoting" >&2; exit 1;; esac
done
DETAIL="window $W, so the last true frame leaves at rollout step $((W-8)), which is \
$(awk "BEGIN{printf \"%.1f\", ($W-8)*10/30}") seconds. Everything else is identical to bs_stride10, \
including proprio on decode_kind flow, p_tf_dynamics at 1.0 and df_scale at 0.0 -- the window is the \
ONE variable. Autobatch picks the batch and a wider window costs memory, so the batch may drop below \
the baseline 13; that is a confound to report rather than hide."

echo "[launch] bs_win$W  gpu=$GPU  (anchor leaves at step $((W-8)))"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -m quickdraw.train_world_model \
    model=vl128_blockstack_2cam environments=recorded \
    environments.obs_dim=17 environments.action_dim=5 \
    data.root=logs/recording_2026_09_10_10_04_45_longhand data.repo_id=block_stack \
    data.subsample=10 data.F=64 data.autobatch=true data.action_aggregate=mean \
    trainer.check_val_every_n_epoch=2 eval.during_train.every_epochs=2 \
    trainer.checkpoint_monitor=val/metric/cam_scene/mse \
    model.window=$W "experiment=bs_win$W" \
    "+run_summary.problem='$PROBLEM'" "+run_summary.tried='$TRIED'" \
    "+run_summary.trying='$TRYING'" "+run_summary.trying_detail='$DETAIL'" \
    "+run_summary.rationale='$RATIONALE'" \
    > "$OUT/bs_win$W.out" 2>&1 &
echo "[launch] pid $!"
