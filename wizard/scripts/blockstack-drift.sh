#!/usr/bin/env bash
# The transition function has never stepped from its own output. Fix that, two ways.
#
#   ./wizard/scripts/blockstack-drift.sh qdyn <value> <gpu>     structured drift
#   ./wizard/scripts/blockstack-drift.sh df   <value> <gpu>     isotropic noise
#
# THE FAILURE, measured on bs_stride10 ep9 rather than assumed:
#   * an UNTOUCHED, STATIONARY object disappears over a rollout (operator, watching the videos). That
#     rules out every uncertainty story -- there is no ambiguity about a block that is not moving.
#   * the rolled latent decorrelates FAST: distance from the true encoded latent, relative to the
#     latent's own norm, 0.58 at step 0 -> 1.22 by 5 s -> 1.31 by 21 s.
#   * and yet the frames stay SHARP: gradient energy 0.90-1.01x that of real frames at every horizon,
#     and encode->decode of a real frame is 0.95x. Nothing is blurring. The model renders a confidently
#     WRONG scene, and an untouched block is one of the things that scene no longer contains.
#
# FIVE HYPOTHESES ARE DEAD, all measured, none of them this:
#   loss reweighting (LPIPS vs DINOv3, derivative terms)  -- moved little, and w10/w25 actively hurt
#   mode collapse in the flow                             -- 8 draws spread to 2.8x their own step motion
#   scoring away real diversity                           -- best-of-8 gains only 2.5% at 21 s, and the
#                                                            gain SHRINKS with horizon, the opposite of
#                                                            what multimodality predicts
#   codec capacity                                        -- recon error is 0.29x the patch's own
#                                                            variation on dynamic patches vs 2.48x on
#                                                            still ones: the codec holds moving content
#                                                            PROPORTIONALLY BETTER. It is not dropping
#                                                            the blocks.
#   off-manifold decode                                   -- see the sharpness numbers above
#
# WHAT THE CODE ACTUALLY DOES, read rather than taken from the comments (multimodal.py:1029-1066):
#   q_dyn = p_tf if p_tf_dynamics is None else p_tf_dynamics      <- q_dyn is a LOCAL, not a knob
#   q_dyn == 1.0 (ours):  the substitution block is SKIPPED. s = z[:,:-1] (clean encoder latents),
#                         target = (z[:,1:] - s_ref) = the true clean->clean residual. The flow is
#                         trained on exactly ONE mapping: clean latent -> true next residual.
#   q_dyn <  1.0:         s_ref becomes the FEED -- what the rollout actually stood on -- so
#                         target = z[:,1:] - (drifted latent). THE TARGET BECOMES A CORRECTION, NOT A
#                         STEP: from where you are, produce the jump that lands on the truth.
#
# WHY THAT MATCHES THE SYMPTOM. A vanished static object means the rolled latent lost the "object is
# here" component. Under q_dyn=1 the flow has only ever seen latents where that component is intact
# and learned present -> present; handed a degraded latent it has NO learned behaviour for restoring
# anything, so the loss persists and compounds. Nothing in the objective is a contraction toward the
# data manifold. q_dyn<1 is the only change here that trains restoration.
#
# WHY IT MIGHT BLOW UP, and why the dose is 0.8. `feeds` is ATTACHED on purpose (multimodal.py:633),
# so the flow-loss gradient runs back through the rollout, and each rollout step runs sampling_steps=6
# Euler passes through the same net. At detach_every=8 that is 8 x 6 = 48 sequential applications of
# one network in the gradient path; a product of 48 Jacobians explodes or vanishes. That alone
# explains the recorded grad/norm/flow 0.42 -> 1.3e7 at full substitution. At q_dyn=0.8 only 20% of
# positions carry that path, so the contribution is diluted AT SOURCE.
# detach_every is deliberately NOT touched: it is SHARED with the decode loss -- the only
# autoregressive gradient in the model -- so shortening it would trade away the compounding training
# we are trying to buy.
#
# WHY THE ROBOCASA NEGATIVE DOES NOT TRANSFER -- checked before relaunching, after killing these arms
# once on it. Record section 21.1 ran a five-dose p_tf_dynamics curve, every dose lost, gradient norm
# monotone to 3.33e13 and inf. But that curve ran on the `dyn512` base: flow_hidden=512, PURE L2.
# Record section 23 is titled "THE COLLAPSE WAS flow_hidden=512" and finds that EVERY L1+LPIPS run
# which destroyed itself had flow_hidden=512; st_fh128 was byte-identical to vl_keep10 except that one
# number and went from dying at ev12 to still setting records at ev15. Their own conclusion: "under
# PURE MSE it won by a hair. It is only under the perceptual loss that it becomes fatal. Capacity that
# is harmless with one loss is lethal with another."
#   ours: flow_hidden=128 (the fix) AND L1+LPIPS (the combination where 512 was fatal).
# So the dose curve measured substitution on top of a base with a latent instability, under a loss
# that did not expose it. Those gradient blow-ups are at least as plausibly flow_hidden=512
# detonating EARLIER under substitution as they are proof that substitution is unstable per se.
# AND THE DATA IS DIFFERENT IN KIND: robocasa is a simulator replaying scripted demos -- deterministic
# by construction. That difference already invalidated one transferred finding today (the "flow learned
# a near-deterministic map" result, which measured 2.8x sample spread when re-run here).
#
# THE TWO ARMS TEST A DISAGREEMENT.
#   qdyn 0.8  trains on STRUCTURED drift -- the real error directions the dynamics produces.
#   df   0.1  replaces the context with (1-l)*s + l*eps, ISOTROPIC Gaussian at a random level, and
#             tells the backbone the level. No recurrent gradient at all, so it CANNOT explode -- but
#             real drift is not isotropic, so it trains robustness to the wrong error distribution.
# conf/model/mm_flow.yaml recommends df. My reading of the code says qdyn should win. If df wins, the
# mechanistic argument above is wrong, which is worth knowing.
#
# READ: the drift curve (scratchpad/collapse_probe.py -- relative latent distance at 0/5/10/21 s) is
# the quantity this targets directly; latent_cos@+32; and open-loop lpips at long horizon as the
# decider. TRIPWIRE: grad/norm/flow. The blow-up is loud, not silent.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a

MODE="${1:?usage: $0 <qdyn|df> <value> <gpu>}"
VAL="${2:?usage: $0 <qdyn|df> <value> <gpu>}"
GPU="${3:?usage: $0 <qdyn|df> <value> <gpu>}"
case "$MODE" in
  qdyn) OV=(model.p_tf_dynamics=$VAL);  NAME="bs_qdyn$VAL" ;;
  df)   OV=(+model.df_scale=$VAL);      NAME="bs_df$VAL" ;;
  *) echo "mode must be qdyn or df" >&2; exit 2 ;;
esac
for _pid in $(pgrep -f "quickdr[a]w.train_world_model" 2>/dev/null || true); do
  tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null | grep -q "experiment=$NAME\b" && {
    echo "REFUSING: pid $_pid already runs $NAME." >&2; exit 1; }
done
OUT=wizard/scripts/out; mkdir -p "$OUT"

PROBLEM="On block-stack an UNTOUCHED STATIONARY object disappears over an open-loop rollout. That \
rules out every uncertainty explanation, because there is no ambiguity about a block that is not \
moving. Measured on the bs_stride10 checkpoint at epoch 9: the rolled latent distance from the true \
encoded latent, relative to the latent own norm, goes 0.58 at step zero to 1.22 by five seconds to \
1.31 by twenty one seconds, while the decoded frames stay SHARP at 0.90 to 1.01 times the gradient \
energy of real frames. Nothing is blurring. The model renders a confidently wrong scene and the \
untouched block is one of the things that scene no longer contains."
TRIED="Five hypotheses were tested and killed. Loss reweighting between LPIPS and a DINOv3 per patch \
cosine moved little and the high dose arms actively hurt. Mode collapse was ruled out: eight draws \
from one context spread to 2.8 times their own step motion. Scoring away real diversity was ruled \
out: best of eight gains only 2.5 percent at twenty one seconds and the gain SHRINKS with horizon, \
the opposite of what multimodality predicts. Codec capacity was ruled out: reconstruction error is \
0.29 times the patch own variation on dynamic patches against 2.48 on still ones, so the codec holds \
moving content proportionally better. Off manifold decode was ruled out by the sharpness numbers."
TRYING="Train the transition function on contexts it produced itself, which it has never seen."
RATIONALE="Reading multimodal.py lines 1029 to 1066 rather than the comments: with p_tf_dynamics at \
1.0 the substitution block is skipped entirely, so the flow is trained on exactly one mapping, clean \
latent to true next residual. Below 1.0 the reference becomes the feed the rollout actually stood on, \
and the target becomes z next minus the drifted latent -- a CORRECTION rather than a STEP. A vanished \
static object means the rolled latent lost the object is here component, and under the current \
setting the flow has no learned behaviour for restoring anything because it has only ever seen \
latents where that component was intact. Nothing in the present objective is a contraction toward the \
data manifold."
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE"; do
  case "$v" in *\'*) echo "REFUSING: apostrophe breaks hydra quoting" >&2; exit 1;; esac
done
if [ "$MODE" = "qdyn" ]; then
DETAIL="p_tf_dynamics set to $VAL, so $VAL of positions keep the clean latent and the rest condition \
on the rollout own feed. detach_every deliberately UNCHANGED at 8: it is shared with the decode loss, \
the only autoregressive gradient in the model, so shortening it would trade away the compounding \
training this run exists to buy. The dose is below one because feeds are attached on purpose, so the \
flow loss gradient runs back through the rollout and each step runs six Euler passes through the same \
net -- at detach_every 8 that is 48 sequential applications in the gradient path, which is a \
sufficient explanation for the recorded grad norm flow going from 0.42 to 1.3e7 under full \
substitution. At 0.8 only a fifth of positions carry that path."
else
DETAIL="df_scale set to $VAL. This replaces the context with one minus lambda times s plus lambda \
times eps, isotropic Gaussian noise at a random level per sample and position, and tells the backbone \
the level. There is no recurrent gradient path at all so it CANNOT explode, which is why the model \
config recommends it. The counter argument this arm tests is that real drift is not isotropic -- it \
lies along the directions the dynamics actually errs in -- so this trains robustness to the wrong \
error distribution and should underperform the structured version."
fi

echo "[launch] $NAME  gpu=$GPU  (${OV[*]})"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -m quickdraw.train_world_model \
    model=vl128_blockstack_2cam environments=recorded \
    environments.obs_dim=17 environments.action_dim=5 \
    data.root=logs/recording_2026_09_10_10_04_45_longhand data.repo_id=block_stack \
    data.subsample=10 data.F=64 data.autobatch=true data.action_aggregate=mean \
    trainer.check_val_every_n_epoch=2 eval.during_train.every_epochs=2 \
    trainer.checkpoint_monitor=val/metric/cam_scene/mse \
    "${OV[@]}" "experiment=$NAME" \
    "+run_summary.problem='$PROBLEM'" "+run_summary.tried='$TRIED'" \
    "+run_summary.trying='$TRYING'" "+run_summary.trying_detail='$DETAIL'" \
    "+run_summary.rationale='$RATIONALE'" \
    > "$OUT/$NAME.out" 2>&1 &
echo "[launch] pid $!"
