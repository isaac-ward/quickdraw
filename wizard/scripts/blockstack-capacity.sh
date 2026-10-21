#!/usr/bin/env bash
# CAPACITY. The one dimension never varied, and the failure looks like a capacity failure in transit.
#
#   ./wizard/scripts/blockstack-capacity.sh tokens <n> <gpu>    more tokens in the bag
#   ./wizard/scripts/blockstack-capacity.sh width  <d> <gpu>    wider tokens
#
# THE FAILURE, measured over two days (record 8.15-8.19): blocks CHANGE COLOUR and merge over a
# rollout. Not blur, not vanishing -- sharp, confident, wrong. The model knows a block is there and
# not which block. Decisively:
#   * the CODEC can hold identity   -- per-frame encode->decode of the TRUE frames keeps blue blue
#                                      and green green for the whole clip
#   * the FLOW has learned identity -- at 0-4 s the rollout matches that codec ceiling exactly
#   * it COMPOUNDS away by ~8 s
# So identity is present at the start and lost IN TRANSIT.
#
# NINE HYPOTHESES DEAD, none of them capacity: loss reweighting (LPIPS / DINOv3 / derivative), mode
# collapse (8 draws spread 2.8x), scoring away diversity (best-of-8 gains 2.5% and SHRINKS with
# horizon), codec capacity (recon error 0.29x the patch's own variation on dynamic patches vs 2.48x
# on still), off-manifold decode (rolled frames 0.90-1.01x as sharp as real), identity being a
# low-variance latent direction (a recolour moves the latent 2.3x MORE per pixel than a reposition),
# the flow never having learned it, p_tf_dynamics, df_scale, and the attention window -- the last
# falsified twice, with the STRONGER manipulation (window 128, anchor never leaves) breaking EARLIEST.
#
# WHY CAPACITY, AND WHY NOW. Nothing in this architecture binds one token to one object: the bag is
# undifferentiated, so "which block is which" must be carried diffusely across all of it, and
# re-derived at every one of ~30 rollout steps. We run 32 tokens per camera for an arm, a table and
# several individually-identifiable blocks. It also predicts the right SHAPE: the codec is fine
# (all 32 tokens serve one frame) while the rollout degrades (each step re-derives the scene from a
# bag at its limit).
#
# AND IT IS A DOCUMENTED OPEN QUESTION, not a new idea. Record 21.2: num_tokens 32->64 "was killed
# after 2 evals. At matched eval index 1 it had the BEST codec floor of any run on this dataset and
# the second-best @+128." It was stopped for COST, not evidence, and the record closes: "If
# num_tokens=64 is retried, it is an OPEN question, not a closed one. Log it that way."
#
# TWO AXES, because they are different claims. `tokens` gives the bag more SLOTS -- more places for
# distinct objects to live. `width` gives each slot more DIMENSIONS -- more that each slot can say.
# If only one helps, that tells us which kind of room was missing.
#
# WATCH THE BATCH. The window sweep cut it 13 -> 8 -> 4 and the batch-4 arm was uninterpretable.
# num_tokens doubles the bag (n_state 65 -> 129), so spatial attention roughly quadruples; record
# 21.2 saw 2,339 batches/epoch against 1,254. Report batch and steps up front, never discover the
# confound afterwards.
#
# DECIDER: OL LPIPS on cam_scene at matched GRADIENT STEPS (not epochs), plus the colour-swap onset
# read off filmstrips. Everything else is identical to bs_stride10, including proprio decode_kind=flow.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a

MODE="${1:?usage: $0 <tokens|width> <value> <gpu>}"
VAL="${2:?usage: $0 <tokens|width> <value> <gpu>}"
GPU="${3:?usage: $0 <tokens|width> <value> <gpu>}"
case "$MODE" in
  tokens) OV=(model.modalities.1.num_tokens=$VAL model.modalities.2.num_tokens=$VAL); NAME="bs_tok$VAL" ;;
  width)  OV=(model.d=$VAL);                                                          NAME="bs_d$VAL" ;;
  *) echo "mode must be tokens or width" >&2; exit 2 ;;
esac
for _pid in $(pgrep -f "quickdr[a]w.train_world_model" 2>/dev/null || true); do
  tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null | grep -q "experiment=$NAME\b" && {
    echo "REFUSING: pid $_pid already runs $NAME." >&2; exit 1; }
done
OUT=wizard/scripts/out; mkdir -p "$OUT"

PROBLEM="On block-stack the blocks CHANGE COLOUR and merge over an open-loop rollout -- sharp, \
confident and wrong rather than blurred or faded. The model knows a block is there and not which \
block. Identity is demonstrably present at the start and lost in transit: a per frame encode decode \
of the true frames keeps colours correct for a whole clip, and at zero to four seconds the rollout \
matches that codec ceiling exactly, then it compounds away by about eight seconds."
TRIED="Nine hypotheses measured and killed on checkpoints. Loss reweighting between LPIPS, a DINOv3 \
per patch cosine and a first order derivative term. Mode collapse, ruled out because eight draws \
spread to 2.8 times their own step motion. Scoring away diversity, ruled out because best of eight \
gains 2.5 percent at twenty one seconds and the gain shrinks with horizon. Codec capacity, ruled out \
because reconstruction error is 0.29 times the patch own variation on dynamic patches against 2.48 \
on still ones. Off manifold decode, ruled out because rolled frames are 0.90 to 1.01 times as sharp \
as real. Identity being a low variance latent direction, ruled out because a recolour moves the \
latent 2.3 times more per pixel than a reposition. The flow never having learned identity, ruled out \
by the zero to four second match. And p_tf_dynamics, df_scale and the attention window all ran as \
real arms and all lost, the window falsified twice with the stronger manipulation breaking earliest."
TRYING="Give the token bag more room, on two different axes, and see whether identity survives the \
transit."
RATIONALE="Nothing in this architecture binds one token to one object. The bag is undifferentiated, \
so which block is which must be carried diffusely across all of it and re derived at every one of \
about thirty rollout steps, from 32 tokens per camera covering an arm, a table and several \
individually identifiable blocks. That predicts the exact shape observed: the codec is fine because \
all its tokens serve a single frame, while the rollout degrades because each step re derives the \
scene from a bag at its limit. Capacity is also the one dimension never varied here, and it is a \
documented open question rather than a new idea: record section 21.2 records that num_tokens 64 was \
killed after two evals with the best codec floor of any run on that dataset, stopped for cost and \
not for evidence, with the explicit note that retrying it is an open question."
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE"; do
  case "$v" in *\'*) echo "REFUSING: apostrophe breaks hydra quoting" >&2; exit 1;; esac
done
if [ "$MODE" = "tokens" ]; then
DETAIL="num_tokens $VAL on both image heads, against the baseline 32. This gives the bag more SLOTS, \
so distinct objects have distinct places to live. It doubles n_state from 65 to about 129 so spatial \
attention roughly quadruples; record 21.2 saw 2339 batches per epoch against 1254 on a comparable \
change, so expect the batch to fall from the baseline 13 and read results at matched gradient steps \
rather than matched epochs."
else
DETAIL="model d $VAL, against the baseline 128. This gives each token more DIMENSIONS rather than \
adding tokens, so it tests whether the missing room was in how much each slot can say rather than in \
how many slots there are. If tokens helps and width does not, or the reverse, that identifies which \
kind of capacity was binding."
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
