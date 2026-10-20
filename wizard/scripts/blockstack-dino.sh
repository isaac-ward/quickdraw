#!/usr/bin/env bash
# Swap LPIPS for the DINOv3 per-patch cosine, at a given weight.
#
#   ./wizard/scripts/blockstack-dino.sh <weight> <gpu>
#
# IDENTICAL to the retained `bs_stride10` baseline except: visual_lpips 1.0 -> 0, visual_dino_v3
# 0 -> $W, on BOTH image heads. proprio keeps decode_kind=flow and there is no derivative term, so
# the ONE variable against that baseline is which perceptual term is used.
#
# The knob is inert on the proprio head -- VisualLoss lives on ImageModality only (modalities.py:398,
# inside class ImageModality; VectorModality at :328 has none). Verified: setting it there leaves
# decode/proprio at 0.658890 vs 0.658890, exactly.
#
# WHY 2.5 IS THE MATCHED WEIGHT, measured on real frames with the shared l1 term subtracted out:
#     LPIPS w=1 contributes   cam_scene 0.8041   cam_wrist 0.8646
#     DINO  w=1 contributes   cam_scene 0.3545   cam_wrist 0.3101
#     -> to match:                      2.27               2.79
# At weight 1.0 the swap would have run the perceptual term at ~40% of the strength LPIPS had, and
# we would have measured a weakened version of it rather than the thing itself.
#
# WHY THE TERM SHOULD HELP, measured on a real block-stack frame perturbed two ways -- one 16x16
# patch replaced with genuinely different CONTENT from a later frame, versus a global brightness
# shift that moves every pixel while nothing IS different. The local change is 75x SMALLER in pixels:
#     LPIPS  ranks it 0.59x the global one   <- the dilution: LPIPS spatially averages over frames
#                                               that are 84.6% static here, diluting motion 5x
#     DINOv3 ranks it 6.61x                  <- 11.2x better
# The mechanism is the reduction ORDER: cosine taken over the feature axis PER PATCH, then averaged,
# so a dim 4-pixel block counts as much as a bright table texture.
#
# READ: eval_ood_horizon/open_loop/cam_scene/lpips at a LONG horizon (@+824 = 275 s, @+1236 = 412 s)
# plus lpips_mean, at matched epochs against bs_stride10. NOTE lpips is still the METRIC even with
# the LPIPS term removed from the loss -- which makes it a fair, if unflattering, yardstick: the arm
# is being scored on the very quantity it no longer optimises. If DINO wins on LPIPS, that is strong.
# Also watch motion_ratio_mean against the baseline's 0.858 (scene) / 0.810 (wrist).
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . "$HOME/.env"; set +a          # HF_TOKEN: the DINOv3 repo is gated

W="${1:?usage: $0 <dino weight> <gpu> [lpips weight, default 0]}"
GPU="${2:?usage: $0 <dino weight> <gpu> [lpips weight, default 0]}"
LP="${3:-0}"       # LPIPS weight. 0 = swap. >0 = run BOTH, which is what PixelGen actually does.
#
# WHY YOU MAY WANT LPIPS ON. A cosine is EXACTLY scale-invariant -- measured: scaling the true
# change by 0.3, 0.5 or 2.0 all score 1.000. That is why it does not dilute small changes, and it is
# also why it exerts ZERO gradient pressure on output MAGNITUDE. With visual_lpips=0 the only thing
# bounding the unbounded conv decoder is L1 at weight 3. At dino=2.5 L1 still wins and the range
# stays sane (max 1.34, min -1.70). At dino=25 the cosine outvotes it 8:1 and the decoder RAN AWAY:
# max 308, min -98, 58% of pixels outside [0,1], roundtrip_cam_scene_mse 1326 against 0.0037, latent
# dynamics 155 against 0.289 -- dead by epoch 1. A purely scale-invariant perceptual term should not
# be the ONLY perceptual term.
for _pid in $(pgrep -f "quickdr[a]w.train_world_model" 2>/dev/null || true); do
  _cl=$(tr '\0' ' ' < "/proc/$_pid/cmdline" 2>/dev/null || true)
  if echo "$_cl" | grep -q "visual_dino_v3=$W\b" && echo "$_cl" | grep -q "visual_lpips=$LP\b"; then
    echo "REFUSING: pid $_pid already trains dino=$W with lpips=$LP." >&2; exit 1
  fi
done
OUT=wizard/scripts/out; mkdir -p "$OUT"

PROBLEM="On block-stack the open-loop rollout flickers: objects pop in and out over long horizons. \
Measured on the bs_stride10 baseline, the predicted frame-to-frame change reaches 0.87 of the true \
change in magnitude by epoch 9 while its cosine with the true change stays at 0.05, which is \
equivalent to a 6 to 8 pixel displacement on a 128 wide frame. LPIPS is structurally poorly placed \
to see this because it reduces with a spatial mean over frames that are 84.6 percent static, which \
dilutes the moving region fivefold."
TRIED="A first-order derivative term at weights 1 and 10 was tried. At weight 10 it lost on the \
decider, open-loop lpips plus 19.6 percent at epoch 3 and worst at the longest horizon, because the \
mean-seeking term froze the prediction and moved 32 percent less. The mechanism engaged but the cure \
cost more than the disease. Nothing so far changes the fact that the perceptual term itself averages \
away the region we care about."
TRYING="Replace LPIPS with a DINOv3 per-patch cosine on both image heads, holding everything else \
identical to bs_stride10, at two doses."
RATIONALE="The difference is the reduction order. LPIPS takes a squared difference per position and \
channel and averages all of it at once, so magnitude leaks across positions and bright large regions \
dominate. A per-patch cosine normalises within each patch, so a dim four pixel block counts as much \
as a bright table texture. Measured on a real frame perturbed two ways, where the local semantic \
change is 75 times smaller in pixels than a global brightness shift, LPIPS ranks the local change at \
0.59 times the global one while DINOv3 ranks it at 6.61 times, an 11.2 fold swing. DINOv3 rather \
than v2 because its Gram anchoring exists to stop patch to patch similarity structure degrading, \
which is exactly the quantity this loss computes, and it is worth plus 6.7 J and F on video tracking."
for v in "$PROBLEM" "$TRIED" "$TRYING" "$RATIONALE"; do
  case "$v" in *\'*) echo "REFUSING: apostrophe breaks hydra quoting" >&2; exit 1;; esac
done
DETAIL="visual_lpips zeroed and visual_dino_v3 set to $W on cam_scene and cam_wrist alike, ViT-S/16, \
final block, plain mean over the 48 patches. Patch 16 divides 96x128 exactly into a 6 by 8 grid so \
there is no resize and no pad, and it is the same grid as the decoder query grid. The matched weight \
is 2.5, measured with the shared l1 term subtracted: LPIPS at weight 1 contributes 0.8041 on \
cam_scene where DINO at weight 1 contributes 0.3545. This arm is dino weight $W with lpips $LP. A previous arm at dino 25 with lpips 0 DIVERGED by epoch \
1 -- decoder output reached 308 against targets in zero to one, with 58 percent of pixels out of \
range -- because a cosine is exactly scale invariant and therefore exerts no pressure at all on \
output magnitude, leaving only the l1 term at weight 3 to bound an unbounded decoder."

NAME="bs_dino_w${W}$([ "$LP" = "0" ] || echo "_lp$LP")"
echo "[launch] $NAME  gpu=$GPU  (lpips=$LP, dino_v3=$W on both image heads)"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -m quickdraw.train_world_model \
    model=vl128_blockstack_2cam environments=recorded \
    environments.obs_dim=17 environments.action_dim=5 \
    data.root=logs/recording_2026_09_10_10_04_45_longhand data.repo_id=block_stack \
    data.subsample=10 data.F=64 data.autobatch=true data.action_aggregate=mean \
    trainer.check_val_every_n_epoch=2 eval.during_train.every_epochs=2 \
    trainer.checkpoint_monitor=val/metric/cam_scene/mse \
    model.modalities.1.visual_lpips=$LP model.modalities.2.visual_lpips=$LP \
    +model.modalities.1.visual_dino_v3=$W +model.modalities.2.visual_dino_v3=$W \
    "experiment=$NAME" \
    "+run_summary.problem='$PROBLEM'" "+run_summary.tried='$TRIED'" \
    "+run_summary.trying='$TRYING'" "+run_summary.trying_detail='$DETAIL'" \
    "+run_summary.rationale='$RATIONALE'" \
    > "$OUT/$NAME.out" 2>&1 &
echo "[launch] pid $!"
