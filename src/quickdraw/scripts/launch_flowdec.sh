#!/usr/bin/env bash
# Real training run with the generative flow DECODE heads (design/models/flow_heads.md P1+P2): mm_flow with
# decode_kind=flow on BOTH trunks (proprio + image), decode_shortcut=true (K=1). TransportHead refactor;
# dynamics unchanged (K=6 rectified flow, shortcut off). Mirrors vis_refactor3 (d=128, depth=4, heads=8,
# window=32, 100 epochs, evals every 20).
#
# NOTE: flow decode is ~2-4x costlier than the old linear-MSE decode (a ViT denoiser per frame), so the
# batch=1024 baseline OOMs. batch=128 + recon_frac=0.25 (decode 1/4 of the F frames, unbiased over epochs)
# runs cleanly (verified). Optional noise injection (existing additive site): +variations.noise_injection.std=0.05
#
# Run INSIDE the container:  docker exec quickdraw-app-1 bash -lc 'cd /app && CUDA_VISIBLE_DEVICES=0 bash src/quickdraw/scripts/launch_flowdec.sh flowdec_v1'
set -euo pipefail
EXP="${1:-flowdec_v1}"
DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"

uv run python -m quickdraw.train_world \
  model=mm_flow model.d=128 model.depth=4 model.heads=8 model.window=32 \
  data.batch=128 model.recon_frac=0.25 \
  data.root="$DATA" \
  trainer.max_epochs=100 eval.during_train.every_epochs=20 \
  +run_summary.problem="MSE_image_decode_blur_prone_deterministic_readout_cannot_commit_to_sharp_detail" \
  +run_summary.tried="deterministic_vit_mse_decode_vis_refactor3_soft_reconstructions" \
  +run_summary.trying="generative_flow_decode_heads_both_trunks_shortcut_1step" \
  +run_summary.trying_detail="transporthead_refactor_decode_kind_flow_b128_reconfrac025_d128" \
  +run_summary.rationale="mode_committing_generative_decoder_should_sharpen_frames_vs_L2_mean_blur_at_1x_cost" \
  experiment="$EXP"
