#!/usr/bin/env bash
# Real training run with the NEW generative flow decode heads (design/models/flow_heads.md P1+P2):
# mm_diffusion with decode_kind=flow on BOTH trunks (proprio + image), shortcut 1-step decode; the
# TransportHead refactor; dynamics unchanged (K=6 rectified flow). Mirrors the proven vis_refactor3
# hyperparams (d=128, depth=4, heads=8, window=32, 100 epochs, evals every 20).
#
# Usage: scripts/launch_flowdec.sh [GPU] [EXPERIMENT]
# Run inside the container:  docker exec quickdraw-app-1 bash -lc 'cd /app && scripts/launch_flowdec.sh 0 flowdec_v1'
set -euo pipefail
GPU="${1:-0}"
EXP="${2:-flowdec_v1}"
DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"

# NOTE: flow decode is ~100x costlier than the old linear MSE decode (a ViT denoiser per frame), so the
# batch=1024 baseline OOMs. batch=128 + recon_frac=0.25 (decode 1/4 of the F frames, unbiased over epochs)
# runs cleanly (verified). Optional noise injection (existing "corrupt-and-hide" site): +variations.noise_injection.std=0.05
CUDA_VISIBLE_DEVICES="$GPU" uv run python -m quickdraw.train_world \
  model=mm_diffusion model.d=128 model.depth=4 model.heads=8 model.window=32 \
  data.batch=128 model.recon_frac=0.25 \
  data.root="$DATA" \
  trainer.max_epochs=100 \
  eval.during_train.every_epochs=20 \
  +run_summary.problem="'MSE image decode is blur-prone; the deterministic readout cannot commit to sharp high-frequency detail'" \
  +run_summary.tried="'deterministic ViT MSE decode as in vis_refactor3 — soft reconstructions'" \
  +run_summary.trying="'generative flow decode heads on BOTH trunks, image ViT and proprio MLP, shortcut 1-step'" \
  +run_summary.trying_detail="'TransportHead refactor in flow.py; decode_kind=flow; dynamics unchanged K=6 flow; d=128 depth=4'" \
  +run_summary.rationale="'a mode-committing generative decoder should sharpen frames vs the L2-mean blur, at about 1x decode cost with K=1'" \
  experiment="$EXP"
