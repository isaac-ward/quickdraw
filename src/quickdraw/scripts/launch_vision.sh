#!/usr/bin/env bash
# Vision shakedown: ONE multimodal (proprio + image_fpv) run per GPU, ONE epoch, then evaluate — to see
# what a single vision run costs (GPU mem + util + per-epoch time) before packing a full sweep. The token-
# bag world model (design/models/vision.md): proprio MLP trunk/head + 128^2 ViT-AE image trunk/head, fused
# on the factorized space-time backbone. In-loop eval = the four MM routines (vision rollout, manifold UMAP,
# ood_horizon proprio long-horizon, control MPPI with FPV rendered in the loop).
# ============================================================================================
# !!! DO NOT set TORCHDYNAMO_DISABLE=1. (MM models skip torch.compile anyway — the per-batch image gather +
#     ViT AE aren't compiled; FlexAttention still runs eager. See train.py.)
# ============================================================================================
#
# RUN SUMMARY IS NOT HARDCODED — supply it FRESH each launch via env vars (train.py rejects duplicates).
# Plain words + periods only (Hydra rejects ; , - : = ). Shared: RS_PROBLEM RS_TRIED RS_DETAIL RS_RATIONALE.
# Per-run trying: RS_TRYING_lsar RS_TRYING_diff.
#
# Usage: RS_PROBLEM=.. RS_TRIED=.. RS_DETAIL=.. RS_RATIONALE=.. RS_TRYING_lsar=.. RS_TRYING_diff=.. \
#          bash src/quickdraw/scripts/launch_vision.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root

DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"

: "${RS_PROBLEM:?author + export the run_summary fresh, not hardcoded; missing RS_PROBLEM}"
: "${RS_TRIED:?missing RS_TRIED}"; : "${RS_DETAIL:?missing RS_DETAIL}"; : "${RS_RATIONALE:?missing RS_RATIONALE}"

# BATCH / GPU-FILL: image windows can't be GPU-resident (batch*L*128^2*3), so batch is small vs the vector
# stage. Start at 16 (safe on 80GB H100) and RAISE toward memory once we measure fill (that's the point of
# this shakedown). data.F=24 keeps the image window (and per-step ViT encode count) bounded. One epoch of
# the train split + all four MM eval routines at the end (every_epochs=1). Control params are trimmed so the
# FPV-in-loop MPPI doesn't dominate the eval on this shakedown; raise for the real sweep.
COMMON=( data.root="$DATA" data.batch=16 data.F=24
         trainer.max_epochs=1 eval.during_train.every_epochs=1 eval.during_train.at_epochs=null
         eval.horizon=256 eval.n_episodes=16
         control.n_episodes=4 control.max_steps=64 control.num_samples=128 control.horizon=16 )
RS=( run_summary.problem="$RS_PROBLEM" run_summary.tried="$RS_TRIED"
     run_summary.trying_detail="$RS_DETAIL" run_summary.rationale="$RS_RATIONALE" )

echo "[vision] killing any existing vis_ runs..."
docker compose exec -T app pkill -9 -f "experiment=vis_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3=model-config  $4=trying-env-var-name
  local gpu="$1" name="$2" model="$3" tvar="$4"
  local trying="${!tvar:?missing $tvar the run-specific trying note}"
  echo "[vision] launching $name (model=$model) on GPU $gpu"
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train model="$model" "${COMMON[@]}" experiment="$name" "${RS[@]}" \
      run_summary.trying="$trying"
}

# ONE run per GPU: latent-space AR (GPU0) + latent diffusion (GPU1), both proprio + image_fpv.
launch 0 vis_lsar_image mm_lsar      RS_TRYING_lsar
sleep 45
launch 1 vis_diff_image mm_diffusion RS_TRYING_diff

echo "[vision] launched 2 vision runs (mm_lsar on GPU0, mm_diffusion on GPU1), 1 epoch + eval each."
echo "[vision] watch GPU fill:  watch -n2 nvidia-smi   |  progress: logs/train_*vis_*/progress.log"
