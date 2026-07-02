#!/usr/bin/env bash
# LARGE-AE vision runs: the SAME two matched image world-models as vis_refactor (GPU0 = LSAR recon, GPU1 =
# stock diffusion), but with a WIDER, higher-capacity image codec. Motivated by the AE-only recon diagnostic
# (logs/ae_recon_diag/*.png): at d=32 / num_tokens=8 / patch=16 the autoencoder ALONE reconstructs FPV frames
# at ~26 dB with visible 16px blocking + low-frequency blur -> the world-model predictions inherit that ceiling.
# These three changes lift the codec ceiling (NO convs -- the linear unpatch is fine, it was starved):
#   d 32 -> 64            wider tokens: fixes the blur; head_dim 4 -> 8 (a friendlier attention size too)
#   num_tokens 8 -> 16    more capacity through the Perceiver bottleneck: fixes the 16px stair-steps
#   patch 16 -> 8         finer patch grid: raises the AE ceiling further. OPTIONAL + the biggest memory cost
#                         (4x more AE patch tokens) -- run with PATCH=16 to drop just this one.
# 100 epochs, eval every 20 (20/40/60/80/100). ONE run per GPU: AR-rollout epochs (p_tf<1) balloon activation
# memory, so two cannot share a GPU.
#
# !!! MEMORY: this codec needs MUCH more memory than the d=32 runs (which fit batch 96). d=64 + 16 tokens +
#     patch 8 is several x. Defaults here are CONSERVATIVE (batch 48, F 16). PROBE epoch-0 memory (nvidia-smi)
#     before trusting a full night; if it OOMs drop BATCH (32) and/or F (12). Override via env: BATCH=.. F=.. PATCH=..
#
# RUN SUMMARY IS NOT HARDCODED -- supply fresh each launch via env vars (train.py rejects duplicates; plain
# words + periods only, Hydra rejects ; , - : = ). Shared: RS_PROBLEM RS_TRIED RS_DETAIL RS_RATIONALE.
# Per-run trying: RS_TRYING_lsar RS_TRYING_diff.
#   Usage: RS_PROBLEM=.. RS_TRIED=.. RS_DETAIL=.. RS_RATIONALE=.. RS_TRYING_lsar=.. RS_TRYING_diff=.. \
#            bash src/quickdraw/scripts/launch_vision_large.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root

DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"
GROUP="${GROUP:-vis_refactor}"        # same wandb group as the d=32 runs so large-vs-small compare directly
PATCH="${PATCH:-8}"                   # image patch (8 = higher ceiling + most memory; set 16 to save memory)
BATCH="${BATCH:-48}"                  # conservative for the larger codec; PROBE + drop if OOM
F="${F:-16}"                          # training rollout window; lower = less AR-epoch memory

: "${RS_PROBLEM:?author + export run_summary fresh, not hardcoded; missing RS_PROBLEM}"
: "${RS_TRIED:?missing RS_TRIED}"; : "${RS_DETAIL:?missing RS_DETAIL}"; : "${RS_RATIONALE:?missing RS_RATIONALE}"

# the three codec changes, shared by both models. image_fpv is modalities index 1 (proprio is index 0);
# model.d propagates to the AE width (see ImageModality(spec, d) -> VisionAEConfig(d=d)).
BIG=( model.d=64 model.modalities.1.num_tokens=16 model.modalities.1.patch="$PATCH" )
COMMON=( data.root="$DATA" data.batch="$BATCH" data.F="$F" logging.group="$GROUP" trainer.max_epochs=100 )
RS=( run_summary.problem="$RS_PROBLEM" run_summary.tried="$RS_TRIED"
     run_summary.trying_detail="$RS_DETAIL" run_summary.rationale="$RS_RATIONALE" )

echo "[vision_large] killing any existing vis_refactor_large_ runs (does NOT touch the vis_refactor d=32 runs)..."
docker compose exec -T app pkill -9 -f "experiment=vis_refactor_large_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3=model-config  $4=trying-env-var-name  $5..=extra overrides
  local gpu="$1" name="$2" model="$3" tvar="$4"; shift 4
  local trying="${!tvar:?missing $tvar the run-specific trying note}"
  echo "[vision_large] launching $name (model=$model, d=64 num_tokens=16 patch=$PATCH, batch=$BATCH F=$F) on GPU $gpu"
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train model="$model" "${COMMON[@]}" "${BIG[@]}" "$@" experiment="$name" "${RS[@]}" \
      run_summary.trying="$trying"
}

# GPU0: LSAR with recon grounding (plain mm_lsar, recon is its default collapse), detach_every=16 to match.
# GPU1: stock diffusion (mm_diffusion; shortcut=false + sampling_steps=6 are the config defaults = stock K=6).
launch 0 vis_refactor_large_lsar_recon mm_lsar      RS_TRYING_lsar model.detach_every=16
sleep 45
launch 1 vis_refactor_large_diffusion  mm_diffusion RS_TRYING_diff

echo "[vision_large] launched 2 LARGE-AE runs (GPU0 lsar_recon, GPU1 diffusion), 100 epochs, eval every 20."
echo "[vision_large] group: $GROUP  |  PROBE epoch-0 memory (nvidia-smi) before trusting overnight  |  progress: logs/train_*vis_refactor_large_*/progress.log"
