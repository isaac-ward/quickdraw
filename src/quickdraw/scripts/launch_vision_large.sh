#!/usr/bin/env bash
# LARGE-AE vision runs (codec iteration 2, group vis_refactor2): the SAME two matched image world-models as
# vis_refactor (GPU0 = LSAR recon, GPU1 = stock diffusion), with a WIDER, higher-capacity image codec.
# Iteration history (AE-only recon diagnostic, logs/ae_recon_diag/*.png):
#   iter 0  d=32 num_tokens=8  patch=16 -> ~26 dB, hard 16px BLOCKING (bottleneck starved, seams on the grid).
#   iter 1  d=64 num_tokens=16 patch=8  -> blocking GONE, but BLURRY/desaturated (~16 dB): patch 16->8 quadrupled
#           the output patches while tokens only doubled, so the token/patch density HALVED -> bottleneck too thin.
#   iter 2 (THIS)  d=64 num_tokens=32 patch=8 -> raise tokens to RESTORE density (0.125 tok/patch, = iter-0) so the
#           finer grid is fed enough -> aim: smooth (no seams) AND sharp. head_dim stays 8 (d=64). NO convs.
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
GROUP="${GROUP:-vis_refactor2}"       # 2nd codec iteration group (see below + ae_recon_diag / accelerations.md)
TOKENS="${TOKENS:-32}"                # image latent tokens. 16 de-blocked but blurred (bottleneck spread thin
                                      #   across the patch-8 grid); 32 restores token/patch density -> sharp + smooth
PATCH="${PATCH:-8}"                   # image patch (8 = fine grid, no 16px seams; kept from iter 1)
BATCH="${BATCH:-96}"                  # matches the d=32 baseline; but TOKENS=32 is heavier than the 16-token run
                                      #   (~47 GB @ 96) -> RE-PROBE epoch-1 AR memory, drop to 64/48 if it tightens
F="${F:-24}"                          # training rollout window (matches baseline); lower = less AR-epoch memory
DETACH="${DETACH:-16}"                # BPTT truncation depth (gradient credit-assignment window). ~free on memory;
                                      #   higher = learns to correct longer-horizon drift, but deeper BPTT = more
                                      #   gradient-blow-up risk (clipping is on at 1.0). detach<=F.
D="${D:-64}"                          # token/backbone width (model.d; propagates to AE width)
DEPTH="${DEPTH:-4}"                   # backbone depth
AEDEPTH="${AEDEPTH:-4}"               # image AE enc/dec depth (lower = smaller AE = memory for a longer F)
STRIDE="${STRIDE:-1}"                 # TRAIN window stride: >1 drops near-duplicate overlapping windows -> Nx fewer
                                      #   batches/epoch (STRIDE=4 -> ~4x faster, ~no coverage loss). val stays dense.
RECONFRAC="${RECONFRAC:-1.0}"         # fraction (0-1) of F frames to supervise the decode recon on (all heads, random
                                      #   subset/step). 1.0=all; <1 saves ViT-AE decode compute. dynamics loss stays full.
# COMBINED long-horizon + codec run (measured, fits): D=64 DEPTH=4 AEDEPTH=2 TOKENS=32 PATCH=8 F=64 DETACH=32
#   BATCH=48 GROUP=vis_refactor3 -> ~0.60M params, ~68 GB AR (batch 64 = 90 GB, risks eval OOM). No contraction.

: "${RS_PROBLEM:?author + export run_summary fresh, not hardcoded; missing RS_PROBLEM}"
: "${RS_TRIED:?missing RS_TRIED}"; : "${RS_DETAIL:?missing RS_DETAIL}"; : "${RS_RATIONALE:?missing RS_RATIONALE}"

# the three codec changes, shared by both models. image_fpv is modalities index 1 (proprio is index 0);
# model.d propagates to the AE width (see ImageModality(spec, d) -> VisionAEConfig(d=d)).
BIG=( model.d="$D" model.depth="$DEPTH" model.modalities.1.num_tokens="$TOKENS" model.modalities.1.patch="$PATCH"
      model.modalities.1.ae_depth="$AEDEPTH" model.detach_every="$DETACH" model.recon_frac="$RECONFRAC" )
COMMON=( data.root="$DATA" data.batch="$BATCH" data.F="$F" data.window_stride="$STRIDE" logging.group="$GROUP" trainer.max_epochs=100 )
RS=( run_summary.problem="$RS_PROBLEM" run_summary.tried="$RS_TRIED"
     run_summary.trying_detail="$RS_DETAIL" run_summary.rationale="$RS_RATIONALE" )

echo "[vision_large] killing any existing ${GROUP}_ runs (does NOT touch other groups' experiments)..."
docker compose exec -T app pkill -9 -f "experiment=${GROUP}_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3=model-config  $4=trying-env-var-name  $5..=extra overrides
  local gpu="$1" name="$2" model="$3" tvar="$4"; shift 4
  local trying="${!tvar:?missing $tvar the run-specific trying note}"
  echo "[vision_large] launching $name (model=$model, d=$D depth=$DEPTH ae_depth=$AEDEPTH tokens=$TOKENS patch=$PATCH, batch=$BATCH F=$F detach=$DETACH) on GPU $gpu"
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train model="$model" "${COMMON[@]}" "${BIG[@]}" "$@" experiment="$name" "${RS[@]}" \
      run_summary.trying="$trying"
}

# GPU0: LSAR with recon grounding (plain mm_lsar, recon is its default collapse), detach_every=16 to match.
# GPU1: stock diffusion (mm_diffusion; shortcut=false + sampling_steps=6 are the config defaults = stock K=6).
launch 0 "${GROUP}_lsar_recon" mm_lsar      RS_TRYING_lsar
sleep 45
launch 1 "${GROUP}_diffusion"  mm_diffusion RS_TRYING_diff

echo "[vision_large] launched 2 LARGE-AE runs (GPU0 lsar_recon, GPU1 diffusion), 100 epochs, eval every 20."
echo "[vision_large] group: $GROUP  |  F=$F detach=$DETACH tokens=$TOKENS batch=$BATCH  |  RE-PROBE epoch-1 AR memory  |  progress: logs/train_*${GROUP}_*/progress.log"
