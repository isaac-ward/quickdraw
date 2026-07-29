#!/usr/bin/env bash
# Vision runs: TWO multimodal (proprio + image) runs, ONE per H100, 50 epochs, eval every 10 (skipping 0).
# GPU 0: LSAR with RECON grounding (plain mm_lsar).  GPU 1: diffusion with shortcut (K=1 self-consistency).
# Matched for a fair head-to-head: identical spine/tokens/data/hyperparams; only the next-state head differs
# (MLP-residual latent step vs rectified-flow step). recon = the 06-28 shootout's best collapse strategy;
# physical loss is OFF (it diverged even on recon in those runs). detach_every=16 on both.
# Packing is OFF: autoregressive-rollout epochs (p_tf<1) balloon activation memory and OOM when two share a GPU.
# The token-bag world model (design/models/vision.md): proprio MLP trunk/head + 128^2 ViT-AE image trunk/head,
# fused on the factorized space-time backbone. In-loop eval = the four MM routines (vision rollout, manifold
# UMAP, ood_horizon proprio long-horizon, control MPPI with FPV rendered in the loop).
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

# BATCH / GPU (measured 2026-07-01, 128^2, GPU-resident loader). At batch 96 while p_tf=1 (parallel, epoch 0):
#   LSAR+EMA+phys ~25.6 GB, diffusion ~33 GB. BUT once the p_tf curriculum turns on AUTOREGRESSIVE rollout
#   (epoch 1+), activation memory balloons (F sequential steps of ViT decode/re-encode held for BPTT) and two
#   runs sharing a GPU OOM (killed vis_dsar this way). So ONE run per GPU. AR epochs are ~7-8x slower than the
#   parallel epoch 0 (~15 min/ep at F=24) -> ~2 days for 200 ep, hence 50 epochs. data.F=24 is only the TRAINING
#   window; eval rolls the full long horizon. eval every 10 skipping 0; control un-trimmed (mppi.yaml defaults).
GROUP="${GROUP:-vis_shootout}"    # wandb group: all runs of this sweep grouped in the UI (override with GROUP=..)
COMMON=( data.root="$DATA" data.batch=96 data.F=24 logging.group="$GROUP"
         eval.horizon=256 eval.vision_horizon=256 eval.n_episodes=16 )
RS=( run_summary.problem="$RS_PROBLEM" run_summary.tried="$RS_TRIED"
     run_summary.trying_detail="$RS_DETAIL" run_summary.rationale="$RS_RATIONALE" )

echo "[vision] killing any existing vis_ runs..."
docker compose exec -T app pkill -9 -f "experiment=vis_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3=model-config  $4=trying-env-var-name  $5..=extra overrides
  local gpu="$1" name="$2" model="$3" tvar="$4"; shift 4
  local trying="${!tvar:?missing $tvar the run-specific trying note}"
  echo "[vision] launching $name (model=$model) on GPU $gpu  ${*:+[+ $*]}"
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train_world_model model="$model" "${COMMON[@]}" "$@" experiment="$name" "${RS[@]}" \
      run_summary.trying="$trying"
}

# ONE run per GPU, matched head-to-head.  GPU 0: LSAR with recon grounding (plain mm_lsar).  GPU 1: diffusion
# with shortcut (K=1 sampling via self-consistency).  detach_every=16 on both so the BPTT window matches too.
launch 0 vis_lsar_recon         mm_lsar      RS_TRYING_lsar model.detach_every=16
sleep 45
launch 1 vis_diffusion_shortcut mm_flow RS_TRYING_diff model.diffusion.shortcut=true

echo "[vision] launched 2 vision runs (GPU0: lsar_recon; GPU1: diffusion_shortcut), 50 epochs, eval every 10."
echo "[vision] wandb group: $GROUP   |   watch: nvidia-smi   |   progress: logs/train_*vis_*/progress.log"
