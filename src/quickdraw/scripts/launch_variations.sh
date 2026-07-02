#!/usr/bin/env bash
# Variation campaign: ONE model (recon LSAR) x { baseline + physical + 2 noise scales + 2 contraction
# targets } = 6 runs, 3 per GPU, staggered (design/models/variations.md). Compares each train-time
# variation against a fresh baseline on long-horizon OOD + control.
# ============================================================================================
# !!! DO NOT disable compile / DO NOT set TORCHDYNAMO_DISABLE=1 — FlexAttention REQUIRES torch.compile
# to build its kernel. The contraction variation runs its penalty on a SEPARATE eager sdpa(MATH) path;
# the main rollout/eval stay on compiled FlexAttention. See launch_shootout.sh banner for the full why.
# ============================================================================================
#
# RUN SUMMARY IS NOT HARDCODED HERE. The operator (the LLM driving the session) supplies the note FRESH
# each launch via env vars, authored to reflect the CURRENT dev cycle (a baked-in note goes stale and
# train.py rejects duplicates). Plain words + periods only (Hydra rejects ; , - : = etc.). The shared
# fields come from RS_PROBLEM/RS_TRIED/RS_DETAIL/RS_RATIONALE; the per-run `trying` (what THIS run tests)
# comes from RS_TRYING_<short> (e.g. RS_TRYING_physical), so every run's summary is distinct + run-specific.
#
# Usage: RS_PROBLEM=.. RS_TRIED=.. RS_DETAIL=.. RS_RATIONALE=.. RS_TRYING_baseline=.. RS_TRYING_physical=.. \
#          RS_TRYING_noise10=.. RS_TRYING_noise30=.. RS_TRYING_contract_soft=.. RS_TRYING_contract_hard=.. \
#          bash src/quickdraw/scripts/launch_variations.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root (this script lives in src/quickdraw/scripts/)

DATA="logs/data_generation_2026_06_27_04_49_59_regen_dyn_v8"

: "${RS_PROBLEM:?author + export the run_summary fresh, not hardcoded; missing RS_PROBLEM}"
: "${RS_TRIED:?missing RS_TRIED}"; : "${RS_DETAIL:?missing RS_DETAIL}"; : "${RS_RATIONALE:?missing RS_RATIONALE}"

# fixed across all 6: recon LSAR, full BPTT, batch 256, v8 data. Per-run variation overrides are passed
# to launch(). All variations default OFF, so var_recon_baseline is a clean (unmodified) recon run.
COMMON=( data.root="$DATA" data.batch=256 model.detach_every=16   # truncated BPTT (16-step chunks):
         # ~4x faster than full BPTT (detach_every=0); near-Markov torus loses little long-range credit.
         model=mm_lsar_proprio +collapse=reconstruction )
RS=( run_summary.problem="$RS_PROBLEM" run_summary.tried="$RS_TRIED"
     run_summary.trying_detail="$RS_DETAIL" run_summary.rationale="$RS_RATIONALE" )

echo "[variations] killing any existing var_ runs..."
docker compose exec -T app pkill -9 -f "experiment=var_" 2>/dev/null || true
sleep 4

launch () {  # $1=gpu  $2=experiment-name  $3..=variation overrides
  local gpu="$1" name="$2"; shift 2
  local short="${name#var_recon_}" tvar
  tvar="RS_TRYING_${short}"                       # per-run `trying` note (operator-supplied, run-specific)
  local trying="${!tvar:?missing $tvar the run-specific trying note}"
  echo "[variations] launching $name on GPU $gpu"
  # isolated compile caches per run (concurrent runs sharing the default cache contend + stall) +
  # in-process inductor compile (fork-after-CUDA deadlock otherwise). See launch_shootout.sh.
  docker compose exec -T -d -e CUDA_VISIBLE_DEVICES="$gpu" -e TORCHINDUCTOR_COMPILE_THREADS=1 \
    -e TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_$name" -e TRITON_CACHE_DIR="/tmp/triton_$name" app \
    uv run python -m quickdraw.train "${COMMON[@]}" "$@" experiment="$name" "${RS[@]}" \
      run_summary.trying="$trying"
}

STAGGER=45
launch 0 var_recon_baseline;                                                                                sleep $STAGGER
launch 1 var_recon_physical       variations.physical_loss.weight=0.3 variations.physical_loss.continuity=0.3; sleep $STAGGER
launch 0 var_recon_noise10        variations.noise_injection.std=0.10;                                       sleep $STAGGER
launch 1 var_recon_noise30        variations.noise_injection.std=0.30;                                       sleep $STAGGER
launch 0 var_recon_contract_soft  variations.contraction.weight=1.0 variations.contraction.target=1.05 variations.contraction.n_sample_steps=2; sleep $STAGGER
launch 1 var_recon_contract_hard  variations.contraction.weight=1.0 variations.contraction.target=1.0 variations.contraction.n_sample_steps=2

echo "[variations] launched 6 (3 per GPU, staggered). Monitor: logs/train_*var_*/progress.log or wandb."
