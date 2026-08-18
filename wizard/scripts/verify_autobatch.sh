#!/usr/bin/env bash
# =============================================================================================
# VERIFICATION for the 2026-08-18 autobatch/memory work. RUN THIS ON AN IDLE GPU.
#
# It calls autobatch_find directly (no training) for three configs whose OLD answers we know, so the new
# sizing can be compared against them. It PROBES LARGE BATCHES BY DESIGN, so it must not share a card with a
# live run -- it refuses to start if a trainer is running (pass FORCE=1 to override, at your own risk).
#
# What it checks, per config:
#   * chosen batch vs the old choice
#   * the fitted slope/intercept, and the analytic persistent-memory figure they should agree with
#   * reserved-vs-allocated ratio (the fragmentation that used to be invisible)
#   * the eval-phase probe -- the number we have NEVER had, and the input to sizing autobatch_reserve_gb
#
# OLD ANSWERS (measured, headroom 0.35, allocated-based, res-8 bisection above base / no upward search below):
#   bsp32mse             batch 8   probe 41.2 GB
#   bsp32mse+base64      batch 8   probe 60.3 GB
#   anch128 (pretrained) batch 32  probe ~45 GB
# =============================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."; cd ..

if [ "${FORCE:-0}" != "1" ]; then
  if docker compose exec -T app bash -c 'pgrep -f "[t]rain_world_model" >/dev/null'; then
    echo "REFUSING: a trainer is running. This probes large batches and would risk OOMing it."
    echo "Wait for it to finish, or FORCE=1 $0"
    exit 1
  fi
fi

DATA=/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71eb0b7d9ecbf1a7ada843da026d4bc0785
COMMON="data.root=$DATA data.repo_id=robocasa-scene4-4h data.cam=robot0_agentview_left data.subsample=5 data.F=64
        environments=recorded environments.obs_dim=16 environments.action_dim=12 environments.position_idx=[7,8,9]
        model.action_dim=12 model.modalities.0.dim=16"

probe_one () {  # $1=label  $2=old_batch  $3...=overrides
  local label="$1" old="$2"; shift 2
  echo; echo "==================== $label   (old choice: batch $old) ===================="
  docker compose exec -T -e CUDA_VISIBLE_DEVICES=0 -e WANDB_MODE=disabled app uv run python - "$@" <<'PY' 2>&1 | grep -E "autobatch|CHOSE|persistent|ERROR"
import sys, torch, hydra
from omegaconf import OmegaConf
ov = sys.argv[1:]
with hydra.initialize(config_path="../../conf", version_base=None):
    cfg = hydra.compose(config_name="config", overrides=ov)
from quickdraw.training.setup import autobatch_find
b = autobatch_find(cfg, torch.device("cuda"), log=print)
print(f"CHOSE data.batch={b}")
PY
}

probe_one "bsp32mse (the recipe)" 8 model=bsp32mse $COMMON
probe_one "bsp32mse + decode_base=64" 8 model=bsp32mse model.modalities.1.decode_base=64 $COMMON
probe_one "anch128 (pretrained TAESD 128px)" 32 model=mm_flow model.d=128 model.heads=8 \
   model.modalities.1.num_tokens=8 model.modalities.1.img_size=128 model.modalities.1.latent_loss_weight=10 \
   model.diffusion.flow_arch=transformer model.diffusion.flow_arch_depth=2 model.diffusion.flow_arch_heads=4 \
   model.action_head.enabled=false model.compile_rollout=true model.recon_frac=0.25 $COMMON

cat <<'NOTE'

==================== WHAT TO DO WITH THE OUTPUT ====================
1. If a chosen batch came out LOWER than the old one, that is expected and correct: we now budget on RESERVED
   memory instead of allocated, so the same config measures larger. Read the logged frag % to see how much.
2. Take the eval-probe number and compare it against mem/peak_infer_reserved_gb from a real run (new metric).
   If they disagree by more than ~2 GB the probe is not modelling the eval path and must be fixed before it is
   trusted. THIS is the calibration that was previously impossible -- the old mem/peak_gb was a process max
   that mixed autobatch's own rejected probes with training and eval.
3. Only after 1 and 2 look right, set data.autobatch_reserve_gb from the measurements rather than from the
   placeholder 12, and do one 2-epoch live run asserting mem/peak_train_reserved_gb and
   mem/peak_infer_reserved_gb both sit under (total - reserve).
NOTE
