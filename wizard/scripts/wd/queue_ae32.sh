#!/usr/bin/env bash
# Wait for the ae_nt8 precheck to exit, then run ae_nt32 on the same GPU (they don't fit together).
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOG=wizard/scripts/wd/queue_ae32.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
say "waiting for ae_nt8 to finish..."
while docker compose exec -T app bash -c 'pgrep -f "[e]xperiment=ae_nt8" >/dev/null'; do sleep 60; done
say "ae_nt8 done -> launching ae_nt32"
DATA=/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71eb0b7d9ecbf1a7ada843da026d4bc0785
docker compose exec -T -e QUICKDRAW_LOG_ROOT=logs/aeprecheck -e CUDA_VISIBLE_DEVICES=0 -e WANDB_MODE=disabled app \
 uv run python -m quickdraw.train_world_model model=mm_flow model.d=128 model.heads=8 \
 model.modalities.1.pretrained=false model.modalities.1.encode_arch=conv \
 model.modalities.1.decode_kind=flow model.modalities.1.decode_arch=unet \
 model.modalities.1.num_tokens=32 model.modalities.1.latent_loss_weight=10 \
 model.modalities.1.img_size=128 model.latent_norm=layernorm \
 model.lambda_flow=0 model.action_head.enabled=false model.compile_rollout=false \
 model.action_dim=12 model.modalities.0.dim=16 \
 data.root=$DATA data.repo_id=robocasa-scene4-4h data.cam=robot0_agentview_left \
 data.subsample=5 data.F=8 data.autobatch=false data.batch=16 \
 environments=recorded environments.obs_dim=16 environments.action_dim=12 'environments.position_idx=[7,8,9]' \
 trainer.max_epochs=3 trainer.check_val_every_n_epoch=1 \
 eval.during_train.every_epochs=1 'eval.during_train.at_epochs=[]' \
 eval.during_train.evals.ae_floor=true eval.during_train.evals.ood_horizon=false \
 eval.during_train.evals.manifold=false eval.during_train.evals.denoising_filmstrip=false \
 eval.during_train.evals.denoising_multistep=false eval.during_train.evals.denoising_aggregate=false \
 eval.during_train.evals.control=false eval.during_train.evals.action_distribution=false \
 experiment=ae_nt32 \
 '+run_summary.problem="Pure autoencoder pre check at thirty two tokens, twelve times compression, to measure the reconstruction floor a from scratch bespoke autoencoder reaches on four hours of one scene before committing two twenty hour runs."' \
 '+run_summary.tried="The eight token arm of this same pre check measures the forty eight times compression case. The only historical bespoke number is fourteen point three to fifteen point three decibels, measured at forty eight times compression with the codec anchor at one, which we now know lets the dynamics erode the tokenizer."' \
 '+run_summary.trying="Bespoke conv encoder with a trainable flow U-Net decoder at thirty two tokens of width one hundred and twenty eight, four thousand and ninety six latent floats, layer normalization, round trip anchor ten, dynamics weight zero so only reconstruction trains."' \
 '+run_summary.trying_detail="One hundred and twenty eight pixels, F of eight, batch sixteen with autobatch off, three epochs, only the autoencoder floor evaluation enabled so the run stays small enough to share a GPU."' \
 '+run_summary.rationale="Twenty seven decibels is the number needed to match the two hundred and fifty six pixel pretrained arm, and twelve times compression is the strongest reason to think a learned encoder could get there where the forty eight times version reached only fifteen."' \
 > wizard/scripts/out/ae_nt32.out 2>&1
say "ae_nt32 exited ($?)"
