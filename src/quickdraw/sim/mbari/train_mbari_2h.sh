#!/bin/bash
# MBARI world-model run. The 5-point run_summary is REQUIRED (train_world_model.py:52) and its values
# must avoid parentheses, commas and apostrophes -- Hydra's override grammar rejects them with a bare
# error naming one character.
#
# EVAL SIZED DOWN, and it took reading ood_horizon_shapes to do it correctly. Autobatch put
# closed_loop_16_steps at 52.0 GB against the 6 h arm's 42 GB budget (the budget is card - reserve
# - resident frame store, and 6 h of frames eats ~54 GB, which is why only that arm was affected).
# Two obvious knobs are NO-OPS here: `horizon_n_episodes` is never read on this path -- n_ep is
# hardcoded to 8 when there is an image head -- and rows = min(max(n_ep,64), n_ep*n_seg) is CAPPED
# at 64, so moving closed_loop_horizon 256 -> 128 only took n_seg 16 -> 8 and left 8*8 = 64 rows.
# Getting under the cap needs n_seg < 8, i.e. closed_loop_horizon=64 -> rows 32, about 26 GB.
# decode_chunk 64 -> 32 does work and halved open_loop from 29.3 to 15.9 GB.
# Both arms carry the IDENTICAL eval shape, or their closed-loop numbers are not comparable, which
# is the whole point of the ablation.
#
# BATCH PINNED AT 6 ON BOTH ARMS, and epochs set so the two match on GRADIENT STEPS.
# Autobatch sizes per-arm from `card - reserve - resident frame store`, and 6 h of frames is ~43 GB
# resident, so it offered batch 6 to run A and batch 12 to run B. Two arms at different batch sizes
# are not comparable -- different gradient noise, different effective learning rate -- and the step
# counts diverge too (1.11 M vs 663 k). Pinned to 6, the smaller of the two, which is what the 6 h
# arm can actually fit:
#     run A  477,504 windows / 6 =  79,584 steps/epoch  x 6 epochs = 477,504 steps
#     run B  198,960 windows / 6 =  33,160 steps/epoch  x 14 epochs = 464,240 steps
# Within 2.9%, so the arms differ in DATA VOLUME and not in optimisation.
#
# Packing a separate smaller dataset -- which is what this did first -- gives the small arm its own
# val split AND its own normalization statistics, so two things move at once and a val difference
# cannot be attributed to data volume. One pack with `data.train_episodes` capped means identical
# val episodes, identical norm stats, identical preprocessing, and the single difference is how
# much training data the model is shown. The cap takes the first 240, and the generator numbers
# episodes so any prefix is class-balanced (job i carries seed 1000+i, class CLASSES[i%3]).
#
# 40 epochs x 11,937 steps = 477,480 against run A's 501,368, within 5% -- so the arms differ in
# DATA VOLUME and not in compute, which is the only way the comparison means anything.
cd /app
R=/app/logs/recording_2026_10_19_00_30_50_mbari-tank
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m quickdraw.train_world_model \
  model=mbari_fpv data=mbari environments=recorded \
  environments.obs_dim=13 environments.action_dim=6 environments.dt=0.0333333 \
  data.root=$R data.repo_id=mbari-tank data.cam=fpv data.subsample=1 data.action_aggregate=concat data.autobatch=false data.batch=6 +data.train_episodes=240 \
  trainer.max_epochs=14 trainer.check_val_every_n_epoch=1 \
  trainer.checkpoint_monitor=val/metric/fpv/mse \
  eval.during_train.evals.manifold=false \
  eval.during_train.evals.denoising_multistep=false \
  eval.during_train.evals.denoising_aggregate=false \
  eval.during_train.evals.denoising_filmstrip=false \
  eval.during_train.every_epochs=2 \
  eval.decode_chunk=32 eval.closed_loop_horizon=64 \
  +run_summary.problem="No world model exists for the MBARI tank underwater domain so we cannot do uncertainty based OOD detection or model based control there" \
  +run_summary.tried="Nothing on this dataset yet. The longhand branch records thirteen arms that all failed to move OL minus floor off 0.063 on 2.06 h of block-stack and concluded the remaining lever was more data which a real robot could not supply - this dataset is simulated so 6 h cost 65 minutes" \
  +run_summary.trying="vl128_blockstack_2cam transplanted to one fpv image head at 144x192 plus a 13 dim proprio head on 2 hours of 30 Hz AUV data" \
  +run_summary.trying_detail="Only dataset forced fields changed - proprio dim 13 action dim 6 one image head bottleneck 9. subsample 1 because delta_k 0.112 already clears the codec floor at 1.76x. Evals manifold and denoising off because they read torus only fields" \
  +run_summary.rationale="That recipe holds the open loop records on robocasa and block-stack and every inherited value was bought with an experiment so a minimal transplant isolates the new domain as the only variable. This is the 2 h CONTROL arm of the data-volume ablation - 40 epochs on a third of the data is 477480 steps against the 6 h arm 501368 so the two differ in DATA not compute" \
  experiment=mbari_fpv_2h
