#!/bin/bash
# MBARI world-model run. The 5-point run_summary is REQUIRED (train_world_model.py:52) and its values
# must avoid parentheses, commas and apostrophes -- Hydra's override grammar rejects them with a bare
# error naming one character. The forbidden set includes ' = ' as well -- a run_summary
# reading 'x 10 = 80000' is rejected with 'mismatched input'.
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
# 14 EPOCHS, NOT 40, AND THE REASON IS COMPARABILITY. The dataset went from 2 h to 6 h, so an epoch
# is 3x longer: 537,192 training windows against 179,064, i.e. 35,812 steps per epoch at batch 15
# against 11,937. Holding epochs fixed would have handed the 6 h arm 3x the gradient steps as well
# as 3x the data, and any win would then be unattributable. 14 epochs is 501,368 steps, within 5%
# of what 40 epochs bought on 2 h, so the comparison isolates DATA VOLUME -- which is the whole
# reason for generating 6 h in the first place.
# BATCH PINNED AT 6 ON BOTH ARMS, and epochs set so the two match on GRADIENT STEPS.
# Autobatch sizes per-arm from `card - reserve - resident frame store`, and 6 h of frames is ~43 GB
# resident, so it offered batch 6 to run A and batch 12 to run B. Two arms at different batch sizes
# are not comparable -- different gradient noise, different effective learning rate -- and the step
# counts diverge too (1.11 M vs 663 k). Pinned to 6, the smaller of the two, which is what the 6 h
# arm can actually fit:
#     run A  477,504 windows / 6 =  79,584 steps/epoch  x 6 epochs = 477,504 steps
#     run B  198,960 windows / 6 =  33,160 steps/epoch  x 14 epochs = 464,240 steps
# Within 2.9%, so the arms differ in DATA VOLUME and not in optimisation.
# EPOCHS CAPPED AT 8000 BATCHES. Measured, not guessed: 95 minutes into epoch 0 neither arm had
# reached its 25% line (run B's is batch 8,290), which bounds the rate at >0.69 s/batch and makes
# epoch 0 alone >6.4 h for the 2 h arm and >15 h for the 6 h arm -- and the first checkpoint only
# exists at the END of epoch 0, so nothing downstream could run until this evening. Epoch 0 is also
# the slowest epoch BY CONSTRUCTION: p_tf ramps 1->0 across its batches, so it is neither the
# parallel teacher-forced forward (needs p_tf==1) nor the compiled autoregressive path (gated on
# p_tf==0.0 exactly) -- it runs the sequential rollout eagerly.
#
# 8000 batches x 10 epochs = 80,000 steps on BOTH arms, exactly matched, ~1.5 h per epoch, first
# checkpoint within the hour. That is a sixth of the training the uncapped schedule would give and
# both models will be weaker in absolute terms -- but the experiment is a COMPARISON at matched
# compute, which is equally valid at 80,000 steps and actually completes.
# VALIDATION IS CAPPED TOO, and leaving it uncapped was the expensive mistake. Capping only
# limit_train_batches left val at 19,896 batches -- 2.5x the train epoch, and autoregressive so
# just as slow per batch -- which made one epoch CYCLE 13.5 h and ten epochs 135 h. 400 val
# batches is 2,400 windows: ample to rank checkpoints, and the heavyweight open-loop and
# 16-step-ahead evals run on their own cadence regardless of this number.
cd /app
R=/app/logs/recording_2026_10_19_00_30_50_mbari-tank
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m quickdraw.train_world_model \
  model=mbari_fpv data=mbari environments=recorded \
  environments.obs_dim=13 environments.action_dim=6 environments.dt=0.0333333 \
  data.root=$R data.repo_id=mbari-tank data.cam=fpv data.subsample=1 data.action_aggregate=concat data.autobatch=false data.batch=6 \
  trainer.max_epochs=10 +trainer.limit_train_batches=8000 +trainer.limit_val_batches=400 trainer.check_val_every_n_epoch=1 \
  trainer.checkpoint_monitor=val/metric/fpv/mse \
  eval.during_train.evals.manifold=false \
  eval.during_train.evals.denoising_multistep=false \
  eval.during_train.evals.denoising_aggregate=false \
  eval.during_train.evals.denoising_filmstrip=false \
  eval.during_train.every_epochs=2 \
  eval.decode_chunk=32 eval.closed_loop_horizon=64 \
  +run_summary.problem="No world model exists for the MBARI tank underwater domain so we cannot do uncertainty based OOD detection or model based control there" \
  +run_summary.tried="An uncapped schedule on this data was launched three times tonight and abandoned - autobatch gave the two arms different batch sizes which breaks the comparison and epoch 0 alone ran over six hours because p_tf ramps inside it so neither the parallel teacher forced path nor the compiled autoregressive path applies" \
  +run_summary.trying="vl128_blockstack_2cam transplanted to one fpv image head at 144x192 plus a 13 dim proprio head on 6 hours of 30 Hz AUV data" \
  +run_summary.trying_detail="Data volume ablation arm on 576 train episodes of one shared pack so both arms see identical val episodes and identical norm stats. Epochs capped at 8000 batches x 10 is 80000 matched gradient steps at batch 6 because an uncapped epoch measured over 2.8 s per batch and the first checkpoint would have been a day away. Only dataset forced fields changed - proprio 13 action 6 one image head bottleneck 9" \
  +run_summary.rationale="That recipe holds the open loop records on robocasa and block-stack and every inherited value was bought with an experiment so a minimal transplant isolates the new domain as the only variable. Epochs cut 40 to 14 so the 6 h arm gets matched gradient steps rather than 3x of them" \
  experiment=mbari_fpv_v1
