#!/usr/bin/env bash
# =============================================================================================
# HOLIDAY ORCHESTRATOR (user, 2026-08-13: "fire off the tests. set up 15 epoch monitors, if stuff
# doesn't work try new configs. don't write new code unless its bug fixes. keep the train a rollin!")
#
# Keeps BOTH GPUs busy with the non-pretrained flow-image-decoder queue (queue.txt). Per GPU:
#   free GPU        -> pop the next config and launch it
#   running, ep<15  -> leave it alone
#   running, ep>=15 -> judge ONCE: KEEP (run to 40) or KILL (free the GPU for the next config)
#   died on its own -> log it and move on
# Config-only: every queue line is hydra overrides. NO code is written by this script.
#
# ADOPTS runs it did not launch: anch256 (the 256px pretrained arm, SNR 2.33x) was already on GPU 1 and is
# judged by the same rule. It is the control that says whether ANY SNR in the torus regime yields motion.
#
# JUDGEMENT BAR, calibrated on measured baselines (not invented):
#   anch128 (128px pretrained, anchored, SNR 1.43x) plateaued at floor 24.4 / 1step 16.6 / ol@64 13.95 with
#   mot@64 stuck at 0.16-0.20 across 9 epochs. So 1.43x is NOT enough and ~0.17 motion is the FAILURE mode.
#   A bespoke arm must clear that, and its only routes are less compression or more AE capacity.
#     KILL if 1step < 12          -> collapsed (every 20 Hz run did this by ep4)
#     KILL if ae_floor < 21 dB    -> codec hopeless; 24.4 dB already fails, below 21 cannot win
#     KILL if mot@64 < 0.22       -> same static-prediction failure as all 14 prior runs
#     else KEEP to max_epochs
#   All five numbers are logged at judgement so every decision is auditable.
#
# NOTE the bracket trick in every pgrep/pkill pattern ([t]rain not train): a bare pattern also matches THIS
# script's own command line, and a plain `pkill -f watchdog.sh` already killed one of my own shells today.
# =============================================================================================
set -uo pipefail
cd "$(dirname "$0")/../../.."

HD=wizard/scripts/holiday
QUEUE=$HD/queue.txt
STATE=$HD/state
LOG=$HD/orchestrator.log
OUT=wizard/scripts/out
POLL=600            # 10 min
JUDGE_EPOCH=15
MAX_EPOCHS=50
mkdir -p "$STATE" "$OUT"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

DATA=/caches/hf/hub/datasets--isaac-ronald-ward--robocasa-scene4-4h/snapshots/5a3df71eb0b7d9ecbf1a7ada843da026d4bc0785

BASE=(
  model=mm_flow model.d=128 model.heads=8
  model.diffusion.flow_arch=transformer model.diffusion.flow_arch_depth=2 model.diffusion.flow_arch_heads=4
  model.diffusion.concat_action_embedding=true
  model.action_fourier_freqs=0 model.action_squash=none
  model.recon_frac=0.25 model.compile_rollout=true
  model.action_head.enabled=false model.p_tf_warmup_epochs=1
  model.action_dim=12 model.modalities.0.dim=16
  model.modalities.1.img_size=128
  model.modalities.1.pretrained=false          # <-- the program: NO pretrained trunk
  model.modalities.1.encode_arch=conv
  model.modalities.1.decode_kind=flow          # <-- trainable GENERATIVE decoder (proprio-style)
  model.modalities.1.decode_arch=unet
  model.modalities.1.latent_loss_weight=10     # codec anchor; never tried on a learned encoder before
  model.latent_norm=layernorm                  # affine RAISES without a frozen pretrained latent
  data.root=$DATA data.repo_id=robocasa-scene4-4h data.cam=robot0_agentview_left
  data.subsample=5 data.F=64
  data.autobatch=true data.autobatch_headroom=0.35
  environments=recorded environments.obs_dim=16 environments.action_dim=12
  environments.position_idx=[7,8,9]
  trainer.max_epochs=$MAX_EPOCHS trainer.check_val_every_n_epoch=1
  eval.during_train.every_epochs=1 eval.during_train.at_epochs=[]
  eval.horizon=128 eval.closed_loop_steps=[1,16]
  eval.during_train.evals.ae_floor=true
  eval.during_train.evals.ood_horizon=true
  eval.during_train.evals.manifold=false          # off: judged on ae_floor + ood_horizon, keeps evals ~2 min
  eval.during_train.evals.denoising_filmstrip=false
  eval.during_train.evals.denoising_multistep=false
  eval.during_train.evals.denoising_aggregate=false
  eval.during_train.evals.control=false
  eval.during_train.evals.action_distribution=false
)

alive(){ pgrep -f -- "CUDA_VISIBLE_DEVICES=$1 app uv run python -m quickdraw.train_world_model" >/dev/null; }

run_dir_for(){ ls -dt logs/holiday/*_"$1" 2>/dev/null | head -1; }

# last completed epoch that has BOTH an ae_floor and an ood_horizon reading (i.e. a judgeable epoch)
metrics(){ # $1 = run dir -> "ep floor 1step ol64 frz64 mot64 lpips" or empty
  local mj="$1/logs/metrics.jsonl"
  [[ -f "$mj" ]] || return 1
  python3 - "$mj" <<'PY'
import json,sys
per={}
for line in open(sys.argv[1]):
    try: r=json.loads(line)
    except Exception: continue
    per.setdefault(r.get("step"),{})[r.get("tag")]=r.get("value")
F="eval_ae_floor/image/psnr/@+1"; O="eval_ood_horizon/open_loop/image/"
ok=[e for e,d in per.items() if F in d and O+"psnr/@+64" in d and isinstance(e,int)]
if not ok: sys.exit(1)
e=max(ok); d=per[e]
print(e, f"{d[F]:.2f}", f"{d.get('val/metric/image/psnr',0):.2f}",
      f"{d[O+'psnr/@+64']:.2f}", f"{d.get(O+'psnr_frozen/@+64',0):.2f}",
      f"{d.get(O+'motion_ratio/@+64',0):.3f}", f"{d.get(O+'lpips_mean',9):.3f}")
PY
}

judge(){ # $1=exp $2=run_dir ; echo KEEP|KILL
  local m; m=$(metrics "$2") || { echo "WAIT"; return; }
  read -r ep floor step1 ol64 frz64 mot64 lpips <<<"$m"
  (( ep < JUDGE_EPOCH )) && { echo "WAIT"; return; }
  # RULES REVISED 2026-08-18 after the first pass. The old `floor < 21 dB` rule killed ALL EIGHT bespoke arms
  # and was WRONG: every model sits 4+ dB BELOW its own floor, so the floor was never the binding constraint,
  # and the arms at floor ~19.5-20 dB MATCHED the 24.4-27.2 dB pretrained arms on open-loop PSNR while beating
  # them on LPIPS and motion. Three of them (bsp8/bsp16/bsp32llw30) peaked at the very epoch they were killed.
  # Floor rule DELETED. motion rule DELETED too -- it is direction-blind and a COLLAPSED model scores HIGHER
  # on it (bsp16 read 1.046 at ep12). LPIPS is what actually tracks the goal, so judge on that.
  local v="KEEP" why="clears the bar"
  if (( $(echo "$step1 < 12" | bc -l) )); then v=KILL; why="COLLAPSED (1step $step1 < 12)"
  elif (( $(echo "$lpips > 0.50" | bc -l) )); then v=KILL; why="no perceptual gain (lpips $lpips > 0.50; every pretrained arm sits at 0.525-0.542, bespoke reaches 0.383-0.395)"
  fi
  say "[judge $1 @ep$ep] $v -- $why | floor=$floor 1step=$step1 ol@64=$ol64 mot@64=$mot64 LPIPS=$lpips"
  echo "$v"
}

kill_exp(){ # $1=experiment name
  # BUG FIXED 2026-08-18: 'experiment=bsp32' is a PREFIX of bsp32wide/bsp32none/bsp32mse/... so killing bsp32
  # ALSO killed bsp32wide on the other GPU 22 min into its life (SIGTERM at ep0 50%, 15.8M params, healthy).
  # Same prefix trap the watchdog's run_dir_for already guarded against. Anchor on a trailing space or EOL so
  # only the EXACT experiment token matches.
  docker compose exec -T app bash -c "pkill -f '[e]xperiment=$1( |\$)'" >/dev/null 2>&1
  sleep 10
  docker compose exec -T app bash -c "pkill -9 -f '[e]xperiment=$1( |\$)'" >/dev/null 2>&1
  sleep 5
}

launch(){ # $1=gpu $2=exp $3=overrides(string)
  local gpu="$1" exp="$2" ov="$3"
  local uniq="Config $exp with overrides $ov, launched by the holiday orchestrator on GPU $gpu."
  # shellcheck disable=SC2206
  local EXTRA=($ov)
  docker compose exec -T -e QUICKDRAW_LOG_ROOT=logs/holiday -e CUDA_VISIBLE_DEVICES="$gpu" app \
    uv run python -m quickdraw.train_world_model "${BASE[@]}" "${EXTRA[@]}" experiment="$exp" \
    "+run_summary.problem=\"The image head cannot express motion because its codec error exceeds the motion itself. At four hertz the per step image change is zero point zero eight six root mean square while the frozen tokenizer reconstructs at zero point zero six, a ratio of one point four three, and the anchored one hundred and twenty eight pixel arm plateaued at motion ratio zero point one seven with that ratio. The proprio head, which has no compression bottleneck at all, reaches a ratio of eleven point five and its rollouts visibly move, so the bottleneck rather than the dynamics is the suspect.\"" \
    "+run_summary.tried=\"Fourteen runs on this dataset have produced a static image rollout regardless of denoiser architecture, latent normalization, action conditioning, teacher forcing or sampling. Subsampling to four hertz raised the ratio from zero point six one to one point four three and anchoring the codec stopped it eroding, but motion ratio still sat at zero point one six to zero point two zero. The only historical bespoke autoencoder measurement is about fifteen decibels at forty eight times compression and unanchored.\"" \
    "+run_summary.trying=\"$uniq A trainable convolutional encoder feeding a trainable generative flow U-Net decoder, with no pretrained tokenizer anywhere, which is structurally the same design as the proprio head that works. Dropping the pretrained trunk frees the latent size from the tokenizer, so the compression ratio becomes a design choice rather than something inherited.\"" \
    "+run_summary.trying_detail=\"One hundred and twenty eight pixels, four hertz subsampling with actions aggregated across skipped frames, F of sixty four, layer normalization since affine requires a fixed pretrained latent, round trip anchor weight ten, transformer denoiser at depth two with four heads, action slot and raw action channels on, fourier bands and action squashing off, forty epochs with validation and evaluation every epoch, judged at epoch fifteen.\"" \
    "+run_summary.rationale=\"If the image head can reach a codec error well below the per step motion, the same dynamics that already moves the proprio token should move the image. This arm tests whether a learned autoencoder specialised to four hours of one scene can beat a generic pretrained tokenizer once it is allowed less compression and is protected from the dynamics by the anchor.\"" \
    > "$OUT/$exp.out" 2>&1 &
  echo "$exp" > "$STATE/gpu$gpu"
  rm -f "$STATE/judged_$exp"
  say "[launch gpu$gpu] $exp | $ov"
}

next_config(){ # echoes "exp|overrides" and marks it consumed
  local line
  line=$(grep -vE '^\s*#|^\s*$' "$QUEUE" | while read -r l; do
    local e="${l%%|*}"
    [[ -f "$STATE/done_$e" ]] || { echo "$l"; break; }
  done | head -1)
  [[ -n "$line" ]] || return 1
  touch "$STATE/done_${line%%|*}"
  echo "$line"
}

say "=========================================================================="
say "orchestrator up | queue=$(grep -vcE '^\s*#|^\s*$' "$QUEUE") configs | judge at ep$JUDGE_EPOCH | poll ${POLL}s"
say "adopting anch256 on gpu1 (pretrained 256px, SNR 2.33x -- the control for whether ANY high SNR moves)"
[[ -f "$STATE/gpu1" ]] || echo "anch256" > "$STATE/gpu1"

while :; do
  for gpu in 0 1; do
    exp=$(cat "$STATE/gpu$gpu" 2>/dev/null || true)

    if alive "$gpu"; then
      [[ -n "$exp" ]] || continue
      [[ -f "$STATE/judged_$exp" ]] && continue          # already judged; leave it to finish
      dir=$(run_dir_for "$exp"); [[ -n "$dir" ]] || dir=$(ls -dt logs/robocasa-anchor/*_"$exp" 2>/dev/null | head -1)
      [[ -n "$dir" ]] || continue
      v=$(judge "$exp" "$dir")
      case "$v" in
        KILL) touch "$STATE/judged_$exp"; say "[gpu$gpu] killing $exp to free the GPU"; kill_exp "$exp" ;;
        KEEP) touch "$STATE/judged_$exp"; say "[gpu$gpu] $exp KEEPS running to ep$MAX_EPOCHS" ;;
      esac
      continue
    fi

    # GPU is free
    if [[ -n "$exp" ]]; then
      dir=$(run_dir_for "$exp"); [[ -n "$dir" ]] || dir=$(ls -dt logs/robocasa-anchor/*_"$exp" 2>/dev/null | head -1)
      m=$([[ -n "$dir" ]] && metrics "$dir" || echo "no metrics")
      say "[gpu$gpu] $exp is no longer running (final: $m)"
      : > "$STATE/gpu$gpu"
    fi
    if nc=$(next_config); then
      launch "$gpu" "${nc%%|*}" "${nc#*|}"
      sleep 120                                          # let it grab VRAM before the other GPU is considered
    else
      say "[gpu$gpu] QUEUE EMPTY -- idle. Add lines to queue.txt to keep going."
    fi
  done
  sleep "$POLL"
done
