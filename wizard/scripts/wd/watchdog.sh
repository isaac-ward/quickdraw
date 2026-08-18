#!/usr/bin/env bash
# =============================================================================================
# OVERNIGHT WATCHDOG for the action-conditioning pair (user asked for it 2026-08-12: "anticipate
# further problems that could kill the runs ... i want them progressed when i wake up").
#
# WHAT IT DOES: every CHECK_S, for each arm, if no training process is on that arm's GPU and the
# arm has not finished its epochs, resume it from its own last.ckpt. Nothing else. It does not
# kill, throttle or reconfigure anything.
#
# WHY A WATCHDOG AND NOT MORE CODE FIXES: the code-level killers are closed (all six enabled eval
# routines were probed green against a same-architecture checkpoint, and a failing routine now
# disables itself instead of raising). What is left is the class we cannot enumerate: a CUDA OOM
# in a later epoch, a dataloader worker dying, the container restarting, the host OOM-killer. For
# those, recovery beats prediction -- Lightning restores model, optimizer, LR schedule and epoch
# counter from last.ckpt, so a resume costs at most the partial epoch in flight.
#
# HOW THE RESUME COMMAND IS BUILT: replayed from the arm's OWN captured argv (/proc/PID/cmdline,
# NUL-separated, captured while it was alive) so it cannot drift from what actually launched. Two
# edits are applied:
#   * data.autobatch=false + data.batch=<the value autobatch chose, read from config.resolved.yaml>
#     -- MANDATORY. train_world_model skips autobatch on resume ("a resume keeps its original
#     batch"), but nothing re-injects the chosen batch, so cfg.data.batch falls back to the config
#     DEFAULT of 1024 (conf/data/torus.yaml) against a chosen 32. A naive resume OOMs instantly.
#   * +resume=<run_dir>/checkpoints/last.ckpt -- continues the SAME run_dir, so ModelCheckpoint
#     keeps its dirpath and best.ckpt keeps tracking, and the unique-run-note gate is skipped.
# Existing data.batch / data.autobatch* tokens are FILTERED OUT rather than overridden twice:
# hydra can reject a duplicated override key, which would turn recovery into a second failure.
#
# SAFETY: at most MAX_RESUMES per arm (a deterministic crash must not loop all night), and a
# GRACE_S settling window after each resume before liveness counts again. A stall (process alive
# but no new epoch for STALL_S) is REPORTED ONLY -- never killed, because a slow eval is not a
# hang and killing a healthy run is the exact mistake this file exists to stop repeating.
# =============================================================================================
set -uo pipefail
cd "$(dirname "$0")/../../.."

WD=wizard/scripts/wd
LOG=$WD/watchdog.log
CHECK_S=300          # liveness poll
GRACE_S=900          # settling window after a resume (compile + autobatch-free startup ~2-3 min)
STALL_S=18000        # 5h with no new epoch -> report (an epoch is ~1.9h + ~4 min of eval)
MAX_RESUMES=3
MAX_EPOCHS=12        # matches trainer.max_epochs in the captured argv

declare -A RUN=( [0]=tfz_act [1]=tfz_act_fourier )
declare -A RESUMES=( [0]=0 [1]=0 )
declare -A NEXT_CHECK=( [0]=0 [1]=0 )

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# newest run dir for an arm, matched EXACTLY on the trailing experiment name so tfz_act does not
# also match tfz_act_fourier
run_dir_for() {
  ls -dt logs/robocasa-act/*_"$1" 2>/dev/null | head -1
}

last_epoch() {   # last completed epoch index, or -1
  local mj="$1/logs/metrics.jsonl"
  [[ -f "$mj" ]] || { echo -1; return; }
  python3 -c "
import json,sys
best=-1
for line in open(sys.argv[1]):
    try: r=json.loads(line)
    except Exception: continue
    if str(r.get('tag','')).startswith('time/'):
        best=max(best,int(r.get('step',-1)))
print(best)" "$mj" 2>/dev/null || echo -1
}

alive() {   # any training process pinned to this GPU index
  pgrep -f -- "CUDA_VISIBLE_DEVICES=$1 app uv run python -m quickdraw.train_world_model" >/dev/null
}

resume_arm() {
  local gpu="$1" name="${RUN[$1]}" dir ck batch
  dir="$(run_dir_for "$name")"
  # NEWEST last*.ckpt, not the literal last.ckpt. Verified 2026-08-12: resuming into an existing
  # checkpoints/ dir makes Lightning write its rolling checkpoint as last-v1.ckpt (then -v2, ...) and
  # leaves last.ckpt FROZEN at the pre-resume epoch. Hardcoding last.ckpt would make the 2nd resume
  # rewind to the 1st resume's starting point and re-lose the same epochs on every retry.
  # NOT epoch=*.ckpt: those are ModelCheckpoint's top-k by val metric, so the highest-numbered one can
  # be stale (the dead tfz_act run held only epoch=0 after dying in epoch 1). last* is the rolling one.
  ck="$(ls -t "$dir"/checkpoints/last*.ckpt 2>/dev/null | head -1)"
  if [[ -z "$ck" ]]; then say "[$name] DEAD but no last*.ckpt under $dir/checkpoints -- cannot resume, giving up on this arm"; return 1; fi
  batch="$(python3 -c "
import re,sys
for l in open(sys.argv[1]):
    m=re.match(r'^  batch:\s*(\d+)',l)
    if m: print(m.group(1)); break
else: print('')" "$dir/checkpoints/config.resolved.yaml" 2>/dev/null)"
  if [[ -z "$batch" ]]; then say "[$name] could not read the chosen batch from config.resolved.yaml -- NOT resuming blind (default is 1024, it would OOM)"; return 1; fi

  mapfile -d '' -t ARGV < "$WD/cmdline.gpu$gpu"
  local CMD=()
  for a in "${ARGV[@]}"; do
    [[ -n "$a" ]] || continue
    case "$a" in data.batch=*|data.autobatch=*|data.autobatch_reserve_gb=*|+resume=*) continue ;; esac
    CMD+=("$a")
  done
  CMD+=("data.autobatch=false" "data.batch=$batch" "+resume=$ck")

  RESUMES[$gpu]=$(( ${RESUMES[$gpu]} + 1 ))
  say "[$name] RESUMING (attempt ${RESUMES[$gpu]}/$MAX_RESUMES) from $ck at batch=$batch, ep$(last_epoch "$dir") done"
  nohup "${CMD[@]}" >> "wizard/scripts/out/${name}.resume${RESUMES[$gpu]}.out" 2>&1 &
  NEXT_CHECK[$gpu]=$(( $(date +%s) + GRACE_S ))
}

say "watchdog up: arms=${RUN[0]}(gpu0) ${RUN[1]}(gpu1) | poll ${CHECK_S}s | max ${MAX_RESUMES} resumes/arm | target ${MAX_EPOCHS} epochs"
while :; do
  live_any=0
  for gpu in 0 1; do
    name="${RUN[$gpu]}"
    [[ "${RESUMES[$gpu]}" == "gaveup" ]] && continue
    dir="$(run_dir_for "$name")"; [[ -n "$dir" ]] || { say "[$name] no run dir found"; continue; }
    ep="$(last_epoch "$dir")"

    if (( ep >= MAX_EPOCHS - 1 )); then
      if [[ "${NEXT_CHECK[$gpu]}" != "done" ]]; then
        say "[$name] FINISHED: epoch $ep of $MAX_EPOCHS complete. No longer watching."
        NEXT_CHECK[$gpu]=done
      fi
      continue
    fi
    live_any=1

    if alive "$gpu"; then
      mj="$dir/logs/metrics.jsonl"
      if [[ -f "$mj" ]]; then
        age=$(( $(date +%s) - $(stat -c %Y "$mj") ))
        (( age > STALL_S )) && say "[$name] STALLED? alive but metrics.jsonl untouched for $((age/60)) min (ep$ep). REPORTING ONLY, not killing."
      fi
      continue
    fi

    now=$(date +%s)
    (( now < ${NEXT_CHECK[$gpu]} )) && continue        # still inside the post-resume grace window
    if (( ${RESUMES[$gpu]} >= MAX_RESUMES )); then
      say "[$name] DEAD again after $MAX_RESUMES resumes (ep$ep). Deterministic -- STOPPING so it does not loop. Needs a human."
      RESUMES[$gpu]=gaveup
      continue
    fi
    say "[$name] DEAD: no process on gpu$gpu, last completed epoch $ep of $MAX_EPOCHS"
    resume_arm "$gpu"
  done
  (( live_any == 0 )) && { say "both arms finished or given up -- watchdog exiting"; exit 0; }
  sleep "$CHECK_S"
done
