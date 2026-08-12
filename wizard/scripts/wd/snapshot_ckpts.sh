#!/usr/bin/env bash
# Read-only checkpoint snapshotter for the §11 action-conditioning pair (2026-08-12).
#
# WHY: trainer.save_top_k=2, so ModelCheckpoint DELETES older epoch=*.ckpt as training proceeds -- we have
# already lost tfz_act ep0/ep3. The literature question we most need answered is whether action
# CONTROLLABILITY emerges late in training (a phase transition) or is simply absent; answering it means
# running the action-sensitivity probe at ep1, ep2, ... ep11 AFTER the fact, which requires those checkpoints
# to still exist. 30 MB each, 12 epochs, 2 arms = ~720 MB against 1.1 TB free. Cheap insurance.
#
# Read-only: `cp -n` (never clobber) out of the live run dirs into logs/ckpt_snapshots. Touches nothing the
# trainer owns and cannot affect the runs.
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOG=wizard/scripts/wd/snapshot.log
while :; do
  docker compose exec -T app bash -c '
    mkdir -p /app/logs/ckpt_snapshots
    for a in tfz_act tfz_act_fourier; do
      d=$(ls -dt /app/logs/robocasa-act/*_$a 2>/dev/null | head -1)
      [ -n "$d" ] || continue
      for c in "$d"/checkpoints/epoch=*.ckpt; do
        [ -f "$c" ] || continue
        b=$(basename "$c")
        t="/app/logs/ckpt_snapshots/${a}__${b}"
        [ -f "$t" ] || { cp -n "$c" "$t" && echo "SNAP ${a}__${b}"; }
      done
    done' 2>/dev/null | while read -r line; do
      echo "[$(date '+%m-%d %H:%M:%S')] $line" >> "$LOG"
    done
  # both arms done? (12 epochs -> epoch=11 snapshotted for each) then stop
  n=$(ls logs/ckpt_snapshots/ 2>/dev/null | grep -c "epoch=11-" || true)
  if [ "${n:-0}" -ge 2 ]; then echo "[$(date '+%m-%d %H:%M:%S')] both arms reached ep11 -- snapshotter exiting" >> "$LOG"; exit 0; fi
  sleep 600
done
