#!/bin/bash
# Everything downstream of generation, chained so it runs unattended:
#   wait for the shards -> preview videos -> pack all four cameras -> push to HF -> train.
# Each stage logs under /app/logs/mbari/, and each is skippable by hand if one needs redoing.
set -u
cd /app
V=/app/logs/mbari/data/v6
L=/app/logs/mbari

echo "=== waiting for generation shards  $(date -u +%H:%M) ==="
while pgrep -f "sim.mbari.generate" >/dev/null; do sleep 60; done
for s in train ood_thruster ood_fish ood_buoy; do
  printf "  %-14s %4s episodes\n" $s $(ls $V/$s/*.npz 2>/dev/null | wc -l)
done
du -sh $V

echo "=== preview videos, one per episode  $(date -u +%H:%M) ==="
bash /app/src/quickdraw/sim/mbari/gen_previews.sh $V 16 > $L/data/previews.log 2>&1
tail -1 $L/data/previews.log

# PACK FPV ONLY, and this is a throughput decision, not a scope cut. write_lerobot_split DECODES
# and RE-ENCODES every clip a second time, serially per split: 4 cameras x 900 frames per episode
# is ~43 s, so 648 train episodes is 7.7 hours. The dataloader reads ONE camera (`data.cam=fpv`),
# and step 1 has already written all four as per-episode clips under media/<cam>/<split>/ in
# parallel -- so chase, topdown and sideon still ship, still play on the Hub, and are still there
# for the control benchmark. Only the LeRobot video KEYS are fpv. Re-run with --cams to add the
# others if a consumer ever needs them as loadable keys.
echo "=== packaging (fpv keys; all four cameras ship as clips)  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.sim.mbari.package --root $V --name mbari-tank --cams fpv \
  > $L/data/package.log 2>&1
R=$(grep "^run_dir:" $L/data/package.log | tail -1 | awk '{print $2}')
echo "  run_dir = $R"
[ -z "$R" ] && { echo "PACKAGING FAILED"; tail -20 $L/data/package.log; exit 1; }
cp $L/data/PROCESSING.md $R/PROCESSING.md 2>/dev/null
# the clips ride along INSIDE the dataset repo, so they play on the Hub page itself
mkdir -p $R/media/previews && cp -r $L/data/previews/* $R/media/previews/
echo "  previews shipped: $(find $R/media/previews -name '*.mp4' | wc -l) mp4, $(du -sh $R/media/previews | cut -f1)"
echo "$R" > $L/data/run_dir.txt

echo "=== push to hugging face  $(date -u +%H:%M) ==="
.venv/bin/python -u -m quickdraw.push_to_hub data.root=$R +hub.name=mbari-tank \
  > $L/data/push.log 2>&1
tail -3 $L/data/push.log

echo "=== training  $(date -u +%H:%M) ==="
sed -i "s|^R=.*|R=$R|" /app/src/quickdraw/sim/mbari/train_mbari.sh
bash /app/src/quickdraw/sim/mbari/train_mbari.sh > $L/worldmodel/train_v2.log 2>&1
echo "=== pipeline done  $(date -u +%H:%M) ==="
