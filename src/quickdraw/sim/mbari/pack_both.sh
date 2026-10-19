#!/bin/bash
# The two arms of the data-volume ablation, packed from the SAME episodes at 20% val:
#   6 h  = all 720 nominal   -> mbari-tank
#   2 h  = the first 240     -> mbari-tank-2h   (a strict prefix, so class-balanced and same seeds)
cd /app
V=/app/logs/mbari/data/v6
L=/app/logs/mbari
.venv/bin/python -u -m quickdraw.sim.mbari.package --root $V --name mbari-tank --cams fpv \
  > $L/data/package_6h.log 2>&1 &
P6=$!
.venv/bin/python -u -m quickdraw.sim.mbari.package --root $V --name mbari-tank-2h --cams fpv \
  --limit 240 > $L/data/package_2h.log 2>&1 &
P2=$!
wait $P6 $P2
echo "6h run_dir: $(grep '^run_dir:' $L/data/package_6h.log | awk '{print $2}')"
echo "2h run_dir: $(grep '^run_dir:' $L/data/package_2h.log | awk '{print $2}')"
