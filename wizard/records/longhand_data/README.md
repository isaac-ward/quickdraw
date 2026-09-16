# Raw evidence for `longhand.md` §8.20–§8.28

Committed because `logs/` is gitignored and the box it lived on was going down. These are the
numbers and images every table in §8.20–§8.28 was computed from, so the conclusions stay checkable
without the instance.

- `bs_*.metrics.jsonl` — long-format `{step, tag, value}`. **`step` is the EPOCH INDEX, not the
  gradient step** (reads 1,3,5,7,9 for evals every 2 epochs). Real steps = epoch × batches/epoch:
  1094 for `bs_stride10` (batch 13), 1778 for every batch-8 arm.
- `bs_*.config.json` — the resolved config per arm, so each one's single changed variable is
  recoverable.
- `*_filmstrip_0.png` — the visual evidence §8.27 was written from and §8.28 retracts. Note these
  sample **8 frames out of 1651**: glitching between samples is invisible in them, which is exactly
  what the operator saw in the MP4s and what made the filmstrip read wrong.

Re-derive the tables with `wizard/scripts/olcmp.py` (it reads `logs/`, so point it at a restored
run dir, or read these directly).

NOT preserved, too large: `best.ckpt` (386 MB for tok16+tok8), the MP4 rollouts (117 MB), and the
full per-epoch eval products. The checkpoints are the real loss — `tok8` had only 2 eval points
when the box went down and cannot be resumed from here.
