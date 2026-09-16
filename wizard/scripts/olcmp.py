#!/usr/bin/env python
"""Open-loop LPIPS on cam_scene, ALWAYS printed next to the codec floor.

THE DECIDER IS OL LPIPS ON cam_scene. But it is only comparable across arms whose CODEC FLOOR has
converged, and that is the mistake this script now exists to prevent (record 8.24).

`eval_ae_floor/cam_scene/lpips_mean` is a per-frame encode->decode of the TRUE frames: the best the
rollout could possibly score. Six arms were judged at ep1-ep3 without it. bs_tok64 turned out to
have a floor of 0.1634 against a rollout of 0.1627 -- its codec could not reconstruct a frame it was
SHOWN, so the arm carried no information about prediction, yet it had been written up as a clean
capacity result. bs_d256 reached the same degenerate state by ep3.

RULES THIS ENCODES:
  1. Never quote OL without the floor beside it. Both columns always print.
  2. An arm is not readable until its floor plateaus -- ~ep7 on the baseline (0.0313 -> 0.0305).
     Rows below that are marked `unconverged` and must not be compared.
  3. floor >= OL means the arm is DEGENERATE: report "codec did not converge", never a dynamics
     result. Marked `DEGENERATE`.
  4. Matched gradient steps is necessary but NOT sufficient. Two arms at the same step with
     different floors are not comparable -- the delta column is suppressed when floors differ >15%.

OL/floor is NOT a rescue normalisation: a high floor leaves no headroom so the ratio -> 1
mechanically, and the baseline's own ratio climbs 1.58 -> 3.05 purely as its codec converges.

NOTE: metrics.jsonl's `step` field is the EPOCH INDEX, not the gradient step (it reads 1,3,5,7,9 for
evals every 2 epochs). Real steps = epoch * batches-per-epoch, applied from BPE below by hand.

Usage: olcmp.py [exp ...]     first exp is the reference
"""
import json, glob, os, re, sys, collections

EXPS = sys.argv[1:] or ['bs_batch8ctl', 'bs_tok16', 'bs_wd01']

# Batches per epoch is READ FROM THE LOG, never hand-maintained. A hand-kept dict silently
# defaulted two new arms to 1 batch/epoch, printing `gstep 1` and making them look like they sat
# outside every comparison range. Twice: bs_tok16, then bs_tok8.
def _bpe(run_dir: str) -> int | None:
    try:
        txt = open(os.path.join(run_dir, 'progress.log'), errors='ignore').read(400_000)
    except OSError:
        return None
    m = re.search(r'(\d+)\s+batches', txt)
    return int(m.group(1)) if m else None
OL = 'eval_ood_horizon/open_loop/cam_scene/lpips'
AE = 'eval_ae_floor/cam_scene/lpips_mean'
HOR = ['@+8', '@+412', '@+824', '@+1651']        # 0.27 s, 13.7 s, 27.5 s, 55 s at 30 Hz
FLOOR_PLATEAU = 0.035        # baseline plateaus at 0.0305-0.0313; above this, treat as unconverged

def load(exp):
    ds = sorted(glob.glob(f'logs/train_world_model_*{exp}'), key=os.path.getmtime)
    if not ds: return None, {}
    f = os.path.join(ds[-1], 'logs', 'metrics.jsonl')
    if not os.path.exists(f): return ds[-1], {}
    out = collections.defaultdict(dict)
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        t, ep = r.get('tag', ''), r.get('step')
        if t == f'{OL}_mean': out[ep]['ol'] = r['value']
        elif t == AE: out[ep]['floor'] = r['value']
        elif t.startswith(OL + '/') and t[len(OL) + 1:] in HOR: out[ep][t[len(OL) + 1:]] = r['value']
    return ds[-1], out

rows, BPE = {}, {}
for e in EXPS:
    d, o = load(e); rows[e] = o
    b = _bpe(d) if d else None
    if b is None:
        print(f'{e:14s} !! batches/epoch UNKNOWN -- gstep column will be wrong; fix before comparing')
        b = 1
    BPE[e] = b
    print(f'{e:14s} {len(o)} eval points   {b} batches/ep   {os.path.basename(d) if d else "MISSING"}')

hdr = f'\n{"run":14s} {"ep":>3s} {"gstep":>6s} {"OL":>7s} {"floor":>7s} {"OL-fl":>7s} ' + \
      ' '.join(f'{h:>8s}' for h in HOR) + '   state'
print(hdr); print('-' * (len(hdr) + 4))
for e in EXPS:
    for ep in sorted(rows[e]):
        v = rows[e][ep]
        ol, fl = v.get('ol'), v.get('floor')
        if ol is None: continue
        f_s = f'{fl:7.4f}' if fl is not None else f'{"-":>7s}'
        d_s = f'{ol - fl:7.4f}' if fl is not None else f'{"-":>7s}'
        cells = ' '.join(f'{v[h]:8.4f}' if h in v else f'{"-":>8s}' for h in HOR)
        if fl is None:            st = 'no floor yet'
        elif fl >= ol:            st = 'DEGENERATE (codec did not converge)'
        elif fl > FLOOR_PLATEAU:  st = f'unconverged (floor {fl:.4f} > {FLOOR_PLATEAU})'
        else:                     st = 'readable'
        print(f'{e:14s} {ep:3d} {ep*BPE[e]:6d} {ol:7.4f} {f_s} {d_s} {cells}   {st}')
    print()

ref = EXPS[0]
if len(EXPS) > 1 and rows.get(ref):
    print(f'vs {ref} at matched gradient steps -- suppressed where floors are not comparable:')
    rg = {ep * BPE[ref]: rows[ref][ep] for ep in rows[ref] if 'ol' in rows[ref][ep]}
    keys = sorted(rg)
    for e in EXPS[1:]:
        for ep in sorted(k for k in rows[e] if 'ol' in rows[e][k]):
            g, v = ep * BPE[e], rows[e][ep]
            if not keys or g < keys[0] or g > keys[-1]:
                print(f'  {e:12s} ep{ep:<3d} gstep {g:6d}  outside {ref} step range'); continue
            hi = next(k for k in keys if k >= g); lo = max(k for k in keys if k <= g)
            w = 0.0 if hi == lo else (g - lo) / (hi - lo)
            bo = rg[lo]['ol'] + (rg[hi]['ol'] - rg[lo]['ol']) * w
            bf = (rg[lo].get('floor'), rg[hi].get('floor'))
            note = ''
            if v.get('floor') is not None and None not in bf:
                bfi = bf[0] + (bf[1] - bf[0]) * w
                if abs(v['floor'] - bfi) / bfi > 0.15:
                    note = f'  <-- NOT COMPARABLE: floor {v["floor"]:.4f} vs {bfi:.4f}'
                if v['floor'] >= v['ol']: note = '  <-- DEGENERATE, codec did not converge'
            d = v['ol'] - bo
            print(f'  {e:12s} ep{ep:<3d} gstep {g:6d}  OL={v["ol"]:.4f}  ref~{bo:.4f}  '
                  f'{"BETTER" if d < 0 else "WORSE ":6s} {abs(d)/bo*100:5.1f}%{note}')
