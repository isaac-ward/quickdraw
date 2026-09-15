"""Does a 4x4 pool localise better than 8x8? Same 23 frames, same ground truth."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
sys.path.insert(0, os.path.dirname(__file__))   # sibling analyses in this package
from localise_ood_pixels import SPLIT, SUB, auc, ensemble, pink_mask
from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.data.ood_windows import kept, window_steps
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                      normalizer, resolve_data_root)
PS = [1, 2, 4, 6, 8, 12, 16]
ckpt = sys.argv[1]
run = os.path.dirname(os.path.dirname(ckpt))
cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
set_subsample(SUB); set_action_aggregate("concat")
m = build_model(cfg).cuda(); load_checkpoint(m, ckpt); m.eval()
core = getattr(m, "_orig_mod", m); norm = normalizer(cfg); P = int(cfg.data.P)
key = next((n for n, _ in core.layout if n != "proprio"))
eps = load_split_episodes_mm(resolve_data_root(cfg), SPLIT, img_size=image_head_sizes(cfg),
                             cam=image_head_cams(cfg), repo_id="starling-2")
pool = {p: ([], []) for p in PS}
n = 0
for i in kept(SPLIT):
    o, a, fr = eps[i]
    w0, w1 = window_steps(SPLIT, i, SUB)
    for t in [x for x in range(max(P, w0), min(w1, len(o)))][:3]:
        gt = pink_mask(fr[key][t])
        if gt.sum() < 50:
            continue
        obs = torch.from_numpy(fr[key][t]).float().div(255.0).cuda()
        s = ensemble(core, norm, o, a, fr[key], key, P, t, "cuda", n=32)
        mu, sd = s.mean(0), s.std(0)
        sur = ((obs - mu).abs() / (sd + 1e-3)).mean(-1)
        n += 1
        for p in PS:
            v = sur if p == 1 else F.avg_pool2d(sur[None, None], p, 1, p // 2)[0, 0][:sur.shape[0], :sur.shape[1]]
            v = v.cpu().numpy()
            pool[p][0].extend(v[gt].tolist()); pool[p][1].extend(v[~gt].tolist())
print(f"\n  pixel-level AUC by pool size ({n} frames):")
for p in PS:
    print(f"    {p:>2d}x{p:<2d}  {auc(np.array(pool[p][0]), np.array(pool[p][1])):.4f}")
