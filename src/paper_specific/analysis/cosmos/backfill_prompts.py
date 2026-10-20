"""Write prompts_<i>.txt into a run that predates the prompt log.

Legitimate because build_action_prose is a pure function: the same recorded actions, the same axes file
and the same window boundaries give the same sentence, every time. Windows come from the adapter's own
arithmetic -- num_frames - _cond_len(P) new frames per call.
"""
import sys
import numpy as np
sys.path.insert(0, "/app/src")
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download
from quickdraw.data.dataset import load_split_episodes, set_action_aggregate, set_subsample
from quickdraw.evaluation.interpret import build_action_prose

run, H, P, NEW = sys.argv[1], 2048, 8, 88          # 88 = num_frames 93 - cond_len 5
set_subsample(8); set_action_aggregate("concat")
it = OmegaConf.load("/app/conf/interpret/block_stack.yaml")
axes = OmegaConf.to_container(it.action_axes)
kw = dict(scene=str(it.get("scene_prompt", "")), subject=str(it.get("prose_subject", "")), max_phases=4)
eps = load_split_episodes(snapshot_download(repo_id="isaac-ronald-ward/block-stack",
                                            repo_type="dataset"), "val", repo_id="block-stack", dcfg=DataConfig.legacy())

import os
from quickdraw.data.dataset import DataConfig
d = os.path.join(run, "logs/epoch_0000/eval_ood_horizon/open_loop/action")
os.makedirs(d, exist_ok=True)
for i in range(2):
    a = eps[i][1]
    h = min(H, len(a) - P)
    rows, g0 = [], 0
    while g0 < h:
        g1 = min(g0 + NEW, h)
        rows.append((g0, g1, build_action_prose(a[P - 1 + g0:P - 1 + g1], axes, **kw)))
        g0 = g1
    body = "\n\n".join(f"[{k+1}] predicted steps +{lo+1}..+{hi}\n{t}" for k, (lo, hi, t) in enumerate(rows))
    p = os.path.join(d, f"prompts_{i}.txt")
    open(p, "w").write(
        f"# eval_ood_horizon/open_loop — episode {i}\n"
        f"# {len(rows)} prompt window(s), one per generated chunk, in the order the model saw\n"
        f"# them. Rendered from the RECORDED future actions by build_action_prose; the numbers\n"
        f"# are ground truth, the wording comes from conf/interpret/<env>.yaml action_axes.\n"
        f"# BACKFILLED: this run predates writer.text; regenerated from the same pure function.\n\n{body}\n")
    print(f"  wrote {p}  ({len(rows)} windows, H={h})")
