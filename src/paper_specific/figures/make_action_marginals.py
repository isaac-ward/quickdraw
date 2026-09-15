"""figures/marginals.png -- does the Action Model actually learn the action distribution?

The scalars in the Action Model table cannot show this. Energy skill measures conditioning, $W_1$
measures distance to the marginal and rest AUC measures the atom at zero, and no row wins all three
because the trade between them is real. This figure answers the same question by eye: for each of the
four sticks (columns) at four lead times into the chunk (rows), the RECORDED marginal in green under the
SAMPLED marginal in red, same bins, same axis.

What to look for: the spike at zero -- the stick held still -- which is the thing a rectified flow cannot
place without the percentile transform, and whether the red spreads as the lead time grows.

    CUDA_VISIBLE_DEVICES=0 python -m paper_specific.figures.make_action_marginals [<action_run>]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from check_prior_smoothness import _h_ctx                                          # noqa: E402

from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.evaluation.steering import PriorProposal
from quickdraw.training.setup import (build_model, effective_action_dim, image_head_cams,
                                      image_head_sizes, load_checkpoint, normalizer, resolve_data_root)

RUN = "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full"
OUT = "/app/logs/paper_icra_2027/marginals.png"
AXES = [a["name"] for a in yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]]
NA = len(AXES)
LEADS = (0, 7, 15, 31)        # slots into the 32-step chunk: +1, +8, +16, +32
N_CTX, N_DRAW, BINS = 48, 64, 61
FS = 9.0


def main(run: str = RUN) -> int:
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    set_subsample(int(cfg.data.get("subsample", 1) or 1)); set_action_aggregate("concat")
    m = build_model(cfg).to("cuda")
    load_checkpoint(m, os.path.join(run, "checkpoints", "last.ckpt")); m.eval()
    core = getattr(m, "_orig_mod", m); norm = normalizer(cfg)
    P, A = int(cfg.data.P), effective_action_dim(cfg)
    K = int(core.action_head_chunk)
    img = next((n for n, _ in core.layout if n != "proprio"), None)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")

    # RECORDED: every chunk in val, folded over the concat sub-steps the same way every other readout
    # folds them, so `real[:, k, j]` is stick j at lead k
    real = []
    for _, a, _ in eps:
        f = a.reshape(len(a), -1, NA).mean(axis=1)
        for i in range(0, len(f) - K, max(1, K // 4)):
            real.append(f[i:i + K])
    real = np.stack(real)

    # SAMPLED: many contexts spread over the split, many draws each, no reward and no selection
    ctxs = [(ei, t) for ei in range(len(eps))
            for t in np.linspace(P + 4, len(eps[ei][0]) - K - 2, N_CTX // len(eps)).astype(int)]
    with torch.no_grad():
        hs = []
        for ei, t in ctxs:
            o, a, fr = eps[ei]
            hs.append(_h_ctx(core, norm,
                             torch.from_numpy(o[t - P:t]).float()[None].cuda(),
                             torch.from_numpy(fr[img][t - P:t]).float().div(255.).unsqueeze(0).cuda(),
                             torch.from_numpy(a[t - P:t - 1]).float()[None].cuda(), img))
        h = torch.cat(hs, 0)
        pr = PriorProposal(core, a_max=1e9, norm=norm, prefix_guidance=False)     # raw stick units
        g = torch.Generator(device="cuda"); g.manual_seed(0)
        d = pr.sample(torch.zeros(len(h), K, A, device="cuda"), N_DRAW, ctx=h, g=g)
        pred = d.reshape(-1, K, A).cpu().numpy().reshape(-1, K, A // NA, NA).mean(axis=2)
    print(f"  chunk {K} | recorded {real.shape} | sampled {pred.shape} "
          f"({len(h)} contexts x {N_DRAW} draws)")

    fig, ax = plt.subplots(len(LEADS), NA, figsize=(7.1, 5.6), sharex="col")
    for r, k in enumerate(LEADS):
        for c in range(NA):
            A_ = ax[r, c]
            lo = min(real[:, k, c].min(), pred[:, k, c].min())
            hi = max(real[:, k, c].max(), pred[:, k, c].max())
            bins = np.linspace(lo, hi, BINS)
            A_.hist(real[:, k, c], bins=bins, density=True, color="tab:green", alpha=0.45,
                    label="recorded" if (r == 0 and c == 0) else None)
            A_.hist(pred[:, k, c], bins=bins, density=True, color="tab:red", alpha=0.45,
                    label="sampled" if (r == 0 and c == 0) else None)
            A_.set_yticks([]); A_.tick_params(labelsize=FS - 2.5)
            for sp in ("top", "right", "left"):
                A_.spines[sp].set_visible(False)
            if r == 0:
                A_.set_title(AXES[c], fontsize=FS)
            if c == 0:
                A_.set_ylabel(f"lead $+${k + 1}", fontsize=FS)
            if r == len(LEADS) - 1:
                A_.set_xlabel("stick", fontsize=FS - 1)
    ax[0, 0].legend(fontsize=FS - 1.5, frameon=False, loc="upper left")
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    fig.savefig(OUT, dpi=450, bbox_inches="tight"); plt.close(fig)
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
