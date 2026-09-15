"""figures/marginals.png -- does the Action Model actually learn the action distribution?

The scalars in the Action Model table cannot show this. Energy skill measures conditioning, $W_1$
measures distance to the marginal and rest AUC measures the atom at zero, and no row wins all three
because the trade between them is real. This figure answers the same question by eye: for each of the
four sticks (rows) at the far end of the chunk, the RECORDED marginal in green under the SAMPLED
marginal in red, same bins, same axis.

What to look for: the spike at zero -- the stick held still -- which is the thing a rectified flow cannot
place without the percentile transform.

ONE LEAD TIME, NOT FOUR. The recorded marginal is pooled over every chunk in the split at stride K/4, so
slot +1 and slot +32 are the same frames offset by 31 (Jaccard 0.979) and the green is stationary in the
slot BY CONSTRUCTION -- per-slot means move by <0.04. Showing four rows of it implied a lookahead
dependence the marginal cannot have. What does decay with lookahead is CONDITIONING, and that is the
energy skill in the Action Model table (0.640 at +1 -> 0.292 at +32), not this figure.

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
LOOKAHEAD = 31                 # the far end of the 32-step chunk: +32
N_CTX, N_DRAW, BINS = 48, 64, 41   # coarser bins: on a log axis 61 bins read as noise
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
        # KEEP THE CONTEXT GROUPING: (contexts, draws, K, axes). The marginals pool it away, but the
        # conditioning panel needs to compare spread WITHIN a context against spread ACROSS contexts.
        pred_g = d.cpu().numpy().reshape(len(h), N_DRAW, K, A // NA, NA).mean(axis=3)
        pred = pred_g.reshape(-1, K, NA)
    print(f"  chunk {K} | recorded {real.shape} | sampled {pred.shape} "
          f"({len(h)} contexts x {N_DRAW} draws)")

    fig = plt.figure(figsize=(3.4, 2.9))
    # ONE SHARED X AXIS. Every stick is now on -1..1, including fore/aft -- which never goes positive in
    # this corpus, so half its panel is empty, but a shared axis is worth more than the space. The four
    # panels are therefore JOINED, only the bottom one is labelled, and the axis name goes inside each
    # panel because a title would land on the panel above it.
    gh = fig.add_gridspec(NA, 1, hspace=0.0, left=0.055, right=0.99, top=0.995, bottom=0.135)
    k = LOOKAHEAD
    for r in range(NA):
        A_ = fig.add_subplot(gh[r, 0])
        x0, x1 = -1.0, 1.0
        bins = np.linspace(x0, x1, BINS)
        last = (r == NA - 1)
        A_.hist(real[:, k, r], bins=bins, density=True, color="tab:green", alpha=0.45,
                label="Truth" if last else None)
        A_.hist(pred[:, k, r], bins=bins, density=True, color="tab:red", alpha=0.45,
                label="Prediction" if last else None)
        A_.set_yticks([])
        A_.set_xlim(x0, x1)
        A_.tick_params(labelsize=FS - 2.5, labelbottom=last)
        if last:
            A_.set_xlabel("Normalised stick deflection", fontsize=FS)
        for sp in A_.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("black")
        A_.text(0.015, 0.90, AXES[r].capitalize() if not AXES[r].startswith("fore") else "Fore/aft",
                transform=A_.transAxes, ha="left", va="top", fontsize=FS)
        if last:
            # fore/aft never goes positive, so the right half of the bottom panel is free
            A_.legend(fontsize=FS - 2.0, frameon=False, loc="upper right")
    # density=True, so the bars ARE a probability density over stick deflection and the four panels are
    # directly comparable now that they share one binning. Ticks stay off -- the shape is the message.
    fig.supylabel("Probability density", fontsize=FS, x=0.014)
    fig.savefig(OUT, dpi=450, bbox_inches="tight"); plt.close(fig)
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
