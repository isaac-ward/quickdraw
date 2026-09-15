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

    fig = plt.figure(figsize=(3.4, 2.2))
    # 2x2, FLUSH, HALF HEIGHT. Every stick is on -1..1, including fore/aft -- which never goes positive
    # in this corpus, so half its panel is empty, but a shared axis is worth more than the space. The
    # panels touch on all four sides, so only the bottom row carries tick labels and the ticks stop at
    # +-0.5: at +-1.0 the left panel's last label and the right panel's first would have collided on the
    # seam. The axis name goes inside each panel, because a title there would land on the panel above.
    gh = fig.add_gridspec(2, 2, hspace=0.0, wspace=0.0,
                          left=0.085, right=0.995, top=0.875, bottom=0.115)
    k = LOOKAHEAD
    for r in range(NA):
        A_ = fig.add_subplot(gh[r // 2, r % 2])
        x0, x1 = -1.0, 1.0
        bins = np.linspace(x0, x1, BINS)
        last, bottom_row = (r == NA - 1), (r >= NA - 2)
        A_.hist(real[:, k, r], bins=bins, density=True, color="tab:green", alpha=0.45,
                label="Truth" if last else None)
        A_.hist(pred[:, k, r], bins=bins, density=True, color="tab:red", alpha=0.45,
                label="Prediction" if last else None)
        A_.set_yticks([])
        A_.set_xlim(x0, x1)
        A_.tick_params(labelsize=FS - 2.5, labelbottom=bottom_row)
        A_.set_xticks([-0.5, 0.0, 0.5])
        for sp in A_.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("black")
        # EQUAL PADDING ON BOTH SIDES. In axes fractions the same number is a different distance in x
        # and in y whenever the panel is not square, and these are wide -- so the inset is given in
        # POINTS off the top-left corner, which is the same gap left and above by construction.
        A_.annotate(AXES[r].capitalize() if not AXES[r].startswith("fore") else "Fore/aft",
                    xy=(0.0, 1.0), xycoords="axes fraction", xytext=(3, -3),
                    textcoords="offset points", ha="left", va="top", fontsize=FS)
        if last:
            # ABOVE THE PANELS, at the author's ask, sharing one band with the axis name: with the panels
            # flush there is no interior corner a legend can sit in without covering a distribution.
            _h, _l = A_.get_legend_handles_labels()
            fig.legend(_h, _l, loc="upper left", bbox_to_anchor=(0.085, 0.998), ncol=2,
                       fontsize=FS - 2.0, frameon=False, handlelength=1.3, columnspacing=1.1,
                       borderaxespad=0.0)
    # density=True, so the bars ARE a probability density over stick deflection and the four panels are
    # directly comparable now that they share one binning. Ticks stay off -- the shape is the message.
    # THE LEGEND AND THE X LABEL SHARE ONE BAND. supxlabel centres the label under the axes, which put
    # it straight on top of the tick labels; side by side with the legend the band does two jobs in the
    # height of one, which is the point of halving the figure.
    fig.supylabel("Probability density", fontsize=FS, x=0.016, y=0.495)
    # ...on the SAME baseline as the legend, so the header is one band and not two
    fig.text(0.995, 0.998, "Normalised stick deflection", ha="right", va="top", fontsize=FS)
    fig.savefig(OUT, dpi=700, bbox_inches="tight"); plt.close(fig)   # raised at the author's ask
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
