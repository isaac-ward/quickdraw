"""Readable proprio OOD-surprise viz (proprio-only -> fast, no image decode). Fixes the all-black strip:
sigma-floor eps=0.1 (so sigma->0 dims don't explode to 1e8) + SEMANTIC GROUPING of the 49-dim obs
(state 0-16, cube R 17-24, G 25-32, B 33-40, P 41-48 -- cube-major, 2 cams x 4 feats each).
Outputs (logs/ood/purple_surprise/): proprio_groups_bar.png (mean surprise per group, val vs purple)
+ proprio_strip_<split>.png (grouped, labeled, readable).
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/proprio_surprise_viz.py <ckpt>
"""
import os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.evaluation.surprise import surprise_proprio
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

N_ENS, N_FRAMES, OUT = 32, 40, "logs/ood/purple_surprise"
SPLITS = ["val", "eval_purple_play", "eval_purple_stack"]
# cube-major: state + R,G,B,P (each 8 dims = 2 cams x 4 feats)
GROUPS = [("state", 0, 17), ("cube R", 17, 25), ("cube G", 25, 33), ("cube B", 33, 41), ("cube P", 41, 49)]
GCOL = ["#888888", "#d62728", "#2ca02c", "#1f77b4", "#9467bd"]


def main():
    ck = sys.argv[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
    rcfg = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    m = build_model(rcfg).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck)
    m.stochastic_eval = True
    norm = normalizer(rcfg); P = rcfg.data.P
    os.makedirs(OUT, exist_ok=True)

    group_means, group_strips = {}, {}                 # split -> (n_groups,) mean / (n_groups, T) time strip
    for split in SPLITS:
        eps = load_split_episodes_mm(resolve_data_root(rcfg), split, dcfg=DataConfig.from_cfg(rcfg),
                                     img_size=image_head_sizes(rcfg) or 128,
                                     cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                     repo_id=rcfg.data.get("repo_id", "torus"))[:1]
        o, a, fr = eps[0]
        ts = np.linspace(P, len(o) - 2, min(N_FRAMES, len(o) - 2 - P)).astype(int)
        S = np.stack([surprise_proprio(m, norm, o, a, fr, P, int(t), n=N_ENS, device=dev).cpu().numpy() for t in ts]).T  # (49,T)
        # CLIP at 30 sigma: purple dims explode to ~1e7 because the normalizer's std for purple is ~0 (purple
        # absent in training), so norm_obs divides by ~0. The ranking/localization is correct; the magnitude is a
        # normalizer artifact -> cap it so the strip/bar are readable ("30 sigma" = maximally surprising).
        S = np.clip(S, 0, 30.0)
        # COLLAPSE 49 noisy dims -> 5 semantic group rows (mean over each group's dims). The old 49-row strip was
        # mostly black (most dims ~0) and looked different per episode just from which raw dim was picked; 5 group
        # rows on a SHARED scale (below) make val/purple directly comparable.
        group_means[split] = np.array([S[a0:a1].mean() for _, a0, a1 in GROUPS])
        group_strips[split] = np.stack([S[a0:a1].mean(0) for _, a0, a1 in GROUPS])            # (5, T)
        print(f"[{split}] group surprise: " + ", ".join(f"{g}={v:.2f}" for (g, _, _), v in zip(GROUPS, group_means[split])), flush=True)

    # ONE figure, one strip per split, SHARED vmax -> directly comparable (fixes "why so much black / ep0 != ep1")
    vmax = max(1e-3, max(np.percentile(gs, 99) for gs in group_strips.values()))
    fig, axs = plt.subplots(len(SPLITS), 1, figsize=(10, 2.1 * len(SPLITS)), sharex=True)
    for ax, split in zip(np.atleast_1d(axs), SPLITS):
        im = ax.imshow(group_strips[split], aspect="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels([g for g, _, _ in GROUPS])
        ax.set_ylabel(split.replace("eval_purple_", "purple-"), fontsize=9)
    np.atleast_1d(axs)[-1].set_xlabel("frame")
    fig.suptitle("proprio surprise by obs group over time (shared scale) — in-dist val vs purple-cube")
    fig.colorbar(im, ax=list(np.atleast_1d(axs)), label="group-mean |obs-mu|/(sigma+0.1)", fraction=0.025)
    fig.savefig(os.path.join(OUT, "proprio_strips.png"), dpi=120); plt.close(fig)

    # grouped bar: per-group mean surprise, val vs purple splits
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(GROUPS)); w = 0.26
    for k, split in enumerate(SPLITS):
        ax.bar(x + (k - 1) * w, group_means[split], w, label=split.replace("eval_purple_", "purple-"),
               color=["#4c78a8", "#e8820c", "#d62728"][k], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _, _ in GROUPS])
    ax.set_ylabel("mean proprio surprise (z)"); ax.set_title("proprio OOD surprise by obs group — in-dist val vs purple-cube")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "proprio_groups_bar.png"), dpi=120); plt.close(fig)
    print("[proprio_surprise_viz] wrote proprio_groups_bar.png + proprio_strips.png", flush=True)


if __name__ == "__main__":
    main()
