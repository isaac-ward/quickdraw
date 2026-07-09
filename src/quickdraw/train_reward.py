"""Train the language reward head R(latent, text) = cos(f_z(latent), f_t(text)), distilled from an
eval_interpret run's (latent, label) pairs. `python -m quickdraw.train_reward reward.interpret_run=<run>`

torch-only: MiniLM embeds the vocabulary via `transformers` (mean-pooled), everything else is a tiny MLP.
Logs (local + wandb, via make_writer) mirror train/ and val/; writes a guide.md explaining each metric.
See design/language_steering.md."""

from __future__ import annotations

import glob
import json
import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .logging import viz
from .logging.writer import make_writer
from .utils.logging import make_run_dir


def _mlp(d_in, d_hidden, d_out):
    return torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.GELU(), torch.nn.Linear(d_hidden, d_out))


@torch.no_grad()
def _embed_texts(texts, model_name, device):
    """MiniLM sentence embeddings via transformers: tokenize -> forward -> attention-masked mean-pool -> L2-norm."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval().to(device)
    enc = tok(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)
    out = mdl(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return F.normalize(emb, dim=-1).cpu()


def _load_interpret(run: str, factor: str):
    """Load the per-point latents + per-point factor labels from an eval_interpret run."""
    base = glob.glob(os.path.join(run, "logs", "epoch_*", "eval_interpret"))
    assert base, f"no eval_interpret outputs under {run}/logs/epoch_*/"
    d = base[0]
    latents = np.load(os.path.join(d, "projections", "latents.npy"))              # (N, D)
    clip_idx = np.load(os.path.join(d, "projections", "clip_index.npy"))          # (N,) -> clip position in `ok`
    recs = json.load(open(os.path.join(d, "labels.json")))                        # per-clip, `ok` order
    labels = [recs[int(c)]["label"][factor] for c in clip_idx]                    # per-point factor label
    return latents.astype(np.float32), clip_idx.astype(int), labels


_GUIDE = """# train_reward metrics guide

Reward head: `R(z, text) = cos(f_z(z), f_t(text))`. `f_z` (latent->d) and `f_t` (MiniLM 384->d) are trained;
MiniLM + the world model are frozen. Trained by cross-entropy over the K vocabulary text prototypes:
`logits_k = cos(f_z(z), f_t(text_k)) / temperature`, target = the clip's label. Every scalar is logged under
both `train/` and `val/` (val = a held-out split BY CLIP, so a clip's per-step latents never leak).

- `loss/total` — the cosine cross-entropy above (lower = better).
- `acc/argmax` — **headline**. Score a latent against every vocab text, take argmax; fraction matching the
  label. Chance = 1/K. This is "did the reward learn the alignment"; if ~chance, don't plan with it.
- `acc/rank_mean` — mean rank (1 = best) of the correct text among all K (per-class detail is the confusion figure).
- `sim/pos_mean` — mean `cos(f_z(z), f_t(correct text))` (want -> 1).
- `sim/neg_mean` — mean `cos` to the WRONG texts (want low).
- `sim/gap` = `pos_mean - neg_mean` — the separation margin; the health signal (near 0 = can't tell classes apart).
- `confusion/<factor>` (val) — argmax-predicted vs true; the per-class view.
- `embedding/<factor>_{2,3}d` (val) — `f_z(latents)` projected (PCA), colored by label; does the reward space separate?
"""


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    rc = cfg.reward
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    factor = rc.factor
    fac_cfg = OmegaConf.to_container(cfg.interpret.factors[factor], resolve=True)
    buckets = list(fac_cfg["buckets"])
    bmap = {b: i for i, b in enumerate(buckets)}
    K = len(buckets)

    # ---- data: per-point latents + labels from the eval_interpret run; split BY CLIP ----
    latents, clip_idx, labels = _load_interpret(rc.interpret_run, factor)
    X = torch.from_numpy(latents)
    y = torch.tensor([bmap[l] for l in labels])
    rng = np.random.RandomState(int(rc.seed))
    clips = np.unique(clip_idx); rng.shuffle(clips)
    n_val = max(1, int(len(clips) * float(rc.val_frac)))
    val_clips = set(clips[:n_val].tolist())
    is_val = np.array([c in val_clips for c in clip_idx])
    tr, va = torch.from_numpy(~is_val), torch.from_numpy(is_val)

    # ---- text prototypes: MiniLM mean-pool over each bucket's paraphrases ----
    protos = [" / ".join(str(t).format(c=b) for t in rc.paraphrases) for b in buckets]   # combined phrasing per bucket
    T = _embed_texts(protos, rc.text_model, dev).to(dev)                                  # (K, 384)

    run_dir = make_run_dir("train_reward", cfg.experiment)
    writer = make_writer(run_dir, cfg, job_type="train_reward")
    writer.config(OmegaConf.to_container(cfg, resolve=True))
    open(os.path.join(writer.dir, "guide.md"), "w").write(_GUIDE)

    def plog(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(run_dir, "progress.log"), "a") as f:
            f.write(line + "\n")

    plog(f"[train_reward] {factor}: {len(X)} points ({int(tr.sum())} train / {int(va.sum())} val by clip), "
         f"{K} classes, latent dim {X.shape[1]}, {int(rc.epochs)} epochs")

    f_z = _mlp(X.shape[1], int(rc.hidden), int(rc.embed_dim)).to(dev)
    f_t = _mlp(384, int(rc.hidden), int(rc.embed_dim)).to(dev)
    opt = torch.optim.Adam(list(f_z.parameters()) + list(f_t.parameters()), lr=float(rc.lr))
    Xd, yd = X.to(dev), y.to(dev)
    temp = float(rc.temperature)

    def evaluate(mask):
        f_z.eval()
        with torch.no_grad():
            zc = F.normalize(f_z(Xd[mask]), dim=-1)                 # (n, e)
            tc = F.normalize(f_t(T), dim=-1)                        # (K, e)
            sims = zc @ tc.T                                        # (n, K) cosine
            yy = yd[mask]
            loss = F.cross_entropy(sims / temp, yy).item()
            pred = sims.argmax(1)
            acc = (pred == yy).float().mean().item()
            pos = sims[torch.arange(len(yy)), yy]
            neg = (sims.sum(1) - pos) / (K - 1)
            rank = (sims >= pos[:, None]).sum(1).float().mean().item()   # 1 = correct is top
        f_z.train()
        return dict(loss=loss, acc=acc, pos=pos.mean().item(), neg=neg.mean().item(),
                    gap=(pos - neg).mean().item(), rank=rank, pred=pred.cpu(), y=yy.cpu(), zc=zc.cpu())

    for ep in range(int(rc.epochs)):
        f_z.train()
        idx = torch.randperm(int(tr.sum()), device=dev)
        tr_idx = torch.where(tr.to(dev))[0][idx]
        for c0 in range(0, len(tr_idx), int(rc.batch)):
            b = tr_idx[c0:c0 + int(rc.batch)]
            zc = F.normalize(f_z(Xd[b]), dim=-1)
            tc = F.normalize(f_t(T), dim=-1)
            loss = F.cross_entropy((zc @ tc.T) / temp, yd[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % int(rc.get("report_every", 10)) == 0 or ep == int(rc.epochs) - 1:
            m = {"train": evaluate(tr.to(dev)), "val": evaluate(va.to(dev))}
            writer.scalars({f"{s}/loss/total": m[s]["loss"] for s in m} |
                           {f"{s}/acc/argmax": m[s]["acc"] for s in m} |
                           {f"{s}/acc/rank_mean": m[s]["rank"] for s in m} |
                           {f"{s}/sim/pos_mean": m[s]["pos"] for s in m} |
                           {f"{s}/sim/neg_mean": m[s]["neg"] for s in m} |
                           {f"{s}/sim/gap": m[s]["gap"] for s in m}, ep)
            plog(f"[ep {ep:3d}/{int(rc.epochs)}] train acc {m['train']['acc']:.2f} gap {m['train']['gap']:.2f} | "
                 f"val acc {m['val']['acc']:.2f} gap {m['val']['gap']:.2f}")

    # ---- final val figures: confusion + reward-space embedding ----
    from .evaluation import interpret as I
    from .evaluation.manifold import pad_lims, reduce_dims
    v = evaluate(va.to(dev))
    cm = I.confusion([buckets[i] for i in v["y"].tolist()], [buckets[i] for i in v["pred"].tolist()], buckets)
    cf = viz.fig_confusion(cm, buckets, title=f"{factor}: reward argmax vs true (val acc {v['acc']:.2f})",
                           xlabel="reward argmax", ylabel="true")
    writer.figure(f"val/confusion/{factor}", cf, int(rc.epochs)); plt.close(cf)
    lab = [buckets[i] for i in v["y"].tolist()]
    for nd in (3, 2):
        e = reduce_dims(v["zc"].numpy(), "pca", n_components=nd, seed=0)
        rgb, legend = I.point_colors(lab, fac_cfg)
        fig_fn = viz.fig_points_9view if nd == 3 else viz.fig_points_2d
        fig = fig_fn(e, color=rgb, lims=pad_lims(e), point_size=6.0, legend=legend,
                     title=f"train_reward — PCA of f_z(latent) to {nd}D, colored by {factor} (val)")
        writer.figure(f"val/embedding/{factor}_{nd}d", fig, int(rc.epochs)); plt.close(fig)

    torch.save({"f_z": f_z.state_dict(), "f_t": f_t.state_dict(), "buckets": buckets,
                "text_prototypes": T.cpu(), "embed_dim": int(rc.embed_dim), "hidden": int(rc.hidden),
                "latent_dim": int(X.shape[1]), "text_model": rc.text_model}, os.path.join(run_dir, "reward_head.pt"))
    writer.finalize()
    plog(f"[train_reward] done -> {run_dir} (reward_head.pt, val acc {v['acc']:.2f}, gap {v['gap']:.2f})")


if __name__ == "__main__":
    main()
