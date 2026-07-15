"""Train the language reward head R(latent, text) = cos(f_z(latent), f_t(text)), distilled from an
eval_interpret run via CLIP-style caption contrastive learning. `python -m quickdraw.train_reward reward.interpret_run=<run>`

Each imagined clip carries N free-form VLM captions. We align f_z(latent) with f_t(MiniLM(caption)) by an
in-batch contrastive loss over (latent, caption) pairs — NO per-factor prototypes in the loss. Two modes,
toggled by reward.soft_targets:
  - false: vanilla CLIP InfoNCE (each latent's positive = its own caption; every other caption is a negative,
           INCLUDING captions of same-state clips -> those are false negatives).
  - true:  soft targets — the per-batch caption-caption MiniLM similarity IS the target distribution, so
           semantically-equal captions ("top red" ~ "the upper red area") share credit instead of being false
           negatives. Costs one extra BxB matmul on frozen caption embeddings.
`factors` are used only for eval coloring + a steering-readiness probe (score latents vs bucket words), never
for the loss. torch-only: MiniLM embeds text via `transformers` (mean-pooled). See design/language_steering.md."""

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


def _mlp(d_in, d_hidden, d_out, dropout=0.0):
    layers = [torch.nn.Linear(d_in, d_hidden), torch.nn.GELU()]
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))            # regularize: drop activations in train (off at eval)
    layers.append(torch.nn.Linear(d_hidden, d_out))
    return torch.nn.Sequential(*layers)


@torch.no_grad()
def _embed_texts(texts, model_name, device, bs=256):
    """MiniLM sentence embeddings via transformers: tokenize -> forward -> attention-masked mean-pool -> L2-norm.
    Batched (captions can number in the tens of thousands). Returns a CPU tensor (N, 384)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval().to(device)
    texts = list(texts)
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0, mdl.config.hidden_size)


def _load_interpret(run: str, factors):
    """Load per-point latents, per-factor labels, AND each clip's caption list from an eval_interpret run.
    Returns (latents (N,D), clip_idx (N,), labels_by {factor:[label per point]}, caps_by_clip [[str]*ncap per clip])."""
    base = glob.glob(os.path.join(run, "logs", "epoch_*", "eval_interpret"))
    assert base, f"no eval_interpret outputs under {run}/logs/epoch_*/"
    d = base[0]
    latents = np.load(os.path.join(d, "saved_projections", "latents.npy"))        # (N, D)
    clip_idx = np.load(os.path.join(d, "saved_projections", "clip_index.npy"))    # (N,) -> clip position in `ok`
    recs = json.load(open(os.path.join(d, "labels.json")))                        # per-clip, `ok` order
    labels_by = {f: [recs[int(c)]["label"][f] for c in clip_idx] for f in factors}
    labels_by_clip = {f: [r["label"][f] for r in recs] for f in factors}          # per-CLIP (for the caption plots)
    caps_by_clip = [list(r.get("captions", [])) for r in recs]
    return latents.astype(np.float32), clip_idx.astype(int), labels_by, labels_by_clip, caps_by_clip


def _soft_ce(logits, target_dist):
    """Cross-entropy against a soft (row-normalized) target distribution."""
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()


_GUIDE = """# train_reward metrics guide

Reward head: `R(z, text) = cos(f_z(z), f_t(text))`. `f_z` (latent->d) and `f_t` (MiniLM 384->d) are trained;
MiniLM + the world model are frozen. **Objective: CLIP-style caption contrastive** over each clip's N VLM
captions — align `f_z(latent)` with `f_t(MiniLM(caption))`. `reward.soft_targets` toggles vanilla InfoNCE
(own caption only = positive) vs soft targets (per-batch caption-caption similarity = the target, so
same-state captions aren't false negatives). Every scalar is `train/` and `val/` (val = held-out BY CLIP).

- `loss/contrastive` — the (bidirectional) contrastive loss above (lower = better).
- `acc/retrieval` — top-1 latent->own-caption retrieval on a val subset (noisy when captions repeat; a rough signal).
- `probe/<factor>_acc` — **steering-readiness**. Score each latent against the factor's bucket WORDS
  (embedded live through f_t, open-vocab — NOT used in the loss); argmax = predicted bucket; fraction matching
  the label. This is what steering actually queries, so it's the headline for "will `red`/`top` steer".
- `probe/<factor>_gap` — pos-minus-neg cosine margin for that probe (separation health).
- `data/same_combo_frac` (logged once) — measured fraction of in-batch off-diagonal pairs that share the FULL
  factor tuple (e.g. both top-red). Under vanilla CLIP these are the false negatives; soft targets fix them.
- `confusion/<factor>` (val) — probe argmax vs true. `embedding/<factor>_{2,3}d` (val) — f_z(latents) projected, colored by label.
"""


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    rc = cfg.reward
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # factors are for EVAL ONLY (coloring + the bucket-word probe). The training loss is caption-contrastive.
    factors = list(rc.get("factors", None) or [rc.factor])
    fac_cfgs = {f: OmegaConf.to_container(cfg.interpret.factors[f], resolve=True) for f in factors}
    buckets_by = {f: list(fac_cfgs[f]["buckets"]) for f in factors}
    flat = [(f, b) for f in factors for b in buckets_by[f]]       # every (factor, bucket), in probe-prototype order
    ranges, _i = {}, 0                                            # factor -> (start, end) slice into the probe block
    for f in factors:
        ranges[f] = (_i, _i + len(buckets_by[f])); _i += len(buckets_by[f])
    bmap = {f: {b: k for k, b in enumerate(buckets_by[f])} for f in factors}

    # ---- data: per-point latents + per-factor labels + per-clip captions; split BY CLIP ----
    latents, clip_idx, labels_by, labels_by_clip, caps_by_clip = _load_interpret(rc.interpret_run, factors)
    ncap = len(caps_by_clip[0]) if caps_by_clip else 0
    assert ncap and all(len(c) == ncap for c in caps_by_clip), (
        f"caption-contrastive training needs a fixed #captions per clip; got {ncap} "
        f"(re-run eval_interpret with interpret.n_captions>0 so labels.json carries captions)")
    X = torch.from_numpy(latents)
    y_by = {f: torch.tensor([bmap[f][l] for l in labels_by[f]]).to(dev) for f in factors}   # per-factor int labels (eval)
    clip_t = torch.from_numpy(clip_idx).to(dev)
    rng = np.random.RandomState(int(rc.seed))
    clips = np.unique(clip_idx); rng.shuffle(clips)
    n_val = max(1, int(len(clips) * float(rc.val_frac)))
    val_clips = set(clips[:n_val].tolist())
    is_val = np.array([c in val_clips for c in clip_idx])
    tr, va = torch.from_numpy(~is_val).to(dev), torch.from_numpy(is_val).to(dev)

    # ---- caption embeddings: MiniLM over ALL captions, flattened in clip order so clip c -> rows [c*ncap, (c+1)*ncap) ----
    cap_texts = [cap for caps in caps_by_clip for cap in caps]
    cap_emb = _embed_texts(cap_texts, rc.text_model, dev).to(dev)                          # (n_clips*ncap, 384) frozen

    # ---- bucket-word prototypes: EVAL-ONLY probe (score latents vs the factor words), also saved for reload ----
    protos = [" / ".join(str(t).format(c=b) for t in rc.paraphrases) for (f, b) in flat]
    T = _embed_texts(protos, rc.text_model, dev).to(dev)                                    # (K_total, 384)

    run_dir = make_run_dir("train_reward", cfg.experiment)
    writer = make_writer(run_dir, cfg, job_type="train_reward")
    writer.config(OmegaConf.to_container(cfg, resolve=True))
    open(os.path.join(writer.dir, "guide.md"), "w").write(_GUIDE)

    def plog(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(run_dir, "progress.log"), "a") as f:
            f.write(line + "\n")

    soft = bool(rc.get("soft_targets", False))
    ctau = float(rc.get("caption_tau", 0.1))
    plog(f"[train_reward] caption-contrastive ({'SOFT targets' if soft else 'vanilla CLIP'}): {len(X)} points "
         f"({int(tr.sum())} train / {int(va.sum())} val by clip), {ncap} captions/clip, latent dim {X.shape[1]}, "
         f"{int(rc.epochs)} epochs")

    # ---- measured false-negative rate: fraction of in-batch off-diagonal pairs sharing the FULL factor tuple ----
    tuple_key = np.array([hash(tuple(labels_by[f][p] for f in factors)) for p in range(len(clip_idx))])
    tr_pos = np.where(~is_val)[0]
    samp = rng.choice(tr_pos, size=min(int(rc.batch), len(tr_pos)), replace=False)
    k = tuple_key[samp]
    same = (k[:, None] == k[None, :])
    same_combo_frac = float((same.sum() - len(samp)) / max(1, len(samp) * (len(samp) - 1)))
    plog(f"[train_reward] measured same-combo pair fraction in a batch of {len(samp)}: {100 * same_combo_frac:.1f}% "
         f"(these are the vanilla-CLIP false negatives; soft_targets down-weights them)")

    drop = float(rc.get("dropout", 0.0))
    f_z = _mlp(X.shape[1], int(rc.hidden), int(rc.embed_dim), dropout=drop).to(dev)
    f_t = _mlp(384, int(rc.hidden), int(rc.embed_dim), dropout=drop).to(dev)
    opt = torch.optim.Adam(list(f_z.parameters()) + list(f_t.parameters()), lr=float(rc.lr),
                           weight_decay=float(rc.get("weight_decay", 0.0)))
    Xd = X.to(dev)
    temp = float(rc.temperature)
    x_std = float(X.std())                                        # global latent scale for latent-noise augmentation
    lnoise = float(rc.get("latent_noise", 0.0))
    patience = int(rc.get("early_stop_patience", 0))
    best = {"loss": float("inf"), "ep": 0, "fz": None, "ft": None}
    bad = 0

    def contrastive(zc_n, cap_n):
        """Bidirectional contrastive loss between L2-normed latent embeds zc_n and caption embeds cap_n (both (B,e)),
        with the batch's frozen MiniLM caption vectors `craw` (B,384) available via closure for soft targets."""
        logits = (zc_n @ cap_n.T) / temp                          # (B,B)
        if soft:
            S = contrastive.craw @ contrastive.craw.T             # (B,B) caption-caption cosine (frozen MiniLM)
            Q = F.softmax(S / ctau, dim=1)                        # soft target: who means the same thing
            return 0.5 * (_soft_ce(logits, Q) + _soft_ce(logits.T, Q))
        tgt = torch.arange(logits.shape[0], device=dev)
        return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))

    def evaluate(mask):
        f_z.eval(); f_t.eval()
        with torch.no_grad():
            idxs = torch.where(mask)[0]
            zc = F.normalize(f_z(Xd[idxs]), dim=-1)               # (n,e)  (returned for projection)
            Tn = F.normalize(f_t(T), dim=-1)                      # (K,e) bucket-word probe prototypes
            psims = zc @ Tn.T                                     # (n,K)
            per = {}
            for f in factors:
                s0, s1 = ranges[f]
                sf, yy = psims[:, s0:s1], y_by[f][idxs]
                pred = sf.argmax(1)
                pos = sf[torch.arange(len(yy)), yy]
                neg = (sf.sum(1) - pos) / max(1, sf.shape[1] - 1)
                per[f] = dict(acc=(pred == yy).float().mean().item(), gap=(pos - neg).mean().item(),
                              pred=pred.cpu(), y=yy.cpu())
            sub = idxs[torch.randperm(len(idxs), device=dev)[:min(len(idxs), 2048)]]   # contrastive loss/retrieval subset
            zc2 = F.normalize(f_z(Xd[sub]), dim=-1)
            craw = cap_emb[clip_t[sub] * ncap]                    # deterministic: caption 0 per clip
            contrastive.craw = craw
            closs = contrastive(zc2, F.normalize(f_t(craw), dim=-1)).item()
            logits = (zc2 @ F.normalize(f_t(craw), dim=-1).T) / temp
            ret = (logits.argmax(1) == torch.arange(len(sub), device=dev)).float().mean().item()
        f_z.train(); f_t.train()
        return dict(loss=closs, retrieval=ret, per=per, zc=zc.cpu())

    tr_idx_all = torch.where(tr)[0]
    for ep in range(int(rc.epochs)):
        f_z.train(); f_t.train()
        perm = tr_idx_all[torch.randperm(len(tr_idx_all), device=dev)]
        for c0 in range(0, len(perm), int(rc.batch)):
            b = perm[c0:c0 + int(rc.batch)]
            xb = Xd[b]
            if lnoise > 0:
                xb = xb + lnoise * x_std * torch.randn_like(xb)   # augment: f_z invariant to latent jitter
            rows = clip_t[b] * ncap + torch.randint(ncap, (len(b),), device=dev)   # one random caption per point
            craw = cap_emb[rows]
            contrastive.craw = craw
            loss = contrastive(F.normalize(f_z(xb), dim=-1), F.normalize(f_t(craw), dim=-1))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % int(rc.get("report_every", 10)) == 0 or ep == int(rc.epochs) - 1:
            m = {"train": evaluate(tr), "val": evaluate(va)}
            scal = {}
            for s in m:
                scal[f"{s}/loss/contrastive"] = m[s]["loss"]
                scal[f"{s}/acc/retrieval"] = m[s]["retrieval"]
                for f in factors:
                    scal[f"{s}/probe/{f}_acc"] = m[s]["per"][f]["acc"]
                    scal[f"{s}/probe/{f}_gap"] = m[s]["per"][f]["gap"]
            if ep == 0:
                scal["data/same_combo_frac"] = same_combo_frac
            writer.scalars(scal, ep)
            vl = m["val"]["loss"]                                 # early-stop / best on val contrastive loss
            if vl < best["loss"] - 1e-4:
                best.update(loss=vl, ep=ep, fz={k: v.clone() for k, v in f_z.state_dict().items()},
                            ft={k: v.clone() for k, v in f_t.state_dict().items()})
                bad = 0
            else:
                bad += 1
            probes = " ".join(f"{f} tr{m['train']['per'][f]['acc']:.2f}/va{m['val']['per'][f]['acc']:.2f}" for f in factors)
            plog(f"[ep {ep:3d}/{int(rc.epochs)}] probe {probes} | val closs {vl:.3f} "
                 f"(best {best['loss']:.3f}@ep{best['ep']}, bad {bad}/{patience or '-'})")
            if patience and bad >= patience:
                plog(f"[train_reward] early stop @ep{ep}: val loss hasn't improved for {patience} reports "
                     f"(best @ep{best['ep']}, val loss {best['loss']:.3f})")
                break

    if best["fz"] is not None:                                   # deploy the best-val checkpoint, not the (overfit) last
        f_z.load_state_dict(best["fz"]); f_t.load_state_dict(best["ft"])
        plog(f"[train_reward] restored best-val checkpoint (ep {best['ep']}, val loss {best['loss']:.3f})")

    # ---- save the trained head FIRST (before the slow/failable projections, so a reducer crash never loses it) ----
    v = evaluate(va)
    torch.save({"f_z": f_z.state_dict(), "f_t": f_t.state_dict(),
                "factors": {f: buckets_by[f] for f in factors},          # factor -> buckets (probe/coloring order)
                "text_prototypes": T.cpu(), "embed_dim": int(rc.embed_dim), "hidden": int(rc.hidden),
                "latent_dim": int(X.shape[1]), "text_model": rc.text_model, "dropout": drop,
                "soft_targets": soft},                                    # provenance: which objective made this head
               os.path.join(run_dir, "reward_head.pt"))
    probes = ", ".join(f"{f} {v['per'][f]['acc']:.2f}" for f in factors)
    plog(f"[train_reward] saved reward_head.pt (val probe acc: {probes}, {'SOFT' if soft else 'vanilla'} targets)")

    # ---- final val figures: per-factor probe confusion + reward-space (f_z) projection colored by factor ----
    from .evaluation import interpret as I
    from .evaluation.projection import project_and_plot
    for f in factors:
        bk, pv = buckets_by[f], v["per"][f]
        cm = I.confusion([bk[i] for i in pv["y"].tolist()], [bk[i] for i in pv["pred"].tolist()], bk)
        cf = viz.fig_confusion(cm, bk, title=f"{f}: reward probe argmax vs true (val acc {pv['acc']:.2f})",
                               xlabel="reward probe argmax", ylabel="true")
        writer.figure(f"val/confusion/{f}", cf, int(rc.epochs)); plt.close(cf)
    labels_val = {f: [buckets_by[f][i] for i in v["per"][f]["y"].tolist()] for f in factors}
    rs_dir = os.path.join(writer.dir, f"epoch_{int(rc.epochs):04d}", "train_reward", "saved_projections")
    sup_w = [float(w) for w in cfg.interpret.get("umap_sup_weights", [0.1, 0.3, 0.9])]
    project_and_plot(writer, "train_reward", v["zc"].numpy(), labels_val, fac_cfgs,
                     step=int(rc.epochs), point_size=2.5, methods=("pca", "tsne", "umap"), umap_sup_weights=sup_w,
                     save_dir=rs_dir, plots_name="joint_latent_space_plots",
                     subtitle=f"joint latent space f_z(latent), val split ({int(va.sum()):,} points)", log=plog)

    # ---- language-model latent space: project the RAW MiniLM caption embeddings, colored by concept. This is
    #      the language space BEFORE f_t (model-independent, frozen MiniLM). Subsample for a legible/fast plot. ----
    cap_lab = {f: [labels_by_clip[f][c] for c in range(len(caps_by_clip)) for _ in range(ncap)] for f in factors}
    n_cap = cap_emb.shape[0]
    keep = np.arange(n_cap)
    if n_cap > 4000:                                             # cap points: t-SNE/UMAP cost + plot legibility
        keep = rng.choice(n_cap, 4000, replace=False)
    project_and_plot(writer, "train_reward", cap_emb[keep].cpu().numpy(),
                     {f: [cap_lab[f][i] for i in keep] for f in factors}, fac_cfgs,
                     step=int(rc.epochs), point_size=2.5, methods=("pca", "tsne", "umap"), umap_sup_weights=sup_w,
                     n_components=(2,), plots_name="language_model_latent_space_plots",
                     subtitle=f"language space MiniLM(caption), {len(keep):,} captions", log=plog)

    writer.finalize()
    plog(f"[train_reward] done -> {run_dir} (reward_head.pt, val probe acc: {probes}, "
         f"{'SOFT' if soft else 'vanilla'} targets)")


if __name__ == "__main__":
    main()
