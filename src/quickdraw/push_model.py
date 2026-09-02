"""Publish a trained world model to the Hub as a MODEL repo (push_to_hub does datasets only).

  python -m quickdraw.push_model +run_dir=logs/train_world_model_<ts>_<exp> +hub.name=quickdraw-wm-<...>

A checkpoint is NOT a model. `best.ckpt` carries `state_dict` and an EMPTY `hyper_parameters`, so the
weights hold no record of the architecture that produced them and no normalisation statistics. Four things
are needed to imagine, and they live in three places:

    weights          <run>/checkpoints/*.ckpt
    architecture     <run>/checkpoints/config.resolved.yaml     -> build_model(cfg)
    NORM STATS       the DATASET's normalization_stats.json     <- the trap
    context frames   the DATASET                                 <- also the trap

This copies all four into one repo, so a consumer needs neither the dataset (private, for robocasa) nor
any knowledge of the Lightning key prefix. What it writes:

    weights.safetensors        state_dict, `model.` prefix ALREADY stripped, no optimizer states
    training_state.ckpt        the full Lightning checkpoint, so fine-tuning can resume the optimizer
    config.resolved.yaml       verbatim
    normalization_stats.json   COPIED from the dataset -- otherwise unobtainable
    example_context.npz        a few (obs, act, frames) windows so the quickstart runs with no dataset
    metrics.json               GENERATED from metrics.jsonl, never hand-written
    README.md                  the card, with the numbers and a runnable quickstart

WHICH CHECKPOINT. Defaults to `checkpoints/preserved/epoch=*.ckpt` when present, NOT best.ckpt. Both are
open-loop rollout metrics, but best.ckpt tracked `val/metric/<head>/mse` for runs launched before
2026-09-02, and mse is structurally blind to sharpness (record §22) -- measured on vl_l1x3, its best-mse
epoch scored open-loop LPIPS@+128 0.1455 while the best-LPIPS epoch scored 0.1370, so best.ckpt pointed
at a model 6% worse on the metric the run is judged by. `preserved/` holds the best-open-loop-LPIPS epoch,
copied aside because save_top_k prunes by mse and would have deleted it. Override with `+hub.ckpt=<path>`.

WHY safetensors AND the raw ckpt. safetensors for loading: no pickle (a .ckpt is arbitrary code execution
on load), 86 MB instead of 140, and the Hub renders the tensor list. The raw checkpoint as well because
resuming training needs AdamW's moment buffers -- without them a fine-tune re-initialises the optimizer
and the first steps lose all adaptive scaling.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import shutil
import tempfile

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OL_KEY = "eval_ood_horizon/open_loop/image/lpips/@+128"


def _pick_ckpt(run_dir: str, explicit: str | None) -> tuple[str, str]:
    """(path, why). Prefers the preserved best-open-loop checkpoint over best.ckpt -- see the module doc."""
    if explicit:
        return explicit, "explicitly requested via +hub.ckpt"
    pres = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "preserved", "epoch=*.ckpt")))
    if pres:
        return pres[-1], ("best OPEN-LOOP LPIPS@+128 epoch, preserved from save_top_k pruning "
                          "(best.ckpt tracks val mse, which is blind to sharpness)")
    best = os.path.join(run_dir, "checkpoints", "best.ckpt")
    if os.path.exists(best):
        return best, "best.ckpt (no preserved/ dir; NOTE this is the monitored-metric best, not necessarily best open-loop)"
    raise FileNotFoundError(f"no checkpoint in {run_dir}/checkpoints")


def _metrics(run_dir: str) -> dict:
    """Best-so-far per eval key WITH the eval index it came from, read from the run's own metrics.jsonl.

    Generated, never hand-written: numbers in this project have been misquoted several times by pairing a
    score from one epoch with a different quantity from another. Recording the index makes that impossible.
    """
    f = os.path.join(run_dir, "logs", "metrics.jsonl")
    rows: dict[str, dict] = collections.defaultdict(dict)
    if os.path.exists(f):
        for line in open(f):
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("value") is not None:
                rows[j.get("tag", "")][j.get("step")] = float(j["value"])
    # Direction is decided by scanning EVERY path segment, not just the last one. The last segment of
    # `eval_ood_horizon/open_loop/image/psnr/@+128` is "@+128", so a last-segment test silently classified
    # every horizon-suffixed PSNR as lower-is-better and reported its WORST epoch (measured: 10.84 dB at
    # eval 0 instead of the real best).
    HI = ("psnr", "ssim", "cos")             # higher is better
    NEAR_ONE = ("motion_ratio",)             # neither direction is "better" -- 1.0 is the target
    out = {}
    for k, series in rows.items():
        if not k.startswith(("eval_", "val/")):
            continue
        segs = k.split("/")
        if any(seg.startswith(t) for seg in segs for t in NEAR_ONE):
            step = min(series, key=lambda st: abs(series[st] - 1.0))
            direction = "closest to 1.0"
        elif any(seg.startswith(h) for seg in segs for h in HI):
            step = max(series, key=series.get)
            direction = "max"
        else:
            step = min(series, key=series.get)
            direction = "min"
        out[k] = {"best": round(series[step], 6), "at_eval": int(step),
                  "n_evals": len(series), "direction": direction}
    return out


def _example_context(cfg, n_eps: int = 4, steps: int = 96) -> dict | None:
    """A few (obs, act, frames) windows from the run's own val split, for the card's quickstart.

    Without this the quickstart cannot run for anyone lacking the dataset -- which for the robocasa model
    is everyone outside this project. A card whose example does not run is worse than no card.
    """
    from .data.dataset import load_split_episodes_mm
    from .training.setup import image_head_cams, image_head_sizes, resolve_data_root
    cams = image_head_cams(cfg)
    if not cams:
        return None
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=cams, repo_id=str(cfg.data.get("repo_id", "torus")))
    eps = eps[:n_eps]
    n = min(steps, min(len(o) for o, _, _ in eps))
    out = {"obs": np.stack([o[:n] for o, _, _ in eps]).astype(np.float32),
           "act": np.stack([a[:n] for _, a, _ in eps]).astype(np.float32)}
    for head in cams:                                  # one entry per image head, named by HEAD
        out[f"frames__{head}"] = np.stack([fr[head][:n] for _, _, fr in eps]).astype(np.uint8)
    return out


def _frame_strip(model, norm, cfg, ctx_np: dict, head: str, out_png: str, horizon: int = 48) -> dict | None:
    """Render a pred-vs-truth strip from the PUBLISHED checkpoint and save it next to the card.

    Exists because a card that says "imagines the frames that follow" and shows only a good LPIPS number
    oversells the model. On these checkpoints the prediction is faithful for the first few steps and then
    holds the original viewpoint while the truth pans away, which is visible instantly in a strip and
    invisible in a scalar. Fail-soft: a card without a picture is worse than no card, but a failed publish
    is worse still.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch as _t
        dev = next(model.parameters()).device
        P = int(cfg.data.P)
        n = min(2, ctx_np["obs"].shape[0])
        ctx = {"proprio": norm.norm_obs(_t.from_numpy(ctx_np["obs"][:n, :P])).float().to(dev),
               head: _t.from_numpy(ctx_np[f"frames__{head}"][:n, :P]).float().div(255).to(dev)}
        acts = norm.norm_act(_t.from_numpy(ctx_np["act"][:n, :P + horizon - 1])).float().to(dev)
        with _t.no_grad():
            out = model.imagine_eval(ctx, acts, horizon=horizon, decode_chunk=8)
        pred = out[head].clamp(0, 1).float().cpu().numpy()
        gt = ctx_np[f"frames__{head}"][:n, P:P + horizon].astype("float32") / 255.0
        ts = [0, 4, 11, 23, 35, horizon - 1]
        ts = sorted({t for t in ts if t < horizon})
        rows = 2 * n
        fig, ax = plt.subplots(rows, len(ts), figsize=(1.9 * len(ts), 2.0 * rows))
        ax = np.atleast_2d(ax)
        for e in range(n):
            for k, src in ((0, pred[e]), (1, gt[e])):
                r = 2 * e + k
                for j, t in enumerate(ts):
                    ax[r, j].imshow(src[t]); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
                    if r == 0:
                        ax[r, j].set_title(f"+{t + 1}", fontsize=9)
                ax[r, 0].set_ylabel(("imagined" if k == 0 else "truth") + f"\nep{e}", fontsize=8)
        fig.suptitle("open-loop imagination vs ground truth", fontsize=11)
        plt.tight_layout(); plt.savefig(out_png, dpi=100, bbox_inches="tight"); plt.close(fig)
        mse = float(((pred - gt) ** 2).mean())
        per = ((pred - gt) ** 2).mean(axis=(0, 2, 3, 4))
        return {"horizon": horizon,
                "psnr_overall": round(float(-10 * np.log10(mse)), 2),
                "psnr_per_step": {f"+{t + 1}": round(float(-10 * np.log10(per[t])), 2) for t in ts}}
    except Exception as e:                      # a missing GPU, an OOM, no matplotlib -- publish anyway
        print(f"[push_model] frame strip skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
        return None


def _card(name: str, cfg, metrics: dict, why_ckpt: str, heads: list[str], strip: dict | None = None) -> str:
    def g(k, d="?"):
        v = metrics.get(k)
        return f"**{v['best']}** (eval {v['at_eval']} of {v['n_evals']})" if v else d
    h0 = heads[0] if heads else "image"
    img = next((m for m in cfg.model.get("modalities", []) if str(m.get("kind", "")) == "image"), {})
    P = int(cfg.data.P)
    if strip:
        ps = "  ".join(f"`+{k.lstrip('+')}` {v} dB" for k, v in strip["psnr_per_step"].items())
        strip_md = (f"![open-loop imagination](imagination.png)\n\n"
                    f"Top row of each pair is imagined, bottom is ground truth, over "
                    f"{strip['horizon']} open-loop steps from a single context.\n"
                    f"PSNR {strip['psnr_overall']} dB overall; per step: {ps}.\n\n"
                    f"**The known failure is large viewpoint change.** Early steps track well; when the "
                    f"robot base drives, the prediction tends to hold the original view while the truth "
                    f"pans away. Judge the model on this, not only on the table below.\n\n")
    else:
        strip_md = ""
    # An EXPLICIT list. A permissive substring filter dumped every ae_floor sub-metric (l1 at seven
    # horizons), which buries the two numbers a reader actually wants.
    want = []
    for h in heads:
        want += [(f"eval_ood_horizon/open_loop/{h}/lpips/@+128", "open-loop LPIPS @+128 (headline, lower better)"),
                 (f"eval_ood_horizon/open_loop/{h}/lpips/@+64", "open-loop LPIPS @+64"),
                 (f"eval_ood_horizon/open_loop/{h}/psnr/@+128", "open-loop PSNR @+128 (dB, higher better)"),
                 (f"eval_ae_floor/{h}/lpips_mean", "autoencoder floor LPIPS (perfect-dynamics bound)"),
                 (f"eval_ae_floor/{h}/psnr_mean", "autoencoder floor PSNR (dB)")]
    rows = "\n".join(f"| {desc} | `{k.split('/')[-1]}` | {metrics[k]['best']} | {metrics[k]['at_eval']} |"
                     for k, desc in want if k in metrics)
    return f"""---
license: mit
library_name: quickdraw
tags:
- world-models
- robotics
- video-prediction
---

# {name}

A latent world model: it takes {P} steps of context (proprioceptive vector + camera frame{'s' if len(heads) > 1 else ''})
plus a sequence of actions, and rolls forward **open-loop** — predicting the frames and states that follow
with no further observations. Faithful for the first several steps; see the strip below for where it stops
being faithful, which matters more than the headline number.

Trained with `quickdraw` (`model={cfg.model.get('name', '?')}`, recipe `vl128`). Image head{'s' if len(heads) > 1 else ''}: {', '.join(f'`{h}`' for h in heads)}
at {img.get('img_size', '?')}px, {img.get('num_tokens', '?')} latent tokens each.

{strip_md}## Headline numbers

Raw open-loop image prediction at +128 steps, and the autoencoder's own reconstruction floor:

| what | key | best | at eval |
|---|---|---|---|
{rows}

`metrics.json` in this repo carries every logged metric with the eval index it came from. **Lower is
better for `lpips`, higher for `psnr`.** The `@+128` suffix means 128 prediction steps with no
re-grounding — the hardest of the reported horizons.

Checkpoint selection: {why_ckpt}.

## Quickstart

```python
from quickdraw import load_pretrained, load_example_context
import torch

model, norm, cfg = load_pretrained("{name}", device="cuda")
ex = load_example_context("{name}")          # ships with the repo; no dataset needed

P, H = int(cfg.data.P), 64
ctx = {{"proprio": norm.norm_obs(torch.from_numpy(ex["obs"][:, :P])).float().cuda(),
       "{h0}": torch.from_numpy(ex["frames__{h0}"][:, :P]).float().div(255).cuda()}}
acts = norm.norm_act(torch.from_numpy(ex["act"][:, :P + H - 1])).float().cuda()

out = model.imagine_eval(ctx, acts, horizon=H, decode_chunk=16)
frames  = out["{h0}"].clamp(0, 1)            # (B, H, HW, HW, 3) imagined frames
proprio = norm.denorm_obs(out["proprio"])    # (B, H, obs_dim) in physical units
```

## Three things that will bite you

1. **Vectors are normalised, images are not.** `proprio` in and out goes through `norm_obs`/`denorm_obs`;
   frames are plain `[0, 1]` floats. Mixing these up produces plausible garbage rather than an error.
2. **`acts` needs `P + H - 1` steps, not `H`.** The context steps consume actions too.
3. **Pass `decode_chunk`.** The image decoder is ~78% of per-sample memory; a long horizon without it will
   exhaust the GPU.

## Fine-tuning

`training_state.ckpt` is the full Lightning checkpoint including AdamW moment buffers, so training can
resume rather than restart the optimizer. `weights.safetensors` is inference-only.

If you fine-tune: **keep `diffusion.flow_hidden` at 128.** Every run of this recipe at 512 destroyed
itself between epochs 5 and 12 under the perceptual loss, while 128 ran past epoch 21 healthy. Likewise
do not move `latent_loss_weight` from 10 — it is bracketed on both sides (0.4 erased the codec in one
epoch; 25 froze it).

## Honest caveats

* The reported `lpips` uses a **SqueezeNet** backbone while training used **VGG**. Different networks, but
  both are ImageNet feature stacks and therefore correlated, so the score is partly self-referential.
  Read `psnr`/`ssim` and look at the frames too.
* **The rollout is the weaker half.** The autoencoder reconstructs far better than the dynamics predicts;
  most of the remaining error at +128 is the codec floor, not drift.
* Trained on one dataset with a fixed camera geometry. Nothing here has been tested off-distribution
  beyond the eval splits reported above.
"""


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_dir = str(cfg.get("run_dir", "") or "")
    assert run_dir and os.path.isdir(run_dir), "pass +run_dir=logs/train_world_model_<ts>_<exp> (the + is required: run_dir is not in conf/config.yaml)"
    hub = cfg.get("hub", None) or {}
    name = str(hub.get("name", "") or "")
    assert name, "pass +hub.name=<model-repo-name>"
    private = bool(hub.get("private", False))

    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    ckpt, why = _pick_ckpt(run_dir, hub.get("ckpt", None))
    print(f"[push_model] checkpoint: {ckpt}\n[push_model] reason: {why}", flush=True)

    from .training.setup import image_head_cams, resolve_data_root
    heads = list(image_head_cams(rcfg))
    stats_src = os.path.join(resolve_data_root(rcfg), "normalization_stats.json")
    assert os.path.exists(stats_src), (
        f"normalization_stats.json not found at {stats_src}. Publishing without it would produce a model "
        f"whose rollouts are SILENTLY WRONG for anyone who cannot reach the dataset -- refusing.")

    with tempfile.TemporaryDirectory() as tmp:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        # DROP THE LPIPS LOSS NETWORK. `VisualLoss._net` is a frozen pretrained VGG used only to COMPUTE the
        # training loss -- it is not part of the world model, it is rebuilt lazily on first use, and
        # torchvision downloads it. Shipping it is pure bloat, and safetensors refuses it outright because
        # torchmetrics' LPIPS aliases `lins.N` onto `linN` (shared storage, which safetensors will not save).
        # `load_state_dict(strict=False)` reports these as missing, which is correct and harmless.
        LOSS_NET = ".visual._net."
        weights = {k[len("model."):]: v.contiguous() for k, v in sd.items()
                   if k.startswith("model.") and LOSS_NET not in k}
        dropped = sum(v.numel() for k, v in sd.items() if LOSS_NET in k)
        if dropped:
            print(f"[push_model] excluded the LPIPS loss network from the weights: "
                  f"{dropped/1e6:.1f}M params ({dropped*4/1e6:.0f} MB) -- not part of the model", flush=True)
        assert weights, f"no publishable `model.`-prefixed tensors in {ckpt}"
        from safetensors.torch import save_file
        save_file(weights, os.path.join(tmp, "weights.safetensors"))
        shutil.copy2(ckpt, os.path.join(tmp, "training_state.ckpt"))
        shutil.copy2(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"), tmp)
        shutil.copy2(stats_src, os.path.join(tmp, "normalization_stats.json"))

        metrics = _metrics(run_dir)
        with open(os.path.join(tmp, "metrics.json"), "w") as f:
            json.dump({"source_run": os.path.basename(run_dir), "checkpoint": os.path.basename(ckpt),
                       "checkpoint_selected_by": why, "metrics": metrics}, f, indent=2, sort_keys=True)

        ctx = _example_context(rcfg)
        if ctx is not None:
            np.savez_compressed(os.path.join(tmp, "example_context.npz"), **ctx)

        strip = None
        if ctx is not None and heads:
            from .pretrained import load_pretrained
            shutil.copy2(os.path.join(tmp, "weights.safetensors"), os.path.join(tmp, ".w.tmp"))
            os.remove(os.path.join(tmp, ".w.tmp"))
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mdl, nrm, lcfg = load_pretrained(tmp, device=dev)
            strip = _frame_strip(mdl, nrm, lcfg, ctx, heads[0], os.path.join(tmp, "imagination.png"))
            del mdl
            if dev == "cuda":
                torch.cuda.empty_cache()
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(_card(name, rcfg, metrics, why, heads, strip))

        sizes = {p: os.path.getsize(os.path.join(tmp, p)) / 1e6 for p in sorted(os.listdir(tmp))}
        print("[push_model] staged:\n" + "\n".join(f"    {k:26s} {v:8.1f} MB" for k, v in sizes.items()),
              flush=True)

        if bool(hub.get("dry_run", False)):
            dest = str(hub.get("out", "") or os.path.join(run_dir, "publish"))
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(tmp, dest)
            print(f"[push_model] DRY RUN -> {dest} (nothing uploaded)", flush=True)
            return

        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = name if "/" in name else f"{api.whoami()['name']}/{name}"
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=tmp)
        print(f"[push_model] {run_dir} -> https://huggingface.co/{repo_id} "
              f"({'private' if private else 'PUBLIC'})", flush=True)


if __name__ == "__main__":
    main()
