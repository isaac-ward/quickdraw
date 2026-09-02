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


def _logged_products(run_dir: str, ckpt_path: str, dest: str) -> dict:
    """Copy the run's OWN logged eval artifacts for the PUBLISHED checkpoint's epoch.

    Ships what the evaluation actually produced rather than an ad-hoc re-render. Better three ways: it is
    the canonical artifact (the same picture the run was judged on), it covers the FULL eval horizon
    (+1..+128, where a hand-rolled strip covered 48 steps on two episodes and led me to overstate a
    failure mode), and it includes the error-vs-step curve and a rollout video for free.

    Returns {name: relative filename} for whatever was found; missing products are simply omitted.
    """
    m = re.search(r"epoch=(\d+)", os.path.basename(ckpt_path))
    if not m:
        return {}
    ep = int(m.group(1))
    base = os.path.join(run_dir, "logs", f"epoch_{ep:04d}", "eval_ood_horizon", "open_loop")
    if not os.path.isdir(base):
        print(f"[push_model] no logged eval products for epoch {ep} at {base}", flush=True)
        return {}
    head = next((d for d in sorted(os.listdir(base)) if d != "proprio"), None)
    if head is None:
        return {}
    src = os.path.join(base, head)
    want = {"filmstrip.png": "filmstrip_0.png",
            "filmstrip_2.png": "filmstrip_1.png",
            "error_vs_step.png": "error_vs_step_avg_log.png",
            "rollout.mp4": "rollout_0.mp4"}
    out = {}
    for newname, orig in want.items():
        p = os.path.join(src, orig)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dest, newname))
            out[newname] = newname
    print(f"[push_model] logged eval products from epoch {ep}: {sorted(out)}", flush=True)
    return {"epoch": ep, "head": head, "files": out}


def _versions() -> dict:
    """The quickdraw version and git commit the weights were produced by, so a consumer can pin it.

    `pyproject.toml`'s version has never been bumped (0.1.0), so on its own it identifies nothing -- the
    COMMIT is the useful pin, and the card prints an install line using it.
    """
    out = {}
    try:
        from importlib.metadata import version
        out["quickdraw"] = version("quickdraw")
    except Exception:
        out["quickdraw"] = "unknown"
    # The container's /app is a baked copy with no .git, so `git rev-parse` there returns nothing. Accept
    # the sha from the environment (the launcher reads it on the host) and fall back to git only if that
    # actually works -- reporting "unknown" gives a consumer nothing to pin.
    out["git_commit"] = os.environ.get("QUICKDRAW_GIT_COMMIT", "") or "unknown"
    if out["git_commit"] == "unknown":
        try:
            import subprocess
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            out["git_commit"] = subprocess.check_output(
                ["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            pass
    out["torch"] = torch.__version__
    return out


CARD_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_card_template.md")


def _metrics_at(run_dir: str, epoch: int) -> dict:
    """{key: value} at ONE eval index — the epoch whose checkpoint is being published.

    The card used to show each metric's own best epoch, which is incoherent when you ship a single
    checkpoint: a reader would see an LPIPS from epoch 11 next to a PSNR from epoch 8 and reasonably
    assume both describe the weights in the repo. They did not. These are the numbers that checkpoint
    actually scored. Run-wide bests stay in metrics.json, where they are labelled as such.
    """
    f = os.path.join(run_dir, "logs", "metrics.jsonl")
    out = {}
    if os.path.exists(f):
        for line in open(f):
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("step") == epoch and j.get("value") is not None:
                out[j.get("tag", "")] = float(j["value"])
    return out


def _card(name: str, cfg, at_epoch: dict, epoch: int, why_ckpt: str, heads: list[str],
          products: dict | None = None, vers: dict | None = None, dataset: str = "?") -> str:
    """Fill the EDITABLE template at model_card_template.md. Keep prose there, not here."""
    vers = vers or {}
    img = next((m for m in cfg.model.get("modalities", []) if str(m.get("kind", "")) == "image"), {})
    prop = next((m for m in cfg.model.get("modalities", []) if str(m.get("kind", "")) == "vector"), {})

    want = []
    for h in heads:
        want += [(f"eval_ood_horizon/open_loop/{h}/lpips/@+128", f"`{h}` open-loop LPIPS @+128 (headline)"),
                 (f"eval_ood_horizon/open_loop/{h}/lpips/@+64", f"`{h}` open-loop LPIPS @+64"),
                 (f"eval_ood_horizon/open_loop/{h}/psnr/@+128", f"`{h}` open-loop PSNR @+128 (dB)"),
                 (f"eval_ae_floor/{h}/lpips_mean", f"`{h}` autoencoder floor LPIPS (perfect-dynamics bound)"),
                 (f"eval_ae_floor/{h}/psnr_mean", f"`{h}` autoencoder floor PSNR (dB)")]
    rows = "\n".join(f"| {desc} | {round(at_epoch[k], 5)} |" for k, desc in want if k in at_epoch) \
           or "| (no eval metrics logged at this epoch) | — |"

    parts = []
    files = (products or {}).get("files", {})
    if "filmstrip.png" in files:
        parts.append(f"![open-loop filmstrip](filmstrip.png)\n\n**Top row predicted, bottom row ground "
                     f"truth**, over the full +1..+128 open-loop horizon on a held-out validation episode. "
                     f"This is the run's own logged evaluation artifact at epoch {products['epoch']}.\n")
    if "filmstrip_2.png" in files:
        parts.append("A second episode: [`filmstrip_2.png`](filmstrip_2.png).\n")
    if "error_vs_step.png" in files:
        parts.append("Error against horizon (log axis): [`error_vs_step.png`](error_vs_step.png).\n")
    if "rollout.mp4" in files:
        parts.append("Rollout video: [`rollout.mp4`](rollout.mp4).\n")
    products_md = ("\n".join(parts) + "\n") if parts else ""

    with open(CARD_TEMPLATE) as f:
        tpl = f.read()
    return tpl.format(
        name=name, P=int(cfg.data.P), plural=("s" if len(heads) > 1 else ""),
        model_name=cfg.model.get("name", "?"), recipe="vl128", dataset=dataset,
        head_list=", ".join(f"`{h}`" for h in heads), head0=(heads[0] if heads else "image"),
        img_size=img.get("img_size", "?"), num_tokens=img.get("num_tokens", "?"),
        obs_dim=prop.get("dim", "?"), products_md=products_md, epoch=epoch, rows=rows,
        why_ckpt=why_ckpt, git_commit=vers.get("git_commit", "main"),
        pkg_version=vers.get("quickdraw", "?"), torch_version=vers.get("torch", "?"))


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

        products = _logged_products(run_dir, ckpt, tmp)
        vers = _versions()
        with open(os.path.join(tmp, "versions.json"), "w") as f:
            json.dump(vers, f, indent=2)
        ep = products.get("epoch")
        if ep is None:
            m = re.search(r"epoch=(\d+)", os.path.basename(ckpt))
            ep = int(m.group(1)) if m else -1
        at_epoch = _metrics_at(run_dir, ep)
        dataset = str(rcfg.data.get("hf_repo", None) or rcfg.data.get("repo_id", "?"))
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(_card(name, rcfg, at_epoch, ep, why, heads, products, vers, dataset))

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
        # create_repo(exist_ok=True) does NOT change the visibility of a repo that already exists, so a
        # re-push with a different `private` silently kept the old setting -- measured: a push reporting
        # PUBLIC left the repo private. Set it explicitly every time.
        try:
            api.update_repo_settings(repo_id=repo_id, repo_type="model", private=private)
        except Exception as e:
            print(f"[push_model] could not set visibility ({type(e).__name__}: {str(e)[:100]}) -- "
                  f"CHECK IT MANUALLY", flush=True)
        # delete_patterns="*": a re-push must REPLACE the repo, not union with it. Without this an
        # artifact from an earlier layout survives forever -- the first push shipped `imagination.png`
        # (an ad-hoc re-render) and it persisted alongside the logged filmstrip that replaced it,
        # leaving two conflicting pictures in one repo. push_to_hub.py does the same for datasets.
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=tmp, delete_patterns="*")
        print(f"[push_model] {run_dir} -> https://huggingface.co/{repo_id} "
              f"({'private' if private else 'PUBLIC'})", flush=True)


if __name__ == "__main__":
    main()
