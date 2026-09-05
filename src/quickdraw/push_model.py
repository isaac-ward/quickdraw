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
    # PICK BY THE OBJECTIVE, from the run's own metrics. This became possible when save_top_k went to -1
    # (conf/trainer/default.yaml): with every epoch on disk there is no reason to accept whichever epoch a
    # PROXY metric happened to rank first. best.ckpt tracks `checkpoint_monitor` (val-time L1+LPIPS at
    # horizon F); the objective is open-loop LPIPS@+128 from the eval suite. Measured, they disagree enough
    # to cost 8-15%. Reads metrics.jsonl, so it needs no Lightning plumbing and works on finished runs.
    obj = _best_objective_epoch(run_dir)
    if obj is not None:
        ep, key, val = obj
        cands = glob.glob(os.path.join(run_dir, "checkpoints", "preserved", f"epoch={ep}-step=*.ckpt")) \
            or glob.glob(os.path.join(run_dir, "checkpoints", f"epoch={ep}-step=*.ckpt"))
        if cands:
            return sorted(cands)[-1], f"best {key} = {val:.5f}, at epoch {ep} (chosen from metrics.jsonl)"
        print(f"[push_model] epoch {ep} is best on {key} ({val:.5f}) but its checkpoint is GONE -- pruned by "
              f"save_top_k before it was set to -1. Falling back.", flush=True)
    pres = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "preserved", "epoch=*.ckpt")))
    if pres:
        return pres[-1], "preserved/ checkpoint (objective epoch unavailable)"
    best = os.path.join(run_dir, "checkpoints", "best.ckpt")
    if os.path.exists(best):
        return best, "best.ckpt (NOTE: the MONITORED-metric best, which is a proxy -- not necessarily best open-loop)"
    raise FileNotFoundError(f"no checkpoint in {run_dir}/checkpoints")


def _best_objective_epoch(run_dir: str):
    """(epoch, key, value) of the best raw open-loop LPIPS@+128, or None if the run logged no such metric.

    THE objective (memory/goal-open-loop-sharpness-not-ae-floor): raw open-loop perceptual distance 128
    prediction steps out. On a multi-head model the FIRST head alphabetically decides -- with two cameras
    the scene view is the one the model is judged on, and picking per-head would need two checkpoints.
    """
    f = os.path.join(run_dir, "logs", "metrics.jsonl")
    if not os.path.exists(f):
        return None
    best = {}
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("tag", "")
            if t.startswith("eval_ood_horizon/open_loop/") and t.endswith("/lpips/@+128"):
                best.setdefault(t, {})[int(r["step"])] = float(r["value"])
    if not best:
        return None
    key = sorted(best)[0]
    ep = min(best[key], key=best[key].get)
    return ep, key, best[key][ep]


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
    # EVERY image head, not just the first. A multi-camera model has one filmstrip and one rollout video
    # PER HEAD, and shipping only `sorted()[0]` silently dropped `cam_wrist` from the two-camera model --
    # i.e. half the thing the model does would have been invisible on its page.
    cams = [d for d in sorted(os.listdir(base)) if d != "proprio"]
    if not cams:
        return {}
    want = {"filmstrip": "filmstrip_0.png",
            "filmstrip_2": "filmstrip_1.png",
            "error_vs_step": "error_vs_step_avg_log.png",
            "rollout": "rollout_0.mp4"}
    out = {}
    for head in cams:
        src = os.path.join(base, head)
        for key, orig in want.items():
            sp = os.path.join(src, orig)
            if not os.path.exists(sp):
                continue
            ext = os.path.splitext(orig)[1]
            newname = f"{head}_{key}{ext}" if len(cams) > 1 else f"{key}{ext}"
            shutil.copy2(sp, os.path.join(dest, newname))
            out.setdefault(head, {})[key] = newname
    print(f"[push_model] logged eval products from epoch {ep}: "
          f"{ {h: sorted(v) for h, v in out.items()} }", flush=True)
    return {"epoch": ep, "heads": cams, "files": out}


def _repo_id(name: str) -> str:
    """`<namespace>/<name>`, asking the Hub who we are when the namespace is not already given.

    Falls back to the bare name if there is no token (a dry run on a machine without credentials). The
    card's embedded <video> src needs the namespace, so getting this wrong renders a 404 rather than an
    error -- hence the loud warning rather than a silent fallback.
    """
    if "/" in name:
        return name
    try:
        from huggingface_hub import HfApi
        return f"{HfApi().whoami()['name']}/{name}"
    except Exception as e:
        print(f"[push_model] could not resolve the Hub namespace ({type(e).__name__}) -- the card's video "
              f"URL will be WRONG. Pass +hub.name=<namespace>/{name} to fix.", flush=True)
        return name


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


def _card(name: str, cfg, at_epoch: dict, epoch: int, why_ckpt: str, heads: list[str],   # `name` = FULL repo id
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

    # PRODUCTS. The rollout video is EMBEDDED with a <video> tag, not linked (user, 2026-09-05: "i want you
    # to include rollout videos (rollout_0) not just filmstrips in the main hugging face page"). A model
    # card renders raw HTML, but relative srcs do not resolve there -- the tag needs the absolute
    # `resolve/main` URL, which is why `name` is threaded in. A filmstrip is eight sampled frames; the video
    # is the whole horizon at frame rate, and drift is a temporal failure, so it is the more honest artifact.
    parts = []
    files = (products or {}).get("files", {})
    multi = len(files) > 1
    for head, f in files.items():
        if multi:
            parts.append(f"### `{head}`\n")
        if "rollout" in f:
            parts.append(
                f'<video controls loop muted playsinline width="100%" '
                f'src="https://huggingface.co/{name}/resolve/main/{f["rollout"]}"></video>\n\n'
                f"Open-loop rollout, full horizon at frame rate — predicted beside ground truth. If your "
                f"viewer does not play it inline: [`{f['rollout']}`]({f['rollout']}).\n")
        if "filmstrip" in f:
            parts.append(f"![open-loop filmstrip]({f['filmstrip']})\n\n**Top row predicted, bottom row "
                         f"ground truth**, over the full +1..+128 open-loop horizon on a held-out validation "
                         f"episode. The run's own logged artifact at epoch {products['epoch']}.\n")
        extra = []
        if "filmstrip_2" in f:
            extra.append(f"a second episode [`{f['filmstrip_2']}`]({f['filmstrip_2']})")
        if "error_vs_step" in f:
            extra.append(f"error against horizon [`{f['error_vs_step']}`]({f['error_vs_step']})")
        if extra:
            parts.append("Also: " + ", ".join(extra) + ".\n")
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
        # FULL repo id (namespace included) is needed BEFORE the card is written: the embedded <video> tag
        # takes an absolute `resolve/main` URL, and a bare name produces huggingface.co/<name>/... which
        # 404s. Resolved here rather than at upload time so a dry run renders the same URL as a real push.
        repo_id = _repo_id(name)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(_card(repo_id, rcfg, at_epoch, ep, why, heads, products, vers, dataset))

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
