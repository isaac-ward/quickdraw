"""Evaluate a trained checkpoint under tweaked EVAL-ONLY flags, WITHOUT trainer.fit.

    python -m quickdraw.eval_checkpoint \
        '+eval_ckpt="<run>/checkpoints/epoch=NN-step=MM.ckpt"' \
        +eval_decode_stochastic=true +eval_routines=[ood_horizon,ae_floor] \
        +eval_out=logs/_eval_ckpt +eval_step=9999

Why this exists: `+resume` is built to CONTINUE a run (rebuilds from the BASE config, needs the data group,
runs trainer.fit) — the wrong tool for "load a checkpoint, flip an eval flag, run the eval once." This reads
the checkpoint's OWN `config.resolved.yaml` (so the model is rebuilt EXACTLY as trained — no model-class
guessing), applies the eval-only overrides, loads the weights, and calls the eval routines directly.

Overrides supported:
    +eval_ckpt=...              (required) the checkpoint to load; quote it — the `epoch=..=..` name trips hydra.
    +eval_decode_stochastic=T   flip decode_stochastic ON for image heads (SAMPLE the decoder, not the mean).
    +eval_routines=[a,b]        which evaluation.routines.REGISTRY entries to run (default ood_horizon, ae_floor).
    +eval_out=<dir>             where products go: <dir>/logs/epoch_<eval_step>/ (default <run>/eval_ckpt).
    +eval_step=<int>            the epoch label stamped on the products (default 9999).
"""
from __future__ import annotations

import os

import hydra
import torch
from omegaconf import OmegaConf


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    ckpt = os.path.expanduser(str(cfg.get("eval_ckpt", "") or ""))
    assert ckpt and os.path.exists(ckpt), "pass +eval_ckpt=<...>/checkpoints/epoch=NN-step=MM.ckpt (quote it)"
    run_dir = os.path.dirname(os.path.dirname(ckpt))                      # <run>/checkpoints/<ckpt>
    rpath = os.path.join(run_dir, "checkpoints", "config.resolved.yaml")
    assert os.path.exists(rpath), f"no config.resolved.yaml beside the checkpoint at {rpath}"
    rcfg = OmegaConf.load(rpath)                                          # rebuild EXACTLY as trained
    OmegaConf.set_struct(rcfg, False)

    # eval_decode_shared_noise implies stochastic (shared noise is only meaningful while sampling).
    _shared = bool(cfg.get("eval_decode_shared_noise", False))
    if bool(cfg.get("eval_decode_stochastic", False)) or _shared:        # the eval-only flags under test
        n = sum(1 for mod in rcfg.model.modalities if str(mod.get("kind", "")) == "image")
        for mod in rcfg.model.modalities:
            if str(mod.get("kind", "")) == "image":
                mod["decode_stochastic"] = True
                if _shared:
                    mod["decode_shared_noise"] = True                    # (2026-09-22) time-coherent decode; §8.29
        print(f"[eval_ckpt] decode_stochastic=TRUE{' + shared_noise' if _shared else ''} on {n} image head(s)",
              flush=True)

    from .environments.registry import make_env  # noqa: F401 (imported for parity with the train env build)
    from .evaluation.routines import REGISTRY
    from .logging.writer import make_writer
    from .training.setup import build_model, env_cfg, load_checkpoint, normalizer

    # NB (2026-09-21): there is nothing to set up here. The data config (subsample / action_aggregate /
    # obs_keep) is carried EXPLICITLY into normalizer(rcfg) and every eval routine via DataConfig.from_cfg(rcfg),
    # so this entrypoint evaluates at the trained rate by construction. This used to require a set_subsample
    # call that eval_checkpoint forgot -> it silently evaluated at subsample=1 (10x wrong). The explicit,
    # keyword-only-no-default loader signature is what makes that failure mode impossible now. See DataConfig.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(rcfg).to(dev).eval()
    load_checkpoint(model, ckpt)                                         # strips `model.` prefix, strict=False
    norm = normalizer(rcfg)
    e = env_cfg(rcfg)                                                    # the `ecfg` the routines expect (as in training)
    out = str(cfg.get("eval_out", "") or os.path.join(run_dir, "eval_ckpt"))
    writer = make_writer(out, rcfg, job_type="eval")
    step = int(cfg.get("eval_step", 9999))
    names = list(cfg.get("eval_routines", None) or ["ood_horizon", "ae_floor"])
    print(f"[eval_ckpt] {ckpt}\n[eval_ckpt] routines={names} -> {os.path.join(out, 'logs', f'epoch_{step:04d}')}",
          flush=True)
    for name in names:
        print(f"[eval_ckpt] === {name} @step{step} ===", flush=True)
        REGISTRY[name](rcfg, model, norm, e, writer, dev, step)          # SAME signature the callback uses
    print("[eval_ckpt] DONE", flush=True)


if __name__ == "__main__":
    main()
