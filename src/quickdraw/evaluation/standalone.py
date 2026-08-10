"""Shared post-hoc runner for the standalone eval entrypoints (best-checkpoint)."""

from __future__ import annotations

import json
import os

import torch

from ..environments.base import log_env_capabilities
from ..environments.registry import make_env
from ..logging.writer import make_writer
from ..training.setup import build_model, env_cfg, load_checkpoint, normalizer
from ..utils.logging import make_run_dir
from .routines import REGISTRY


def run_standalone(cfg, routines, label: str | None = None):
    """Run one or more eval routines under a single run dir/writer. `routines` is a name or a list of
    names (REGISTRY keys); `label` names the run dir (defaults to the sole routine name)."""
    if isinstance(routines, str):
        routines = [routines]
    label = label or routines[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # rebuild the model ARCHITECTURE from the run's saved config (logs/config.json) so ANY checkpoint loads
    # regardless of the CLI default model (modalities/d/name may differ). Falls back to cfg.model if absent.
    ck = cfg.get("checkpoint", None)
    run = os.path.dirname(os.path.dirname(ck)) if ck and str(ck).endswith(".ckpt") else ck
    cfgj = os.path.join(run, "logs", "config.json") if run else None
    if cfgj and os.path.exists(cfgj):
        from omegaconf import OmegaConf
        saved = OmegaConf.create(json.load(open(cfgj)))
        OmegaConf.set_struct(cfg, False)
        cfg.model = saved.model                                  # adopt the trained model config (arch + modalities)
        # ...but adopting it WHOLESALE silently discarded any `model.*` the caller passed on the CLI, so
        # `model.diffusion.stochastic_eval=true` (evaluate a trained checkpoint under stochastic sampling
        # instead of the committed mean) looked like it applied and did nothing. Re-apply CLI model overrides
        # ON TOP of the saved arch: the saved config still wins for everything the caller did not name.
        try:
            from hydra.core.hydra_config import HydraConfig
            for ov in HydraConfig.get().overrides.task:
                key, _, val = str(ov).lstrip("+~").partition("=")
                if key.startswith("model.") and val != "":
                    # PARSE the value as YAML -- OmegaConf.create({"v": val}) would keep the raw STRING, and
                    # bool("false") is True, so a `...=false` override would silently arrive as True.
                    OmegaConf.update(cfg, key, OmegaConf.create(f"v: {val}").v, merge=False)
                    print(f"[standalone] re-applied CLI override after adopting the saved model cfg: {key}={val}")
        except Exception as e:
            print(f"[standalone] could not re-apply CLI model overrides ({type(e).__name__}: {e})")
    model = build_model(cfg).to(device)
    load_checkpoint(model, cfg.checkpoint)  # .ckpt file or train run dir (-> best.ckpt)
    model.eval()
    run_dir = make_run_dir(f"eval_{label}", cfg.experiment)

    writer = make_writer(run_dir, cfg, job_type=f"eval_{label}")
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    # WorldEnv contract self-report (environments/base.py): one ✓/✗ line at eval start (additive, log-only)
    env_name = cfg.environments.get("name", "torus_world")
    log_env_capabilities(make_env(env_name, cfg.environments, batch=1),
                         lambda m: print(m, flush=True), name=env_name)
    summary = {}
    for name in routines:
        summary.update(REGISTRY[name](cfg, model, norm, ecfg, writer, device, 0))
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    writer.finalize()
    print(f"[eval_{label}] {json.dumps(summary, indent=2)}  run_dir={run_dir}")
