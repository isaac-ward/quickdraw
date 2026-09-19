"""Are the model's heads actually CONFIGURABLE, and did making them so change training?

Two independent things, because they fail independently:

  MATRIX   build/roll/train each head combination -- image only, two images and no proprio, proprio only,
           one of each -- and assert the layout, the rollout keys and the eval head selection follow the
           CONFIG rather than a hardcoded "proprio". A proprio-only FEATURE (physics prior, relativize)
           requested without a proprio modality must raise a NAMED error, not a KeyError and not silence.

  REGRESSION  N forward passes through lit.LitWorldModel._step -- the same code path the head work
           modifies -- against frozen initial weights, compared to refs/train_starling_200.json.

           NO OPTIMIZER STEP, and that is the whole design. The first version took real training steps
           and MEASURED ITS OWN NOISE FLOOR at 0.81 in loss units: step 0 reproduced bit-identically,
           then the non-deterministic backward (cuDNN autotune, bf16 accumulation order, flex_attention)
           compounded through the weights until step 107 differed by more than any refactor would. A
           trajectory that drifts by 0.81 cannot detect a change worth detecting. Holding the weights
           fixed removes the compounding: every batch is scored against the same parameters, so the
           only thing that can move a loss is the computation itself -- which is exactly what is on
           trial. The backward path is untouched by the head work, so nothing is given up.

    python -m quickdraw.smoke.head_matrix            # both
    python -m quickdraw.smoke.head_matrix --capture  # rewrite the reference (only on purpose)
"""
from __future__ import annotations

import json
import os
import sys

import torch

REF = os.path.join(os.path.dirname(__file__), "refs", "train_starling_200.json")
TOL = 1e-5          # loss units. Forward-only against frozen weights is deterministic in practice;
#                     this is headroom for bf16 reduction order, not for training drift.
STEPS = 200

ok = bad = 0


def check(name, cond, extra=""):
    global ok, bad
    mark = "✓" if cond else "✗"
    print(f"  {mark} {name}{(' — ' + extra) if extra else ''}")
    if cond:
        ok += 1
    else:
        bad += 1
    return cond


def _cfg(modalities, **over):
    """The settled starling recipe with the modality list swapped out."""
    from hydra import compose, initialize_config_dir
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf"))
    with initialize_config_dir(config_dir=root, version_base=None):
        cfg = compose(config_name="config", overrides=[
            "model=vl128_starling", "data=starling2", "environments=recorded",
            "data.subsample=4", "data.action_aggregate=concat",
            "data.autobatch=false", "data.batch=2", "seed=0", *[f"{k}={v}" for k, v in over.items()]])
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    if modalities is not None:
        cfg.model.modalities = modalities
    return cfg


def _mods(cfg_full, names):
    """Pick modality specs out of the settled config by name, preserving their settings."""
    from omegaconf import OmegaConf
    allm = OmegaConf.to_container(cfg_full.model.modalities, resolve=True)
    by = {m["name"]: m for m in allm}
    out = []
    for n in names:
        base = dict(by.get(n.split("#")[0], by["image"]))
        base["name"] = n
        out.append(base)
    return out


def matrix():
    """Every head combination builds, rolls out, and reports exactly the heads it was configured with."""
    from ..training.setup import build_model
    print("\nHEAD MATRIX")
    full = _cfg(None)
    combos = {
        "proprio + image (the settled recipe)": ["proprio", "image"],
        "image only":                           ["image"],
        "two images, no proprio":               ["image", "image2"],
        "proprio only":                         ["proprio"],
    }
    for label, names in combos.items():
        try:
            cfg = _cfg(_mods(full, names))
            m = build_model(cfg)
            got = [n for n, _ in m.layout]
            check(f"{label}: layout == config", got == names, f"got {got}")
        except Exception as e:                                  # noqa: BLE001 -- the failure IS the result
            check(f"{label}: builds", False, f"{type(e).__name__}: {e}")

    # A PROPRIO-ONLY FEATURE WITHOUT PROPRIO MUST BE LOUD. Silently dropping the physics prior is exactly
    # the quiet wrongness this whole exercise is about.
    # Through build_model, which is where the guard lives: it must fire on the MODALITY LIST before the
    # environment/physics schema is touched, or the reader gets `Missing key quat_idx` and goes looking
    # in the wrong file.
    cfg = _cfg(_mods(full, ["image"]))
    cfg.environments.dynamics_prior = True
    try:
        build_model(cfg)
        check("dynamics_prior without proprio raises", False, "it built silently")
    except Exception as e:                                      # noqa: BLE001
        check("dynamics_prior without proprio raises a NAMED error",
              isinstance(e, ValueError) and "proprio" in str(e).lower(),
              f"{type(e).__name__}: {str(e)[:100]}")


def train_losses(n=STEPS):
    """N forward+loss passes through lit.LitWorldModel._step on FROZEN weights. Returns [loss per batch]."""
    import lightning as L
    from ..environments.registry import make_env
    from ..training.lit import LitWorldModel
    from ..training.setup import build_model, env_cfg, normalizer, window_loaders
    # BATCH 8, NOT THE DEPLOYED 22. This gate detects whether the refactor changed the training
    # COMPUTATION, which needs the same batch before and after -- not the same batch as production. 22
    # sits at 92.9/94GB on this card and OOMs the moment anything else touches the GPU, and a smoke test
    # you cannot afford to run is not a smoke test.
    cfg = _cfg(None, **{"data.batch": 8})
    # THE PROCESS-WIDE DATASET SETTERS the real entrypoints call. Without them the loader yields raw
    # 4-dim actions while the normalizer was fitted on the 16-dim concatenated ones, and the first batch
    # dies in transforms.apply on a shape mismatch.
    from ..data.dataset import set_action_aggregate, set_subsample
    set_subsample(int(cfg.data.get("subsample", 1) or 1))
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    L.seed_everything(int(cfg.get("seed", 0) or 0), workers=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm, e = normalizer(cfg), env_cfg(cfg)
    model = build_model(cfg).to(dev)
    lit = LitWorldModel(model, norm, e.R, e.r, e.init_speed, cfg.data.P, cfg.data.F,
                        cfg.model.p_tf_start, cfg.model.p_tf_end, cfg.model.p_tf_warmup_epochs,
                        cfg.optim.lr, cfg.optim.weight_decay, cfg.model.detach_every,
                        variations=cfg.get("variations"), dt=e.dt,
                        recon_frac=float(cfg.model.get("recon_frac", 1.0)),
                        lr_warmup_steps=int(cfg.optim.get("lr_warmup_steps", 0)),
                        env=make_env(cfg.environments.get("name", "torus_world"), cfg.environments, 1)).to(dev)
    # LIGHTNING SCAFFOLDING, stubbed rather than instantiated. `_step` calls self.log (needs a Trainer)
    # and `_cur_p_tf` reads current_epoch and trainer.num_training_batches for the batch-granular ramp.
    # A real Trainer would drag in checkpointing, callbacks and the eval cadence -- none of which this
    # gate is measuring -- so the two attributes it actually reads are supplied directly. num_training_batches
    # is pinned to n so the teacher-forcing ramp is a function of the step index and nothing else.
    class _T:
        num_training_batches = float(n)
        current_epoch = 0
        global_step = 0
    lit._trainer = _T()
    lit.log = lambda *a, **k: None
    lit.log_dict = lambda *a, **k: None
    loader = window_loaders(cfg, norm)["train"]
    out = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        lit._batch_idx = i                       # the p_tf ramp reads this, so the schedule still varies
        with torch.no_grad():
            out.append(float(lit._step(batch, "train").detach()))
    return out


def regression():
    print(f"\nTRAINING REGRESSION ({STEPS} steps, tol {TOL})")
    if not os.path.exists(REF):
        check("reference exists", False, f"{REF} missing — run with --capture first")
        return
    ref = json.load(open(REF))["losses"]
    got = train_losses(len(ref))
    check("step count matches", len(got) == len(ref), f"{len(got)} vs {len(ref)}")
    n = min(len(got), len(ref))
    dev = [abs(got[i] - ref[i]) for i in range(n)]
    worst = max(dev) if dev else 0.0
    check(f"every step within {TOL}", worst <= TOL,
          f"worst |Δ| {worst:.2e} at step {dev.index(worst) if dev else -1}; "
          f"first {got[0]:.6f} vs {ref[0]:.6f}, last {got[n-1]:.6f} vs {ref[n-1]:.6f}")


def _capture():
    print(f"capturing {STEPS}-step reference -> {REF}")
    losses = train_losses(STEPS)
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    json.dump({"steps": len(losses), "recipe": "vl128_starling / starling2 / sub4 / concat / batch8 / seed0",
               "losses": losses}, open(REF, "w"), indent=1)
    print(f"  wrote {len(losses)} steps; first {losses[0]:.6f} last {losses[-1]:.6f}")


if __name__ == "__main__":
    if "--capture" in sys.argv:
        _capture(); raise SystemExit(0)
    matrix()
    regression()
    print(f"\n{ok} passed, {bad} failed")
    raise SystemExit(1 if bad else 0)
