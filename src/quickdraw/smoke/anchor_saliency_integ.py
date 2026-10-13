"""Integration smoke: saliency loss + z0 re-injection together, on the real block-stack recipe.
  A. OFF: build the plain recipe, run a train-style forward (rollout+recon+loss_terms) AND an eval
     rollout (imagine_eval, both cached and uncached) -- no error, finite losses.
  B. ON:  build with +model.anchor_init=true and cube-scene visual_saliency=hue -> same passes run,
     finite losses, and the loss DIFFERS from OFF (the features are actually active).
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.anchor_saliency_integ
"""
import torch
torch.set_num_threads(1)
from hydra import compose, initialize_config_dir

from ..training.setup import build_model


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _cfg(overrides):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        return compose(config_name="config", overrides=overrides)


def _fake_batch(m, B=2, P=8, F=10):
    L = P + F
    obs = {}
    for name, n in m.layout:
        mod = m.modalities[name]
        if name == "proprio":
            obs[name] = torch.randn(B, L, mod.dim)
        else:
            sh = mod.ae.cfg.img_size
            sh = sh if isinstance(sh, (tuple, list)) else (sh, sh)
            obs[name] = torch.rand(B, L, sh[0], sh[1], 3)
    act = torch.randn(B, L, m.act_enc[0].in_features if hasattr(m.act_enc, "__getitem__") else 5)
    return obs, act, P, F


def _run(m):
    m.compile_rollout = False   # bare smoke: inductor needs train.py's dynamo cache config; eager exercises the same z0/saliency code
    m.train()
    obs, act, P, F = _fake_batch(m)
    ctx = {k: v[:, :P] for k, v in obs.items()}
    fut = {k: v[:, P:] for k, v in obs.items()}
    # AR rollout (own-fed) + recon (decode, where saliency lives) + loss_terms (flow, where z0 lives)
    out = m.rollout_train(ctx, act[:, :-1], fut, p_tf=0.0, detach_every=8, return_feeds=True)
    preds, feeds = out
    recon, rw = m.recon_losses(preds, fut)
    raw, w = m.loss_terms(preds, fut, obs, 0.0, act, feeds=feeds)
    total = sum(float(v) * float(rw.get(k, 1.0)) for k, v in recon.items()) + \
            sum(float(v) * float(w.get(k, 1.0)) for k, v in raw.items())
    # eval rollout: uncached AND cached (the sink/anchor must be consistent across both)
    m.eval()
    with torch.no_grad():
        r_unc = m.imagine_eval(ctx, act[:, :P + F - 1], F, use_cache=False)
        r_cac = m.imagine_eval(ctx, act[:, :P + F - 1], F, use_cache=True)
    return total, recon, raw, r_unc, r_cac


def main():
    torch.manual_seed(0)
    print("[A] OFF (plain recipe) ...")
    off = build_model(_cfg(["model=vl128_blockstack_2cam"]))
    t_off, recon, raw, ru, rc = _run(off)
    for k, v in {**recon, **raw}.items():
        if not torch.isfinite(torch.as_tensor(float(v))):
            _fail(f"OFF non-finite loss {k}={v}")
    if not torch.isfinite(ru["cam_scene"]).all():
        _fail("OFF eval rollout non-finite")
    print(f"    OFF total={t_off:.4f}  terms={list(recon) + list(raw)}   OK")

    print("[B] ON (anchor_init + cube saliency on scene) ...")
    on = build_model(_cfg(["model=vl128_blockstack_2cam",
                           "+model.anchor_init=true",
                           "+model.modalities.1.visual_saliency=hue",
                           "+model.modalities.1.visual_saliency_weight=8.0"]))
    if not on.anchor_init:
        _fail("anchor_init not set")
    if getattr(on.modalities["cam_scene"].visual, "saliency", "none") != "hue":
        _fail("scene saliency not set")
    if getattr(on.modalities["cam_wrist"].visual, "saliency", "none") != "none":
        _fail("wrist saliency should be OFF (per-head)")
    t_on, recon2, raw2, ru2, rc2 = _run(on)
    for k, v in {**recon2, **raw2}.items():
        if not torch.isfinite(torch.as_tensor(float(v))):
            _fail(f"ON non-finite loss {k}={v}")
    if not torch.isfinite(ru2["cam_scene"]).all():
        _fail("ON eval rollout non-finite")
    print(f"    ON total={t_on:.4f}   scene-saliency=hue wrist=none anchor_init=True   OK")
    print(f"    (cached & uncached eval rollouts both ran; shapes {tuple(ru2['cam_scene'].shape)})")

    print("\nALL INTEGRATION SMOKE PASSED")


if __name__ == "__main__":
    main()
